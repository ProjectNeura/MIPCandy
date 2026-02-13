from typing import Literal

import torch
from torch import nn

from mipcandy.data import convert_ids_to_logits, convert_logits_to_ids
from mipcandy.metrics import do_reduction, soft_dice_coefficient, dice_similarity_coefficient_binary


class FocalBCEWithLogits(nn.Module):
    def __init__(self, alpha: float, gamma: float, *, reduction: Literal["mean", "sum", "none"] = "mean") -> None:
        super().__init__()
        self.alpha: float = alpha
        self.gamma: float = gamma
        self.reduction: Literal["mean", "sum", "none"] = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p = torch.sigmoid(logits)
        p_t = torch.where(targets.bool(), p, 1 - p)
        alpha_t = torch.where(targets.bool(), torch.as_tensor(self.alpha, device=logits.device), torch.as_tensor(
            1 - self.alpha, device=logits.device))
        loss = alpha_t * (1 - p_t).pow(self.gamma) * bce
        return do_reduction(loss, self.reduction)


class _Loss(nn.Module):
    def __init__(self, include_background: bool) -> None:
        super().__init__()
        self.validation_mode: bool = False
        self.include_background: bool = include_background

    def forward(self, masks: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        if not self.validation_mode:
            return self._forward(masks, labels)
        with torch.no_grad():
            c, metrics = self._forward(masks, labels)
            masks = convert_logits_to_ids(masks)
            dice = 0
            for i in range(0 if self.include_background else 1, self.num_classes):
                class_dice = dice_similarity_coefficient_binary(masks == i, labels == i).item()
                dice += class_dice
                metrics[f"dice {i}"] = class_dice
            metrics["dice"] = dice / (self.num_classes - (0 if self.include_background else 1))
            return c, metrics


class _SegmentationLoss(_Loss):
    def __init__(self, num_classes: int, include_background: bool) -> None:
        super().__init__(include_background)
        self.num_classes: int = num_classes

    def logitfy(self, labels: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            if self.num_classes != 1 and labels.shape[1] == 1:
                d = labels.ndim - 2
                if d not in (1, 2, 3):
                    raise ValueError(f"Expected labels to be 1D, 2D, or 3D, got {d} spatial dimensions")
                return convert_ids_to_logits(labels.int(), d, self.num_classes)
        return labels.float()


class DiceCELossWithLogits(_SegmentationLoss):
    def __init__(self, num_classes: int, *, lambda_ce: float = 1, lambda_soft_dice: float = 1,
                 smooth: float = 1e-5, include_background: bool = True) -> None:
        super().__init__(num_classes, include_background)
        self.lambda_ce: float = lambda_ce
        self.lambda_soft_dice: float = lambda_soft_dice
        self.smooth: float = smooth

    def _forward(self, masks: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        labels = self.logitfy(labels)
        ce = nn.functional.cross_entropy(masks, labels)
        if not self.include_background:
            masks = masks[:, 1:]
            labels = labels[:, 1:]
        masks = masks.softmax(1)
        soft_dice = soft_dice_coefficient(masks, labels, smooth=self.smooth)
        metrics = {"soft dice": soft_dice.item(), "ce loss": ce.item()}
        c = self.lambda_ce * ce + self.lambda_soft_dice * (1 - soft_dice)
        return c, metrics


class DiceBCELossWithLogits(_SegmentationLoss):
    def __init__(self, *, lambda_bce: float = 1, lambda_soft_dice: float = 1,
                 smooth: float = 1e-5, include_background: bool = True) -> None:
        super().__init__(1, include_background)
        self.lambda_bce: float = lambda_bce
        self.lambda_soft_dice: float = lambda_soft_dice
        self.smooth: float = smooth

    def _forward(self, masks: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        labels = self.logitfy(labels)
        if not self.include_background:
            masks = masks[:, 1:]
            labels = labels[:, 1:]
        bce = nn.functional.binary_cross_entropy(masks, labels)
        masks.sigmoid_()
        soft_dice = soft_dice_coefficient(masks, labels, smooth=self.smooth)
        metrics = {"soft dice": soft_dice.item(), "bce loss": bce.item()}
        c = self.lambda_bce * bce + self.lambda_soft_dice * (1 - soft_dice)
        return c, metrics
