from typing import Literal

import torch
from torch import nn

from mipcandy.data import convert_ids_to_logits, convert_logits_to_ids
from mipcandy.metrics import do_reduction, soft_dice_coefficient, dice_similarity_coefficient_binary, \
    dice_similarity_coefficient_with_logits


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
    def __init__(self) -> None:
        super().__init__()
        self.validation_mode: bool = False


class _SegmentationLoss(_Loss):
    def __init__(self, num_classes: int, include_background: bool) -> None:
        super().__init__()
        self.num_classes: int = num_classes
        self.include_background: bool = include_background

    def logitfy_no_grad(self, ids: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            if self.num_classes != 1 and ids.shape[1] == 1:
                if (d := ids.ndim - 2) not in (1, 2, 3):
                    raise ValueError(f"Expected labels to be 1D, 2D, or 3D, got {d} spatial dimensions")
                return convert_ids_to_logits(ids.int(), d, self.num_classes)
        return ids.float()

    def forward(self, outputs: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        if not self.validation_mode:
            return self._forward(outputs, labels)
        with torch.no_grad():
            c, metrics = self._forward(outputs, labels)
            outputs = convert_logits_to_ids(outputs)
            dice = 0
            for i in range(0 if self.include_background else 1, self.num_classes):
                class_dice = dice_similarity_coefficient_binary(outputs == i, labels == i).item()
                dice += class_dice
                metrics[f"dice {i}"] = class_dice
            outputs = self.logitfy_no_grad(outputs)
            metrics["dice"] = dice_similarity_coefficient_with_logits(outputs, labels).item()
            return c, metrics


class DiceCELossWithLogits(_SegmentationLoss):
    def __init__(self, num_classes: int, *, lambda_ce: float = 1, lambda_soft_dice: float = 1,
                 smooth: float = 1e-5, include_background: bool = True,
                 min_percentage_per_class: float | None = None) -> None:
        super().__init__(num_classes, include_background)
        self.lambda_ce: float = lambda_ce
        self.lambda_soft_dice: float = lambda_soft_dice
        self.smooth: float = smooth
        self.min_percentage_per_class: float | None = min_percentage_per_class

    def _forward(self, outputs: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        ce = nn.functional.cross_entropy(outputs, labels[:, 0].long())
        outputs = outputs.softmax(1)
        labels = self.logitfy_no_grad(labels)
        if not self.include_background:
            outputs = outputs[:, 1:]
            labels = labels[:, 1:]
        soft_dice = soft_dice_coefficient(outputs, labels, smooth=self.smooth,
                                          min_percentage_per_class=self.min_percentage_per_class)
        metrics = {"soft dice": soft_dice.item(), "ce loss": ce.item()}
        c = self.lambda_ce * ce + self.lambda_soft_dice * (1 - soft_dice)
        return c, metrics


class DiceBCELossWithLogits(_SegmentationLoss):
    def __init__(self, *, lambda_bce: float = 1, lambda_soft_dice: float = 1,
                 smooth: float = 1e-5, min_percentage_per_class: float | None = None) -> None:
        super().__init__(1, True)
        self.lambda_bce: float = lambda_bce
        self.lambda_soft_dice: float = lambda_soft_dice
        self.smooth: float = smooth
        self.min_percentage_per_class: float | None = min_percentage_per_class

    def _forward(self, outputs: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        outputs = outputs.sigmoid()
        labels = labels.float()
        bce = nn.functional.binary_cross_entropy(outputs, labels)
        soft_dice = soft_dice_coefficient(outputs, labels, smooth=self.smooth,
                                          min_percentage_per_class=self.min_percentage_per_class)
        metrics = {"soft dice": soft_dice.item(), "bce loss": bce.item()}
        c = self.lambda_bce * bce + self.lambda_soft_dice * (1 - soft_dice)
        return c, metrics
