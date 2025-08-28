from typing import Literal

import torch
from torch import nn

from mip_candy.metrics import do_reduction, soft_dice_coefficient


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


class DiceBCELossWithLogits(nn.Module):
    def __init__(self, num_classes: int, *, lambda_bce: float = .5, lambda_soft_dice: float = 1,
                 smooth: float = 1e-5) -> None:
        super().__init__()
        self.num_classes: int = num_classes
        self.lambda_bce: float = lambda_bce
        self.lambda_soft_dice: float = lambda_soft_dice
        self.smooth: float = smooth

    def forward(self, masks: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        labels = labels.float()
        bce = nn.functional.binary_cross_entropy_with_logits(masks, labels)
        masks = masks.sigmoid()
        soft_dice = soft_dice_coefficient(masks, labels, smooth=self.smooth)
        c = self.lambda_bce * bce + self.lambda_soft_dice * (1 - soft_dice)
        return c, {"soft dice": soft_dice.item(), "bce loss": bce.item()}
