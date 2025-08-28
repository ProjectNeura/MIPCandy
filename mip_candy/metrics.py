from typing import Protocol, Literal

import torch

try:
    from cupy import from_dlpack as _dlpack2np
    from cupyx.scipy.ndimage import distance_transform_edt as _distance_transform_edt
except ImportError:
    from numpy import from_dlpack as _dlpack2np
    from scipy.ndimage import distance_transform_edt as _distance_transform_edt


def _args_check(mask: torch.Tensor, label: torch.Tensor, *, dtype: torch.dtype | None = None,
                device: torch.device | None = None) -> tuple[torch.dtype, torch.device]:
    if mask.shape != label.shape:
        raise ValueError(f"Mask ({mask.shape}) and label ({label.shape}) must have the same shape")
    if (mask_dtype := mask.dtype) != label.dtype or dtype and mask_dtype != dtype:
        raise TypeError(f"Mask ({mask_dtype}) and label ({label.dtype}) must both be {dtype}")
    if (mask_device := mask.device) != label.device:
        raise RuntimeError(f"Mask ({mask.device}) and label ({label.device}) must be on the same device")
    if device and mask_device != device:
        raise RuntimeError(f"Tensors are expected to be on {device}, but instead they are on {mask.device}")
    return mask_dtype, mask_device


class Metric(Protocol):
    def __call__(self, mask: torch.Tensor, label: torch.Tensor, *, if_empty: float = ...) -> torch.Tensor: ...


def do_reduction(x: torch.Tensor, method: Literal["mean", "median", "sum", "none"] = "mean") -> torch.Tensor:
    match method:
        case "mean":
            return x.mean()
        case "median":
            return x.median()
        case "sum":
            return x.sum()
        case "none":
            return x


def apply_multiclass_to_binary(metric: Metric, mask: torch.Tensor, label: torch.Tensor, num_classes: int | None,
                               if_empty: float, *, reduction: Literal["mean", "sum"] = "mean") -> torch.Tensor:
    _args_check(mask, label, dtype=torch.int)
    if not num_classes:
        num_classes = max(mask.max().item(), label.max().item())
    if num_classes == 0:
        return torch.tensor(if_empty, dtype=torch.float)
    else:
        x = torch.tensor([metric(mask == cls, label == cls, if_empty=if_empty) for cls in range(1, num_classes + 1)])
        return do_reduction(x, reduction)


def dice_similarity_coefficient_binary(mask: torch.Tensor, label: torch.Tensor, *, if_empty: float = 1) -> torch.Tensor:
    _args_check(mask, label, dtype=torch.bool)
    volume_sum = mask.sum() + label.sum()
    if volume_sum == 0:
        return torch.tensor(if_empty, dtype=torch.float)
    return 2 * (mask & label).sum() / volume_sum


def dice_similarity_coefficient_multiclass(mask: torch.Tensor, label: torch.Tensor, *, num_classes: int | None = None,
                                           if_empty: float = 1) -> torch.Tensor:
    return apply_multiclass_to_binary(dice_similarity_coefficient_binary, mask, label, num_classes, if_empty)


def soft_dice_coefficient(mask: torch.Tensor, label: torch.Tensor, *, smooth: float = 1e-5) -> torch.Tensor:
    _args_check(mask, label)
    num = label.size(0)
    mask = mask.view(num, -1)
    label = label.view(num, -1)
    intersection = (mask * label)
    dice = (2 * intersection.sum(1) + smooth) / (mask.sum(1) + label.sum(1) + smooth)
    return dice.sum() / num


def accuracy_binary(mask: torch.Tensor, label: torch.Tensor, *, if_empty: float = 1) -> torch.Tensor:
    _args_check(mask, label, dtype=torch.bool)
    numerator = (mask & label).sum() + (~mask & ~label).sum()
    denominator = numerator + (mask & ~label).sum() + (label & ~mask).sum()
    return torch.tensor(if_empty, dtype=torch.float) if denominator == 0 else numerator / denominator


def accuracy_multiclass(mask: torch.Tensor, label: torch.Tensor, *, num_classes: int | None = None,
                        if_empty: float = 1) -> torch.Tensor:
    return apply_multiclass_to_binary(accuracy_binary, mask, label, num_classes, if_empty)


def _precision_or_recall(mask: torch.Tensor, label: torch.Tensor, if_empty: float, is_precision: bool) -> torch.Tensor:
    _args_check(mask, label, dtype=torch.bool)
    tp = (mask & label).sum()
    denominator = mask.sum() if is_precision else label.sum()
    return torch.tensor(if_empty, dtype=torch.float) if denominator == 0 else tp / denominator


def precision_binary(mask: torch.Tensor, label: torch.Tensor, *, if_empty: float = 1) -> torch.Tensor:
    return _precision_or_recall(mask, label, if_empty, True)


def precision_multiclass(mask: torch.Tensor, label: torch.Tensor, *, num_classes: int | None = None,
                         if_empty: float = 1) -> torch.Tensor:
    return apply_multiclass_to_binary(precision_binary, mask, label, num_classes, if_empty)


def recall_binary(mask: torch.Tensor, label: torch.Tensor, *, if_empty: float = 1) -> torch.Tensor:
    return _precision_or_recall(mask, label, if_empty, False)


def recall_multiclass(mask: torch.Tensor, label: torch.Tensor, *, num_classes: int | None = None,
                      if_empty: float = 1) -> torch.Tensor:
    return apply_multiclass_to_binary(recall_binary, mask, label, num_classes, if_empty)


def iou_binary(mask: torch.Tensor, label: torch.Tensor, *, if_empty: float = 1) -> torch.Tensor:
    _args_check(mask, label, dtype=torch.bool)
    denominator = (mask | label).sum()
    return torch.tensor(if_empty, dtype=torch.float) if denominator == 0 else (mask & label).sum() / denominator


def iou_multiclass(mask: torch.Tensor, label: torch.Tensor, *, num_classes: int | None = None,
                   if_empty: float = 1) -> torch.Tensor:
    return apply_multiclass_to_binary(iou_binary, mask, label, num_classes, if_empty)
