from typing import Protocol, Literal

import torch

from mipcandy.types import Device


def _args_check(output: torch.Tensor, label: torch.Tensor, *, dtype: torch.dtype | None = None,
                device: Device | None = None) -> tuple[torch.dtype, Device]:
    if output.shape != label.shape:
        raise ValueError(f"Output ({output.shape}) and label ({label.shape}) must have the same shape")
    if (output_dtype := output.dtype) != label.dtype or dtype and output_dtype != dtype:
        raise TypeError(f"Output ({output_dtype}) and label ({label.dtype}) must both be {dtype}")
    if (output_device := output.device) != label.device:
        raise RuntimeError(f"Output ({output.device}) and label ({label.device}) must be on the same device")
    if device and output_device != device:
        raise RuntimeError(f"Tensors are expected to be on {device}, but instead they are on {output.device}")
    return output_dtype, output_device


class Metric(Protocol):
    def __call__(self, output: torch.Tensor, label: torch.Tensor, *, if_empty: float = ...) -> torch.Tensor: ...


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


def apply_multiclass_to_binary(metric: Metric, output: torch.Tensor, label: torch.Tensor, num_classes: int | None,
                               if_empty: float, *, reduction: Literal["mean", "sum"] = "mean") -> torch.Tensor:
    _args_check(output, label, dtype=torch.int)
    if not num_classes:
        num_classes = max(output.max().item(), label.max().item())
    if num_classes == 0:
        return torch.tensor(if_empty, dtype=torch.float)
    else:
        x = torch.tensor([metric(output == cls, label == cls, if_empty=if_empty) for cls in range(1, num_classes + 1)])
        return do_reduction(x, reduction)


def dice_similarity_coefficient_binary(output: torch.Tensor, label: torch.Tensor, *,
                                       if_empty: float = 1) -> torch.Tensor:
    _args_check(output, label, dtype=torch.bool)
    volume_sum = output.sum() + label.sum()
    if volume_sum == 0:
        return torch.tensor(if_empty, dtype=torch.float)
    return 2 * (output & label).sum() / volume_sum


def dice_similarity_coefficient_multiclass(output: torch.Tensor, label: torch.Tensor, *, num_classes: int | None = None,
                                           if_empty: float = 1) -> torch.Tensor:
    return apply_multiclass_to_binary(dice_similarity_coefficient_binary, output, label, num_classes, if_empty)


def soft_dice_coefficient(output: torch.Tensor, label: torch.Tensor, *, smooth: float = 1,
                          batch: bool = True) -> torch.Tensor:
    _args_check(output, label)
    axes = tuple(range(2, output.ndim))
    with torch.no_grad():
        label_sum = label.sum(axes)
    intersection = (output * label).sum(axes)
    output_sum = output.sum(axes)
    if batch:
        intersection = intersection.sum(0)
        output_sum = output_sum.sum(0)
        label_sum = label_sum.sum(0)
    dice = (2 * intersection + smooth) / (torch.clip(label_sum + output_sum + smooth, 1e-8))
    return dice.mean()


def accuracy_binary(output: torch.Tensor, label: torch.Tensor, *, if_empty: float = 1) -> torch.Tensor:
    _args_check(output, label, dtype=torch.bool)
    numerator = (output & label).sum() + (~output & ~label).sum()
    denominator = numerator + (output & ~label).sum() + (label & ~output).sum()
    return torch.tensor(if_empty, dtype=torch.float) if denominator == 0 else numerator / denominator


def accuracy_multiclass(output: torch.Tensor, label: torch.Tensor, *, num_classes: int | None = None,
                        if_empty: float = 1) -> torch.Tensor:
    return apply_multiclass_to_binary(accuracy_binary, output, label, num_classes, if_empty)


def _precision_or_recall(output: torch.Tensor, label: torch.Tensor, if_empty: float,
                         is_precision: bool) -> torch.Tensor:
    _args_check(output, label, dtype=torch.bool)
    tp = (output & label).sum()
    denominator = output.sum() if is_precision else label.sum()
    return torch.tensor(if_empty, dtype=torch.float) if denominator == 0 else tp / denominator


def precision_binary(output: torch.Tensor, label: torch.Tensor, *, if_empty: float = 1) -> torch.Tensor:
    return _precision_or_recall(output, label, if_empty, True)


def precision_multiclass(output: torch.Tensor, label: torch.Tensor, *, num_classes: int | None = None,
                         if_empty: float = 1) -> torch.Tensor:
    return apply_multiclass_to_binary(precision_binary, output, label, num_classes, if_empty)


def recall_binary(output: torch.Tensor, label: torch.Tensor, *, if_empty: float = 1) -> torch.Tensor:
    return _precision_or_recall(output, label, if_empty, False)


def recall_multiclass(output: torch.Tensor, label: torch.Tensor, *, num_classes: int | None = None,
                      if_empty: float = 1) -> torch.Tensor:
    return apply_multiclass_to_binary(recall_binary, output, label, num_classes, if_empty)


def iou_binary(output: torch.Tensor, label: torch.Tensor, *, if_empty: float = 1) -> torch.Tensor:
    _args_check(output, label, dtype=torch.bool)
    denominator = (output | label).sum()
    return torch.tensor(if_empty, dtype=torch.float) if denominator == 0 else (output & label).sum() / denominator


def iou_multiclass(output: torch.Tensor, label: torch.Tensor, *, num_classes: int | None = None,
                   if_empty: float = 1) -> torch.Tensor:
    return apply_multiclass_to_binary(iou_binary, output, label, num_classes, if_empty)
