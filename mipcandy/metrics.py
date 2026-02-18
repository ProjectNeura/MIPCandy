from typing import Protocol, Literal

import torch

from mipcandy.types import Device


def _args_check(outputs: torch.Tensor, labels: torch.Tensor, *, dtype: torch.dtype | None = None,
                device: Device | None = None) -> tuple[torch.dtype, Device]:
    if outputs.shape != labels.shape:
        raise ValueError(f"Outputs ({outputs.shape}) and labels ({labels.shape}) must have the same shape")
    if (outputs_dtype := outputs.dtype) != labels.dtype or dtype and outputs_dtype != dtype:
        raise TypeError(f"Outputs({outputs_dtype}) and labels ({labels.dtype}) must both be {dtype}")
    if (outputs_device := outputs.device) != labels.device:
        raise RuntimeError(f"Outputs ({outputs.device}) and labels ({labels.device}) must be on the same device")
    if device and outputs_device != device:
        raise RuntimeError(f"Tensors are expected to be on {device}, but instead they are on {outputs.device}")
    return outputs_dtype, outputs_device


class Metric(Protocol):
    def __call__(self, outputs: torch.Tensor, labels: torch.Tensor, *, if_empty: float = ...) -> torch.Tensor:
        """
        :param outputs: prediction of shape (B, C, ...)
        :param labels: ground truth of shape (B, C, ...)
        :param if_empty: the value to return if both outputs and labels are empty
        """
        ...


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


def apply_multiclass_to_binary(metric: Metric, outputs: torch.Tensor, labels: torch.Tensor, num_classes: int | None,
                               if_empty: float, *, reduction: Literal["mean", "sum"] = "mean") -> torch.Tensor:
    _args_check(outputs, labels, dtype=torch.int)
    if not num_classes:
        num_classes = max(outputs.max().item(), labels.max().item())
    if num_classes == 0:
        return torch.tensor(if_empty, dtype=torch.float)
    else:
        x = torch.tensor(
            [metric(outputs == cls, labels == cls, if_empty=if_empty) for cls in range(1, num_classes + 1)])
        return do_reduction(x, reduction)


def dice_similarity_coefficient_binary(outputs: torch.Tensor, labels: torch.Tensor, *,
                                       if_empty: float = 1) -> torch.Tensor:
    _args_check(outputs, labels, dtype=torch.bool)
    volume_sum = outputs.sum() + labels.sum()
    if volume_sum == 0:
        return torch.tensor(if_empty, dtype=torch.float)
    return 2 * (outputs & labels).sum() / volume_sum


def dice_similarity_coefficient_multiclass(outputs: torch.Tensor, labels: torch.Tensor, *,
                                           num_classes: int | None = None, if_empty: float = 1) -> torch.Tensor:
    return apply_multiclass_to_binary(dice_similarity_coefficient_binary, outputs, labels, num_classes, if_empty)


def _dice_with_logits(outputs: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    _args_check(outputs, labels, dtype=torch.float)
    axes = tuple(range(2, outputs.ndim))
    tp = (outputs * labels).sum(axes)
    fp = (outputs * (1 - labels)).sum(axes)
    fn = ((1 - outputs) * labels).sum(axes)
    return tp, 2 * tp + fp + fn


def dice_similarity_coefficient_with_logits(outputs: torch.Tensor, labels: torch.Tensor, *,
                                            if_empty: float = 1) -> torch.Tensor:
    tp, volume_sum = _dice_with_logits(outputs, labels)
    if (volume_sum == 0).any():
        return torch.tensor(if_empty, dtype=torch.float)
    dice = 2 * tp / volume_sum
    return dice.mean()


def dice_similarity_coefficient_with_logits_clip(outputs: torch.Tensor, labels: torch.Tensor, *,
                                                 clip_min: float = 1e-8) -> torch.Tensor:
    tp, volume_sum = _dice_with_logits(outputs, labels)
    dice = 2 * tp / torch.clip(volume_sum, clip_min)
    return dice.mean()


def soft_dice_coefficient(outputs: torch.Tensor, labels: torch.Tensor, *, smooth: float = 1, clip_min: float = 1e-8,
                          batch_dice: bool = True) -> torch.Tensor:
    _args_check(outputs, labels)
    axes = tuple(range(2, outputs.ndim))
    if batch_dice:
        axes = (0,) + axes
    label_sum = labels.sum(axes)
    intersection = (outputs * labels).sum(axes)
    output_sum = outputs.sum(axes)
    if batch_dice:
        intersection = intersection.sum(0)
        output_sum = output_sum.sum(0)
        label_sum = label_sum.sum(0)
    dice = (2 * intersection + smooth) / torch.clip(label_sum + output_sum + smooth, clip_min)
    return dice.mean()


def accuracy_binary(outputs: torch.Tensor, labels: torch.Tensor, *, if_empty: float = 1) -> torch.Tensor:
    _args_check(outputs, labels, dtype=torch.bool)
    numerator = (outputs & labels).sum() + (~outputs & ~labels).sum()
    denominator = numerator + (outputs & ~labels).sum() + (labels & ~outputs).sum()
    return torch.tensor(if_empty, dtype=torch.float) if denominator == 0 else numerator / denominator


def accuracy_multiclass(outputs: torch.Tensor, labels: torch.Tensor, *, num_classes: int | None = None,
                        if_empty: float = 1) -> torch.Tensor:
    return apply_multiclass_to_binary(accuracy_binary, outputs, labels, num_classes, if_empty)


def _precision_or_recall(outputs: torch.Tensor, labels: torch.Tensor, if_empty: float,
                         is_precision: bool) -> torch.Tensor:
    _args_check(outputs, labels, dtype=torch.bool)
    tp = (outputs & labels).sum()
    denominator = outputs.sum() if is_precision else labels.sum()
    return torch.tensor(if_empty, dtype=torch.float) if denominator == 0 else tp / denominator


def precision_binary(outputs: torch.Tensor, labels: torch.Tensor, *, if_empty: float = 1) -> torch.Tensor:
    return _precision_or_recall(outputs, labels, if_empty, True)


def precision_multiclass(outputs: torch.Tensor, labels: torch.Tensor, *, num_classes: int | None = None,
                         if_empty: float = 1) -> torch.Tensor:
    return apply_multiclass_to_binary(precision_binary, outputs, labels, num_classes, if_empty)


def recall_binary(outputs: torch.Tensor, labels: torch.Tensor, *, if_empty: float = 1) -> torch.Tensor:
    return _precision_or_recall(outputs, labels, if_empty, False)


def recall_multiclass(outputs: torch.Tensor, labels: torch.Tensor, *, num_classes: int | None = None,
                      if_empty: float = 1) -> torch.Tensor:
    return apply_multiclass_to_binary(recall_binary, outputs, labels, num_classes, if_empty)


def iou_binary(outputs: torch.Tensor, labels: torch.Tensor, *, if_empty: float = 1) -> torch.Tensor:
    _args_check(outputs, labels, dtype=torch.bool)
    denominator = (outputs | labels).sum()
    return torch.tensor(if_empty, dtype=torch.float) if denominator == 0 else (outputs & labels).sum() / denominator


def iou_multiclass(outputs: torch.Tensor, labels: torch.Tensor, *, num_classes: int | None = None,
                   if_empty: float = 1) -> torch.Tensor:
    return apply_multiclass_to_binary(iou_binary, outputs, labels, num_classes, if_empty)
