import torch

from mipcandy.types import Device, Reduction


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


def do_reduction(x: torch.Tensor, method: Reduction) -> torch.Tensor:
    match method:
        case "mean":
            return x.mean()
        case "median":
            return x.median()
        case "sum":
            return x.sum()
        case "none":
            return x


def binary_dice(outputs: torch.Tensor, labels: torch.Tensor, *, if_empty: float = 1,
                reduction: Reduction = "mean") -> torch.Tensor:
    """
    :param outputs: boolean class ids (B, 1, ...)
    :param labels: boolean class ids (B, 1, ...)
    :param if_empty: the value to return if both outputs and labels are empty
    :param reduction: the reduction method to apply to the dice score
    """
    _args_check(outputs, labels, dtype=torch.bool)
    axes = tuple(range(2, outputs.ndim))
    volume_sum = outputs.sum(axes) + labels.sum(axes)
    if volume_sum == 0:
        return torch.tensor(if_empty, dtype=torch.float)
    return do_reduction(2 * (outputs & labels).sum(axes) / volume_sum, reduction)


def dice_similarity_coefficient(outputs: torch.Tensor, labels: torch.Tensor, *, if_empty: float = 1,
                                reduction: Reduction = "mean") -> torch.Tensor:
    """
    :param outputs: one-hot (B, N, ...)
    :param labels: one-hot (B, N, ...)
    :param if_empty: the value to return if both outputs and labels are empty
    :param reduction: the reduction method to apply to the dice score
    """
    _args_check(outputs, labels, dtype=torch.float)
    axes = tuple(range(2, outputs.ndim))
    tp = (outputs * labels).sum(axes)
    fp = (outputs * (1 - labels)).sum(axes)
    fn = ((1 - outputs) * labels).sum(axes)
    volume_sum = 2 * tp + fp + fn
    if (volume_sum == 0).any():
        return torch.tensor(if_empty, dtype=torch.float)
    return do_reduction(2 * tp / volume_sum, reduction)


def soft_dice(outputs: torch.Tensor, labels: torch.Tensor, *, smooth: float = 1, batch_dice: bool = True,
              reduction: Reduction = "mean") -> torch.Tensor:
    """
    :param outputs: logits (B, C, ...)
    :param labels: logits (B, C, ...)
    :param smooth: the smoothness term to avoid division by zero
    :param batch_dice: whether to compute dice score for each batch separately
    :param reduction: the reduction method to apply to the dice score
    """
    _args_check(outputs, labels, dtype=torch.float)
    axes = tuple(range(2, outputs.ndim))
    if batch_dice:
        axes = (0,) + axes
    label_sum = labels.sum(axes)
    intersection = (outputs * labels).sum(axes)
    output_sum = outputs.sum(axes)
    return do_reduction((2 * intersection + smooth) / (label_sum + output_sum + smooth), reduction)
