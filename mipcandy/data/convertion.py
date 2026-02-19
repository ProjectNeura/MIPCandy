import torch

from mipcandy.common import Normalize


def convert_ids_to_logits(ids: torch.Tensor, num_classes: int, *, channel_dim: int = 1) -> torch.Tensor:
    """
    :param ids: class ids (..., 1, ...)
    :param num_classes: number of classes
    :param channel_dim: the index of the channel dimension
    :return: logits (..., num_classes, ...)
    """
    shape = list(ids.shape)
    shape.insert(channel_dim, num_classes)
    logits = torch.zeros(shape, device=ids.device, dtype=torch.float32)
    logits.scatter_(channel_dim, ids.long(), 1)
    return logits


def convert_logits_to_ids(logits: torch.Tensor, *, channel_dim: int = 1) -> torch.Tensor:
    """
    :param logits: logits (..., num_classes, ...)
    :param channel_dim: the index of the channel dimension
    :return: class ids (..., 1, ...)
    """
    return logits.round().int() if logits.shape[channel_dim] < 2 else logits.argmax(channel_dim, keepdim=True)


def auto_convert(image: torch.Tensor) -> torch.Tensor:
    return (image * 255 if 0 <= image.min() <= image.max() <= 1 else Normalize(domain=(0, 255))(image)).int()
