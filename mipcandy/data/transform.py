import torch
from torch import nn

from mipcandy.types import Transform


class JointTransform(nn.Module):
    def __init__(self, transform: Transform, *, image_only: Transform | None = None,
                 label_only: Transform | None = None) -> None:
        super().__init__()
        self._transform: Transform = transform
        self._image_only: Transform | None = image_only
        self._label_only: Transform | None = label_only

    def __call__(self, image: torch.Tensor, label: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        data = {"image": image, "label": label}
        data = self._transform(data)
        if self._image_only:
            data["image"] = self._image_only(data["image"])
        if self._label_only:
            data["label"] = self._label_only(data["label"])
        return data["image"], data["label"]
