import torch
from torch import nn

from mipcandy.types import Transform


class JointTransform(nn.Module):
    def __init__(self, *, transform: Transform | None = None, image_only: Transform | None = None,
                 label_only: Transform | None = None, keys: tuple[str, str] = ("image", "label"),
                 order: tuple[str, str, str] = ("transform", "image_only", "label_only")) -> None:
        super().__init__()
        self.transform: Transform | None = transform
        self.image_only: Transform | None = image_only
        self.label_only: Transform | None = label_only
        self._keys: tuple[str, str] = keys
        self._order: tuple[str, str, str] = order

    def forward(self, image: torch.Tensor, label: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        ik, lk = self._keys
        data = {ik: image, lk: label}
        for t in self._order:
            transform = getattr(self, t)
            if not transform:
                continue
            match t:
                case "transform":
                    data = transform(data)
                case "image_only":
                    data[ik] = transform(data[ik])
                case "label_only":
                    data[lk] = transform(data[lk])
        return data[ik], data[lk]


class MONAITransform(nn.Module):
    def __init__(self, transform: Transform, *, keys: tuple[str, str] = ("image", "label")) -> None:
        super().__init__()
        self.transform: Transform = transform
        self._keys: tuple[str, str] = keys

    def forward(self, data: torch.Tensor | dict[str, torch.Tensor]) -> torch.Tensor | dict[str, torch.Tensor]:
        if isinstance(data, torch.Tensor):
            return self.transform(data)
        ik, lk = self._keys
        image, label = data[ik], data[lk]
        return {ik: self.transform(image), lk: self.transform(label)}
