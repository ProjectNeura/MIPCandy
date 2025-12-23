from torch import nn

from mipcandy.types import Transform


class JointTransform(nn.Module):
    def __init__(self, transform: Transform, image_only: Transform | None = None,
                 label_only: Transform | None = None) -> None:
        super().__init__()
        self._transform: Transform = transform
        self._image_only: Transform | None = image_only
        self._label_only: Transform | None = label_only
