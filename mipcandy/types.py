from importlib.util import find_spec
from os import PathLike
from typing import Any, Iterable, Sequence

import torch
from torch import nn
from torchvision.transforms import Compose

type Setting = str | int | float | bool | None | dict[str, Setting] | list[Setting]
type Settings = dict[str, Setting]
type Params = Iterable[torch.Tensor] | Iterable[dict[str, Any]]
if find_spec("monai"):
    from monai.transforms import Transform as _Transform

    type Transform = nn.Module | Compose | _Transform
else:
    type Transform = nn.Module | Compose
type SupportedPredictant = Sequence[torch.Tensor] | str | PathLike[str] | Sequence[str] | torch.Tensor
type Colormap = Sequence[int | tuple[int, int, int]]
type Device = torch.device | str
type Shape2d = tuple[int, int]
type Shape3d = tuple[int, int, int]
type Shape = Shape2d | Shape3d
type AmbiguousShape = tuple[int, ...]
type Paddings2d = tuple[int, int, int, int]
type Paddings3d = tuple[int, int, int, int, int, int]
type Paddings = Paddings2d | Paddings3d
