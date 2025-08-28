from os import PathLike
from typing import Any, Iterable, Sequence

import torch
from torch import nn
from torchvision.transforms import Compose

type Secret = str | int | float | bool | None | dict[str, Secret] | list[Secret]
type Secrets = dict[str, Secret]
type Params = Iterable[torch.Tensor] | Iterable[dict[str, Any]]
type Transform = nn.Module | Compose
type SupportedPredictant = Sequence[torch.Tensor] | str | PathLike[str] | Sequence[str] | torch.Tensor
type Colormap = Sequence[int | tuple[int, int, int]]
