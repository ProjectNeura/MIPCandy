from dataclasses import dataclass
from typing import Sequence

import torch
from torch import nn

from mipcandy.training import Trainer


def num_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_complexity_info(model: nn.Module, example_shape: Sequence[int]) -> tuple[float, float, str]:
    return Trainer.model_complexity_info(model, example_shape)


@dataclass
class SanityCheckResult(object):
    macs: float
    params: float
    layer_stats: str
    output: torch.Tensor


def sanity_check(model: nn.Module, input_shape: Sequence[int], *,
                 device: torch.device | str | None = None) -> SanityCheckResult:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    macs, params, layer_stats = model_complexity_info(model, input_shape)
    output = model.to(device)(torch.randn(1, *input_shape, device=device))
    return SanityCheckResult(macs, params, layer_stats, output)
