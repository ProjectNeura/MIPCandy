from dataclasses import dataclass
from io import StringIO
from typing import Sequence, override

import torch
from ptflops import get_model_complexity_info
from torch import nn


def num_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_complexity_info(model: nn.Module, example_shape: Sequence[int]) -> tuple[float | None, float | None, str]:
    layer_stats = StringIO()
    macs, params = get_model_complexity_info(model, tuple(example_shape), ost=layer_stats, as_strings=False)
    return macs, params, layer_stats.getvalue()


@dataclass
class SanityCheckResult(object):
    num_macs: float
    num_params: float
    layer_stats: str
    output: torch.Tensor

    @override
    def __str__(self) -> str:
        return f"MACs: {self.num_macs / 1e9:.1f} G / Params: {self.num_params / 1e6:.1f} M"


def sanity_check(model: nn.Module, input_shape: Sequence[int], *,
                 device: torch.device | str | None = None) -> SanityCheckResult:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    num_macs, num_params, layer_stats = model_complexity_info(model, input_shape)
    if num_macs is None or num_params is None:
        raise RuntimeError("Failed to validate model")
    output = model.to(device)(torch.randn(1, *input_shape, device=device))
    return SanityCheckResult(num_macs, num_params, layer_stats, output)
