from typing import Sequence

import torch
from torch import nn

from mipcandy.training import Trainer


def num_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_complexity_info(model: nn.Module, example_shape: tuple[int, ...]) -> tuple[float, float]:
    return Trainer.model_complexity_info(model, example_shape)


def sanity_check(model: nn.Module, input_shape: Sequence[int], *, device: str = "cpu") -> torch.Tensor:
    return model.to(device)(torch.randn(*input_shape).to(device))
