from typing import Sequence

import torch
from torch import nn


def num_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def sanity_check(model: nn.Module, input_shape: Sequence[int], *, device: str = "cpu") -> torch.Tensor:
    return model.to(device)(torch.randn(*input_shape).to(device))
