from typing import Literal

import torch


def ensure_num_dimensions(x: torch.Tensor, num_dimensions: int) -> torch.Tensor:
    d = num_dimensions - x.ndim
    if d == 0:
        return x
    return x.reshape(*((1,) * d + x.shape)) if d > 0 else x.reshape(x.shape[-num_dimensions:])


def orthographic_views(x: torch.Tensor, reduction: Literal["mean", "sum"] = "mean") -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor]:
    match reduction:
        case "mean":
            return x.mean(dim=-3), x.mean(dim=-2), x.mean(dim=-1)
        case "sum":
            return x.sum(dim=-3), x.sum(dim=-2), x.sum(dim=-1)


def aggregate_orthographic_views(d: torch.Tensor, h: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    d, h, w = d.unsqueeze(-3), h.unsqueeze(-2), w.unsqueeze(-1)
    return d * h * w


def crop(t: torch.Tensor, bbox: tuple[int, ...]) -> torch.Tensor:
    return t[:, :, bbox[0]:bbox[1], bbox[2]:bbox[3]] if len(bbox) == 4 else t[:, :, bbox[0]:bbox[1], bbox[2]:bbox[3],
    bbox[4]:bbox[5]]
