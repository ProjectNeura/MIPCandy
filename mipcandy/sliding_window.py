from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn

from mipcandy.layer import HasDevice
from mipcandy.types import Shape


def _extract_3d_windows_kernel(t: torch.Tensor, sd: int, sh: int, sw: int,
                               kd: int, kh: int, kw: int, num_windows: int) -> torch.Tensor:
    """Optimized kernel for 3D window extraction - can be compiled with torch.compile"""
    b, c, d, h, w = t.shape
    windows = torch.empty((num_windows, b, c, kd, kh, kw), device=t.device, dtype=t.dtype)

    idx = 0
    for z in range(0, d - kd + 1, sd):
        for y in range(0, h - kh + 1, sh):
            for x in range(0, w - kw + 1, sw):
                windows[idx] = t[:, :, z:z + kd, y:y + kh, x:x + kw]
                idx += 1

    return windows


def _reconstruct_3d_windows_kernel(t_weighted: torch.Tensor, w3d: torch.Tensor,
                                   d: int, h: int, w: int, sd: int, sh: int, sw: int,
                                   kd: int, kh: int, kw: int, b: int, c: int,
                                   dtype: torch.dtype, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Optimized kernel for 3D window reconstruction - can be compiled with torch.compile"""
    canvas = torch.zeros((b, c, d, h, w), dtype=dtype, device=device)
    acc_w = torch.zeros((b, 1, d, h, w), dtype=dtype, device=device)

    idx = 0
    for z in range(0, d - kd + 1, sd):
        for y in range(0, h - kh + 1, sh):
            for x in range(0, w - kw + 1, sw):
                canvas[:, :, z:z + kd, y:y + kh, x:x + kw] += t_weighted[idx]
                acc_w[:, :, z:z + kd, y:y + kh, x:x + kw] += w3d
                idx += 1

    return canvas, acc_w


@dataclass
class SWMetadata(object):
    kernel: Shape
    stride: tuple[int, int] | tuple[int, int, int]
    ndim: Literal[2, 3]
    batch_size: int
    out_size: Shape
    n: int


class SlidingWindow(HasDevice, metaclass=ABCMeta):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Try to compile the kernels for better performance
        try:
            self._extract_3d_compiled = torch.compile(_extract_3d_windows_kernel, mode="max-autotune")
            self._reconstruct_3d_compiled = torch.compile(_reconstruct_3d_windows_kernel, mode="max-autotune")
        except Exception:
            # Fallback to non-compiled versions if compilation fails
            self._extract_3d_compiled = _extract_3d_windows_kernel
            self._reconstruct_3d_compiled = _reconstruct_3d_windows_kernel

    @abstractmethod
    def get_window_shape(self) -> Shape:
        raise NotImplementedError

    @abstractmethod
    def get_batch_size(self) -> int | None:
        raise NotImplementedError

    def gaussian_1d(self, k: int, *, sigma_scale: float = 0.5) -> torch.Tensor:
        x = torch.linspace(-1.0, 1.0, steps=k, device=self._device)
        sigma = sigma_scale
        g = torch.exp(-0.5 * (x / sigma) ** 2)
        g /= g.max()
        return g

    def do_sliding_window(self, t: torch.Tensor) -> tuple[torch.Tensor, SWMetadata]:
        window_shape = self.get_window_shape()
        if not (len(window_shape) + 2 == t.ndim):
            raise RuntimeError("Unmatched number of dimensions")
        stride = window_shape
        if len(stride) == 2:
            kernel = stride[0] * 2, stride[1] * 2
            b, c, h, w = t.shape
            t = nn.functional.unfold(t, kernel, stride=stride)
            n = t.shape[-1]
            kh, kw = kernel
            return (t.transpose(1, 2).contiguous().view(b * n, c, kh, kw),
                    SWMetadata(kernel, stride, 2, b, (h, w), n))
        else:
            b, c, d, h, w = t.shape
            sd, sh, sw = stride = window_shape
            kd, kh, kw = kernel = sd * 2, sh * 2, sw * 2

            # Calculate number of windows
            nz = (d - kd) // sd + 1
            ny = (h - kh) // sh + 1
            nx = (w - kw) // sw + 1
            num_windows = nz * ny * nx

            # Use compiled kernel for extraction
            windows = self._extract_3d_compiled(t, sd, sh, sw, kd, kh, kw, num_windows)
            windows = windows.view(num_windows * b, c, kd, kh, kw)
            return windows, SWMetadata(kernel, stride, 3, b, (d, h, w), num_windows)

    def revert_sliding_window(self, t: torch.Tensor, metadata: SWMetadata, *, clamp_min: float = 1e-8) -> torch.Tensor:
        kernel = metadata.kernel
        stride = metadata.stride
        dims = metadata.ndim
        b = metadata.batch_size
        out_size = metadata.out_size
        n = metadata.n
        dtype = t.dtype
        if dims == 2:
            kh, kw = kernel
            gh = self.gaussian_1d(kh)
            gw = self.gaussian_1d(kw)
            w2d = (gh[:, None] * gw[None, :]).to(dtype)
            w2d /= w2d.max()
            w2d = w2d.view(1, 1, kh, kw)
            bn, c, _, _ = t.shape
            if bn != b * n:
                raise RuntimeError("Inconsistent number of windows for reverting sliding window")
            weighted = t * w2d
            patches = weighted.view(b, n, c, kh, kw)
            cols = patches.view(b, n, c * kh * kw).transpose(1, 2).contiguous()
            numerator = nn.functional.fold(cols, out_size, kernel, stride=stride)
            w_cols = w2d.expand(b, n, 1, kh, kw).contiguous().view(b, n, 1 * kh * kw).transpose(1, 2)
            denominator = nn.functional.fold(w_cols, out_size, kernel, stride=stride)
            denominator = denominator.clamp_min(clamp_min)
            return numerator / denominator
        else:
            kd, kh, kw = kernel
            sd, sh, sw = stride
            d, h, w = out_size

            # Compute Gaussian weights once
            gd = self.gaussian_1d(kd)
            gh = self.gaussian_1d(kh)
            gw = self.gaussian_1d(kw)
            w3d = (gd[:, None, None] * gh[None, :, None] * gw[None, None, :]).to(dtype)
            w3d /= w3d.max()
            w3d = w3d.view(1, 1, kd, kh, kw)

            bn, c, _, _, _ = t.shape
            if bn != b * n:
                raise RuntimeError("Inconsistent number of windows for reverting sliding window")

            # Pre-multiply all windows by weights before loop
            t_weighted = t.view(n, b, c, kd, kh, kw) * w3d

            # Use compiled kernel for reconstruction
            canvas, acc_w = self._reconstruct_3d_compiled(
                t_weighted, w3d, d, h, w, sd, sh, sw, kd, kh, kw, b, c, dtype, self._device
            )

            acc_w = acc_w.clamp_min(clamp_min)
            return canvas / acc_w
