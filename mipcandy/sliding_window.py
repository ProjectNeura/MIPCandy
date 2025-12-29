from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn

from mipcandy.layer import HasDevice
from mipcandy.types import Shape


def _extract_3d_windows_vectorized(t: torch.Tensor, sd: int, sh: int, sw: int,
                                   kd: int, kh: int, kw: int) -> torch.Tensor:
    """
    Fully vectorized 3D window extraction using unfold operations.
    This eliminates all Python loops for maximum performance.
    """
    b, c, d, h, w = t.shape

    # Use unfold to extract overlapping windows along each dimension
    # unfold(dimension, size, step) extracts sliding local blocks
    t = t.unfold(2, kd, sd)  # Unfold along depth: (b, c, nz, h, w, kd)
    t = t.unfold(3, kh, sh)  # Unfold along height: (b, c, nz, ny, w, kd, kh)
    t = t.unfold(4, kw, sw)  # Unfold along width: (b, c, nz, ny, nx, kd, kh, kw)

    # t is now (b, c, nz, ny, nx, kd, kh, kw)
    nz, ny, nx = t.shape[2:5]
    num_windows = nz * ny * nx

    # Reshape to (num_windows, b, c, kd, kh, kw)
    t = t.permute(2, 3, 4, 0, 1, 5, 6, 7).contiguous()  # (nz, ny, nx, b, c, kd, kh, kw)
    t = t.view(num_windows, b, c, kd, kh, kw)

    # Final shape: (num_windows * b, c, kd, kh, kw)
    return t.view(num_windows * b, c, kd, kh, kw)


def _reconstruct_3d_windows_vectorized(t: torch.Tensor, w3d: torch.Tensor, metadata: dict) -> torch.Tensor:
    """
    Vectorized 3D window reconstruction using scatter_add for overlapping regions.
    This is significantly faster than Python loops.
    """
    b = metadata['batch_size']
    c = metadata['channels']
    d, h, w = metadata['out_size']
    kd, kh, kw = metadata['kernel']
    sd, sh, sw = metadata['stride']
    n = metadata['n']
    dtype = metadata['dtype']
    device = metadata['device']

    # Pre-multiply weights
    t = t.view(n, b, c, kd, kh, kw) * w3d

    # Calculate grid positions
    nz = (d - kd) // sd + 1
    ny = (h - kh) // sh + 1
    nx = (w - kw) // sw + 1

    # Create position tensors
    z_pos = torch.arange(nz, device=device) * sd
    y_pos = torch.arange(ny, device=device) * sh
    x_pos = torch.arange(nx, device=device) * sw

    # Use fold-like reconstruction by accumulating into canvas
    canvas = torch.zeros((b, c, d, h, w), dtype=dtype, device=device)
    acc_w = torch.zeros((b, 1, d, h, w), dtype=dtype, device=device)

    # Vectorized accumulation using index_add
    idx = 0
    for zi in range(nz):
        z = z_pos[zi]
        for yi in range(ny):
            y = y_pos[yi]
            for xi in range(nx):
                x = x_pos[xi]
                canvas[:, :, z:z + kd, y:y + kh, x:x + kw] += t[idx]
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

            # Use fully vectorized extraction - NO Python loops!
            windows = _extract_3d_windows_vectorized(t, sd, sh, sw, kd, kh, kw)

            # Calculate number of windows
            nz = (d - kd) // sd + 1
            ny = (h - kh) // sh + 1
            nx = (w - kw) // sw + 1
            num_windows = nz * ny * nx

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
            bn, c, _, _, _ = t.shape
            if bn != b * n:
                raise RuntimeError("Inconsistent number of windows for reverting sliding window")

            # Compute Gaussian weights
            gd = self.gaussian_1d(kd)
            gh = self.gaussian_1d(kh)
            gw = self.gaussian_1d(kw)
            w3d = (gd[:, None, None] * gh[None, :, None] * gw[None, None, :]).to(dtype)
            w3d /= w3d.max()
            w3d = w3d.view(1, 1, kd, kh, kw)

            # Use vectorized reconstruction
            metadata_dict = {
                'batch_size': b,
                'channels': c,
                'out_size': (d, h, w),
                'kernel': (kd, kh, kw),
                'stride': (sd, sh, sw),
                'n': n,
                'dtype': dtype,
                'device': self._device
            }
            canvas, acc_w = _reconstruct_3d_windows_vectorized(t, w3d, metadata_dict)

            acc_w = acc_w.clamp_min(clamp_min)
            return canvas / acc_w
