from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn

from mip_candy.layer import HasDevice


@dataclass
class SWMetadata(object):
    kernel: tuple[int, int] | tuple[int, int, int]
    stride: tuple[int, int] | tuple[int, int, int]
    ndim: Literal[2, 3]
    batch_size: int
    out_size: tuple[int, int] | tuple[int, int, int]
    n: int


class SlidingWindow(HasDevice, metaclass=ABCMeta):
    @abstractmethod
    def get_window_shape(self) -> tuple[int, int] | tuple[int, int, int]:
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
            sd, sh, sw = stride
            kd, kh, kw = kernel = sd * 2, sh * 2, sw * 2
            image_windows = []
            for z in range(0, d - kd + 1, sd):
                for y in range(0, h - kh + 1, sh):
                    for x in range(0, w - kw + 1, sw):
                        image_windows.append(t[:, :, z:z + kd, y:y + kh, x:x + kw])
            t = torch.stack(image_windows, dim=0)
            n = t.shape[0]
            return (t.permute(0, 1, 2, 3, 4, 5).contiguous().view(b * n, c, kd, kh, kw),
                    SWMetadata(kernel, stride, 3, b, (h, w), n))

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
            gd = self.gaussian_1d(kd)
            gh = self.gaussian_1d(kh)
            gw = self.gaussian_1d(kw)
            w3d = (gd[:, None, None] * gh[None, :, None] * gw[None, None, :]).to(dtype)
            w3d /= w3d.max()
            w3d = w3d.view(1, 1, kd, kh, kw)
            bn, c, _, _, _ = t.shape
            if bn != b * n:
                raise RuntimeError("Inconsistent number of windows for reverting sliding window")
            canvas = torch.zeros((b, c, d, h, w), dtype=dtype, device=self._device)
            acc_w = torch.zeros((b, 1, d, h, w), dtype=dtype, device=self._device)
            idx = 0
            for z in range(0, d - kd + 1, sd):
                for y in range(0, h - kh + 1, sh):
                    for x in range(0, w - kw + 1, sw):
                        window = t[idx * b:(idx + 1) * b]
                        window *= w3d
                        canvas[:, :, z:z + kd, y:y + kh, x:x + kw] += window
                        acc_w[:, :, z:z + kd, y:y + kh, x:x + kw] += w3d
                        idx += 1
            acc_w = acc_w.clamp_min(clamp_min)
            return canvas / acc_w
