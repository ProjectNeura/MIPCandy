from math import ceil
from typing import Literal

import torch
from torch import nn

from mip_candy.types import Colormap


class Pad(nn.Module):
    def __init__(self, *, value: int = 0, mode: str = "constant", batch: bool = True) -> None:
        super().__init__()
        self._value: int = value
        self._mode: str = mode
        self.batch: bool = batch
        self._paddings: tuple[int, int, int, int, int, int] | tuple[int, int, int, int] | None = None
        self.requires_grad_(False)

    @staticmethod
    def _c_t(size: int, min_factor: int) -> int:
        """
        Compute target on a single dimension
        """
        return ceil(size / min_factor) * min_factor

    @staticmethod
    def _c_p(size: int, min_factor: int) -> tuple[int, int]:
        """
        Compute padding on a single dimension
        """
        excess = Pad._c_t(size, min_factor) - size
        before = excess // 2
        return before, excess - before


class Pad2d(Pad):
    def __init__(self, min_factor: int | tuple[int, int], *, value: int = 0, mode: str = "constant",
                 batch: bool = True) -> None:
        super().__init__(value=value, mode=mode, batch=batch)
        self._min_factor: tuple[int, int] = (min_factor,) * 2 if isinstance(min_factor, int) else min_factor

    def paddings(self) -> tuple[int, int, int, int] | None:
        return self._paddings

    def padded_shape(self, in_shape: tuple[int, int, ...]) -> tuple[int, int, ...]:
        return *in_shape[:-2], Pad._c_t(in_shape[-2], self._min_factor[0]), Pad._c_t(in_shape[-1], self._min_factor[1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.batch:
            _, _, h, w = x.shape
            suffix = (0,) * 4
        else:
            _, h, w = x.shape
            suffix = (0,) * 2
        self._paddings = Pad._c_p(h, self._min_factor[0]) + Pad._c_p(w, self._min_factor[1])
        return nn.functional.pad(x, self._paddings[::-1] + suffix, self._mode, self._value)


class Pad3d(Pad):
    def __init__(self, min_factor: int | tuple[int, int, int], *, value: int = 0, mode: str = "constant",
                 batch: bool = True) -> None:
        super().__init__(value=value, mode=mode, batch=batch)
        self._min_factor: tuple[int, int, int] = (min_factor,) * 3 if isinstance(min_factor, int) else min_factor

    def paddings(self) -> tuple[int, int, int, int, int, int] | None:
        return self._paddings

    def padded_shape(self, in_shape: tuple[int, int, int, ...]) -> tuple[int, int, int, ...]:
        return (*in_shape[:-3], Pad._c_t(in_shape[-3], self._min_factor[0]), Pad._c_t(
            in_shape[-2], self._min_factor[1]), Pad._c_t(in_shape[-1], self._min_factor[2]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.batch:
            _, _, d, h, w = x.shape
            suffix = (0,) * 4
        else:
            _, d, h, w = x.shape
            suffix = (0,) * 2
        self._paddings = Pad._c_p(d, self._min_factor[2]) + Pad._c_p(h, self._min_factor[0]) + Pad._c_p(
            w, self._min_factor[1])
        return nn.functional.pad(x, self._paddings[::-1] + suffix, self._mode, self._value)


class Restore2d(nn.Module):
    def __init__(self, conjugate_padding: Pad2d) -> None:
        super().__init__()
        self.conjugate_padding: Pad2d = conjugate_padding
        self.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        paddings = self.conjugate_padding.paddings()
        if not paddings:
            raise UserWarning("Paddings seem to be unset, did you forget to pad before restoring?")
        pad_h0, pad_h1, pad_w0, pad_w1 = paddings
        if self.conjugate_padding.batch:
            _, _, h, w = x.shape
            return x[:, :, pad_h0: h - pad_h1, pad_w0: w - pad_w1]
        _, h, w = x.shape
        return x[:, pad_h0: h - pad_h1, pad_w0: w - pad_w1]


class Restore3d(nn.Module):
    def __init__(self, conjugate_padding: Pad3d) -> None:
        super().__init__()
        self.conjugate_padding: Pad3d = conjugate_padding
        self.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        paddings = self.conjugate_padding.paddings()
        if not paddings:
            raise UserWarning("Paddings seem to be unset, did you forget to pad before restoring?")
        pad_h0, pad_h1, pad_w0, pad_w1, pad_d0, pad_d1 = paddings
        if self.conjugate_padding.batch:
            _, _, d, h, w = x.shape
            return x[:, :, pad_d0: d - pad_d1, pad_h0: h - pad_h1, pad_w0: w - pad_w1]
        _, d, h, w = x.shape
        return x[:, pad_d0: d - pad_d1, pad_h0: h - pad_h1, pad_w0: w - pad_w1]


class Normalize(nn.Module):
    def __init__(self, *, domain: tuple[float | None, float | None] = (0, None), strict: bool = False,
                 method: Literal["linear", "intercept", "cut"] = "linear") -> None:
        super().__init__()
        self._domain: tuple[float | None, float | None] = domain
        self._strict: bool = strict
        self._method: Literal["linear", "intercept", "cut"] = method

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        left, right = self._domain
        if left is None and right is None:
            return x
        r_l, r_r = x.min(), x.max()
        match self._method:
            case "linear":
                if left is None or (left < r_l and not self._strict):
                    left = r_l
                if right is None or (right > r_r and not self._strict):
                    right = r_r
                numerator = right - left
                if numerator == 0:
                    numerator = 1
                denominator = r_r - r_l
                if denominator == 0:
                    denominator = 1
                return (x - r_l) * numerator / denominator + left
            case "intercept":
                if left is not None and right is None:
                    return x - r_l + left if r_l < left or self._strict else x
                elif left is None and right is not None:
                    return x - r_r + right if r_r > right or self._strict else x
                else:
                    raise ValueError("Cannot use intercept normalization when both ends are fixed")
            case "cut":
                if self._strict:
                    raise ValueError("Method \"cut\" cannot be strict")
                if left is not None:
                    x[x < left] = left
                if right is not None:
                    x[x > right] = right
                return x


class ColorizeLabel(nn.Module):
    def __init__(self, *, colormap: Colormap | None = None) -> None:
        super().__init__()
        if not colormap:
            colormap = []
            for r in range(8):
                for g in range(8):
                    for b in range(32):
                        colormap.append([r * 32, g * 32, 255 - b * 32])
        self._colormap: torch.Tensor = torch.tensor(colormap)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cmap = self._colormap.to(x.device)
        return (
            torch.cat([cmap[(x > 0).int()].permute(2, 0, 1), x.unsqueeze(0)]) if 0 <= x.min() < x.max() <= 1
            else cmap[x.int()].permute(2, 0, 1)
        )
