from typing import override

import torch
from torch import nn

from mipcandy.layer import LayerT


class AbstractConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, *, stride: int = 1, padding: int = 0,
                 dilation: int = 1, groups: int = 1, bias: bool = True, padding_mode: str = "zeros",
                 conv: LayerT = ..., norm: LayerT = ..., act: LayerT = ...) -> None:
        super().__init__()
        self.conv: nn.Module = conv.assemble(in_ch, out_ch, kernel_size, stride, padding, dilation, groups, bias,
                                             padding_mode)
        self.norm: nn.Module = norm.assemble(in_ch=out_ch)
        self.act: nn.Module = act.assemble()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


def _conv_block(default_conv: LayerT, default_norm: LayerT, default_act: LayerT) -> type[AbstractConvBlock]:
    class ConvBlock(AbstractConvBlock):
        def __init__(self, *args, **kwargs) -> None:
            if "conv" not in kwargs:
                kwargs["conv"] = default_conv
            if "norm" not in kwargs:
                kwargs["norm"] = default_norm
            if "act" not in kwargs:
                kwargs["act"] = default_act
            super().__init__(*args, **kwargs)

    return ConvBlock


ConvBlock2d: type[AbstractConvBlock] = _conv_block(
    LayerT(nn.Conv2d), LayerT(nn.BatchNorm2d, num_features="in_ch"), LayerT(nn.ReLU, inplace=True)
)

ConvBlock3d: type[AbstractConvBlock] = _conv_block(
    LayerT(nn.Conv3d), LayerT(nn.BatchNorm3d, num_features="in_ch"), LayerT(nn.ReLU, inplace=True)
)


class WSConv2d(nn.Conv2d):
    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight
        w = (w - w.mean(dim=(1, 2, 3), keepdim=True)) / (w.std(dim=(1, 2, 3), keepdim=True) + 1e-5)
        return nn.functional.conv2d(x, w, self.bias, self.stride, self.padding, self.dilation, self.groups)


class WSConv3d(nn.Conv3d):
    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight
        w = (w - w.mean(dim=(1, 2, 3, 4), keepdim=True)) / (w.std(dim=(1, 2, 3, 4), keepdim=True) + 1e-5)
        return nn.functional.conv3d(x, w, self.bias, self.stride, self.padding, self.dilation, self.groups)
