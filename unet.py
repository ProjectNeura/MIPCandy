from typing import override

from monai.networks.nets import BasicUNet
from torch import nn

from mipcandy import SlidingTrainer, AmbiguousShape, DiceBCELossWithLogits


class UNetTrainer(SlidingTrainer):
    @override
    def build_criterion(self) -> nn.Module:
        return DiceBCELossWithLogits(self.num_classes, include_background=False)

    @override
    def build_network(self, example_shape: AmbiguousShape) -> nn.Module:
        return BasicUNet(3, example_shape[0], self.num_classes)
