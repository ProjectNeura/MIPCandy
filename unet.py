from typing import override

from monai.networks.nets import BasicUNet
from torch import nn

from mipcandy import SegmentationTrainer, SlidingSegmentationTrainer, AmbiguousShape, DiceBCELossWithLogits


class UNetTrainer(SegmentationTrainer):
    @override
    def build_criterion(self) -> nn.Module:
        return DiceBCELossWithLogits(self.num_classes, include_bg=False)

    @override
    def build_network(self, example_shape: AmbiguousShape) -> nn.Module:
        return BasicUNet(3, example_shape[0], 5)


class UNetSlidingTrainer(SlidingSegmentationTrainer, UNetTrainer):
    sliding_window_shape = (64, 64, 64)
    sliding_window_batch_size = 1
