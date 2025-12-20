from typing import override

from monai.networks.nets import BasicUNet
from torch import nn

from mipcandy import SlidingSegmentationTrainer, AmbiguousShape


class UNetTrainer(SlidingSegmentationTrainer):
    sliding_window_shape = (64, 64, 64)
    sliding_window_batch_size = 1

    @override
    def build_network(self, example_shape: AmbiguousShape) -> nn.Module:
        return BasicUNet(3, example_shape[0], 4)
