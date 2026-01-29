from typing import override

from monai.networks.nets import DynUNet
from torch import nn

from mipcandy import SegmentationTrainer, SlidingTrainer, AmbiguousShape


class UNetTrainer(SegmentationTrainer):
    num_classes = 5
    include_background: bool = False

    @override
    def build_network(self, example_shape: AmbiguousShape) -> nn.Module:
        return DynUNet(3, example_shape[0], self.num_classes, [3, 3, 3, 3, 3, 3], [1, 2, 2, 2, 2, [1, 2, 2]],
                       [2, 2, 2, 2, [1, 2, 2]], [32, 64, 128, 256, 320, 320])


class UNetSlidingTrainer(UNetTrainer, SlidingTrainer):
    pass
