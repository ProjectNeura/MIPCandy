from typing import override

from monai.networks.nets import DynUNet
from torch import nn

from mipcandy import SegmentationTrainer, SlidingTrainer, AmbiguousShape


class UNetTrainer(SegmentationTrainer):
    num_classes = 5
    deep_supervision = False
    deep_supervision_scales = [1, 2, 4]
    include_background: bool = False

    @override
    def build_network(self, example_shape: AmbiguousShape) -> nn.Module:
        kernels = [[3, 3, 3]] * 6
        strides = [[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]]
        filters = [32, 64, 128, 256, 512, 1024]
        return DynUNet(
            spatial_dims=3,
            in_channels=example_shape[0],
            out_channels=self.num_classes,
            kernel_size=kernels,
            strides=strides,
            upsample_kernel_size=strides[1:],
            filters=filters,
            norm_name="INSTANCE",
            deep_supervision=self.deep_supervision,
            deep_supr_num=2,
            res_block=True
        )


class UNetSlidingTrainer(UNetTrainer, SlidingTrainer):
    pass
