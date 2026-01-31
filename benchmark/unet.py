from os import makedirs
from typing import override

import torch
from monai.networks.nets import DynUNet
from torch import nn

from mipcandy import SegmentationTrainer, SlidingTrainer, AmbiguousShape
from mipcandy.data import convert_logits_to_ids
from mipcandy.training import TrainerToolbox


class UNetTrainer(SegmentationTrainer):
    num_classes = 5
    include_background: bool = False
    debug_epoch: int | None = None
    _debug_counter: int = 0

    @override
    def build_network(self, example_shape: AmbiguousShape) -> nn.Module:
        return DynUNet(3, example_shape[0], self.num_classes, [3, 3, 3, 3, 3, 3], [1, 2, 2, 2, 2, [1, 2, 2]],
                       [2, 2, 2, 2, [1, 2, 2]], [32, 64, 128, 256, 320, 320])

    def _save_debug_case(self, image: torch.Tensor, label: torch.Tensor, output: torch.Tensor,
                         prefix: str, idx: int) -> None:
        folder: str = f"debug_epoch_{self.debug_epoch}/{prefix}"
        makedirs(f"{self.experiment_folder()}/{folder}", exist_ok=True)
        output = output.detach().sigmoid()
        if output.shape[0] != 1:
            output = convert_logits_to_ids(output.unsqueeze(0)).squeeze(0).int()
        self._save_preview(image, f"{folder}/{idx:04d}_image", .75)
        self._save_preview(label.int(), f"{folder}/{idx:04d}_label", .75, is_label=True)
        self._save_preview(output, f"{folder}/{idx:04d}_prediction", .75, is_label=True)

    @override
    def train_epoch(self, toolbox: TrainerToolbox) -> dict[str, list[float]]:
        self._debug_counter = 0
        return super().train_epoch(toolbox)

    @override
    def backward(self, images: torch.Tensor, labels: torch.Tensor, toolbox: TrainerToolbox) -> tuple[
        float, dict[str, float]]:
        masks: torch.Tensor = toolbox.model(images)
        if self.debug_epoch is not None and self._tracker.epoch == self.debug_epoch:
            for i in range(images.shape[0]):
                self._save_debug_case(images[i], labels[i], masks[i], "train", self._debug_counter)
                self._debug_counter += 1
        loss, metrics = toolbox.criterion(masks, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(toolbox.model.parameters(), 12)
        return loss.item(), metrics

    @override
    def validate_case(self, idx: int, image: torch.Tensor, label: torch.Tensor, toolbox: TrainerToolbox) -> tuple[
        float, dict[str, float], torch.Tensor]:
        score, metrics, output = super().validate_case(idx, image, label, toolbox)
        if self.debug_epoch is not None and self._tracker.epoch == self.debug_epoch:
            self._save_debug_case(image, label, output, "val", idx)
        return score, metrics, output


class UNetSlidingTrainer(UNetTrainer, SlidingTrainer):
    pass
