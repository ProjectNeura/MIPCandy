from abc import ABCMeta
from typing import override

import torch
from torch import nn, optim

from mipcandy.common import AbsoluteLinearLR, DiceBCELossWithLogits
from mipcandy.data import visualize2d, visualize3d, overlay, auto_convert, convert_logits_to_ids
from mipcandy.sliding_window import SWMetadata
from mipcandy.training import Trainer, TrainerToolbox, SlidingTrainer
from mipcandy.types import Params, Shape


class SegmentationTrainer(Trainer, metaclass=ABCMeta):
    num_classes: int = 1
    include_background: bool = True

    def _save_preview(self, x: torch.Tensor, title: str, quality: float) -> None:
        path = f"{self.experiment_folder()}/{title} (preview).png"
        if x.ndim == 3 and x.shape[0] in (1, 3, 4):
            visualize2d(auto_convert(x), title=title, blocking=True, screenshot_as=path)
        elif x.ndim == 4 and x.shape[0] == 1:
            visualize3d(x, title=title, max_volume=int(quality * 1e6), blocking=True, screenshot_as=path)

    @override
    def save_preview(self, image: torch.Tensor, label: torch.Tensor, output: torch.Tensor, *,
                     quality: float = .75) -> None:
        output = output.sigmoid()
        if output.shape[0] != 1:
            output = convert_logits_to_ids(output.unsqueeze(0)).squeeze(0)
        self._save_preview(image, "input", quality)
        self._save_preview(label, "label", quality)
        self._save_preview(output, "prediction", quality)
        if image.ndim == label.ndim == output.ndim == 3 and label.shape[0] == output.shape[0] == 1:
            visualize2d(overlay(image, label), title="expected", blocking=True,
                        screenshot_as=f"{self.experiment_folder()}/expected (preview).png")
            visualize2d(overlay(image, output), title="actual", blocking=True,
                        screenshot_as=f"{self.experiment_folder()}/actual (preview).png")

    @override
    def build_ema(self, model: nn.Module) -> nn.Module:
        return optim.swa_utils.AveragedModel(model)

    @override
    def build_criterion(self) -> nn.Module:
        return DiceBCELossWithLogits(self.num_classes, include_background=self.include_background)

    @override
    def build_optimizer(self, params: Params) -> optim.Optimizer:
        return optim.AdamW(params)

    @override
    def build_scheduler(self, optimizer: optim.Optimizer, num_epochs: int) -> optim.lr_scheduler.LRScheduler:
        return AbsoluteLinearLR(optimizer, -8e-6 / len(self._dataloader), 1e-2)

    @override
    def backward(self, images: torch.Tensor, labels: torch.Tensor, toolbox: TrainerToolbox) -> tuple[float, dict[
        str, float]]:
        masks = toolbox.model(images)
        loss, metrics = toolbox.criterion(masks, labels)
        loss.backward()
        return loss.item(), metrics

    @override
    def validate_case(self, image: torch.Tensor, label: torch.Tensor, toolbox: TrainerToolbox) -> tuple[float, dict[
        str, float], torch.Tensor]:
        image, label = image.unsqueeze(0), label.unsqueeze(0)
        mask = (toolbox.ema if toolbox.ema else toolbox.model)(image)
        loss, metrics = toolbox.criterion(mask, label)
        return -loss.item(), metrics, mask.squeeze(0)


class SlidingSegmentationTrainer(SlidingTrainer, SegmentationTrainer, metaclass=ABCMeta):
    sliding_window_shape: Shape = (128, 128)
    sliding_window_batch_size: int | None = None

    @override
    def backward_windowed(self, images: torch.Tensor, labels: torch.Tensor, toolbox: TrainerToolbox,
                          metadata: SWMetadata) -> tuple[float, dict[str, float]]:
        return SegmentationTrainer.backward(self, images, labels, toolbox)

    @override
    def validate_case_windowed(self, outputs: torch.Tensor, label: torch.Tensor, toolbox: TrainerToolbox,
                               metadata: SWMetadata) -> tuple[float, dict[str, float], torch.Tensor]:
        outputs = self.revert_sliding_window(outputs, metadata)
        loss, metrics = toolbox.criterion(outputs, label.unsqueeze(0))
        return -loss.item(), metrics, outputs.squeeze(0)

    @override
    def get_window_shape(self) -> Shape:
        return self.sliding_window_shape

    @override
    def get_batch_size(self) -> int | None:
        return self.sliding_window_batch_size


class SlidingValidationTrainer(SlidingSegmentationTrainer, metaclass=ABCMeta):
    """
    Use this when training data comes from RandomROIDataset (already patched), but validation data is full volumes
    requiring sliding window inference.
    """
    @override
    def backward_windowed(self, images: torch.Tensor, labels: torch.Tensor, toolbox: TrainerToolbox,
                          metadata: SWMetadata) -> tuple[float, dict[str, float]]:
        raise RuntimeError("`backward_windowed()` should not be called in `SlidingValidationTrainer`")

    @override
    def backward(self, images: torch.Tensor, labels: torch.Tensor, toolbox: TrainerToolbox) -> tuple[float, dict[
        str, float]]:
        return SegmentationTrainer.backward(self, images, labels, toolbox)

    @override
    def validate_case_windowed(self, outputs: torch.Tensor, label: torch.Tensor, toolbox: TrainerToolbox,
                               metadata: SWMetadata) -> tuple[float, dict[str, float], torch.Tensor]:
        return super().validate_case_windowed(outputs, label, toolbox, metadata)
