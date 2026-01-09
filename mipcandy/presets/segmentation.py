from abc import ABCMeta
from typing import override, Self

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from mipcandy.common import AbsoluteLinearLR, DiceBCELossWithLogits
from mipcandy.data import visualize2d, visualize3d, overlay, auto_convert, convert_logits_to_ids, SupervisedDataset, \
    revert_sliding_window, SupervisedSWDataset
from mipcandy.training import Trainer, TrainerToolbox
from mipcandy.types import Params


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
    def validate_case(self, idx: int, image: torch.Tensor, label: torch.Tensor, toolbox: TrainerToolbox) -> tuple[
        float, dict[str, float], torch.Tensor]:
        image, label = image.unsqueeze(0), label.unsqueeze(0)
        mask = (toolbox.ema if toolbox.ema else toolbox.model)(image)
        loss, metrics = toolbox.criterion(mask, label)
        return -loss.item(), metrics, mask.squeeze(0)


class _TemplateDataset(SupervisedDataset[tuple[None]]):
    def __init__(self, dataset: SupervisedDataset) -> None:
        super().__init__((None,), (None,))
        self._base: SupervisedDataset = dataset

    @override
    def __len__(self) -> int:
        return len(self._base)

    @override
    def load_image(self, idx: int) -> torch.Tensor:
        return self._base.load_label(idx)

    @override
    def load_label(self, idx: int) -> torch.Tensor:
        return torch.empty(1)

    @override
    def construct_new(self, images: tuple[None], labels: tuple[None]) -> Self:
        raise NotImplementedError


class SlidingTrainer(SegmentationTrainer, metaclass=ABCMeta):
    overlap: float = .5
    batch_size: int = 1
    _validation_dataset: SupervisedDataset | None = None
    _slided_validation_dataset: SupervisedSWDataset | None = None

    def set_slided_validation_dataset(self, dataset: SupervisedSWDataset) -> None:
        self._slided_validation_dataset = dataset

    def slided_validation_dataset(self) -> SupervisedSWDataset:
        if self._slided_validation_dataset:
            return self._slided_validation_dataset
        raise ValueError("Slided validation dataset is not set")

    @override
    def save_preview(self, image: torch.Tensor, label: torch.Tensor, output: torch.Tensor, *,
                     quality: float = .75) -> None:
        super().save_preview(self._validation_dataset.image(self._tracker.worst_case), image, output, quality=quality)

    @override
    def validate(self, toolbox: TrainerToolbox) -> tuple[float, dict[str, list[float]]]:
        if not self._validation_dataset:
            dataset = self._validation_dataloader.dataset
            if not isinstance(dataset, SupervisedDataset):
                raise ValueError("Validation dataset must be a SupervisedDataset")
            self._validation_dataset = dataset
            self._validation_dataloader = DataLoader(_TemplateDataset(dataset), 1, False)
            self.log("WARNING: Transforms in the validation dataset will not be applied")
        return super().validate(toolbox)

    @override
    def validate_case(self, idx: int, label: torch.Tensor, _: torch.Tensor, toolbox: TrainerToolbox) -> tuple[
        float, dict[str, float], torch.Tensor]:
        model = toolbox.ema if toolbox.ema else toolbox.model
        self.record_profiler()
        self.record_profiler_linebreak(f"Loading case {idx} windows")
        images = self.slided_validation_dataset().images()
        num_windows, layout, original_shape = images.case_meta(idx)
        canvas = None
        self.record_profiler()
        self.record_profiler_linebreak("Inferring windows")
        for i in range(0, num_windows, self.batch_size):
            end = min(i + self.batch_size, num_windows)
            outputs = model(images.case(idx, part=slice(i, end)).to(self._device))
            if canvas is None:
                canvas = torch.empty((num_windows, *outputs.shape[1:]), dtype=outputs.dtype, device=self._device)
            canvas[i:end] = outputs
            self.empty_cache()
        self.record_profiler()
        self.record_profiler_linebreak("Reconstructing windows")
        reconstructed = revert_sliding_window(canvas.to(self._device), layout, original_shape, overlap=self.overlap)
        self.record_profiler()
        self.record_profiler_linebreak("Computing loss")
        loss, metrics = toolbox.criterion(reconstructed.unsqueeze(0), label.unsqueeze(0))
        self.record_profiler()
        return -loss.item(), metrics, reconstructed
