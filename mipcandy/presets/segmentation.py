from abc import ABCMeta
from typing import override, Callable, Sequence, Any

import numpy as np
import torch
from rich.progress import Progress, SpinnerColumn
from torch import nn, optim

from mipcandy.common import PolyLRScheduler, DiceBCELossWithLogits, DiceCELossWithLogits
from mipcandy.data import visualize2d, visualize3d, overlay, auto_convert, convert_logits_to_ids, SupervisedDataset, \
    revert_sliding_window, SupervisedSWDataset, fast_save
from mipcandy.training import Trainer, TrainerToolbox, try_append_all
from mipcandy.types import Params, Shape


class DeepSupervisionWrapper(nn.Module):
    def __init__(self, loss: nn.Module, *, weight_factors: Sequence[float] | None = None) -> None:
        super().__init__()
        if weight_factors and all(x == 0 for x in weight_factors):
            raise ValueError("At least one weight factor should be nonzero")
        self.weight_factors: tuple[float, ...] = tuple(weight_factors)
        self.loss: nn.Module = loss

    @override
    def __getattr__(self, item: str) -> Any:
        return self.loss.validation_mode if item == "validation_mode" else super().__getattr__(item)

    @override
    def __setattr__(self, name: str, value: Any) -> None:
        if name == "validation_mode" and hasattr(self.loss, "validation_mode"):
            self.loss.validation_mode = value
        super().__setattr__(name, value)

    def forward(self, outputs: Sequence[torch.Tensor], targets: Sequence[torch.Tensor]) -> tuple[
        torch.Tensor, dict[str, float]]:
        if not self.weight_factors:
            weights = (1.0,) * len(outputs)
        else:
            weights = self.weight_factors
        total_loss = torch.tensor(0, device=outputs[0].device, dtype=outputs[0].dtype)
        combined_metrics = {}
        for i, (output, target) in enumerate(zip(outputs, targets)):
            if weights[i] == 0:
                continue
            loss, metrics = self.loss(output, target)
            total_loss += weights[i] * loss
            for key, value in metrics.items():
                metric_key = f"{key}_ds{i}" if len(outputs) > 1 else key
                combined_metrics[metric_key] = value
        if combined_metrics:
            main_loss_key = next(iter(combined_metrics.keys())).replace("_ds0", "")
            if f"{main_loss_key}_ds0" in combined_metrics:
                combined_metrics[main_loss_key] = combined_metrics[f"{main_loss_key}_ds0"]
        return total_loss, combined_metrics


class SegmentationTrainer(Trainer, metaclass=ABCMeta):
    num_classes: int = 1
    include_background: bool = True
    deep_supervision: bool = False
    deep_supervision_scales: Sequence[float] | None = None
    deep_supervision_weights: Sequence[float] | None = None

    def _save_preview(self, x: torch.Tensor, title: str, quality: float, *, is_label: bool = False) -> None:
        path = f"{self.experiment_folder()}/{title} (preview).png"
        if x.ndim == 3 and x.shape[0] in (1, 3, 4):
            visualize2d(auto_convert(x), title=title, is_label=is_label, blocking=True, screenshot_as=path)
        elif x.ndim == 4 and x.shape[0] == 1:
            visualize3d(x, title=title, max_volume=int(quality * 1e6), is_label=is_label, blocking=True,
                        screenshot_as=path)

    @override
    def save_preview(self, image: torch.Tensor, label: torch.Tensor, output: torch.Tensor, *,
                     quality: float = .75) -> None:
        output = output.sigmoid()
        if output.shape[0] != 1:
            output = convert_logits_to_ids(output.unsqueeze(0)).squeeze(0).int()
        self._save_preview(image, "input", quality)
        self._save_preview(label.int(), "label", quality, is_label=True)
        self._save_preview(output, "prediction", quality, is_label=True)
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
        if self.num_classes < 2:
            loss = DiceBCELossWithLogits(include_background=self.include_background)
        else:
            loss = DiceCELossWithLogits(self.num_classes, include_background=self.include_background)
        if self.deep_supervision:
            if not self.deep_supervision_weights and self.deep_supervision_scales:
                weights = np.array([1 / (2 ** i) for i in range(len(self.deep_supervision_scales))])
                weights = weights / weights.sum()
                self.deep_supervision_weights = tuple(weights.tolist())
            loss = DeepSupervisionWrapper(loss, weight_factors=self.deep_supervision_weights)
            self.log(f"Deep supervision enabled with weights: {self.deep_supervision_weights}")
        return loss

    @override
    def build_optimizer(self, params: Params) -> optim.Optimizer:
        return optim.SGD(params, 1e-2, weight_decay=3e-5, momentum=.99, nesterov=True)

    @override
    def build_scheduler(self, optimizer: optim.Optimizer, num_epochs: int) -> optim.lr_scheduler.LRScheduler:
        return PolyLRScheduler(optimizer, 1e-2, num_epochs * len(self._dataloader))

    @override
    def backward(self, images: torch.Tensor, labels: torch.Tensor, toolbox: TrainerToolbox) -> tuple[float, dict[
        str, float]]:
        outputs = toolbox.model(images)
        if self.deep_supervision and outputs.ndim == labels.ndim + 1:
            masks_list = list(torch.unbind(outputs, dim=1))
            targets = self.prepare_deep_supervision_targets(labels, [m.shape[2:] for m in masks_list])
            loss, metrics = toolbox.criterion(masks_list, targets)
        elif self.deep_supervision and isinstance(outputs, (list, tuple)):
            targets = self.prepare_deep_supervision_targets(labels, [m.shape[2:] for m in outputs])
            loss, metrics = toolbox.criterion(outputs, targets)
        else:
            loss, metrics = toolbox.criterion(outputs, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(toolbox.model.parameters(), 12)
        return loss.item(), metrics

    @staticmethod
    def prepare_deep_supervision_targets(labels: torch.Tensor, output_shapes: list[tuple[int, ...]]) -> list[
        torch.Tensor]:
        targets = []
        for shape in output_shapes:
            if labels.shape[2:] == shape:
                targets.append(labels)
            else:
                downsampled = nn.functional.interpolate(labels.float(), shape,
                                                        mode="nearest-exact" if labels.ndim == 4 else "nearest")
                targets.append(downsampled if labels.dtype == torch.float32 else downsampled.to(labels.dtype))
        return targets

    @override
    def validate_case(self, idx: int, image: torch.Tensor, label: torch.Tensor, toolbox: TrainerToolbox) -> tuple[
        float, dict[str, float], torch.Tensor]:
        image, label = image.unsqueeze(0), label.unsqueeze(0)
        output = (toolbox.ema if toolbox.ema else toolbox.model)(image)
        # (B, N, C, H, W, D) with the highest resolution is at index 0
        if self.deep_supervision and output.ndim == label.ndim + 1:
            mask_for_loss = output[:, 0]
            mask_output = output[:, 0]
        elif self.deep_supervision and isinstance(output, (list, tuple)):
            mask_for_loss = output[0]
            mask_output = output[0]
        else:
            mask_for_loss = output
            mask_output = output
        if hasattr(toolbox.criterion, "validation_mode"):
            toolbox.criterion.validation_mode = True
        if self.deep_supervision and isinstance(toolbox.criterion, DeepSupervisionWrapper):
            loss, metrics = toolbox.criterion([mask_for_loss], [label])
        else:
            loss, metrics = toolbox.criterion(mask_for_loss, label)
        if hasattr(toolbox.criterion, "validation_mode"):
            toolbox.criterion.validation_mode = False
        self.log(f"Metrics for case {idx}: {metrics}")
        return -loss.item(), metrics, mask_output.squeeze(0)


class SlidingTrainer(SegmentationTrainer, metaclass=ABCMeta):
    overlap: float = .5
    window_batch_size: int = 1
    full_validation_at_epochs: list[Callable[[int], int]] = [lambda num_epochs: num_epochs - 1]
    compute_loss_on_device: bool = False
    _full_validation_dataset: SupervisedDataset | None = None
    _slided_validation_dataset: SupervisedSWDataset | None = None

    def set_datasets(self, full_dataset: SupervisedDataset, slided_dataset: SupervisedSWDataset) -> None:
        self.set_full_validation_dataset(full_dataset)
        self.set_slided_validation_dataset(slided_dataset)

    def set_full_validation_dataset(self, dataset: SupervisedDataset) -> None:
        dataset.device(device=self._device if self.compute_loss_on_device else "cpu")
        self._full_validation_dataset = dataset

    def full_validation_dataset(self) -> SupervisedDataset:
        if self._full_validation_dataset:
            return self._full_validation_dataset
        raise ValueError("Full validation dataset is not set")

    def set_slided_validation_dataset(self, dataset: SupervisedSWDataset) -> None:
        self._slided_validation_dataset = dataset

    def slided_validation_dataset(self) -> SupervisedSWDataset:
        if self._slided_validation_dataset:
            return self._slided_validation_dataset
        raise ValueError("Slided validation dataset is not set")

    @override
    def validate(self, toolbox: TrainerToolbox) -> tuple[float, dict[str, list[float]]]:
        if self._tracker.epoch not in self.full_validation_at_epochs:
            return super().validate(toolbox)
        self.log("Performing full-resolution validation")
        return self.fully_validate(toolbox)

    def fully_validate(self, toolbox: TrainerToolbox) -> tuple[float, dict[str, list[float]]]:
        self.record_profiler_linebreak(f"Fully validating epoch {self._tracker.epoch}")
        self.record_profiler()
        self.record_profiler_linebreak("Emptying cache")
        self.empty_cache()
        self.record_profiler()
        toolbox.model.eval()
        if toolbox.ema:
            toolbox.ema.eval()
        score = 0
        worst_score = float("+inf")
        metrics = {}
        num_cases = len(self._full_validation_dataset)
        with torch.no_grad(), Progress(
                *Progress.get_default_columns(), SpinnerColumn(), console=self._console
        ) as progress:
            task = progress.add_task(f"Fully validating", total=num_cases)
            for idx in range(num_cases):
                progress.update(task, description=f"Validating epoch {self._tracker.epoch} case {idx}")
                case_score, case_metrics, output = self.fully_validate_case(idx, toolbox)
                self.record_profiler()
                self.record_profiler_linebreak("Emptying cache")
                self.empty_cache()
                self.record_profiler()
                score += case_score
                if case_score < worst_score:
                    self._tracker.worst_case = idx
                    fast_save(output, f"{self.experiment_folder()}/worst_full_output.pt")
                    worst_score = case_score
                try_append_all(case_metrics, metrics)
                progress.update(task, advance=1,
                                description=f"Validating epoch {self._tracker.epoch} case {idx} ({case_score:.4f})")
                self.record_profiler()
        return score / num_cases, metrics

    def infer_validation_case(self, idx: int, toolbox: TrainerToolbox) -> tuple[torch.Tensor, Shape, Shape]:
        model = toolbox.ema if toolbox.ema else toolbox.model
        images = self.slided_validation_dataset().images()
        num_windows, layout, original_shape = images.case_meta(idx)
        canvas = None
        for i in range(0, num_windows, self.window_batch_size):
            end = min(i + self.window_batch_size, num_windows)
            outputs = model(images.case(idx, part=slice(i, end)).to(self._device))

            # For deep supervision, only use the highest resolution output
            # DynUNet stacks as (B, N, C, H, W, D), extract first output at dim 1
            if self.deep_supervision and outputs.ndim == 6:  # 6D tensor for 3D images with deep supervision
                outputs = outputs[:, 0]  # Extract (B, C, H, W, D)
            elif self.deep_supervision and outputs.ndim == 5:  # 5D tensor for 2D images with deep supervision
                outputs = outputs[:, 0]  # Extract (B, C, H, W)
            elif self.deep_supervision and isinstance(outputs, (list, tuple)):
                outputs = outputs[0]

            if canvas is None:
                canvas = torch.empty((num_windows, *outputs.shape[1:]), dtype=outputs.dtype, device=self._device)
            canvas[i:end] = outputs
        return canvas, layout, original_shape

    def fully_validate_case(self, idx: int, toolbox: TrainerToolbox) -> tuple[
        float, dict[str, float], torch.Tensor]:
        windows, layout, original_shape = self.infer_validation_case(idx, toolbox)
        self.empty_cache()
        reconstructed = revert_sliding_window(windows, layout, original_shape, overlap=self.overlap)
        if self.compute_loss_on_device:
            self.empty_cache()
        else:
            reconstructed = reconstructed.cpu()
        label = self._full_validation_dataset.label(idx)
        loss, metrics = toolbox.criterion(reconstructed.unsqueeze(0), label.unsqueeze(0))
        return -loss.item(), metrics, reconstructed
