from abc import ABCMeta
from typing import override, Sequence, Any

import numpy as np
import torch
from torch import nn, optim

from mipcandy.common import PolyLRScheduler, DiceBCELossWithLogits, DiceCELossWithLogits
from mipcandy.data import visualize2d, visualize3d, overlay, auto_convert, convert_logits_to_ids
from mipcandy.training import Trainer, TrainerToolbox
from mipcandy.types import Params


class DeepSupervisionWrapper(nn.Module):
    def __init__(self, loss: nn.Module, *, weight_factors: Sequence[float] | None = None) -> None:
        super().__init__()
        if weight_factors and all(x == 0 for x in weight_factors):
            raise ValueError("At least one weight factor should be nonzero")
        self.weight_factors: tuple[float, ...] | None = tuple(weight_factors) if weight_factors else None
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

    def apply_non_linearity(self, x: torch.Tensor, channel_dim: int) -> torch.Tensor:
        return x.sigmoid() if self.num_classes < 2 else x.softmax(channel_dim)

    @override
    def save_preview(self, image: torch.Tensor, label: torch.Tensor, output: torch.Tensor, *,
                     quality: float = .75) -> None:
        output = convert_logits_to_ids(self.apply_non_linearity(output, 0), channel_dim=0)
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
            if not self.include_background:
                raise ValueError("Binary segmentation models must include background class")
            loss = DiceBCELossWithLogits()
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
        if self.deep_supervision:
            if outputs.ndim == labels.ndim + 1:
                outputs = list(torch.unbind(outputs, dim=1))
            labels = self.prepare_deep_supervision_targets(labels, [m.shape[2:] for m in outputs])
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
                downsampled = nn.functional.interpolate(labels, shape,
                                                        mode="nearest-exact" if labels.ndim == 4 else "nearest")
                targets.append(downsampled)
        return targets

    def class_percentages(self, ids: torch.Tensor) -> dict[int, float]:
        bin_count = torch.bincount(ids.flatten(), minlength=self.num_classes)
        distribution = (bin_count / bin_count.sum()).cpu().tolist()
        return dict(enumerate(distribution))

    @staticmethod
    def format_class_percentages(percentages: dict[int, float], prefix: str) -> dict[str, float]:
        return {f"% {prefix} class {class_id}": percentage for class_id, percentage in percentages.items()}

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
        label_percentages = self.class_percentages(label)
        metrics.update(self.format_class_percentages(label_percentages, "label"))
        output_logits = self.apply_non_linearity(mask_output, 1)
        output_percentages = self.class_percentages(convert_logits_to_ids(output_logits))
        metrics.update(self.format_class_percentages(output_percentages, "output"))
        return -loss.item(), metrics, mask_output.squeeze(0)
