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
        output = output.sigmoid() if self.num_classes < 2 else output.softmax(0)
        if output.shape[0] != 1:
            output = convert_logits_to_ids(output, channel_dim=0).int()
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
            loss = DiceBCELossWithLogits(include_background=self.include_background, min_percentage_per_class=1e-5)
        else:
            loss = DiceCELossWithLogits(self.num_classes, include_background=self.include_background,
                                        min_percentage_per_class=1e-5)
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

    def log_stats_of_class_ids(self, ids: torch.Tensor, name: str) -> None:
        binc_p = torch.bincount(ids.flatten(), minlength=self.num_classes)
        self.log(f"{name} unique values: {ids.unique()}")
        self.log(f"{name} class distribution: {(binc_p / binc_p.sum()).cpu().tolist()}")

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
        self.log_stats_of_class_ids(label, "Label")
        self.log_stats_of_class_ids(mask_output, "Output")
        return -loss.item(), metrics, mask_output.squeeze(0)
