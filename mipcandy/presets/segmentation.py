from abc import ABCMeta
from typing import override

import torch
from torch import nn, optim

from mipcandy.common import AbsoluteLinearLR, DiceBCELossWithLogits
from mipcandy.data import visualize2d, visualize3d, overlay
from mipcandy.training import Trainer, TrainerToolbox
from mipcandy.types import Params


class SegmentationTrainer(Trainer, metaclass=ABCMeta):
    num_classes: int = 1

    def _save_preview(self, x: torch.Tensor, title: str) -> None:
        path = f"{self.experiment_folder()}/{title} (preview).png"
        if x.shape[0] not in (1, 3, 4):
            return
        if x.ndim == 3:
            visualize2d((x * 255 / x.max()).to(torch.uint16), title=title, blocking=True, screenshot_as=path)
        elif x.ndim == 4:
            visualize3d(x, title=title, blocking=True, screenshot_as=path)
        else:
            raise ValueError("MIP Candy only intends to support 2D and 3D data")

    @override
    def save_preview(self, image: torch.Tensor, label: torch.Tensor, mask: torch.Tensor) -> None:
        mask = mask.sigmoid()
        self._save_preview(image, "input")
        self._save_preview(label, "label")
        self._save_preview(mask, "prediction")
        if image.ndim == label.ndim == mask.ndim == 3:
            visualize2d(overlay(image, label), title="expected", blocking=True,
                        screenshot_as=f"{self.experiment_folder()}/expected (preview).png")
            visualize2d(overlay(image, mask), title="actual", blocking=True,
                        screenshot_as=f"{self.experiment_folder()}/actual (preview).png")

    @override
    def build_criterion(self) -> nn.Module:
        return DiceBCELossWithLogits(self.num_classes)

    @override
    def build_optimizer(self, params: Params) -> optim.Optimizer:
        return optim.AdamW(params)

    @override
    def build_scheduler(self, optimizer: optim.Optimizer, num_epochs: int) -> optim.lr_scheduler.LRScheduler:
        return AbsoluteLinearLR(optimizer, -8e-6 / len(self._dataloader), 1e-2)

    @override
    def backward(self, images: torch.Tensor, labels: torch.Tensor, toolbox: TrainerToolbox) -> tuple[float, dict[
        str, float]]:
        mask = toolbox.model(images)
        loss, metrics = toolbox.criterion(mask, labels)
        loss.backward()
        return loss.item(), metrics

    @override
    def validate_case(self, image: torch.Tensor, label: torch.Tensor, toolbox: TrainerToolbox) -> tuple[float, dict[
        str, float], torch.Tensor]:
        image, label = image.unsqueeze(0), label.unsqueeze(0)
        mask = (toolbox.ema if toolbox.ema else toolbox.model)(image)
        loss, metrics = toolbox.criterion(mask, label)
        return -loss.item(), metrics, mask.squeeze(0)
