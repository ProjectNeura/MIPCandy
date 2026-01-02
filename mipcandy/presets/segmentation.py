from abc import ABCMeta
from collections import defaultdict
from typing import override

import torch
from rich.progress import Progress, SpinnerColumn
from torch import nn, optim

from mipcandy.common import AbsoluteLinearLR, DiceBCELossWithLogits
from mipcandy.data import visualize2d, visualize3d, overlay, auto_convert, convert_logits_to_ids, \
    revert_sliding_window, PathBasedSupervisedDataset, SupervisedSWDataset
from mipcandy.training import Trainer, TrainerToolbox, try_append_all
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
    def validate_case(self, image: torch.Tensor, label: torch.Tensor, toolbox: TrainerToolbox) -> tuple[float, dict[
        str, float], torch.Tensor]:
        image, label = image.unsqueeze(0), label.unsqueeze(0)
        mask = (toolbox.ema if toolbox.ema else toolbox.model)(image)
        loss, metrics = toolbox.criterion(mask, label)
        return -loss.item(), metrics, mask.squeeze(0)


class SlidingTrainer(SegmentationTrainer, metaclass=ABCMeta):
    overlap: float = .5
    validation_dataset: PathBasedSupervisedDataset | None = None
    slided_validation_dataset: SupervisedSWDataset | None = None

    def set_validation_datasets(self, dataset: PathBasedSupervisedDataset, slided_dataset: SupervisedSWDataset) -> None:
        self.validation_dataset = dataset
        self.slided_validation_dataset = slided_dataset

    @override
    def validate(self, toolbox: TrainerToolbox) -> tuple[float, dict[str, list[float]]]:
        image_files = self.slided_validation_dataset._images.paths()
        groups = defaultdict(list)
        for idx, filename in enumerate(image_files):
            case_id = filename.split("_")[0]
            groups[case_id].append(idx)
        toolbox.model.eval()
        if toolbox.ema:
            toolbox.ema.eval()
        score = 0.0
        worst_score = float("+inf")
        metrics = {}
        num_cases = len(groups)
        with torch.no_grad(), Progress(
                *Progress.get_default_columns(), SpinnerColumn(), console=self._console
        ) as progress:
            val_prog = progress.add_task("Validating", total=num_cases)
            for case_idx, case_id in enumerate(sorted(groups.keys())):
                patches = [self.slided_validation_dataset[idx][0].to(self._device) for idx in groups[case_id]]
                label = self.validation_dataset[case_idx][1].to(self._device)
                progress.update(val_prog, description=f"Validating case {case_id} ({len(patches)} patches)")
                case_score, case_metrics, output = self.validate_case(patches, label, toolbox)
                score += case_score
                if case_score < worst_score:
                    self._tracker.worst_case = (patches[0], label, output)
                    worst_score = case_score
                try_append_all(case_metrics, metrics)
                progress.update(val_prog, advance=1, description=f"Validating ({case_score:.4f})")
        return score / num_cases, metrics

    @override
    def validate_case(self, patches: list[torch.Tensor], label: torch.Tensor, toolbox: TrainerToolbox) -> tuple[
        float, dict[str, float], torch.Tensor]:
        model = toolbox.ema if toolbox.ema else toolbox.model
        outputs = []
        for patch in patches:
            outputs.append(model(patch.unsqueeze(0)).squeeze(0))
        reconstructed = revert_sliding_window(outputs, overlap=self.overlap)
        pad = []
        for r, l in zip(reversed(reconstructed.shape[2:]), reversed(label.shape[1:])):
            pad.extend([0, r - l])
        label = nn.functional.pad(label, pad)
        loss, metrics = toolbox.criterion(reconstructed, label.unsqueeze(0))
        return -loss.item(), metrics, reconstructed.squeeze(0)
