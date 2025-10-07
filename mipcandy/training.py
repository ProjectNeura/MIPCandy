from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from hashlib import md5
from os import PathLike, urandom, makedirs, environ
from os.path import exists
from random import seed as random_seed, randint
from shutil import copy
from threading import Lock
from time import time
from typing import Sequence, override, Callable

import numpy as np
import torch
from matplotlib import pyplot as plt
from pandas import DataFrame
from rich.console import Console
from rich.progress import Progress, SpinnerColumn
from rich.table import Table
from torch import nn, optim
from torch.utils.data import DataLoader

from mipcandy.common import Pad2d, Pad3d, quotient_regression, quotient_derivative, quotient_bounds
from mipcandy.config import load_settings, load_secrets
from mipcandy.frontend import Frontend
from mipcandy.layer import WithPaddingModule
from mipcandy.sanity_check import sanity_check
from mipcandy.sliding_window import SWMetadata, SlidingWindow
from mipcandy.types import Params, Setting, Device


def try_append(new: float, to: dict[str, list[float]], key: str) -> None:
    if key in to:
        to[key].append(new)
    else:
        to[key] = [new]


def try_append_all(new: dict[str, float], to: dict[str, list[float]]) -> None:
    for key, value in new.items():
        try_append(value, to, key)


@dataclass
class TrainerToolbox(object):
    model: nn.Module
    optimizer: optim.Optimizer
    scheduler: optim.lr_scheduler.LRScheduler
    criterion: nn.Module
    ema: optim.swa_utils.AveragedModel | None = None


class Trainer(WithPaddingModule, metaclass=ABCMeta):
    def __init__(self, trainer_folder: str | PathLike[str], dataloader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
                 validation_dataloader: DataLoader[tuple[torch.Tensor, torch.Tensor]], *,
                 device: Device = "cpu") -> None:
        super().__init__(device)
        self._trainer_folder: str = trainer_folder
        self._trainer_variant: str = self.__class__.__name__
        self._experiment_id: str = "tbd"
        self._dataloader: DataLoader[tuple[torch.Tensor, torch.Tensor]] = dataloader
        self._validation_dataloader: DataLoader[tuple[torch.Tensor, torch.Tensor]] = validation_dataloader
        self._metrics: dict[str, list[float]] = {}
        self._epoch_metrics: dict[str, list[float]] = {}
        self._best_score: float = float("-inf")
        self._worst_case: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None
        self._frontend: Frontend = Frontend({})
        self._lock: Lock = Lock()

    def set_frontend(self, frontend: type[Frontend], *, path_to_secrets: str | PathLike[str] | None = None) -> None:
        self._frontend = frontend(load_secrets(path=path_to_secrets) if path_to_secrets else load_secrets())

    def initialized(self) -> bool:
        return self._experiment_id != "tbd"

    def experiment_folder(self) -> str:
        return f"{self._trainer_folder}/{self._trainer_variant}/{self._experiment_id}"

    def allocate_experiment_folder(self) -> str:
        self._experiment_id = datetime.now().strftime("%Y%m%d-%H-") + md5(urandom(8)).hexdigest()[:4]
        experiment_folder = self.experiment_folder()
        return self.allocate_experiment_folder() if exists(experiment_folder) else experiment_folder

    def init_experiment(self) -> None:
        if self.initialized():
            raise RuntimeError("Experiment already initialized")
        makedirs(self._trainer_folder, exist_ok=True)
        experiment_folder = self.allocate_experiment_folder()
        makedirs(experiment_folder)
        t = datetime.now()
        with open(f"{experiment_folder}/logs.txt", "w") as f:
            f.write(f"File created by FightTumor, copyright (C) {t.year} Project Neura. All rights reserved\n")
        self.log(f"Experiment (ID {self._experiment_id}) created at {t}")
        self.log(f"Trainer: {self.__class__.__name__}")

    def log(self, msg: str, *, console: bool = True) -> None:
        msg = f"[{datetime.now()}] {msg}"
        if self.initialized():
            with open(f"{self.experiment_folder()}/logs.txt", "a") as f:
                f.write(f"{msg}\n")
        if console:
            with self._lock:
                print(msg)

    def record(self, metric: str, value: float) -> None:
        try_append(value, self._epoch_metrics, metric)

    def _record(self, metric: str, value: float) -> None:
        try_append(value, self._metrics, metric)

    def record_all(self, metrics: dict[str, float]) -> None:
        try_append_all(metrics, self._epoch_metrics)

    def _bump_metrics(self) -> None:
        for metric, values in self._epoch_metrics.items():
            epoch_overall = sum(values) / len(values)
            try_append(epoch_overall, self._metrics, metric)
        self._epoch_metrics.clear()

    def save_metrics(self) -> None:
        df = DataFrame(self._metrics)
        df.index = range(1, len(df) + 1)
        df.index.name = "epoch"
        df.to_csv(f"{self.experiment_folder()}/metrics.csv")

    def save_metric_curve(self, name: str, values: Sequence[float]) -> None:
        name = name.capitalize()
        plt.plot(values)
        plt.title(f"{name} over Epoch")
        plt.xlabel("Epoch")
        plt.ylabel(name)
        plt.grid()
        plt.savefig(f"{self.experiment_folder()}/{name.lower()}.png")
        plt.close()

    def save_metric_curve_combo(self, metrics: dict[str, Sequence[float]], *, title: str = "All Metrics") -> None:
        for name, values in metrics.items():
            plt.plot(values, label=name.capitalize())
        plt.title(title)
        plt.xlabel("Epoch")
        plt.legend()
        plt.grid()
        plt.savefig(f"{self.experiment_folder()}/{title.lower()}.png")
        plt.close()

    def save_metric_curves(self, *, names: Sequence[str] | None = None) -> None:
        if names is None:
            for name, values in self._metrics.items():
                self.save_metric_curve(name, values)
        else:
            for name in names:
                self.save_metric_curve(name, self._metrics[name])

    def save_progress(self, *, names: Sequence[str] = ("combined loss", "val score")) -> None:
        self.save_metric_curve_combo({name: self._metrics[name] for name in names}, title="Progress")

    def save_preview(self, image: torch.Tensor, label: torch.Tensor, mask: torch.Tensor, *,
                     quality: float = .75) -> None:
        ...

    @abstractmethod
    def build_network(self, example_shape: tuple[int, ...]) -> nn.Module:
        raise NotImplementedError

    @abstractmethod
    def build_optimizer(self, params: Params) -> optim.Optimizer:
        raise NotImplementedError

    @abstractmethod
    def build_scheduler(self, optimizer: optim.Optimizer, num_epochs: int) -> optim.lr_scheduler.LRScheduler:
        raise NotImplementedError

    @abstractmethod
    def build_criterion(self) -> nn.Module:
        raise NotImplementedError

    @abstractmethod
    def backward(self, images: torch.Tensor, labels: torch.Tensor, toolbox: TrainerToolbox) -> tuple[float, dict[
        str, float]]:
        raise NotImplementedError

    def train_batch(self, images: torch.Tensor, labels: torch.Tensor, toolbox: TrainerToolbox) -> tuple[float, dict[
        str, float]]:
        toolbox.optimizer.zero_grad()
        loss, metrics = self.backward(images, labels, toolbox)
        toolbox.optimizer.step()
        toolbox.scheduler.step()
        if toolbox.ema:
            toolbox.ema.update_parameters(toolbox.model)
        return loss, metrics

    def train_epoch(self, epoch: int, toolbox: TrainerToolbox) -> None:
        toolbox.model.train()
        if toolbox.ema:
            toolbox.ema.train()
        with Progress(*Progress.get_default_columns(), SpinnerColumn()) as progress:
            epoch_prog = progress.add_task(f"Epoch {epoch}", total=len(self._dataloader))
            for images, labels in self._dataloader:
                images, labels = images.to(self._device), labels.to(self._device)
                padding_module = self.get_padding_module()
                if padding_module:
                    images, labels = padding_module(images), padding_module(labels)
                loss, metrics = self.train_batch(images, labels, toolbox)
                self.record("combined loss", loss)
                self.record_all(metrics)
                progress.update(epoch_prog, advance=1, description=f"Training epoch {epoch} ({loss:.4f})")
        self._bump_metrics()

    @abstractmethod
    def validate_case(self, image: torch.Tensor, label: torch.Tensor, toolbox: TrainerToolbox) -> tuple[float, dict[
        str, float], torch.Tensor]:
        raise NotImplementedError

    def validate(self, toolbox: TrainerToolbox) -> tuple[float, dict[str, list[float]]]:
        if self._validation_dataloader.batch_size != 1:
            raise RuntimeError("Validation dataloader should have batch size 1")
        toolbox.model.eval()
        if toolbox.ema:
            toolbox.ema.eval()
        score = 0
        worst_score = float("+inf")
        metrics = {}
        num_cases = len(self._validation_dataloader)
        with Progress(*Progress.get_default_columns(), SpinnerColumn()) as progress:
            val_prog = progress.add_task(f"Validating", total=num_cases)
            for image, label in self._validation_dataloader:
                image, label = image.to(self._device), label.to(self._device)
                padding_module = self.get_padding_module()
                if padding_module:
                    image, label = padding_module(image), padding_module(label)
                image, label = image.squeeze(0), label.squeeze(0)
                case_score, case_metrics, mask = self.validate_case(image, label, toolbox)
                score += case_score
                if case_score < worst_score:
                    self._worst_case = (image, label, mask)
                    worst_score = case_score
                try_append_all(case_metrics, metrics)
                progress.update(val_prog, advance=1, description=f"Validating ({case_score:.4f})")
        return score / num_cases, metrics

    def predict_maximum_validation_score(self, num_epochs: int, *, degree: int = 5) -> tuple[int, float]:
        val_scores = np.array(self._metrics["val score"])
        a, b = quotient_regression(np.arange(len(val_scores)), val_scores, degree, degree)
        da, db = quotient_derivative(a, b)
        max_roc = float(da[0] / db[0])
        max_val_score = float(a[0] / b[0])
        bounds = quotient_bounds(a, b, None, max_val_score * (1 - max_roc), x_start=0, x_stop=num_epochs, x_step=1)
        return round(bounds[1]) + 1, max_val_score if bounds else (0, 0)

    def set_seed(self, seed: int) -> None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        random_seed(seed)
        np.random.seed(seed)
        environ['PYTHONHASHSEED'] = str(seed)
        if self.initialized():
            self.log(f"Set to manual seed {seed}")

    @staticmethod
    def filter_train_params(**kwargs) -> dict[str, Setting]:
        return {k: v for k, v in kwargs.items() if k in (
            "note", "num_checkpoints", "ema", "seed", "early_stop_tolerance", "val_score_prediction",
            "val_score_prediction_degree", "save_preview", "preview_quality"
        )}

    def train_with_settings(self, num_epochs: int, **kwargs) -> None:
        settings = self.filter_train_params(**load_settings())
        settings.update(kwargs)
        self.train(num_epochs, **settings)

    def show_metrics(self, epoch: int, *, metrics: dict[str, list[float]] | None = None, prefix: str = "training",
                     epochwise: bool = True, skip: Callable[[str, list[float]], bool] | None = None) -> None:
        if not metrics:
            metrics = self._metrics
        prefix = prefix.capitalize()
        table = Table(title=f"Epoch {epoch} {prefix}")
        table.add_column("Metric")
        table.add_column("Mean Value", style="green")
        table.add_column("Span", style="cyan")
        table.add_column("Diff", style="magenta")
        for metric, values in metrics.items():
            if skip and skip(metric, values):
                continue
            span = f"[{min(values):.4f}, {max(values):.4f}]"
            if epochwise:
                value = f"{values[-1]:.4f}"
                diff = f"{values[-1] - values[-2]:+.4f}" if len(values) > 1 else "N/A"
            else:
                mean = sum(values) / len(values)
                value = f"{mean:.4f}"
                diff = f"{mean - self._metrics[metric][-1]:+.4f}" if metric in self._metrics else "N/A"
            table.add_row(metric, value, span, diff)
            self.log(f"{prefix} {metric}: {value} @{span} ({diff})")
        console = Console()
        console.print(table)

    def train(self, num_epochs: int, *, note: str = "", num_checkpoints: int = 5, ema: bool = True,
              seed: int | None = None, early_stop_tolerance: int = 5, val_score_prediction: bool = True,
              val_score_prediction_degree: int = 5, save_preview: bool = True, preview_quality: float = .75) -> None:
        self.init_experiment()
        if note:
            self.log(f"Note: {note}")
        if seed is None:
            seed = randint(0, 100)
        self.set_seed(seed)
        example_input = self._dataloader.dataset[0][0].to(self._device).unsqueeze(0)
        padding_module = self.get_padding_module()
        if padding_module:
            example_input = padding_module(example_input)
        example_shape = tuple(example_input.shape[1:])
        self.log(f"Example input shape: {example_shape}")
        model = self.build_network(example_shape).to(self._device)
        model_name = model.__class__.__name__
        sanity_check_result = sanity_check(model, example_shape, device=self._device)
        self.log(f"Model: {model_name}")
        self.log(str(sanity_check_result))
        self.log(f"Example output shape: {tuple(sanity_check_result.output.shape)}")
        optimizer = self.build_optimizer(model.parameters())
        scheduler = self.build_scheduler(optimizer, num_epochs)
        criterion = self.build_criterion().to(self._device)
        toolbox = TrainerToolbox(model, optimizer, scheduler, criterion)
        if ema:
            toolbox.ema = optim.swa_utils.AveragedModel(model)
        checkpoint_path = lambda v: f"{self.experiment_folder()}/checkpoint_{v}.pth"
        es_tolerance = early_stop_tolerance
        self._frontend.on_experiment_created(self._experiment_id, self._trainer_variant, model_name, note,
                                             sanity_check_result.num_macs, sanity_check_result.num_params, num_epochs,
                                             early_stop_tolerance)
        try:
            for epoch in range(1, num_epochs + 1):
                if early_stop_tolerance == -1:
                    epoch -= 1
                    self.log(f"Early stopping triggered because the validation score has not improved for {
                    es_tolerance} epochs")
                    break
                # Training
                t0 = time()
                self.train_epoch(epoch, toolbox)
                lr = scheduler.get_last_lr()[0]
                self._record("learning rate", lr)
                self.log(f"Learning rate: {lr}")
                self.show_metrics(epoch, skip=lambda m, _: m.startswith("val "))
                torch.save(model.state_dict(), checkpoint_path("latest"))
                if epoch % (num_epochs / num_checkpoints) == 0:
                    copy(checkpoint_path("latest"), checkpoint_path(epoch))
                    self.log(f"Epoch {epoch} checkpoint saved")
                self.log(f"Epoch {epoch} training completed in {time() - t0:.1f} seconds")
                # Validation
                score, metrics = self.validate(toolbox)
                self._record("val score", score)
                msg = f"Validation score: {score:.4f}"
                if epoch > 1:
                    msg += f" ({score - self._metrics["val score"][-2]:+.4f})"
                self.log(msg)
                if val_score_prediction and epoch > val_score_prediction_degree:
                    target_epoch, max_score = self.predict_maximum_validation_score(
                        num_epochs, degree=val_score_prediction_degree
                    )
                    self.log(f"Maximum validation score {max_score:.4f} predicted at epoch {target_epoch}")
                    epoch_durations = self._metrics["epoch duration"]
                    etc = sum(epoch_durations) * (target_epoch - epoch) / len(epoch_durations)
                    self.log(f"Estimated time of completion in {etc:.1f} seconds: {datetime.fromtimestamp(
                        time() + etc):%H:%M:%S}")
                self.show_metrics(epoch, metrics=metrics, prefix="validation", epochwise=False)
                if score > self._best_score:
                    copy(checkpoint_path("latest"), checkpoint_path("best"))
                    self.log(f"======== Best checkpoint updated ({self._best_score:.4f} -> {score:.4f}) ========")
                    self._best_score = score
                    early_stop_tolerance = es_tolerance
                    if save_preview:
                        self.save_preview(*self._worst_case, quality=preview_quality)
                else:
                    early_stop_tolerance -= 1
                epoch_duration = time() - t0
                self._record("epoch duration", epoch_duration)
                self.log(f"Epoch {epoch} completed in {epoch_duration:.1f} seconds")
                self.log(f"=============== Best Validation Score {self._best_score:.4f} ===============")
                self.save_metrics()
                self.save_progress()
                self.save_metric_curves()
                self._frontend.on_experiment_updated(self._experiment_id, epoch, self._metrics, early_stop_tolerance)
        except Exception as e:
            self.log("Training interrupted")
            self.log(repr(e))
            self._frontend.on_experiment_interrupted(self._experiment_id, e)
            raise e
        else:
            self.log("Training completed")
            self._frontend.on_experiment_completed(self._experiment_id)

    def __call__(self, *args, **kwargs) -> None:
        self.train(*args, **kwargs)


class SlidingTrainer(Trainer, SlidingWindow, metaclass=ABCMeta):
    @override
    def build_padding_module(self) -> nn.Module | None:
        window_shape = self.get_window_shape()
        return (Pad2d if len(window_shape) == 2 else Pad3d)(window_shape)

    @abstractmethod
    def validate_case_windowed(self, images: torch.Tensor, labels: torch.Tensor, toolbox: TrainerToolbox,
                               metadata: SWMetadata) -> tuple[float, dict[str, float], torch.Tensor]:
        raise NotImplementedError

    @override
    def validate_case(self, image: torch.Tensor, label: torch.Tensor, toolbox: TrainerToolbox) -> tuple[float, dict[
        str, float], torch.Tensor]:
        image, label = image.unsqueeze(0), label.unsqueeze(0)
        images, metadata = self.do_sliding_window(image)
        labels, _ = self.do_sliding_window(label)
        loss, metrics, masks = self.validate_case_windowed(images, labels, toolbox, metadata)
        mask = self.revert_sliding_window(masks, metadata)
        return loss, metrics, mask.squeeze(0)

    @abstractmethod
    def backward_windowed(self, images: torch.Tensor, labels: torch.Tensor, toolbox: TrainerToolbox,
                          metadata: SWMetadata) -> tuple[float, dict[str, float]]:
        raise NotImplementedError

    @override
    def backward(self, images: torch.Tensor, labels: torch.Tensor, toolbox: TrainerToolbox) -> tuple[float, dict[
        str, float]]:
        images, metadata = self.do_sliding_window(images)
        labels, _ = self.do_sliding_window(labels)
        return self.backward_windowed(images, labels, toolbox, metadata)
