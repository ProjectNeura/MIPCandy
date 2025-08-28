from abc import ABCMeta, abstractmethod
from math import log, ceil
from os import PathLike, listdir
from os.path import isdir, basename
from typing import Sequence, Mapping, Any, override

import torch
from torch import nn

from mipcandy.common import Pad2d, Pad3d, Restore2d, Restore3d
from mipcandy.data import save_image, Loader, UnsupervisedDataset
from mipcandy.layer import WithPaddingModule
from mipcandy.sliding_window import SlidingWindow
from mipcandy.types import SupportedPredictant


def parse_predictant(x: SupportedPredictant, loader: type[Loader], *, as_label: bool = False) -> tuple[list[
    torch.Tensor], list[str] | None]:
    if isinstance(x, str):
        if isdir(x):
            cases = listdir(x)
            return [loader.do_load(f"{x}/{case}", is_label=as_label) for case in cases], cases
        return [loader.do_load(x, is_label=as_label)], [basename(x)]
    if isinstance(x, torch.Tensor):
        return [x], None
    r, filenames = [], None
    for case in x:
        if isinstance(case, str):
            if not filenames:
                filenames = []
            r.append(loader.do_load(case, is_label=as_label))
            filenames.append(case[case.rfind("/") + 1:])
        elif filenames:
            raise TypeError("`x` should be single-typed")
        elif isinstance(case, torch.Tensor):
            r.append(case)
        else:
            raise TypeError(f"Unexpected type of element {type(case)}")
    return r, filenames


class Predictor(WithPaddingModule, metaclass=ABCMeta):
    def __init__(self, experiment_folder: str | PathLike[str], *, checkpoint: str = "checkpoint_best.pth",
                 device: torch.device | str = "cpu") -> None:
        super().__init__(device)
        self._experiment_folder: str = experiment_folder
        self._checkpoint: str = checkpoint
        self._model: nn.Module | None = None

    def lazy_load_model(self) -> None:
        if self._model:
            return
        self._model = self.build_network(torch.load(f"{self._experiment_folder}/{self._checkpoint}")).to(self._device)
        self._model.eval()

    @abstractmethod
    def build_network(self, checkpoint: Mapping[str, Any]) -> nn.Module:
        raise NotImplementedError

    def predict_image(self, image: torch.Tensor, *, batch: bool = False) -> torch.Tensor:
        self.lazy_load_model()
        image = image.to(self._device)
        if not batch:
            image = image.unsqueeze(0)
        padding_module = self.get_padding_module()
        if padding_module:
            image = padding_module(image)
        mask = self._model(image)
        restoring_module = self.get_restoring_module()
        if restoring_module:
            mask = restoring_module(mask)
        return mask if batch else mask.squeeze(0)

    def _predict(self, x: SupportedPredictant | UnsupervisedDataset) -> tuple[list[torch.Tensor], list[str] | None]:
        if isinstance(x, UnsupervisedDataset):
            return [self.predict_image(case) for case in x], None
        images, filenames = parse_predictant(x, Loader)
        return [self.predict_image(image) for image in images], filenames

    def predict(self, x: SupportedPredictant | UnsupervisedDataset) -> list[torch.Tensor]:
        return self._predict(x)[0]

    @staticmethod
    def save_prediction(mask: torch.Tensor, path: str | PathLike[str]) -> None:
        save_image(mask, path)

    def save_predictions(self, masks: Sequence[torch.Tensor], folder: str | PathLike[str], *,
                         filenames: Sequence[str | PathLike[str]] | None = None) -> None:
        if not filenames:
            num_cases = len(masks)
            num_digits = ceil(log(num_cases))
            filenames = [f"prediction_{str(i).zfill(num_digits)}" for i in range(num_cases)]
        for i, prediction in enumerate(masks):
            self.save_prediction(prediction, f"{folder}/{filenames[i]}")

    def predict_to_files(self, x: SupportedPredictant | UnsupervisedDataset, folder: str | PathLike[str]) -> None:
        masks, filenames = self._predict(x)
        self.save_predictions(masks, folder, filenames=filenames)

    def __call__(self, x: SupportedPredictant | UnsupervisedDataset) -> list[torch.Tensor]:
        return self.predict(x)


class SlidingPredictor(Predictor, SlidingWindow, metaclass=ABCMeta):
    @override
    def build_padding_module(self) -> nn.Module | None:
        window_shape = self.get_window_shape()
        return (Pad2d if len(window_shape) == 2 else Pad3d)(window_shape)

    @override
    def build_restoring_module(self, padding_module: nn.Module | None) -> nn.Module | None:
        if not isinstance(padding_module, (Pad2d, Pad3d)):
            raise TypeError("`padding_module` should be either `Pad2d` or `Pad3d`")
        window_shape = self.get_window_shape()
        return (Restore2d if len(window_shape) == 2 else Restore3d)(padding_module)

    @override
    def predict_image(self, image: torch.Tensor, *, batch: bool = False) -> torch.Tensor:
        if not batch:
            image = image.unsqueeze(0)
        images, metadata = self.do_sliding_window(image)
        masks = super().predict_image(images, batch=True)
        masks = self.revert_sliding_window(masks, metadata)
        return masks if batch else masks.squeeze(0)
