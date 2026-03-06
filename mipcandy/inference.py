from abc import ABCMeta
from collections.abc import Generator
from math import log, ceil
from os import PathLike, listdir
from os.path import isdir, basename, exists
from typing import Sequence

import torch
from torch import nn

from mipcandy.data import save_image, Loader, UnsupervisedDataset, PathBasedUnsupervisedDataset
from mipcandy.layer import WithPaddingModule, WithNetwork
from mipcandy.types import SupportedPredictant, Device, AmbiguousShape


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


class Predictor(WithPaddingModule, WithNetwork, metaclass=ABCMeta):
    def __init__(self, experiment_folder: str | PathLike[str], example_shape: AmbiguousShape, *,
                 checkpoint: str = "checkpoint_best.pth", device: Device = "cpu") -> None:
        WithPaddingModule.__init__(self, device)
        WithNetwork.__init__(self, device)
        self._experiment_folder: str = experiment_folder
        self._example_shape: AmbiguousShape = example_shape
        self._checkpoint: str = checkpoint
        self._model: nn.Module | None = None

    def lazy_load_model(self) -> None:
        if self._model:
            return
        self._model = self.load_model(self._example_shape, False,
                                      path=f"{self._experiment_folder}/{self._checkpoint}")
        self._model.eval()

    def predict_image(self, image: torch.Tensor, *, batch: bool = False) -> torch.Tensor:
        self.lazy_load_model()
        image = image.to(self._device)
        if not batch:
            image = image.unsqueeze(0)
        padding_module = self.get_padding_module()
        if padding_module:
            image = padding_module(image)
        output = self._model(image)
        restoring_module = self.get_restoring_module()
        if restoring_module:
            output = restoring_module(output)
        return output if batch else output.squeeze(0)

    def _predict(self, x: SupportedPredictant | UnsupervisedDataset) -> Generator[
        tuple[torch.Tensor, str | None], None, None]:
        if isinstance(x, PathBasedUnsupervisedDataset):
            for case, path in zip(x, x.paths()):
                yield self.predict_image(case), path
            return
        if isinstance(x, UnsupervisedDataset):
            for case in x:
                yield self.predict_image(case), None
            return
        if isinstance(x, str):
            if isdir(x):
                for case in listdir(x):
                    yield self.predict_image(Loader.do_load(f"{x}/{case}")), case
            else:
                yield self.predict_image(Loader.do_load(x)), basename(x)
            return
        if isinstance(x, torch.Tensor):
            yield self.predict_image(x), None
            return
        for case in x:
            if isinstance(case, str):
                yield self.predict_image(Loader.do_load(case)), case[case.rfind("/") + 1:]
            elif isinstance(case, torch.Tensor):
                yield self.predict_image(case), None
            else:
                raise TypeError(f"Unexpected type of element {type(case)}")

    def predict(self, x: SupportedPredictant | UnsupervisedDataset) -> list[torch.Tensor]:
        return [output for output, _ in self._predict(x)]

    @staticmethod
    def save_prediction(output: torch.Tensor, path: str | PathLike[str]) -> None:
        save_image(output, path)

    def save_predictions(self, outputs: Sequence[torch.Tensor], folder: str | PathLike[str], *,
                         filenames: Sequence[str | PathLike[str]] | None = None) -> None:
        if not exists(folder):
            raise FileNotFoundError(f"Folder {folder} does not exist")
        if not filenames:
            num_digits = ceil(log(len(outputs)))
            filenames = [f"prediction_{str(i).zfill(num_digits)}.{
            "png" if output.ndim == 3 and output.shape[0] in (1, 3) else "mha"}" for i, output in enumerate(outputs)]
        for i, prediction in enumerate(outputs):
            self.save_prediction(prediction, f"{folder}/{filenames[i]}")

    def predict_to_files(self, x: SupportedPredictant | UnsupervisedDataset,
                         folder: str | PathLike[str]) -> list[str] | None:
        if not exists(folder):
            raise FileNotFoundError(f"Folder {folder} does not exist")
        result: list[str] | None = None
        for i, (output, name) in enumerate(self._predict(x)):
            if name is not None:
                if result is None:
                    result = []
                result.append(name)
            else:
                ext = "png" if output.ndim == 3 and output.shape[0] in (1, 3) else "mha"
                name = f"prediction_{i}.{ext}"
            self.save_prediction(output, f"{folder}/{name}")
        return result

    def __call__(self, x: SupportedPredictant | UnsupervisedDataset) -> list[torch.Tensor]:
        return self.predict(x)
