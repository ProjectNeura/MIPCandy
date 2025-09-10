from abc import ABCMeta, abstractmethod
from os import PathLike, listdir, makedirs
from os.path import exists
from random import choice, randint
from shutil import move, copy
from typing import Literal, override, Self, Sequence, TypeVar, Generic, Any

import torch
from torch.utils.data import Dataset

from mipcandy.data.io import load_image
from mipcandy.layer import HasDevice
from mipcandy.types import Transform

T = TypeVar("T")


class KFPicker(object, metaclass=ABCMeta):
    @staticmethod
    @abstractmethod
    def pick(self, choices: Sequence[T], fold: Literal[0, 1, 2, 3, 4, "all"]) -> Sequence[T]:
        raise NotImplementedError


class OrderedKFPicker(KFPicker):
    @staticmethod
    @override
    def pick(self, choices: Sequence[T], fold: Literal[0, 1, 2, 3, 4, "all"]) -> Sequence[T]:
        if fold == "all":
            return choices[::4]
        size = len(choices) // 5
        return choices[size * fold: size * (fold + 1)]


class Loader(object):
    @staticmethod
    def do_load(path: str | PathLike[str], *, is_label: bool = False, device: torch.device | str = "cpu",
                **kwargs) -> torch.Tensor:
        return load_image(path, is_label=is_label, device=device, **kwargs)


class _AbstractDataset(Dataset, Loader, HasDevice, Generic[T], Sequence[T], metaclass=ABCMeta):
    @abstractmethod
    def load(self, idx: int) -> T:
        raise NotImplementedError

    @override
    def __getitem__(self, idx: int) -> T:
        return self.load(idx)


D = TypeVar("D", bound=Sequence)


class UnsupervisedDataset(_AbstractDataset[torch.Tensor], Generic[D], metaclass=ABCMeta):
    def __init__(self, images: D, *, device: torch.device | str = "cpu") -> None:
        super().__init__(device)
        self._images: D = images

    @override
    def __len__(self) -> int:
        return len(self._images)


class SupervisedDataset(_AbstractDataset[tuple[torch.Tensor, torch.Tensor]], Generic[D], metaclass=ABCMeta):
    def __init__(self, images: D, labels: D, *, device: torch.device | str = "cpu") -> None:
        super().__init__(device)
        if len(images) != len(labels):
            raise ValueError(f"Unmatched number of images {len(images)} and labels {len(labels)}")
        self._images: D = images
        self._labels: D = labels

    @override
    def __len__(self) -> int:
        return len(self._images)

    def fold(self, *, fold: Literal[0, 1, 2, 3, 4, "all"] = "all", picker: type[KFPicker] = OrderedKFPicker) -> Self:
        # Todo
        ...


class DatasetFromMemory(UnsupervisedDataset[Sequence[torch.Tensor]]):
    def __init__(self, images: Sequence[torch.Tensor], device: torch.device | str = "cpu") -> None:
        super().__init__(images, device=device)

    @override
    def load(self, idx: int) -> torch.Tensor:
        return self._images[idx].to(self._device)


class MergedDataset(SupervisedDataset[UnsupervisedDataset]):
    def __init__(self, images: UnsupervisedDataset, labels: UnsupervisedDataset, *,
                 device: torch.device | str = "cpu") -> None:
        super().__init__(images, labels, device=device)

    @override
    def load(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self._images[idx].to(self._device), self._labels[idx].to(self._device)


class NNUNetDataset(SupervisedDataset[list[str]]):
    def __init__(self, folder: str | PathLike[str], *, split: Literal["Tr", "Ts"] = "Tr", prefix: str = "",
                 align_spacing: bool = False, image_transform: Transform | None = None,
                 label_transform: Transform | None = None, device: torch.device | str = "cpu") -> None:
        images: list[str] = [f for f in listdir(f"{folder}/images{split}") if f.startswith(prefix)]
        images.sort()
        labels: list[str] = [f for f in listdir(f"{folder}/labels{split}") if f.startswith(prefix)]
        labels.sort()
        super().__init__(images, labels, device=device)
        self._folder: str = folder
        self._split: str = split
        self._prefix: str = prefix
        self._align_spacing: bool = align_spacing
        self._image_transform: Transform | None = image_transform
        self._label_transform: Transform | None = label_transform

    @staticmethod
    def _create_subset(folder: str) -> None:
        if exists(folder) and len(listdir(folder)) > 0:
            raise FileExistsError(f"{folder} already exists and is not empty")
        makedirs(folder, exist_ok=True)

    def divide(self) -> tuple[list[int], list[int]]:
        positives = []
        negatives = []
        for i in range(len(self)):
            _, label = self[i]
            (positives if label.max() > 0 else negatives).append(i)
        return positives, negatives

    def split(self, split: Literal["Tr", "Ts"], size: int, *, exclusive: bool = True,
              positive_only: bool = False) -> Self:
        if split == self._split:
            raise FileExistsError(f"Split {split} already exists")
        if not (0 < size < len(self)):
            raise ValueError(f"Invalid test set size {size}, expected (0, {len(self)})")
        images_folder = f"{self._folder}/images{split}"
        labels_folder = f"{self._folder}/labels{split}"
        self._create_subset(images_folder)
        self._create_subset(labels_folder)
        op = move if exclusive else copy
        images, labels = self._images.copy(), self._labels.copy()
        num_cases = len(self)
        positives = self.divide()[0]
        if positive_only and len(positives) < size:
            raise RuntimeError(f"Not enough positive cases {len(positives)}/{size}")
        for _ in range(size):
            if positive_only:
                i = choice(positives)
                positives.remove(i)
                image, label = self._images[i], self._labels[i]
                images.remove(image)
                labels.remove(label)
            else:
                num_cases -= 1
                i = randint(0, num_cases)
                image, label = images.pop(i), labels.pop(i)
            op(f"{self._folder}/images{self._split}/{image}", f"{images_folder}/{image}")
            op(f"{self._folder}/labels{self._split}/{label}", f"{labels_folder}/{label}")
        if exclusive:
            self._images, self._labels = images, labels
        return NNUNetDataset(self._folder, split=split, prefix=self._prefix, image_transform=self._image_transform,
                             label_transform=self._label_transform)

    @override
    def load(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        image = self.do_load(
            f"{self._folder}/images{self._split}/{self._images[idx]}", align_spacing=self._align_spacing,
            device=self._device
        )
        label = self.do_load(
            f"{self._folder}/labels{self._split}/{self._labels[idx]}", is_label=True, align_spacing=self._align_spacing,
            device=self._device
        )
        if self._image_transform:
            image = self._image_transform(image)
        if self._label_transform:
            label = self._label_transform(label)
        return image, label


class BinarizedDataset(NNUNetDataset):
    positive_ids: tuple[int, ...] = (1, 2)

    @override
    def load(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        image, label = super().load(idx)
        for pid in self.positive_ids:
            label[label == pid] = -1
        label[label > 0] = 0
        label[label == -1] = 1
        return image, label
