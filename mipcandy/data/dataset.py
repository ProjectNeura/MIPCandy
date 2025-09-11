from abc import ABCMeta, abstractmethod
from os import PathLike, listdir, makedirs
from os.path import exists
from random import choices
from typing import Literal, override, Self, Sequence, TypeVar, Generic, Any

import torch
from torch.utils.data import Dataset

from mipcandy.data.io import load_image
from mipcandy.layer import HasDevice
from mipcandy.types import Transform


class KFPicker(object, metaclass=ABCMeta):
    @staticmethod
    @abstractmethod
    def pick(n: int, fold: Literal[0, 1, 2, 3, 4, "all"]) -> tuple[int, ...]:
        raise NotImplementedError


class OrderedKFPicker(KFPicker):
    @staticmethod
    @override
    def pick(n: int, fold: Literal[0, 1, 2, 3, 4, "all"]) -> tuple[int, ...]:
        if fold == "all":
            return tuple(range(0, n, 4))
        size = n // 5
        return tuple(range(size * fold, size * (fold + 1)))


class RandomKFPicker(OrderedKFPicker):
    @staticmethod
    @override
    def pick(n: int, fold: Literal[0, 1, 2, 3, 4, "all"]) -> tuple[int, ...]:
        return tuple(choices(range(n), k=n // 5)) if fold == "all" else super().pick(n, fold)


class Loader(object):
    @staticmethod
    def do_load(path: str | PathLike[str], *, is_label: bool = False, device: torch.device | str = "cpu",
                **kwargs) -> torch.Tensor:
        return load_image(path, is_label=is_label, device=device, **kwargs)


T = TypeVar("T")


class _AbstractDataset(Dataset, Loader, HasDevice, Generic[T], Sequence[T], metaclass=ABCMeta):
    @abstractmethod
    def load(self, idx: int) -> T:
        raise NotImplementedError

    @override
    def __getitem__(self, idx: int) -> T:
        return self.load(idx)


D = TypeVar("D", bound=Sequence[Any])


class UnsupervisedDataset(_AbstractDataset[torch.Tensor], Generic[D], metaclass=ABCMeta):
    """
    Do not use this as a generic class. Only parameterize it if you are inheriting from it.
    """

    def __init__(self, images: D, *, device: torch.device | str = "cpu") -> None:
        super().__init__(device)
        self._images: D = images

    @override
    def __len__(self) -> int:
        return len(self._images)


class SupervisedDataset(_AbstractDataset[tuple[torch.Tensor, torch.Tensor]], Generic[D], metaclass=ABCMeta):
    """
    Do not use this as a generic class. Only parameterize it if you are inheriting from it.
    """

    def __init__(self, images: D, labels: D, *, device: torch.device | str = "cpu") -> None:
        super().__init__(device)
        if len(images) != len(labels):
            raise ValueError(f"Unmatched number of images {len(images)} and labels {len(labels)}")
        self._images: D = images
        self._labels: D = labels

    @override
    def __len__(self) -> int:
        return len(self._images)

    @abstractmethod
    def construct_new(self, images: D, labels: D) -> Self:
        raise NotImplementedError

    def fold(self, *, fold: Literal[0, 1, 2, 3, 4, "all"] = "all", picker: type[KFPicker] = OrderedKFPicker) -> tuple[
        Self, Self]:
        indexes = picker.pick(len(self), fold)
        images_train = []
        labels_train = []
        images_val = []
        labels_val = []
        for i in range(len(self)):
            if i in indexes:
                images_val.append(self._images[i])
                labels_val.append(self._labels[i])
            else:
                images_train.append(self._images[i])
                labels_train.append(self._labels[i])
        return self.construct_new(images_train, labels_train), self.construct_new(images_val, labels_val)


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

    @override
    def construct_new(self, images: D, labels: D) -> Self:
        return MergedDataset(DatasetFromMemory(images), DatasetFromMemory(labels), device=self._device)


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
        self._split: Literal["Tr", "Ts", "fold"] = split
        self._prefix: str = prefix
        self._align_spacing: bool = align_spacing
        self._image_transform: Transform | None = image_transform
        self._label_transform: Transform | None = label_transform

    @staticmethod
    def _create_subset(folder: str) -> None:
        if exists(folder) and len(listdir(folder)) > 0:
            raise FileExistsError(f"{folder} already exists and is not empty")
        makedirs(folder, exist_ok=True)

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

    @override
    def construct_new(self, images: D, labels: D) -> Self:
        if self._split == "fold":
            raise ValueError("Cannot construct a new dataset from a fold")
        new = NNUNetDataset(self._folder, split=self._split, prefix=self._prefix, align_spacing=self._align_spacing,
                            image_transform=self._image_transform, label_transform=self._label_transform,
                            device=self._device)
        new._split = "fold"
        new._images = images
        new._labels = labels
        return new


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
