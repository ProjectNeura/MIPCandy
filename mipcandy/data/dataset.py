from abc import ABCMeta, abstractmethod
from os import PathLike, listdir, makedirs
from os.path import exists
from random import choice, randint
from shutil import move, copy
from typing import Literal, override, Self, Sized, Sequence

import torch
from torch.utils.data import Dataset

from mipcandy.data.io import load_image
from mipcandy.layer import HasDevice
from mipcandy.types import Transform


class Loader(object):
    @staticmethod
    def do_load(path: str | PathLike[str], *, is_label: bool = False, device: torch.device | str = "cpu",
                **kwargs) -> torch.Tensor:
        return load_image(path, is_label=is_label, device=device, **kwargs)


class _AbstractDataset(Dataset, Loader, HasDevice, Sized, metaclass=ABCMeta):
    pass


class UnsupervisedDataset(_AbstractDataset, metaclass=ABCMeta):
    @abstractmethod
    def load(self, idx: int) -> torch.Tensor:
        raise NotImplementedError

    @override
    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.load(idx)


class SupervisedDataset(_AbstractDataset, metaclass=ABCMeta):
    @abstractmethod
    def load(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    @override
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.load(idx)


class DatasetFromMemory(UnsupervisedDataset):
    def __init__(self, images: Sequence[torch.Tensor],
                 device: torch.device | str = "cpu") -> None:
        super().__init__(device)
        self._images: Sequence[torch.Tensor] = images

    @override
    def load(self, idx: int) -> torch.Tensor:
        return self._images[idx].to(self._device)

    @override
    def __len__(self) -> int:
        return len(self._images)


class MergedDataset(SupervisedDataset):
    def __init__(self, images: UnsupervisedDataset, labels: UnsupervisedDataset, *,
                 device: torch.device | str = "cpu") -> None:
        super().__init__(device)
        if len(images) != len(labels):
            raise ValueError("Unmatched number of images and labels")
        self._images: UnsupervisedDataset = images
        self._labels: UnsupervisedDataset = labels

    @override
    def load(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self._images[idx].to(self._device), self._labels[idx].to(self._device)

    @override
    def __len__(self) -> int:
        return len(self._images)


class NNUNetDataset(SupervisedDataset):
    def __init__(self, folder: str | PathLike[str], *, split: Literal["Tr", "Ts"] = "Tr", prefix: str = "",
                 align_spacing: bool = False, image_transform: Transform | None = None,
                 label_transform: Transform | None = None, device: torch.device | str = "cpu") -> None:
        super().__init__(device)
        self._folder: str = folder
        self._split: str = split
        self._prefix: str = prefix
        self._align_spacing: bool = align_spacing
        self._image_transform: Transform | None = image_transform
        self._label_transform: Transform | None = label_transform
        self._images: list[str] = [f for f in listdir(f"{folder}/images{split}") if f.startswith(prefix)]
        self._images.sort()
        self._labels: list[str] = [f for f in listdir(f"{folder}/labels{split}") if f.startswith(prefix)]
        self._labels.sort()
        self._num_cases: int = len(self._images)
        num_labels = len(self._labels)
        if num_labels != self._num_cases:
            raise FileNotFoundError(f"Inconsistent number of labels ({num_labels}) and images ({self._num_cases})")

    @staticmethod
    def _create_subset(folder: str) -> None:
        if exists(folder) and len(listdir(folder)) > 0:
            raise FileExistsError(f"{folder} already exists and is not empty")
        makedirs(folder, exist_ok=True)

    def divide(self) -> tuple[list[int], list[int]]:
        positives = []
        negatives = []
        for i in range(self._num_cases):
            _, label = self[i]
            (positives if label.max() > 0 else negatives).append(i)
        return positives, negatives

    def split(self, split: Literal["Tr", "Ts"], size: int, *, exclusive: bool = True,
              positive_only: bool = False) -> Self:
        if split == self._split:
            raise FileExistsError(f"Split {split} already exists")
        if not (0 < size < self._num_cases):
            raise ValueError(f"Invalid test set size {size}, expected (0, {self._num_cases})")
        images_folder = f"{self._folder}/images{split}"
        labels_folder = f"{self._folder}/labels{split}"
        self._create_subset(images_folder)
        self._create_subset(labels_folder)
        op = move if exclusive else copy
        images, labels = self._images.copy(), self._labels.copy()
        num_cases = self._num_cases
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
            self._num_cases -= size
        return NNUNetDataset(self._folder, split=split, prefix=self._prefix, image_transform=self._image_transform,
                             label_transform=self._label_transform)

    @override
    def __len__(self) -> int:
        return self._num_cases

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
