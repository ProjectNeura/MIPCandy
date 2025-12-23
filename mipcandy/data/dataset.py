from abc import ABCMeta, abstractmethod
from json import dump
from os import PathLike, listdir, makedirs
from os.path import exists
from random import choices
from shutil import copy2
from typing import Literal, override, Self, Sequence, TypeVar, Generic, Any

import torch
from pandas import DataFrame
from torch.utils.data import Dataset

from mipcandy.data.io import load_image
from mipcandy.data.transform import JointTransform
from mipcandy.layer import HasDevice
from mipcandy.types import Transform, Device


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
            return tuple(range(0, n, 5))
        size = n // 5
        return tuple(range(size * fold, size * (fold + 1)))


class RandomKFPicker(OrderedKFPicker):
    @staticmethod
    @override
    def pick(n: int, fold: Literal[0, 1, 2, 3, 4, "all"]) -> tuple[int, ...]:
        return tuple(choices(range(n), k=n // 5)) if fold == "all" else super().pick(n, fold)


class Loader(object):
    @staticmethod
    def do_load(path: str | PathLike[str], *, is_label: bool = False, device: Device = "cpu", **kwargs) -> torch.Tensor:
        return load_image(path, is_label=is_label, device=device, **kwargs)


T = TypeVar("T")


class _AbstractDataset(Dataset, Loader, HasDevice, Generic[T], Sequence[T], metaclass=ABCMeta):
    @abstractmethod
    def load(self, idx: int) -> T:
        """
        Do not use this directly.
        """
        raise NotImplementedError

    @override
    def __getitem__(self, idx: int) -> T:
        return self.load(idx)


D = TypeVar("D", bound=Sequence[Any])


class UnsupervisedDataset(_AbstractDataset[torch.Tensor], Generic[D], metaclass=ABCMeta):
    """
    Do not use this as a generic class. Only parameterize it if you are inheriting from it.
    """

    def __init__(self, images: D, *, transform: Transform | None = None, device: Device = "cpu") -> None:
        super().__init__(device)
        self._images: D = images
        self._transform: Transform | None = transform.to(device)

    @override
    def __len__(self) -> int:
        return len(self._images)

    @override
    def __getitem__(self, idx: int) -> torch.Tensor:
        item = super().__getitem__(idx).to(self._device)
        return self._transform(item) if self._transform else item


class SupervisedDataset(_AbstractDataset[tuple[torch.Tensor, torch.Tensor]], Generic[D], metaclass=ABCMeta):
    """
    Do not use this as a generic class. Only parameterize it if you are inheriting from it.
    """

    def __init__(self, images: D, labels: D, *, transform: JointTransform | None = None,
                 device: Device = "cpu") -> None:
        super().__init__(device)
        if len(images) != len(labels):
            raise ValueError(f"Unmatched number of images {len(images)} and labels {len(labels)}")
        self._images: D = images
        self._labels: D = labels
        self._transform: JointTransform | None = transform.to(device)

    @override
    def __len__(self) -> int:
        return len(self._images)

    @override
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        image, label = super().__getitem__(idx)
        image, label = image.to(self._device), label.to(self._device)
        return self._transform(image, label) if self._transform else (image, label)

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
    def __init__(self, images: Sequence[torch.Tensor], *, transform: Transform | None = None,
                 device: Device = "cpu") -> None:
        super().__init__(images, transform=transform, device=device)

    @override
    def load(self, idx: int) -> torch.Tensor:
        return self._images[idx]


class MergedDataset(SupervisedDataset[UnsupervisedDataset]):
    def __init__(self, images: UnsupervisedDataset, labels: UnsupervisedDataset, *,
                 transform: JointTransform | None = None, device: Device = "cpu") -> None:
        super().__init__(images, labels, transform=transform, device=device)

    @override
    def load(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self._images[idx], self._labels[idx]

    @override
    def construct_new(self, images: UnsupervisedDataset, labels: UnsupervisedDataset) -> Self:
        return MergedDataset(DatasetFromMemory(images), DatasetFromMemory(labels), transform=self._transform,
                             device=self._device)


class ComposeDataset(_AbstractDataset[tuple[torch.Tensor, torch.Tensor] | torch.Tensor]):
    def __init__(self, bases: Sequence[SupervisedDataset] | Sequence[UnsupervisedDataset], *,
                 device: Device = "cpu") -> None:
        super().__init__(device)
        self._bases: dict[tuple[int, int], SupervisedDataset | UnsupervisedDataset] = {}
        self._len = 0
        for dataset in bases:
            end = len(dataset)
            self._bases[(self._len, self._len + end)] = dataset
            self._len += end

    @override
    def load(self, idx: int) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        for (start, end), base in self._bases.items():
            if start <= idx < end:
                return base.load(idx - start)
        raise IndexError(f"Index {idx} out of range [0, {self._len})")

    @override
    def __len__(self) -> int:
        return self._len


class PathBasedUnsupervisedDataset(UnsupervisedDataset[list[str]], metaclass=ABCMeta):
    def paths(self) -> list[str]:
        return self._images

    def save_paths(self, to: str | PathLike[str]) -> None:
        match (fmt := to.split(".")[-1]):
            case "csv":
                df = DataFrame([{"image": image_path} for image_path in self.paths()])
                df.index = range(len(df))
                df.index.name = "case"
                df.to_csv(to)
            case "json":
                with open(to, "w") as f:
                    dump([{"image": image_path} for image_path in self.paths()], f)
            case "txt":
                with open(to, "w") as f:
                    for image_path in self.paths():
                        f.write(f"{image_path}\n")
            case _:
                raise ValueError(f"Unsupported file extension: {fmt}")


class SimpleDataset(PathBasedUnsupervisedDataset):
    def __init__(self, folder: str | PathLike[str], *, transform: Transform | None = None,
                 device: Device = "cpu") -> None:
        images = listdir(folder)
        images.sort()
        super().__init__(images, transform=transform, device=device)
        self._folder: str = folder

    @override
    def load(self, idx: int) -> torch.Tensor:
        return self.do_load(f"{self._folder}/{self._images[idx]}", device=self._device)


class PathBasedSupervisedDataset(SupervisedDataset[list[str]], metaclass=ABCMeta):
    def paths(self) -> list[tuple[str, str]]:
        return [(self._images[i], self._labels[i]) for i in range(len(self))]

    def save_paths(self, to: str | PathLike[str]) -> None:
        match (fmt := to.split(".")[-1]):
            case "csv":
                df = DataFrame([{"image": image_path, "label": label_path} for image_path, label_path in self.paths()])
                df.index = range(len(df))
                df.index.name = "case"
                df.to_csv(to)
            case "json":
                with open(to, "w") as f:
                    dump([{"image": image_path, "label": label_path} for image_path, label_path in self.paths()], f)
            case "txt":
                with open(to, "w") as f:
                    for image_path, label_path in self.paths():
                        f.write(f"{image_path}\t{label_path}\n")
            case _:
                raise ValueError(f"Unsupported file extension: {fmt}")


class NNUNetDataset(PathBasedSupervisedDataset):
    def __init__(self, folder: str | PathLike[str], *, split: str | Literal["Tr", "Ts"] = "Tr", prefix: str = "",
                 align_spacing: bool = False, transform: JointTransform | None = None, device: Device = "cpu") -> None:
        images: list[str] = [f for f in listdir(f"{folder}/images{split}") if f.startswith(prefix)]
        images.sort()
        labels: list[str] = [f for f in listdir(f"{folder}/labels{split}") if f.startswith(prefix)]
        labels.sort()
        self._multimodal_images: list[list[str]] = []
        if len(images) == len(labels):
            super().__init__(images, labels, transform=transform, device=device)
        else:
            super().__init__([""] * len(labels), labels, transform=transform, device=device)
            current_case = ""
            for image in images:
                case = image[:image.rfind("_")]
                if case != current_case:
                    self._multimodal_images.append([])
                    current_case = case
                self._multimodal_images[-1].append(image)
            if len(self._multimodal_images) != len(self._labels):
                raise ValueError("Unmatched number of images and labels")
        self._folder: str = folder
        self._split: str = split
        self._folded: bool = False
        self._prefix: str = prefix
        self._align_spacing: bool = align_spacing
        self._transform: JointTransform | None = transform

    @staticmethod
    def _create_subset(folder: str) -> None:
        if exists(folder) and len(listdir(folder)) > 0:
            raise FileExistsError(f"{folder} already exists and is not empty")
        makedirs(folder, exist_ok=True)

    @override
    def load(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        image = torch.cat([self.do_load(
            f"{self._folder}/images{self._split}/{path}", align_spacing=self._align_spacing, device=self._device
        ) for path in self._multimodal_images[idx]]) if self._multimodal_images else self.do_load(
            f"{self._folder}/images{self._split}/{self._images[idx]}", align_spacing=self._align_spacing,
            device=self._device
        )
        label = self.do_load(
            f"{self._folder}/labels{self._split}/{self._labels[idx]}", is_label=True, align_spacing=self._align_spacing,
            device=self._device
        )
        return image, label

    def save(self, split: str | Literal["Tr", "Ts"], *, target_folder: str | PathLike[str] | None = None) -> None:
        target_base = target_folder if target_folder else self._folder
        images_target = f"{target_base}/images{split}"
        labels_target = f"{target_base}/labels{split}"
        self._create_subset(images_target)
        self._create_subset(labels_target)
        for image_path, label_path in self.paths():
            copy2(f"{self._folder}/images{self._split}/{image_path}", f"{images_target}/{image_path}")
            copy2(f"{self._folder}/labels{self._split}/{label_path}", f"{labels_target}/{label_path}")
        self._split = split
        self._folded = False

    @override
    def construct_new(self, images: list[str], labels: list[str]) -> Self:
        if self._folded:
            raise ValueError("Cannot construct a new dataset from a fold")
        new = self.__class__(self._folder, split=self._split, prefix=self._prefix, align_spacing=self._align_spacing,
                             transform=self._transform, device=self._device)
        new._images = images
        new._labels = labels
        new._folded = True
        return new


class BinarizedDataset(SupervisedDataset[D]):
    def __init__(self, base: SupervisedDataset[D], positive_ids: tuple[int, ...], *,
                 transform: JointTransform | None = None, device: Device = "cpu") -> None:
        super().__init__(base._images, base._labels, transform=transform, device=device)
        self._base: SupervisedDataset[D] = base
        self._positive_ids: tuple[int, ...] = positive_ids

    @override
    def construct_new(self, images: D, labels: D) -> Self:
        raise NotImplementedError

    @override
    def load(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        image, label = self._base.load(idx)
        for pid in self._positive_ids:
            label[label == pid] = -1
        label[label > 0] = 0
        label[label == -1] = 1
        return image, label
