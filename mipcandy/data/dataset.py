from abc import ABCMeta, abstractmethod
from json import dump
from math import log10
from os import PathLike, listdir, makedirs
from os.path import exists
from random import choices
from shutil import copy2
from typing import Literal, override, Self, Sequence, TypeVar, Generic, Any

import torch
from pandas import DataFrame
from torch import nn
from torch.utils.data import Dataset

from mipcandy.data.io import fast_save, fast_load, load_image
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


class TensorLoader(Loader):
    @staticmethod
    @override
    def do_load(path: str | PathLike[str], *, is_label: bool = False, device: Device = "cpu", **kwargs) -> torch.Tensor:
        return fast_load(path, device=device)


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
        self._transform: Transform | None = None
        self.set_transform(transform)

    @override
    def __len__(self) -> int:
        return len(self._images)

    @override
    def __getitem__(self, idx: int) -> torch.Tensor:
        item = super().__getitem__(idx).to(self._device, non_blocking=True)
        if self._transform:
            item = self._transform(item)
        return item.as_tensor() if hasattr(item, "as_tensor") else item

    def transform(self) -> Transform | None:
        return self._transform

    def set_transform(self, transform: Transform | None) -> None:
        self._transform = transform.to(self._device) if isinstance(transform, nn.Module) else transform


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
        self._transform: JointTransform | None = None
        self.set_transform(transform)
        self._preloaded: str = ""

    @override
    def __len__(self) -> int:
        return len(self._images)

    @abstractmethod
    def load_image(self, idx: int) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def load_label(self, idx: int) -> torch.Tensor:
        raise NotImplementedError

    @override
    def load(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.load_image(idx), self.load_label(idx)

    @override
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self._preloaded:
            nd = int(log10(len(self))) + 1
            idx = str(idx).zfill(nd)
            image, label = fast_load(f"{self._preloaded}/images/{idx}.pt"), fast_load(
                f"{self._preloaded}/labels/{idx}.pt")
        else:
            image, label = super().__getitem__(idx)
        image, label = image.to(self._device, non_blocking=True), label.to(self._device, non_blocking=True)
        if self._transform:
            image, label = self._transform(image, label)
        return image.as_tensor() if hasattr(image, "as_tensor") else image, label.as_tensor() if hasattr(
            label, "as_tensor") else label

    def image(self, idx: int) -> torch.Tensor:
        return self.load_image(idx)

    def label(self, idx: int) -> torch.Tensor:
        return self.load_label(idx)

    def transform(self) -> JointTransform | None:
        return self._transform

    def set_transform(self, transform: JointTransform | None) -> None:
        self._transform = transform.to(self._device) if transform else None

    def _construct_new(self, images: D, labels: D) -> Self:
        new = self.construct_new(images, labels)
        new._preloaded = self._preloaded
        return new

    @abstractmethod
    def construct_new(self, images: D, labels: D) -> Self:
        raise NotImplementedError

    def preload(self, output_folder: str | PathLike[str]) -> None:
        if self._preloaded:
            return
        images_path = f"{output_folder}/images"
        labels_path = f"{output_folder}/labels"
        if not exists(images_path) and not exists(labels_path):
            makedirs(images_path)
            makedirs(labels_path)
            nd = int(log10(len(self))) + 1
            for idx in range(len(self)):
                image, label = self.load(idx)
                idx = str(idx).zfill(nd)
                fast_save(image, f"{images_path}/{idx}.pt")
                fast_save(label, f"{labels_path}/{idx}.pt")
        self._preloaded = output_folder

    def fold(self, *, fold: Literal[0, 1, 2, 3, 4, "all"] = "all", picker: type[KFPicker] = OrderedKFPicker) -> tuple[
        Self, Self]:
        indices = picker.pick(len(self), fold)
        images_train = []
        labels_train = []
        images_val = []
        labels_val = []
        for i in range(len(self)):
            if i in indices:
                images_val.append(self._images[i])
                labels_val.append(self._labels[i])
            else:
                images_train.append(self._images[i])
                labels_train.append(self._labels[i])
        return self._construct_new(images_train, labels_train), self._construct_new(images_val, labels_val)


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
    def load_image(self, idx: int) -> torch.Tensor:
        return self._images[idx]

    @override
    def load_label(self, idx: int) -> torch.Tensor:
        return self._labels[idx]

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
        super().__init__(sorted(listdir(folder)), transform=transform, device=device)
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
        images = sorted([f for f in listdir(f"{folder}/images{split}") if f.startswith(prefix)])
        labels = sorted([f for f in listdir(f"{folder}/labels{split}") if f.startswith(prefix)])
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

    @staticmethod
    def _create_subset(folder: str) -> None:
        if exists(folder) and len(listdir(folder)) > 0:
            raise FileExistsError(f"{folder} already exists and is not empty")
        makedirs(folder, exist_ok=True)

    @override
    def load_image(self, idx: int) -> torch.Tensor:
        return torch.cat([self.do_load(
            f"{self._folder}/images{self._split}/{path}", align_spacing=self._align_spacing, device=self._device
        ) for path in self._multimodal_images[idx]]) if self._multimodal_images else self.do_load(
            f"{self._folder}/images{self._split}/{self._images[idx]}", align_spacing=self._align_spacing,
            device=self._device
        )

    @override
    def load_label(self, idx: int) -> torch.Tensor:
        return self.do_load(
            f"{self._folder}/labels{self._split}/{self._labels[idx]}", is_label=True, align_spacing=self._align_spacing,
            device=self._device
        )

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


class BinarizedDataset(SupervisedDataset[tuple[None]]):
    def __init__(self, base: SupervisedDataset, positive_ids: tuple[int, ...], *,
                 transform: JointTransform | None = None, device: Device = "cpu") -> None:
        super().__init__((None,), (None,), transform=transform, device=device)
        self._base: SupervisedDataset = base
        self._positive_ids: tuple[int, ...] = positive_ids

    @override
    def __len__(self) -> int:
        return len(self._base)

    @override
    def construct_new(self, images: tuple[None], labels: tuple[None]) -> Self:
        raise NotImplementedError

    @override
    def load_image(self, idx: int) -> torch.Tensor:
        return self._base.load_image(idx)

    @override
    def load_label(self, idx: int) -> torch.Tensor:
        label = self._base.load_label(idx)
        for pid in self._positive_ids:
            label[label == pid] = -1
        label[label > 0] = 0
        label[label == -1] = 1
        return label

    @override
    def fold(self, *, fold: Literal[0, 1, 2, 3, 4, "all"] = "all", picker: type[KFPicker] = OrderedKFPicker) -> tuple[
        Self, Self]:
        train, val = self._base.fold(fold=fold, picker=picker)
        return (
            self.__class__(train, self._positive_ids, transform=self._transform, device=self._device),
            self.__class__(val, self._positive_ids, transform=self._transform, device=self._device)
        )
