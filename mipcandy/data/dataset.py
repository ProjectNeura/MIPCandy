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
    def do_load(path: str | PathLike[str], *, is_label: bool = False, device: Device = "cpu", **kwargs) -> torch.Tensor:
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

    def __init__(self, images: D, *, device: Device = "cpu") -> None:
        super().__init__(device)
        self._images: D = images

    @override
    def __len__(self) -> int:
        return len(self._images)


class SupervisedDataset(_AbstractDataset[tuple[torch.Tensor, torch.Tensor]], Generic[D], metaclass=ABCMeta):
    """
    Do not use this as a generic class. Only parameterize it if you are inheriting from it.
    """

    def __init__(self, images: D, labels: D, *, device: Device = "cpu") -> None:
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
    def __init__(self, images: Sequence[torch.Tensor], device: Device = "cpu") -> None:
        super().__init__(images, device=device)

    @override
    def load(self, idx: int) -> torch.Tensor:
        return self._images[idx].to(self._device)


class MergedDataset(SupervisedDataset[UnsupervisedDataset]):
    def __init__(self, images: UnsupervisedDataset, labels: UnsupervisedDataset, *, device: Device = "cpu") -> None:
        super().__init__(images, labels, device=device)

    @override
    def load(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self._images[idx].to(self._device), self._labels[idx].to(self._device)

    @override
    def construct_new(self, images: D, labels: D) -> Self:
        return MergedDataset(DatasetFromMemory(images), DatasetFromMemory(labels), device=self._device)


class NNUNetDataset(SupervisedDataset[list[str]]):
    def __init__(self, folder: str | PathLike[str], *, split: str | Literal["Tr", "Ts"] = "Tr", prefix: str = "",
                 align_spacing: bool = False, image_transform: Transform | None = None,
                 label_transform: Transform | None = None, device: Device = "cpu") -> None:
        images: list[str] = [f for f in listdir(f"{folder}/images{split}") if f.startswith(prefix)]
        images.sort()
        labels: list[str] = [f for f in listdir(f"{folder}/labels{split}") if f.startswith(prefix)]
        labels.sort()
        self._multimodal_images: list[list[str]] = []
        if len(images) == len(labels):
            super().__init__(images, labels, device=device)
        else:
            super().__init__([""] * len(labels), labels, device=device)
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
        self._image_transform: Transform | None = image_transform
        self._label_transform: Transform | None = label_transform

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
        if self._image_transform:
            image = self._image_transform(image)
        if self._label_transform:
            label = self._label_transform(label)
        return image, label

    def iter_paths(self) -> list[tuple[str, str]]:
        return [(self._images[i], self._labels[i]) for i in range(len(self))]

    def save(self, split: str | Literal["Tr", "Ts"], *, target_folder: str | PathLike[str] | None = None) -> None:
        target_base = target_folder if target_folder else self._folder
        images_target = f"{target_base}/images{split}"
        labels_target = f"{target_base}/labels{split}"
        self._create_subset(images_target)
        self._create_subset(labels_target)
        for image_path, label_path in self.iter_paths():
            copy2(f"{self._folder}/images{self._split}/{image_path}", f"{images_target}/{image_path}")
            copy2(f"{self._folder}/labels{self._split}/{label_path}", f"{labels_target}/{label_path}")
        self._split = split
        self._folded = False

    def save_paths(self, path: str | PathLike[str], *, fmt: Literal["csv", "json", "txt"] = "csv") -> None:
        paths = self.iter_paths()
        match fmt.lower():
            case "csv":
                df = DataFrame([{"image": image_path, "label": label_path} for image_path, label_path in paths])
                df.index = range(1, len(df) + 1)
                df.index.name = "case"
                df.to_csv(path)
            case "json":
                with open(path, "w") as f:
                    dump([{"image": image_path, "label": label_path} for image_path, label_path in paths], f)
            case "txt":
                with open(path, "w") as f:
                    for i, l in paths:
                        f.write(f"{i}\t{l}\n")

    @override
    def construct_new(self, images: D, labels: D) -> Self:
        if self._folded:
            raise ValueError("Cannot construct a new dataset from a fold")
        new = self.__class__(self._folder, split=self._split, prefix=self._prefix, align_spacing=self._align_spacing,
                             image_transform=self._image_transform, label_transform=self._label_transform,
                             device=self._device)
        new._images = images
        new._labels = labels
        new._folded = True
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


class RandomPatchDataset(NNUNetDataset):
    def __init__(self, folder: str | PathLike[str], *, precompute_foreground: bool = False,
                 properties_cache_file: str | PathLike[str] | None = None, ignore_label: int = 0, **kwargs) -> None:
        super().__init__(folder, **kwargs)
        self._properties: list[dict] | None = None
        self._ignore_label: int = ignore_label

        if properties_cache_file and exists(f"{properties_cache_file}"):
            self._load_properties(properties_cache_file)
        elif precompute_foreground:
            self._compute_properties_with_progress()

    def _compute_single_property(self, image: torch.Tensor, label: torch.Tensor) -> dict:
        from mipcandy.data.patch_sampling import compute_foreground_locations
        return {
            'shape': image.shape,
            'foreground_locations': compute_foreground_locations(label, ignore_label=self._ignore_label)
        }

    def _compute_properties_with_progress(self) -> None:
        from rich.progress import track
        self._properties = []
        for idx in track(range(len(self)), description="Precomputing foreground locations"):
            image, label = super().load(idx)
            props = self._compute_single_property(image, label)
            self._properties.append(props)

    def save_properties(self, filepath: str | PathLike[str]) -> None:
        from pickle import dump
        with open(filepath, 'wb') as f:
            dump(self._properties, f)

    def _load_properties(self, filepath: str | PathLike[str]) -> None:
        from pickle import load
        with open(filepath, 'rb') as f:
            self._properties = load(f)

    def load_with_properties(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, dict]:
        image, label = super().load(idx)

        if self._properties and idx < len(self._properties):
            props = self._properties[idx]
        else:
            props = self._compute_single_property(image, label)

        return image, label, props

    @override
    def construct_new(self, images: D, labels: D) -> Self:
        new = super().construct_new(images, labels)
        if self._properties:
            indices = [i for i, img in enumerate(self._images) if img in images]
            new._properties = [self._properties[i] for i in indices]
        return new


def compute_patch_size(dataset: NNUNetDataset, *, percentile: int = 95) -> tuple[int, ...]:
    import numpy as np
    import SimpleITK as sitk
    from rich.console import Console

    console = Console()

    spacings = []
    shapes = []

    image_paths = dataset._multimodal_images if dataset._multimodal_images else [[p] for p in dataset._images]

    for img_paths in image_paths:
        img_path = img_paths[0]
        img = sitk.ReadImage(f"{dataset._folder}/images{dataset._split}/{img_path}")
        spacing = np.array(img.GetSpacing()[::-1])
        shape = np.array(img.GetSize()[::-1])

        spacings.append(spacing)
        shapes.append(shape)

    spacings = np.array(spacings)
    shapes = np.array(shapes)

    target_spacing = np.median(spacings, axis=0)

    resampled_shapes = shapes * spacings / target_spacing

    patch_size = np.percentile(resampled_shapes, percentile, axis=0)
    patch_size_tuple = tuple(int(np.round(s)) for s in patch_size)

    console.print(f"Computed patch size (percentile={percentile}): {patch_size_tuple}")

    return patch_size_tuple
