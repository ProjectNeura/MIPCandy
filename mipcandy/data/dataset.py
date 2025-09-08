from abc import ABCMeta, abstractmethod
from os import PathLike, listdir, makedirs
from os.path import exists
from random import choice, randint
from shutil import move, copy
from typing import Literal, override, Self, Sized, Sequence
import json

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
    def __init__(self, device: torch.device | str = "cpu", *,
                 fold_config: dict | str | None = None,
                 current_fold: int | None = None,
                 fold_mode: Literal["train", "val"] | None = None) -> None:
        super().__init__(device)
        self._fold_config = fold_config
        self._current_fold = current_fold
        self._fold_mode = fold_mode
        self._fold_data: dict = {}
        
        if fold_config:
            self._initialize_fold_system()
    
    def _initialize_fold_system(self) -> None:
        if isinstance(self._fold_config, str):
            self._fold_data = self._load_folds_from_json(self._fold_config)
        elif isinstance(self._fold_config, dict):
            self._fold_data = self._fold_config
        else:
            raise ValueError("fold_config must be dict or JSON file path")
    
    def _load_folds_from_json(self, json_path: str) -> dict:
        with open(json_path, 'r') as f:
            return json.load(f)
    
    def generate_folds(self, num_folds: int = 5, *, 
                      stratified: bool = True,
                      positive_ratio_threshold: float = 0.1) -> dict:
        try:
            from sklearn.model_selection import StratifiedKFold, KFold
        except ImportError:
            raise ImportError("scikit-learn is required for fold generation while it is not installed."
                            "To use this feature, run: pip install scikit-learn")
        
        total_cases = self._get_total_cases()
        indices = list(range(total_cases))
        
        if stratified:
            labels = []
            for i in indices:
                _, label = self.load(i)
                is_positive = label.max() > positive_ratio_threshold
                labels.append(1 if is_positive else 0)
            
            skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)
            fold_splits = list(skf.split(indices, labels))
        else:
            kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
            fold_splits = list(kf.split(indices))
        
        fold_data = {}
        for fold_idx, (train_indices, val_indices) in enumerate(fold_splits):
            fold_data[str(fold_idx)] = {
                "train": train_indices.tolist(),
                "val": val_indices.tolist()
            }
        
        return fold_data
    
    def get_effective_indices(self) -> list[int]:
        if self._current_fold is None or self._fold_mode is None:
            return list(range(self._get_total_cases()))
        
        fold_key = str(self._current_fold)
        if fold_key not in self._fold_data:
            raise ValueError(f"Fold {self._current_fold} not found in fold configuration")
        
        return self._fold_data[fold_key][self._fold_mode]
    
    def load_multimodal_image(self, image_paths: list[str]) -> torch.Tensor:
        if len(image_paths) == 1:
            return self.do_load(image_paths[0], device=self._device, 
                              align_spacing=getattr(self, '_align_spacing', False))
        
        modalities = []
        for path in image_paths:
            modal_data = self.do_load(path, device=self._device,
                                    align_spacing=getattr(self, '_align_spacing', False))
            modalities.append(modal_data)
        
        return torch.cat(modalities, dim=0)
    
    @abstractmethod
    def _get_total_cases(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def load(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    @override
    def __len__(self) -> int:
        if self._current_fold is not None:
            return len(self.get_effective_indices())
        return self._get_total_cases()

    @override
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self._current_fold is not None:
            effective_indices = self.get_effective_indices()
            if idx >= len(effective_indices):
                raise IndexError(f"Index {idx} out of range for current fold")
            real_idx = effective_indices[idx]
            return self.load(real_idx)
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
                 device: torch.device | str = "cpu", **kwargs) -> None:
        super().__init__(device, **kwargs)
        if len(images) != len(labels):
            raise ValueError("Unmatched number of images and labels")
        self._images: UnsupervisedDataset = images
        self._labels: UnsupervisedDataset = labels

    @override
    def _get_total_cases(self) -> int:
        return len(self._images)

    @override
    def load(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self._images[idx].to(self._device), self._labels[idx].to(self._device)


class NNUNetDataset(SupervisedDataset):
    def __init__(self, folder: str | PathLike[str], *, split: Literal["Tr", "Ts"] = "Tr", prefix: str = "",
                 align_spacing: bool = False, image_transform: Transform | None = None,
                 label_transform: Transform | None = None, device: torch.device | str = "cpu",
                 num_modalities: int = 1,
                 modality_pattern: str = "_{:04d}",
                 **kwargs) -> None:
        
        self._folder: str = str(folder)
        self._split: str = split
        self._prefix: str = prefix
        self._align_spacing: bool = align_spacing
        self._image_transform: Transform | None = image_transform
        self._label_transform: Transform | None = label_transform
        self._num_modalities: int = num_modalities
        self._modality_pattern: str = modality_pattern
        
        super().__init__(device, **kwargs)
        
        if num_modalities > 1:
            self._cases, self._labels = self._discover_multimodal_cases()
        else:
            self._images = [f for f in listdir(f"{folder}/images{split}") if f.startswith(prefix)]
            self._images.sort()
            self._labels = [f for f in listdir(f"{folder}/labels{split}") if f.startswith(prefix)]
            self._labels.sort()
            self._cases = self._images
        
        self._num_cases: int = len(self._cases)
        num_labels = len(self._labels)
        if num_labels != self._num_cases:
            raise FileNotFoundError(f"Inconsistent number of labels ({num_labels}) and images ({self._num_cases})")

    def _discover_multimodal_cases(self) -> tuple[list[str], list[str]]:
        images_dir = f"{self._folder}/images{self._split}"
        labels_dir = f"{self._folder}/labels{self._split}"
        
        image_files = [f for f in listdir(images_dir) if f.startswith(self._prefix)]
        
        case_bases = set()
        for filename in image_files:
            base_name = self._extract_case_base(filename)
            if self._validate_case_completeness(base_name):
                case_bases.add(base_name)
        
        case_bases = sorted(list(case_bases))
        
        valid_cases = []
        valid_labels = []
        
        for case_base in case_bases:
            label_candidates = [
                f"{case_base}.nii.gz",
                f"{case_base}_seg.nii.gz",
                f"{case_base}.nii",
            ]
            
            label_found = None
            for candidate in label_candidates:
                if exists(f"{labels_dir}/{candidate}"):
                    label_found = candidate
                    break
            
            if label_found:
                valid_cases.append(case_base)
                valid_labels.append(label_found)
            else:
                print(f"Warning: No label found for case {case_base}")
        
        return valid_cases, valid_labels

    def _extract_case_base(self, filename: str) -> str:
        name_without_ext = filename.split('.')[0]
        parts = name_without_ext.split('_')
        
        if len(parts) >= 2 and parts[-1].isdigit() and len(parts[-1]) == 4:
            return '_'.join(parts[:-1])
        
        return name_without_ext

    def _validate_case_completeness(self, case_base: str) -> bool:
        images_dir = f"{self._folder}/images{self._split}"
        
        for modality_idx in range(self._num_modalities):
            expected_file = f"{case_base}{self._modality_pattern.format(modality_idx)}.nii.gz"
            if not exists(f"{images_dir}/{expected_file}"):
                expected_file_uncompressed = f"{case_base}{self._modality_pattern.format(modality_idx)}.nii"
                if not exists(f"{images_dir}/{expected_file_uncompressed}"):
                    return False
        
        return True

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
        NNUNetDataset._create_subset(images_folder)
        NNUNetDataset._create_subset(labels_folder)
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
    def _get_total_cases(self) -> int:
        return self._num_cases

    @override
    def __len__(self) -> int:
        return super().__len__()

    @override
    def load(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self._num_modalities > 1:
            case_base = self._cases[idx]
            image_paths = []
            
            for modality_idx in range(self._num_modalities):
                modal_filename = f"{case_base}{self._modality_pattern.format(modality_idx)}.nii.gz"
                modal_path = f"{self._folder}/images{self._split}/{modal_filename}"
                
                if not exists(modal_path):
                    modal_filename = f"{case_base}{self._modality_pattern.format(modality_idx)}.nii"
                    modal_path = f"{self._folder}/images{self._split}/{modal_filename}"
                
                image_paths.append(modal_path)
            
            image = self.load_multimodal_image(image_paths)
            
        else:
            image = self.do_load(
                f"{self._folder}/images{self._split}/{self._cases[idx]}",
                align_spacing=self._align_spacing,
                device=self._device
            )
        
        label = self.do_load(
            f"{self._folder}/labels{self._split}/{self._labels[idx]}",
            is_label=True,
            align_spacing=self._align_spacing,
            device=self._device
        )
        
        if self._image_transform:
            image = self._image_transform(image)
        if self._label_transform:
            label = self._label_transform(label)
        
        return image, label


class BinarizedDataset(NNUNetDataset):
    def __init__(self, folder: str | PathLike[str], *, 
                 positive_ids: tuple[int, ...] = (1, 2),
                 **kwargs) -> None:
        super().__init__(folder, **kwargs)
        self.positive_ids = positive_ids

    @override
    def load(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        image, label = super().load(idx)
        for pid in self.positive_ids:
            label[label == pid] = -1
        label[label > 0] = 0
        label[label == -1] = 1
        return image, label
