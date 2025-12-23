from dataclasses import dataclass, asdict
from json import dump, load
from os import PathLike
from typing import Sequence, override, Callable, Self, Any

import numpy as np
import torch
from rich.console import Console
from rich.progress import Progress, SpinnerColumn
from torch import nn

from mipcandy.data.dataset import SupervisedDataset
from mipcandy.data.geometric import crop
from mipcandy.layer import HasDevice
from mipcandy.types import Device, Shape, AmbiguousShape


def format_bbox(bbox: Sequence[int]) -> tuple[int, int, int, int] | tuple[int, int, int, int, int, int]:
    if len(bbox) == 4:
        return bbox[0], bbox[1], bbox[2], bbox[3]
    elif len(bbox) == 6:
        return bbox[0], bbox[1], bbox[2], bbox[3], bbox[4], bbox[5]
    else:
        raise ValueError(f"Invalid bbox with {len(bbox)} elements")


@dataclass
class InspectionAnnotation(object):
    shape: AmbiguousShape
    foreground_bbox: tuple[int, int, int, int] | tuple[int, int, int, int, int, int]
    ids: tuple[int, ...]

    def foreground_shape(self) -> Shape:
        r = (self.foreground_bbox[1] - self.foreground_bbox[0], self.foreground_bbox[3] - self.foreground_bbox[2])
        return r if len(self.foreground_bbox) == 4 else r + (self.foreground_bbox[5] - self.foreground_bbox[4],)

    def center_of_foreground(self) -> tuple[int, int] | tuple[int, int, int]:
        r = (round((self.foreground_bbox[1] + self.foreground_bbox[0]) * .5),
             round((self.foreground_bbox[3] + self.foreground_bbox[2]) * .5))
        return r if len(self.shape) == 2 else r + (round((self.foreground_bbox[5] + self.foreground_bbox[4]) * .5),)

    def to_dict(self) -> dict[str, tuple[int, ...]]:
        return asdict(self)


class InspectionAnnotations(HasDevice, Sequence[InspectionAnnotation]):
    def __init__(self, dataset: SupervisedDataset, background: int, *annotations: InspectionAnnotation,
                 device: Device = "cpu") -> None:
        super().__init__(device)
        self._dataset: SupervisedDataset = dataset
        self._background: int = background
        self._annotations: tuple[InspectionAnnotation, ...] = annotations
        self._shapes: tuple[AmbiguousShape | None, AmbiguousShape, AmbiguousShape] | None = None
        self._foreground_shapes: tuple[AmbiguousShape | None, AmbiguousShape, AmbiguousShape] | None = None
        self._statistical_foreground_shape: Shape | None = None
        self._foreground_heatmap: torch.Tensor | None = None
        self._center_of_foregrounds: tuple[int, int] | tuple[int, int, int] | None = None
        self._foreground_offsets: tuple[int, int] | tuple[int, int, int] | None = None
        self._roi_shape: Shape | None = None

    def dataset(self) -> SupervisedDataset:
        return self._dataset

    def background(self) -> int:
        return self._background

    def annotations(self) -> tuple[InspectionAnnotation, ...]:
        return self._annotations

    @override
    def __getitem__(self, item: int) -> InspectionAnnotation:
        return self._annotations[item]

    @override
    def __len__(self) -> int:
        return len(self._annotations)

    def save(self, path: str | PathLike[str]) -> None:
        with open(path, "w") as f:
            dump({"background": self._background, "annotations": [a.to_dict() for a in self._annotations]}, f)

    def _get_shapes(self, get_shape: Callable[[InspectionAnnotation], AmbiguousShape]) -> tuple[
        AmbiguousShape | None, AmbiguousShape, AmbiguousShape]:
        depths = []
        widths = []
        heights = []
        for annotation in self._annotations:
            shape = get_shape(annotation)
            if len(shape) == 2:
                heights.append(shape[0])
                widths.append(shape[1])
            else:
                depths.append(shape[0])
                heights.append(shape[1])
                widths.append(shape[2])
        return tuple(depths) if depths else None, tuple(heights), tuple(widths)

    def shapes(self) -> tuple[AmbiguousShape | None, AmbiguousShape, AmbiguousShape]:
        if self._shapes:
            return self._shapes
        self._shapes = self._get_shapes(lambda annotation: annotation.shape)
        return self._shapes

    def foreground_shapes(self) -> tuple[AmbiguousShape | None, AmbiguousShape, AmbiguousShape]:
        if self._foreground_shapes:
            return self._foreground_shapes
        self._foreground_shapes = self._get_shapes(lambda annotation: annotation.foreground_shape())
        return self._foreground_shapes

    def statistical_foreground_shape(self, *, percentile: float = .95) -> Shape:
        if self._statistical_foreground_shape:
            return self._statistical_foreground_shape
        depths, heights, widths = self.foreground_shapes()
        percentile *= 100
        sfs = (round(np.percentile(heights, percentile)), round(np.percentile(widths, percentile)))
        self._statistical_foreground_shape = (round(np.percentile(depths, percentile)),) + sfs if depths else sfs
        return self._statistical_foreground_shape

    def crop_foreground(self, i: int, *, expand_ratio: float = 1) -> tuple[torch.Tensor, torch.Tensor]:
        image, label = self._dataset[i]
        annotation = self._annotations[i]
        bbox = list(annotation.foreground_bbox)
        shape = annotation.foreground_shape()
        for dim_idx, size in enumerate(shape):
            left = int((expand_ratio - 1) * size // 2)
            right = int((expand_ratio - 1) * size - left)
            bbox[dim_idx * 2] = max(0, bbox[dim_idx * 2] - left)
            bbox[dim_idx * 2 + 1] = min(bbox[dim_idx * 2 + 1] + right, annotation.shape[dim_idx])
        return crop(image.unsqueeze(0), bbox).squeeze(0), crop(label.unsqueeze(0), bbox).squeeze(0)

    def foreground_heatmap(self) -> torch.Tensor:
        if self._foreground_heatmap:
            return self._foreground_heatmap
        depths, heights, widths = self.foreground_shapes()
        max_shape = (max(depths), max(heights), max(widths)) if depths else (max(heights), max(widths))
        accumulated_label = torch.zeros((1, *max_shape), device=self._device)
        for i, (_, label) in enumerate(self._dataset):
            annotation = self._annotations[i]
            paddings = [0, 0, 0, 0]
            shape = annotation.foreground_shape()
            for j, size in enumerate(max_shape):
                left = (size - shape[j]) // 2
                right = size - shape[j] - left
                paddings.append(right)
                paddings.append(left)
            paddings.reverse()
            accumulated_label += nn.functional.pad(
                crop((label != self._background).unsqueeze(0), annotation.foreground_bbox), paddings
            ).squeeze(0)
        self._foreground_heatmap = accumulated_label.squeeze(0)
        return self._foreground_heatmap

    def center_of_foregrounds(self) -> tuple[int, int] | tuple[int, int, int]:
        if self._center_of_foregrounds:
            return self._center_of_foregrounds
        heatmap = self.foreground_heatmap()
        center = (heatmap.sum(dim=1).argmax().item(), heatmap.sum(dim=0).argmax().item()) if heatmap.ndim == 2 else (
            heatmap.sum(dim=(1, 2)).argmax().item(),
            heatmap.sum(dim=(0, 2)).argmax().item(),
            heatmap.sum(dim=(0, 1)).argmax().item(),
        )
        self._center_of_foregrounds = center
        return self._center_of_foregrounds

    def center_of_foregrounds_offsets(self) -> tuple[int, int] | tuple[int, int, int]:
        if self._foreground_offsets:
            return self._foreground_offsets
        center = self.center_of_foregrounds()
        depths, heights, widths = self.foreground_shapes()
        max_shape = (max(depths), max(heights), max(widths)) if depths else (max(heights), max(widths))
        offsets = (round(center[0] - max_shape[0] * .5), round(center[1] - max_shape[1] * .5))
        self._foreground_offsets = offsets + (round(center[2] - max_shape[2] * .5),) if depths else offsets
        return self._foreground_offsets

    def set_roi_shape(self, roi_shape: Shape | None) -> None:
        if roi_shape is not None:
            depths, heights, widths = self.shapes()
            if depths:
                if roi_shape[0] > min(depths) or roi_shape[1] > min(heights) or roi_shape[2] > min(widths):
                    raise ValueError(f"ROI shape {roi_shape} exceeds minimum image shape ({min(depths)}, {min(heights)}, {min(widths)})")
            else:
                if roi_shape[0] > min(heights) or roi_shape[1] > min(widths):
                    raise ValueError(f"ROI shape {roi_shape} exceeds minimum image shape ({min(heights)}, {min(widths)})")
        self._roi_shape = roi_shape

    def roi_shape(self, *, percentile: float = .95) -> Shape:
        if self._roi_shape:
            return self._roi_shape
        sfs = self.statistical_foreground_shape(percentile=percentile)
        if len(sfs) == 2:
            sfs = (None, *sfs)
        depths, heights, widths = self.shapes()
        roi_shape = (min(min(heights), sfs[1]), min(min(widths), sfs[2]))
        if depths:
            roi_shape = (min(min(depths), sfs[0]),) + roi_shape
        self._roi_shape = roi_shape
        return self._roi_shape

    def roi(self, i: int, *, percentile: float = .95) -> tuple[int, int, int, int] | tuple[
        int, int, int, int, int, int]:
        annotation = self._annotations[i]
        roi_shape = self.roi_shape(percentile=percentile)
        offsets = self.center_of_foregrounds_offsets()
        center = annotation.center_of_foreground()
        roi = []
        for i, position in enumerate(center):
            left = roi_shape[i] // 2
            right = roi_shape[i] - left
            offset = min(max(offsets[i], left - position), annotation.shape[i] - right - position)
            roi.append(position + offset - left)
            roi.append(position + offset + right)
        return tuple(roi)

    def crop_roi(self, i: int, *, percentile: float = .95) -> tuple[torch.Tensor, torch.Tensor]:
        image, label = self._dataset[i]
        roi = self.roi(i, percentile=percentile)
        return crop(image.unsqueeze(0), roi).squeeze(0), crop(label.unsqueeze(0), roi).squeeze(0)


def _lists_to_tuples(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    return {k: tuple(v) if isinstance(v, list) else v for k, v in pairs}


def load_inspection_annotations(path: str | PathLike[str], dataset: SupervisedDataset) -> InspectionAnnotations:
    with open(path) as f:
        obj = load(f, object_pairs_hook=_lists_to_tuples)
    return InspectionAnnotations(dataset, obj["background"], *(
        InspectionAnnotation(**row) for row in obj["annotations"]
    ))


def inspect(dataset: SupervisedDataset, *, background: int = 0, console: Console = Console()) -> InspectionAnnotations:
    r = []
    with Progress(*Progress.get_default_columns(), SpinnerColumn(), console=console) as progress:
        task = progress.add_task("Inspecting dataset...", total=len(dataset))
        for _, label in dataset:
            progress.update(task, advance=1, description=f"Inspecting dataset {tuple(label.shape)}")
            indices = (label != background).nonzero()
            mins = indices.min(dim=0)[0].tolist()
            maxs = indices.max(dim=0)[0].tolist()
            bbox = (mins[1], maxs[1] + 1, mins[2], maxs[2] + 1)
            r.append(InspectionAnnotation(
                label.shape[1:], bbox if label.ndim == 3 else bbox + (mins[3], maxs[3] + 1), tuple(label.unique())
            ))
    return InspectionAnnotations(dataset, background, *r, device=dataset.device())


class ROIDataset(SupervisedDataset[list[int]]):
    def __init__(self, annotations: InspectionAnnotations, *, percentile: float = .95) -> None:
        super().__init__(list(range(len(annotations))), list(range(len(annotations))))
        self._annotations: InspectionAnnotations = annotations
        self._percentile: float = percentile

    @override
    def construct_new(self, images: list[torch.Tensor], labels: list[torch.Tensor]) -> Self:
        return self.__class__(self._annotations, percentile=self._percentile)

    @override
    def load(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        i = self._images[idx]
        if i != self._labels[idx]:
            raise ValueError(f"Image {i} and label {self._labels[idx]} indices do not match")
        return self._annotations.crop_roi(i, percentile=self._percentile)


class RandomROIDataset(ROIDataset):
    def __init__(self, annotations: InspectionAnnotations, *, percentile: float = .95,
                 foreground_oversample_percentage: float = .33, min_foreground_samples: int = 500,
                 max_foreground_samples: int = 10000, min_percent_coverage: float = .01) -> None:
        super().__init__(annotations, percentile=percentile)
        self._fg_oversample: float = foreground_oversample_percentage
        self._min_fg_samples: int = min_foreground_samples
        self._max_fg_samples: int = max_foreground_samples
        self._min_coverage: float = min_percent_coverage
        self._fg_locations_cache: dict[int, tuple[tuple[int, ...], ...] | None] = {}

    def _get_foreground_locations(self, idx: int) -> tuple[tuple[int, ...], ...] | None:
        if idx not in self._fg_locations_cache:
            _, label = self._annotations.dataset()[idx]
            indices = (label != self._annotations.background()).nonzero()[:, 1:]
            if len(indices) == 0:
                self._fg_locations_cache[idx] = None
            elif len(indices) <= self._min_fg_samples:
                self._fg_locations_cache[idx] = tuple(tuple(coord.tolist()) for coord in indices)
            else:
                target_samples = min(
                    self._max_fg_samples,
                    max(self._min_fg_samples, int(np.ceil(len(indices) * self._min_coverage)))
                )
                sampled_idx = torch.randperm(len(indices))[:target_samples]
                sampled = indices[sampled_idx]
                self._fg_locations_cache[idx] = tuple(tuple(coord.tolist()) for coord in sampled)
        return self._fg_locations_cache[idx]

    def _random_roi(self, idx: int) -> tuple[int, int, int, int] | tuple[int, int, int, int, int, int]:
        annotation = self._annotations[idx]
        roi_shape = self._annotations.roi_shape(percentile=self._percentile)
        roi = []
        for dim_size, patch_size in zip(annotation.shape, roi_shape):
            left = patch_size // 2
            right = patch_size - left
            min_center = left
            max_center = dim_size - right
            center = torch.randint(min_center, max_center + 1, (1,)).item()
            roi.append(center - left)
            roi.append(center + right)
        return tuple(roi)

    def _foreground_guided_random_roi(self, idx: int) -> tuple[int, int, int, int] | tuple[
        int, int, int, int, int, int]:
        annotation = self._annotations[idx]
        roi_shape = self._annotations.roi_shape(percentile=self._percentile)
        foreground_locations = self._get_foreground_locations(idx)

        if foreground_locations is None or len(foreground_locations) == 0:
            return self._random_roi(idx)

        fg_idx = torch.randint(0, len(foreground_locations), (1,)).item()
        fg_position = foreground_locations[fg_idx]

        roi = []
        for fg_pos, dim_size, patch_size in zip(fg_position, annotation.shape, roi_shape):
            left = patch_size // 2
            right = patch_size - left
            center = max(left, min(fg_pos, dim_size - right))
            roi.append(center - left)
            roi.append(center + right)
        return tuple(roi)

    @override
    def construct_new(self, images: list[torch.Tensor], labels: list[torch.Tensor]) -> Self:
        return self.__class__(self._annotations, percentile=self._percentile,
                              foreground_oversample_percentage=self._fg_oversample,
                              min_foreground_samples=self._min_fg_samples,
                              max_foreground_samples=self._max_fg_samples,
                              min_percent_coverage=self._min_coverage)

    @override
    def load(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        image, label = self._annotations.dataset()[idx]
        force_fg = torch.rand(1).item() < self._fg_oversample
        if force_fg:
            roi = self._foreground_guided_random_roi(idx)
        else:
            roi = self._random_roi(idx)
        return crop(image.unsqueeze(0), roi).squeeze(0), crop(label.unsqueeze(0), roi).squeeze(0)
