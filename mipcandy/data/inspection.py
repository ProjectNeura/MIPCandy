from dataclasses import dataclass, asdict
from json import dump, load
from math import ceil
from os import PathLike
from random import randint, choice
from typing import Sequence, override, Callable, Self, Any, Literal

import numpy as np
import torch
from rich.console import Console
from rich.progress import Progress, SpinnerColumn
from torch import nn

from mipcandy.data.dataset import SupervisedDataset
from mipcandy.data.geometric import crop
from mipcandy.types import Shape, AmbiguousShape


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
    class_ids: tuple[int, ...]
    class_counts: dict[int, int]
    class_bboxes: dict[int, tuple[int, int, int, int] | tuple[int, int, int, int, int, int]]
    class_locations: dict[int, tuple[tuple[int, int] | tuple[int, int, int], ...]]
    spacing: Shape | None = None

    def foreground_shape(self) -> Shape:
        r = (self.foreground_bbox[1] - self.foreground_bbox[0], self.foreground_bbox[3] - self.foreground_bbox[2])
        return r if len(self.foreground_bbox) == 4 else r + (self.foreground_bbox[5] - self.foreground_bbox[4],)

    def center_of_foreground(self) -> tuple[int, int] | tuple[int, int, int]:
        r = (round((self.foreground_bbox[1] + self.foreground_bbox[0]) * .5),
             round((self.foreground_bbox[3] + self.foreground_bbox[2]) * .5))
        return r if len(self.shape) == 2 else r + (round((self.foreground_bbox[5] + self.foreground_bbox[4]) * .5),)


class InspectionAnnotations(Sequence[InspectionAnnotation]):
    def __init__(self, dataset: SupervisedDataset, background: int, intensity_stats: tuple[float, float, float, float],
                 *annotations: InspectionAnnotation) -> None:
        self._dataset: SupervisedDataset = dataset
        self._background: int = background
        self._intensity_stats: tuple[float, float, float, float] = intensity_stats
        self._annotations: tuple[InspectionAnnotation, ...] = annotations
        self._shapes: tuple[AmbiguousShape | None, AmbiguousShape, AmbiguousShape] | None = None
        self._foreground_shapes: tuple[AmbiguousShape | None, AmbiguousShape, AmbiguousShape] | None = None
        self._statistical_foreground_shape: Shape | None = None
        self._center_of_foregrounds: tuple[int, int] | tuple[int, int, int] | None = None
        self._foreground_offsets: tuple[int, int] | tuple[int, int, int] | None = None
        self._roi_shape: Shape | None = None

    def dataset(self) -> SupervisedDataset:
        return self._dataset

    def background(self) -> int:
        return self._background

    def intensity_stats(self) -> tuple[float, float, float, float]:
        """
        :return: mean, std, 0.5th percentile, 99.5th percentile
        """
        return self._intensity_stats

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
            dump({
                "background": self._background, "intensity_stats": self._intensity_stats,
                "annotations": [asdict(a) for a in self._annotations]
            }, f)

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
        image, label = self._dataset.image(i), self._dataset.label(i)
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
        depths, heights, widths = self.foreground_shapes()
        max_shape = (max(depths), max(heights), max(widths)) if depths else (max(heights), max(widths))
        accumulated_label = torch.zeros((1, *max_shape), device=self._dataset.device())
        for i in range(len(self._dataset)):
            label = self._dataset.label(i)
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
        return accumulated_label.squeeze(0).detach()

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
                    raise ValueError(
                        f"ROI shape {roi_shape} exceeds minimum image shape ({min(depths)}, {min(heights)}, {min(widths)})")
            else:
                if roi_shape[0] > min(heights) or roi_shape[1] > min(widths):
                    raise ValueError(
                        f"ROI shape {roi_shape} exceeds minimum image shape ({min(heights)}, {min(widths)})")
        self._roi_shape = roi_shape

    def roi_shape(self, *, clamp: bool = True, percentile: float = .95) -> Shape:
        if self._roi_shape:
            return self._roi_shape
        sfs = self.statistical_foreground_shape(percentile=percentile)
        if clamp:
            if len(sfs) == 2:
                sfs = (None, *sfs)
            depths, heights, widths = self.shapes()
            roi_shape = (min(min(heights), sfs[1]), min(min(widths), sfs[2]))
            if depths:
                roi_shape = (min(min(depths), sfs[0]),) + roi_shape
            self._roi_shape = roi_shape
        else:
            self._roi_shape = sfs
        return self._roi_shape

    def roi(self, i: int, *, clamp: bool = True, percentile: float = .95) -> tuple[int, int, int, int] | tuple[
        int, int, int, int, int, int]:
        annotation = self._annotations[i]
        roi_shape = self.roi_shape(clamp=clamp, percentile=percentile)
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

    def crop_roi(self, i: int, *, clamp: bool = True, percentile: float = .95) -> tuple[torch.Tensor, torch.Tensor]:
        image, label = self._dataset.image(i), self._dataset.label(i)
        roi = self.roi(i, clamp=clamp, percentile=percentile)
        return crop(image.unsqueeze(0), roi).squeeze(0), crop(label.unsqueeze(0), roi).squeeze(0)


def _lists_to_tuples(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    return {k: tuple(v) if isinstance(v, list) else v for k, v in pairs}


def _str_indices_to_int_indices(obj: dict[str, Any]) -> dict[int, Any]:
    return {int(k): v for k, v in obj.items()}


def parse_inspection_annotation(obj: dict[str, Any]) -> InspectionAnnotation:
    obj["class_bboxes"] = _str_indices_to_int_indices(obj["class_bboxes"])
    obj["class_locations"] = _str_indices_to_int_indices(obj["class_locations"])
    return InspectionAnnotation(**obj)


def load_inspection_annotations(path: str | PathLike[str], dataset: SupervisedDataset) -> InspectionAnnotations:
    with open(path) as f:
        obj = load(f, object_pairs_hook=_lists_to_tuples)
    annotations = InspectionAnnotations(dataset, obj["background"], obj["intensity_stats"], *(
        parse_inspection_annotation(row) for row in obj["annotations"]
    ))
    return annotations


def bbox_from_indices(indices: torch.Tensor, num_dim: Literal[2, 3]) -> tuple[int, int, int, int]:
    mins = indices.min(dim=0)[0].tolist()
    maxs = indices.max(dim=0)[0].tolist()
    bbox = (mins[1], maxs[1] + 1, mins[2], maxs[2] + 1)
    if num_dim == 3:
        bbox += (mins[3], maxs[3] + 1)
    return bbox


def inspect(dataset: SupervisedDataset, *, background: int = 0, max_samples: int = 10000,
            console: Console = Console()) -> InspectionAnnotations:
    r = []
    with torch.no_grad(), Progress(*Progress.get_default_columns(), SpinnerColumn(), console=console) as progress:
        task = progress.add_task("Inspecting dataset...", total=len(dataset))
        foreground_voxels = []
        for idx in range(len(dataset)):
            label = dataset.label(idx).int()
            progress.update(task, advance=1, description=f"Inspecting dataset {tuple(label.shape)}")
            ndim = label.ndim - 1
            indices = (label != background).nonzero()
            if len(indices) == 0:
                r.append(InspectionAnnotation(
                    tuple(label.shape[1:]), (0, 0, 0, 0) if ndim == 2 else (0, 0, 0, 0, 0, 0), (), {}, {}, {})
                )
                continue
            foreground_bbox = bbox_from_indices(indices, ndim)
            class_ids = label.unique().tolist()
            class_counts = {}
            class_bboxes = {}
            class_locations = {}
            for class_id in class_ids:
                indices = (label == class_id).nonzero()
                class_counts[class_id] = len(indices)
                class_bboxes[class_id] = bbox_from_indices(indices, ndim)
                if len(indices) > max_samples:
                    target_samples = min(max_samples, len(indices))
                    sampled_idx = torch.randperm(len(indices))[:target_samples]
                    indices = indices[sampled_idx]
                class_locations[class_id] = [tuple(coord.tolist()[1:]) for coord in indices]
            r.append(InspectionAnnotation(
                tuple(label.shape[1:]), foreground_bbox, tuple(
                    class_id for class_id in class_ids if class_id != background
                ), class_counts, class_bboxes, class_locations
            ))
            image = dataset.image(idx)
            fg = image[label != background]
            if len(fg) > 0:
                foreground_voxels.append(fg)
        if len(foreground_voxels) == 0:
            raise ValueError("No foreground voxels found in dataset")
        all_fg = torch.cat(foreground_voxels)
        all_fg_np = all_fg.numpy()
        intensity_stats = (all_fg.mean().item(), all_fg.std().item(), float(np.percentile(all_fg_np, 0.5)),
                           float(np.percentile(all_fg_np, 99.5)))
    return InspectionAnnotations(dataset, background, intensity_stats, *r)


class ROIDataset(SupervisedDataset[list[int]]):
    def __init__(self, annotations: InspectionAnnotations, *, clamp: bool = True, percentile: float = .95) -> None:
        super().__init__(list(range(len(annotations))), list(range(len(annotations))),
                         transform=annotations.dataset().transform(), device=annotations.dataset().device())
        self._annotations: InspectionAnnotations = annotations
        self._clamp: bool = clamp
        self._percentile: float = percentile

    @override
    def construct_new(self, images: list[int], labels: list[int]) -> Self:
        new = self.__class__(self._annotations, percentile=self._percentile)
        new._images = images
        new._labels = labels
        return new

    @override
    def load_image(self, idx: int) -> torch.Tensor:
        raise NotImplementedError

    @override
    def load_label(self, idx: int) -> torch.Tensor:
        raise NotImplementedError

    @override
    def load(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        i = self._images[idx]
        if i != self._labels[idx]:
            raise ValueError(f"Image {i} and label {self._labels[idx]} indices do not match")
        with torch.no_grad():
            return self._annotations.crop_roi(i, clamp=self._clamp, percentile=self._percentile)


def crop_and_pad(x: torch.Tensor, bbox_lbs: list[int], bbox_ubs: list[int], *,
                 pad_value: int | float = 0) -> torch.Tensor:
    shape = x.shape[1:]
    dim = len(shape)
    valid_bbox_lbs = [max(0, bbox_lbs[i]) for i in range(dim)]
    valid_bbox_ubs = [min(shape[i], bbox_ubs[i]) for i in range(dim)]
    slices = tuple([slice(0, x.shape[0])] + [slice(valid_bbox_lbs[i], valid_bbox_ubs[i]) for i in range(dim)])
    cropped = x[slices]
    padding = [(-min(0, bbox_lbs[i]), max(bbox_ubs[i] - shape[i], 0)) for i in range(dim)]
    padding_torch = []
    for left, right in reversed(padding):
        padding_torch.extend([left, right])
    padded = nn.functional.pad(cropped, padding_torch, mode="constant", value=pad_value)
    return padded


class RandomROIDataset(ROIDataset):
    def __init__(self, annotations: InspectionAnnotations, batch_size: int, *, num_patches_per_case: int = 1,
                 oversample_rate: float = .33, clamp: bool = False, percentile: float = .5,
                 min_factor: int = 16) -> None:
        super().__init__(annotations, clamp=clamp, percentile=percentile)
        if num_patches_per_case > 1:
            images = [idx for idx in self._images for _ in range(num_patches_per_case)]
            self._images, self._labels = images, images.copy()
        self._batch_size: int = batch_size
        self._oversample_rate: float = oversample_rate
        sfs = self._annotations.statistical_foreground_shape(percentile=self._percentile)
        sfs = [ceil(s / min_factor) * min_factor for s in sfs]
        self._roi_shape: Shape = (min(sfs[0], 2048), min(sfs[1], 2048)) if len(sfs) == 2 else (
            min(sfs[0], 128), min(sfs[1], 128), min(sfs[2], 128))

    def convert_idx(self, idx: int) -> int:
        idx, idx2 = self._images[idx], self._labels[idx]
        if idx != idx2:
            raise ValueError(f"Image {idx} and label {idx2} indices do not match")
        return idx

    def roi_shape(self, *, roi_shape: Shape | None = None) -> None | Shape:
        if not roi_shape:
            return self._roi_shape
        self._roi_shape = roi_shape

    @override
    def construct_new(self, images: list[int], labels: list[int]) -> Self:
        new = self.__class__(self._annotations, self._batch_size, oversample_rate=self._oversample_rate,
                             clamp=self._clamp, percentile=self._percentile)
        new._images = images
        new._labels = labels
        new._roi_shape = self._roi_shape
        return new

    def random_roi(self, idx: int, force_foreground: bool) -> tuple[list[int], list[int]]:
        idx = self.convert_idx(idx)
        annotation = self._annotations[idx]
        roi_shape = self._roi_shape
        dim = len(annotation.shape)
        need_to_pad = [max(0, roi_shape[i] - annotation.shape[i]) for i in range(dim)]
        lbs = [-need_to_pad[i] // 2 for i in range(dim)]
        ubs = [annotation.shape[i] + need_to_pad[i] // 2 + need_to_pad[i] % 2 - roi_shape[i] for i in range(dim)]
        if force_foreground:
            if len(annotation.class_ids) == 0:
                bbox_lbs = [randint(lbs[j], ubs[j]) for j in range(dim)]
            else:
                selected_class = choice(annotation.class_ids)
                selected_voxel = choice(annotation.class_locations[selected_class])
                bbox_lbs = [max(lbs[i], selected_voxel[i] - roi_shape[i] // 2) for i in range(dim)]
        else:
            bbox_lbs = [randint(lbs[i], ubs[i]) for i in range(dim)]
        return bbox_lbs, [bbox_lbs[i] + roi_shape[i] for i in range(dim)]

    def oversample_foreground(self, idx: int) -> bool:
        return idx % self._batch_size >= round(self._batch_size * (1 - self._oversample_rate))

    @override
    def load(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        force_foreground = self.oversample_foreground(idx)
        lbs, ubs = self.random_roi(idx, force_foreground)
        dataset = self._annotations.dataset()
        idx = self.convert_idx(idx)
        return crop_and_pad(dataset.image(idx), lbs, ubs), crop_and_pad(dataset.label(idx), lbs, ubs)
