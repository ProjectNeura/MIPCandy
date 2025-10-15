from dataclasses import dataclass
from os import PathLike
from typing import Sequence, override, Callable, Self

import numpy as np
import torch
from pandas import DataFrame
from torch import nn

from mipcandy.data.dataset import SupervisedDataset
from mipcandy.data.geometric import crop
from mipcandy.layer import HasDevice
from mipcandy.types import Device


def format_bbox(bbox: Sequence[int]) -> tuple[int, int, int, int] | tuple[int, int, int, int, int, int]:
    if len(bbox) == 4:
        return bbox[0], bbox[1], bbox[2], bbox[3]
    elif len(bbox) == 6:
        return bbox[0], bbox[1], bbox[2], bbox[3], bbox[4], bbox[5]
    else:
        raise ValueError(f"Invalid bbox with {len(bbox)} elements")


@dataclass
class InspectionAnnotation(object):
    shape: tuple[int, ...]
    foreground_bbox: tuple[int, int, int, int] | tuple[int, int, int, int, int, int]
    ids: tuple[int, ...]

    def foreground_shape(self) -> tuple[int, int] | tuple[int, int, int]:
        r = (self.foreground_bbox[1] - self.foreground_bbox[0], self.foreground_bbox[3] - self.foreground_bbox[2])
        return r if len(self.foreground_bbox) == 4 else r + (self.foreground_bbox[5] - self.foreground_bbox[4],)

    def center_of_foreground(self) -> tuple[int, int] | tuple[int, int, int]:
        r = (round((self.foreground_bbox[1] + self.foreground_bbox[0]) * .5),
             round((self.foreground_bbox[3] + self.foreground_bbox[2]) * .5))
        return r if len(self.shape) == 2 else r + (round((self.foreground_bbox[5] + self.foreground_bbox[4]) * .5),)


class InspectionAnnotations(HasDevice, Sequence[InspectionAnnotation]):
    def __init__(self, dataset: SupervisedDataset, background: int, *annotations: InspectionAnnotation,
                 device: Device = "cpu") -> None:
        super().__init__(device)
        self._dataset: SupervisedDataset = dataset
        self._background: int = background
        self._annotations: tuple[InspectionAnnotation, ...] = annotations
        self._shapes: tuple[tuple[int, ...] | None, tuple[int, ...], tuple[int, ...]] | None = None
        self._foreground_shapes: tuple[tuple[int, ...] | None, tuple[int, ...], tuple[int, ...]] | None = None
        self._statistical_foreground_shape: tuple[int, int] | tuple[int, int, int] | None = None
        self._foreground_heatmap: torch.Tensor | None = None
        self._center_of_foregrounds: tuple[int, int] | tuple[int, int, int] | None = None
        self._foreground_offsets: tuple[int, int] | tuple[int, int, int] | None = None
        self._roi_shape: tuple[int, int] | tuple[int, int, int] | None = None

    def annotations(self) -> tuple[InspectionAnnotation, ...]:
        return self._annotations

    @override
    def __getitem__(self, item: int) -> InspectionAnnotation:
        return self._annotations[item]

    @override
    def __len__(self) -> int:
        return len(self._annotations)

    def save(self, path: str | PathLike[str]) -> None:
        r = []
        for annotation in self._annotations:
            r.append({"foreground_bbox": annotation.foreground_bbox, "ids": annotation.ids})
        DataFrame(r).to_csv(path, index=False)

    def _get_shapes(self, get_shape: Callable[[InspectionAnnotation], tuple[int, ...]]) -> tuple[
        tuple[int, ...] | None, tuple[int, ...], tuple[int, ...]]:
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

    def shapes(self) -> tuple[tuple[int, ...] | None, tuple[int, ...], tuple[int, ...]]:
        if self._shapes:
            return self._shapes
        self._shapes = self._get_shapes(lambda annotation: annotation.shape)
        return self._shapes

    def foreground_shapes(self) -> tuple[tuple[int, ...] | None, tuple[int, ...], tuple[int, ...]]:
        if self._foreground_shapes:
            return self._foreground_shapes
        self._foreground_shapes = self._get_shapes(lambda annotation: annotation.foreground_shape())
        return self._foreground_shapes

    def statistical_foreground_shape(self, *, percentile: float = .95) -> tuple[int, int] | tuple[int, int, int]:
        if self._statistical_foreground_shape:
            return self._statistical_foreground_shape
        depths, heights, widths = self.foreground_shapes()
        percentile *= 100
        sfs = (round(np.percentile(heights, percentile)), round(np.percentile(widths, percentile)))
        self._statistical_foreground_shape = (round(np.percentile(heights, percentile)),) + sfs if depths else sfs
        return self._statistical_foreground_shape

    def crop_foreground(self, i: int, *, expand_ratio: float = 1) -> tuple[torch.Tensor, torch.Tensor]:
        image, label = self._dataset[i]
        annotation = self._annotations[i]
        bbox = list(annotation.foreground_bbox)
        shape = annotation.foreground_shape()
        for i, size in enumerate(shape):
            left = (expand_ratio - 1) * size // 2
            right = (expand_ratio - 1) - left
            bbox[i] = max(0, bbox[i] - left)
            i += 1
            bbox[i] = min(bbox[i] + right, label.shape[i])
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
                paddings.append(left)
                paddings.append(size - shape[j] - left)
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

    def set_roi_shape(self, roi_shape: tuple[int, int] | tuple[int, int, int] | None) -> None:
        self._roi_shape = roi_shape

    def roi_shape(self, *, percentile: float = .95) -> tuple[int, int] | tuple[int, int, int]:
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


def load_inspection_annotations(path: str | PathLike[str]) -> InspectionAnnotations:
    df = DataFrame.from_csv(path)
    return InspectionAnnotations(*(
        InspectionAnnotation(
            tuple(row["shape"]), format_bbox(row["foreground_bbox"]), tuple(row["ids"])
        ) for _, row in df.iterrows()
    ))


def inspect(dataset: SupervisedDataset, *, background: int = 0) -> InspectionAnnotations:
    r = []
    for _, label in dataset:
        indices = (label != background).nonzero()
        mins = indices.min(dim=0)[0].tolist()
        maxs = indices.max(dim=0)[0].tolist()
        bbox = (mins[1], maxs[1], mins[2], maxs[2])
        r.append(InspectionAnnotation(
            label.shape[1:],
            bbox if label.ndim == 3 else bbox + (mins[3], maxs[3]),
            tuple(label.unique())
        ))
    return InspectionAnnotations(dataset, background, *r, device=dataset.device())


class ROIDataset(SupervisedDataset[list[torch.Tensor]]):
    def __init__(self, annotations: InspectionAnnotations, *, percentile: float = .95) -> None:
        super().__init__([], [])
        self._annotations: InspectionAnnotations = annotations
        self._percentile: float = percentile

    @override
    def construct_new(self, images: list[torch.Tensor], labels: list[torch.Tensor]) -> Self:
        return ROIDataset(self._annotations)

    @override
    def load(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self._annotations.crop_roi(idx, percentile=self._percentile)
