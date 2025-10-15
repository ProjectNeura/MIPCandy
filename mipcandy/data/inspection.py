from dataclasses import dataclass
from os import PathLike
from typing import Sequence, override

import numpy as np
import torch
from pandas import DataFrame
from torch import nn

from mipcandy.data.dataset import SupervisedDataset
from mipcandy.data.geometric import crop


@dataclass
class InspectionAnnotation(object):
    shape: tuple[int, ...]
    foreground_bbox: tuple[int, ...]
    ids: tuple[int, ...]

    def foreground_shape(self) -> tuple[int, int] | tuple[int, int, int]:
        return (
            self.foreground_bbox[2] - self.foreground_bbox[0], self.shape[3] - self.foreground_bbox[1]
        ) if len(self.shape) == 2 else (
            self.foreground_bbox[3] - self.foreground_bbox[0], self.shape[4] - self.foreground_bbox[1],
            self.foreground_bbox[5] - self.foreground_bbox[2]
        )

    def center_of_foreground(self) -> tuple[int, int] | tuple[int, int, int]:
        return (
            round((self.foreground_bbox[2] + self.foreground_bbox[0]) * .5),
            round((self.foreground_bbox[3] + self.foreground_bbox[1]) * .5)
        ) if len(self.shape) == 2 else (
            round((self.foreground_bbox[3] + self.foreground_bbox[0]) * .5),
            round((self.foreground_bbox[4] + self.foreground_bbox[1]) * .5),
            round((self.foreground_bbox[5] + self.foreground_bbox[2]) * .5)
        )


class InspectionAnnotations(Sequence[InspectionAnnotation]):
    def __init__(self, dataset: SupervisedDataset, background: int, *annotations: InspectionAnnotation) -> None:
        self._dataset: SupervisedDataset = dataset
        self._background: int = background
        self._annotations: tuple[InspectionAnnotation, ...] = annotations
        self._shapes: tuple[tuple[int, ...] | None, tuple[int, ...], tuple[int, ...]] | None = None
        self._statistical_foreground_shape: tuple[int, int] | tuple[int, int, int] | None = None
        self._foreground_heatmap: torch.Tensor | None = None
        self._center_of_foregrounds: tuple[int, int] | tuple[int, int, int] | None = None
        self._foreground_offsets: tuple[int, int] | tuple[int, int, int] | None = None
        self._roi: tuple[int, int, int, int] | tuple[int, int, int, int, int, int] | None = None

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

    def shapes(self) -> tuple[tuple[int, ...] | None, tuple[int, ...], tuple[int, ...]]:
        if self._shapes:
            return self._shapes
        depths = []
        widths = []
        heights = []
        for annotation in self._annotations:
            shape = annotation.foreground_shape()
            if len(shape) == 2:
                heights.append(shape[0])
                widths.append(shape[1])
            else:
                depths.append(shape[0])
                heights.append(shape[1])
                widths.append(shape[2])
        self._shapes = tuple(depths) if depths else None, tuple(heights), tuple(widths)
        return self._shapes

    def statistical_foreground_shape(self, *, percentile: float = .95) -> tuple[int, int] | tuple[int, int, int]:
        if self._statistical_foreground_shape:
            return self._statistical_foreground_shape
        depths, heights, widths = self.shapes()
        self._statistical_foreground_shape = (
            round(np.percentile(depths, percentile * 100)), round(np.percentile(heights, percentile * 100)),
            round(np.percentile(widths, percentile * 100))
        ) if depths else (
            round(np.percentile(heights, percentile * 100)), round(np.percentile(widths, percentile * 100))
        )
        return self._statistical_foreground_shape

    def crop_foreground(self, i: int, *, expand_ratio: float = 1) -> tuple[torch.Tensor, torch.Tensor]:
        image, label = self._dataset[i]
        annotation = self._annotations[i]
        bbox = annotation.foreground_bbox
        shape = annotation.foreground_shape()
        for i, size in enumerate(shape):
            left = expand_ratio * size // 2
            right = expand_ratio * size - left
            bbox[i] = max(0, bbox[i] - left)
            bbox[i + 2] = min(bbox[i + 2] + right, label.shape[i])
        return crop(image, bbox), crop(label, bbox)

    def foreground_heatmap(self) -> torch.Tensor:
        if self._foreground_heatmap:
            return self._foreground_heatmap
        depths, heights, widths = self.shapes()
        max_shape = (max(heights), max(widths)) if depths else (max(depths), max(heights), max(widths))
        accumulated_label = torch.zeros(max_shape)
        for i, (_, label) in enumerate(self._dataset):
            annotation = self._annotations[i]
            paddings = []
            for size in max_shape:
                left = (size - label.shape[0]) // 2
                paddings.append(left)
                paddings.append(size - left)
            accumulated_label += nn.functional.pad(crop(label != self._background, annotation.foreground_bbox),
                                                   paddings)
        self._foreground_heatmap = accumulated_label
        return self._foreground_heatmap

    def center_of_foregrounds(self) -> tuple[int, int] | tuple[int, int, int]:
        if self._center_of_foregrounds:
            return self._center_of_foregrounds
        heatmap = self.foreground_heatmap()
        self._center_of_foregrounds = (
            round(heatmap.sum(dim=0).argmax().item()), round(heatmap.sum(dim=1).argmax().item())
        ) if heatmap.ndim == 2 else (
            round(heatmap.sum(dim=0).argmax().item()), round(heatmap.sum(dim=1).argmax().item()),
            round(heatmap.sum(dim=2).argmax().item())
        )
        return self._center_of_foregrounds

    def center_of_foregrounds_offsets(self) -> tuple[int, int] | tuple[int, int, int]:
        if self._foreground_offsets:
            return self._foreground_offsets
        center = self.center_of_foregrounds()
        depths, heights, widths = self.shapes()
        max_shape = (max(heights), max(widths)) if depths else (max(depths), max(heights), max(widths))
        self._foreground_offsets = (
            round(center[0] - max_shape[0] * .5), round(center[1] - max_shape[1] * .5)
        ) if depths else (
            round(center[0] - max_shape[0] * .5), round(center[1] - max_shape[1] * .5),
            round(center[2] - max_shape[2] * .5)
        )
        return self._foreground_offsets

    def roi(self, i: int) -> tuple[int, int, int, int] | tuple[int, int, int, int, int, int]:
        if self._roi:
            return self._roi
        annotation = self._annotations[i]
        sfs = self.statistical_foreground_shape()
        offsets = self.center_of_foregrounds_offsets()
        center = annotation.center_of_foreground()
        roi = []
        for i, position in enumerate(center):
            left = sfs[i] // 2
            right = sfs[i] - left
            offset = min(max(offsets[i], position - left), annotation.shape[i] - right - position)
            roi += [position + offset - left, position + offset + right]
        self._roi = tuple(roi)
        return self._roi

    def crop_roi(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        image, label = self._dataset[i]
        return crop(image, self.roi(i)), crop(label, self.roi(i))


def load_inspection_annotations(path: str | PathLike[str]) -> InspectionAnnotations:
    df = DataFrame.from_csv(path)
    return InspectionAnnotations(*(
        InspectionAnnotation(
            tuple(row["shape"]), tuple(row["foreground_bbox"]), tuple(row["ids"])
        ) for _, row in df.iterrows()
    ))


def inspect(dataset: SupervisedDataset, *, background: int = 0) -> InspectionAnnotations:
    return InspectionAnnotations(dataset, background, *(
        InspectionAnnotation(
            label.shape,
            tuple((indices := (label != background).nonzero()).min(dim=0)[0].tolist() + indices.max(dim=0)[0].tolist()),
            tuple(label.unique())
        ) for _, label in dataset
    ))
