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
        ) if len(self.shape) == 4 else (
            self.foreground_bbox[3] - self.foreground_bbox[0], self.shape[4] - self.foreground_bbox[1],
            self.foreground_bbox[5] - self.foreground_bbox[2]
        )


class InspectionAnnotations(Sequence[InspectionAnnotation]):
    def __init__(self, dataset: SupervisedDataset, background: int, *annotations: InspectionAnnotation) -> None:
        self._dataset: SupervisedDataset = dataset
        self._background: int = background
        self._annotations: tuple[InspectionAnnotation, ...] = annotations

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
        return tuple(depths) if depths else None, tuple(heights), tuple(widths)

    def foreground_shapes(self, *, percentile: float = .95) -> tuple[int, int] | tuple[int, int, int]:
        depths, heights, widths = self.shapes()
        return (
            round(np.percentile(depths, percentile * 100)), round(np.percentile(heights, percentile * 100)),
            round(np.percentile(widths, percentile * 100))
        ) if depths else (
            round(np.percentile(heights, percentile * 100)), round(np.percentile(widths, percentile * 100))
        )

    def crop_case(self, i: int, *, expand_ratio: float = 1) -> tuple[torch.Tensor, torch.Tensor]:
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
        depths, heights, widths = self.shapes()
        max_shape = (max(heights), max(widths)) if depths else (max(depths), max(heights), max(widths))
        accumulated_label = torch.zeros(max_shape)
        for i, (_, label) in enumerate(self._dataset):
            paddings = []
            for size in max_shape:
                left = (size - label.shape[0]) // 2
                paddings.append(left)
                paddings.append(size - left)
            accumulated_label += nn.functional.pad(crop(label, self._annotations[i].foreground_bbox), paddings,
                                                   value=self._background)
        return accumulated_label


def load_inspection_annotations(path: str | PathLike[str]) -> InspectionAnnotations:
    df = DataFrame.from_csv(path)
    return InspectionAnnotations(*(
        InspectionAnnotation(tuple(row["foreground_bbox"]), tuple(row["ids"])) for _, row in df.iterrows()
    ))


def inspect(dataset: SupervisedDataset, *, background: int = 0) -> InspectionAnnotations:
    return InspectionAnnotations(dataset, background, *(
        InspectionAnnotation(
            label.shape,
            tuple((indices := (label != background).nonzero()).min(dim=0)[0].tolist() + indices.max(dim=0)[0].tolist()),
            tuple(label.unique())
        ) for _, label in dataset
    ))
