from dataclasses import dataclass
from os import PathLike
from typing import Sequence, override

import numpy as np
import torch
from pandas import DataFrame

from mipcandy.data.dataset import SupervisedDataset


@dataclass
class InspectionAnnotation(object):
    dimensions: tuple[int, ...]
    foreground_bbox: tuple[int, ...]
    ids: tuple[int, ...]


class InspectionAnnotations(Sequence[InspectionAnnotation]):
    def __init__(self, dataset: SupervisedDataset, *annotations: InspectionAnnotation) -> None:
        self._dataset: SupervisedDataset = dataset
        self._annotations: tuple[InspectionAnnotation, ...] = annotations

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

    def foreground_dimensions(self, *, percentile: float = .95) -> tuple[int, int] | tuple[int, int, int]:
        depths = []
        widths = []
        heights = []
        for annotation in self._annotations:
            bbox = annotation.foreground_bbox
            if len(bbox) == 4:
                heights.append(bbox[2] - annotation.foreground_bbox[0])
                widths.append(bbox[3] - annotation.foreground_bbox[1])
            else:
                depths.append(bbox[3] - annotation.foreground_bbox[0])
                heights.append(bbox[4] - annotation.foreground_bbox[1])
                widths.append(bbox[5] - annotation.foreground_bbox[2])
        return (
            round(np.percentile(depths, percentile * 100)), round(np.percentile(heights, percentile * 100)),
            round(np.percentile(widths, percentile * 100))
        ) if depths else (
            round(np.percentile(heights, percentile * 100)), round(np.percentile(widths, percentile * 100))
        )

    def center_of_foreground(self) -> tuple[int, int] | tuple[int, int, int]:
        accumulated_label = torch.zeros()
        # TODO


def load_inspection_annotations(path: str | PathLike[str]) -> InspectionAnnotations:
    df = DataFrame.from_csv(path)
    return InspectionAnnotations(*(
        InspectionAnnotation(tuple(row["foreground_bbox"]), tuple(row["ids"])) for _, row in df.iterrows()
    ))


def inspect(dataset: SupervisedDataset, *, background: int = 0) -> InspectionAnnotations:
    return InspectionAnnotations(dataset, *(
        InspectionAnnotation(
            label.shape,
            tuple((indices := (label != background).nonzero()).min(dim=0)[0].tolist() + indices.max(dim=0)[0].tolist()),
            tuple(label.unique())
        ) for _, label in dataset
    ))
