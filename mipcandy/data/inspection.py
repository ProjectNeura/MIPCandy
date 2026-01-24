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


class InspectionAnnotations(Sequence[InspectionAnnotation]):
    def __init__(self, dataset: SupervisedDataset, background: int, *annotations: InspectionAnnotation) -> None:
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
        accumulated_label = torch.zeros((1, *max_shape), device=self._dataset.device())
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
        self._foreground_heatmap = accumulated_label.squeeze(0).detach()
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
    with torch.no_grad(), Progress(*Progress.get_default_columns(), SpinnerColumn(), console=console) as progress:
        task = progress.add_task("Inspecting dataset...", total=len(dataset))
        for _, label in dataset:
            progress.update(task, advance=1, description=f"Inspecting dataset {tuple(label.shape)}")
            indices = (label != background).nonzero()
            mins = indices.min(dim=0)[0].tolist()
            maxs = indices.max(dim=0)[0].tolist()
            bbox = (mins[1], maxs[1] + 1, mins[2], maxs[2] + 1)
            r.append(InspectionAnnotation(
                tuple(label.shape[1:]), bbox if label.ndim == 3 else bbox + (mins[3], maxs[3] + 1),
                tuple(label.unique().tolist())
            ))
    return InspectionAnnotations(dataset, background, *r)


class ROIDataset(SupervisedDataset[list[int]]):
    def __init__(self, annotations: InspectionAnnotations, *, percentile: float = .95) -> None:
        super().__init__(list(range(len(annotations))), list(range(len(annotations))),
                         transform=annotations.dataset().transform(), device=annotations.dataset().device())
        self._annotations: InspectionAnnotations = annotations
        self._percentile: float = percentile

    @override
    def construct_new(self, images: list[torch.Tensor], labels: list[torch.Tensor]) -> Self:
        return self.__class__(self._annotations, percentile=self._percentile)

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
            return self._annotations.crop_roi(i, percentile=self._percentile)


class RandomROIDataset(ROIDataset):
    def __init__(self, annotations: InspectionAnnotations, batch_size: int = 2, *,
                 percentile: float = .95,
                 foreground_oversample_percentage: float = .33,
                 min_foreground_samples: int = 500,
                 max_foreground_samples: int = 10000,
                 min_percent_coverage: float = .01,
                 ignore_label: int = 0) -> None:
        super().__init__(annotations, percentile=percentile)
        self._batch_size: int = batch_size
        self._fg_oversample: float = foreground_oversample_percentage
        self._min_fg_samples: int = min_foreground_samples
        self._max_fg_samples: int = max_foreground_samples
        self._min_coverage: float = min_percent_coverage
        self._ignore_label: int = ignore_label
        self._fg_locations_cache: dict[int, dict[int, tuple[tuple[int, ...], ...]] | None] = {}
        self._batch_counter: int = 0

    def _get_foreground_locations(self, idx: int) -> dict[int, tuple[tuple[int, ...], ...]] | None:
        """
        Precompute and cache foreground voxel locations for each class.
        Matches nnU-Net's class_locations from preprocessing.
        """
        if idx not in self._fg_locations_cache:
            _, label = self._annotations.dataset()[idx]
            background = self._annotations.background()
            class_ids = [c for c in label.unique().tolist() if c != background]

            if len(class_ids) == 0:
                self._fg_locations_cache[idx] = None
            else:
                class_locations: dict[int, tuple[tuple[int, ...], ...]] = {}
                for class_id in class_ids:
                    # Get spatial coordinates (skip channel dimension at index 0)
                    indices = (label == class_id).nonzero()[:, 1:]
                    if len(indices) == 0:
                        continue
                    elif len(indices) <= self._min_fg_samples:
                        class_locations[class_id] = tuple(tuple(coord.tolist()) for coord in indices)
                    else:
                        # Subsample to save memory (like nnU-Net preprocessing)
                        target_samples = min(
                            self._max_fg_samples,
                            max(self._min_fg_samples, int(np.ceil(len(indices) * self._min_coverage)))
                        )
                        sampled_idx = torch.randperm(len(indices))[:target_samples]
                        sampled = indices[sampled_idx]
                        class_locations[class_id] = tuple(tuple(coord.tolist()) for coord in sampled)
                self._fg_locations_cache[idx] = class_locations if class_locations else None
        return self._fg_locations_cache[idx]

    def _get_bbox(self, annotation: InspectionAnnotation, roi_shape: Shape,
                  force_fg: bool, class_locations: dict | None) -> tuple[list[int], list[int]]:
        """
        Get bbox bounds (can be negative or exceed image bounds).
        Exactly matches nnU-Net's get_bbox logic from base_data_loader.py:65-140
        """
        data_shape = annotation.shape
        dim = len(data_shape)

        # Calculate padding needed if image is smaller than patch
        need_to_pad = [0] * dim
        for d in range(dim):
            if need_to_pad[d] + data_shape[d] < roi_shape[d]:
                need_to_pad[d] = roi_shape[d] - data_shape[d]

        # Define sampling bounds (can be negative - nnU-Net line 80-81)
        lbs = [- need_to_pad[i] // 2 for i in range(dim)]
        ubs = [data_shape[i] + need_to_pad[i] // 2 + need_to_pad[i] % 2 - roi_shape[i]
               for i in range(dim)]

        # Random sampling vs foreground-guided sampling
        if not force_fg:
            # Random crop (nnU-Net line 86)
            bbox_lbs = [np.random.randint(lbs[i], ubs[i] + 1) for i in range(dim)]
        else:
            # Foreground-guided sampling (nnU-Net line 96-136)
            if class_locations is None or len(class_locations) == 0:
                # No foreground, fall back to random (nnU-Net line 135-136)
                bbox_lbs = [np.random.randint(lbs[i], ubs[i] + 1) for i in range(dim)]
            else:
                # Get eligible classes with locations (nnU-Net line 103)
                eligible_classes = [c for c in class_locations.keys() if len(class_locations[c]) > 0]

                if len(eligible_classes) == 0:
                    bbox_lbs = [np.random.randint(lbs[i], ubs[i] + 1) for i in range(dim)]
                else:
                    # Randomly select a class (nnU-Net line 121-122)
                    selected_class = eligible_classes[np.random.choice(len(eligible_classes))]
                    locations = class_locations[selected_class]

                    # Randomly select a voxel from that class (nnU-Net line 129)
                    selected_voxel = locations[np.random.choice(len(locations))]

                    # Center patch around selected voxel, clamp to bounds (nnU-Net line 133)
                    bbox_lbs = [max(lbs[i], selected_voxel[i] - roi_shape[i] // 2)
                                for i in range(dim)]

        # Calculate upper bounds (nnU-Net line 138)
        bbox_ubs = [bbox_lbs[i] + roi_shape[i] for i in range(dim)]
        return bbox_lbs, bbox_ubs

    def _crop_and_pad(self, data: torch.Tensor, bbox_lbs: list[int], bbox_ubs: list[int],
                      pad_value: int | float = 0) -> torch.Tensor:
        """
        Crop to bbox (which may be negative/exceed bounds) then pad to target size.
        Matches nnU-Net's workflow from data_loader_3d.py:35-51
        """
        shape = data.shape[1:]  # Skip channel dimension
        dim = len(shape)

        # Clip bbox to valid range (nnU-Net line 35-36)
        valid_bbox_lbs = [max(0, bbox_lbs[i]) for i in range(dim)]
        valid_bbox_ubs = [min(shape[i], bbox_ubs[i]) for i in range(dim)]

        # Crop to valid region (nnU-Net line 42-43)
        slices = tuple([slice(0, data.shape[0])] +
                       [slice(valid_bbox_lbs[i], valid_bbox_ubs[i]) for i in range(dim)])
        cropped = data[slices]

        # Calculate padding needed (nnU-Net line 48)
        padding = [(-min(0, bbox_lbs[i]), max(bbox_ubs[i] - shape[i], 0)) for i in range(dim)]

        # Convert to torch.nn.functional.pad format (reversed order, flattened)
        padding_torch = []
        for left, right in reversed(padding):
            padding_torch.extend([left, right])

        # Pad with constant value (nnU-Net line 50-51)
        padded = nn.functional.pad(cropped, padding_torch, mode='constant', value=pad_value)
        return padded

    def _should_oversample_foreground(self) -> bool:
        """
        Batch-based oversampling decision.
        Matches nnU-Net's _oversample_last_XX_percent from base_data_loader.py:46-50
        """
        sample_idx = self._batch_counter % self._batch_size
        return not sample_idx < round(self._batch_size * (1 - self._fg_oversample))

    @override
    def construct_new(self, images: list[torch.Tensor], labels: list[torch.Tensor]) -> Self:
        return self.__class__(
            self._annotations,
            batch_size=self._batch_size,
            percentile=self._percentile,
            foreground_oversample_percentage=self._fg_oversample,
            min_foreground_samples=self._min_fg_samples,
            max_foreground_samples=self._max_fg_samples,
            min_percent_coverage=self._min_coverage,
            ignore_label=self._ignore_label
        )

    @override
    def load_image(self, idx: int) -> torch.Tensor:
        raise NotImplementedError("NnUNetStyleDataset does not support single image loading")

    @override
    def load_label(self, idx: int) -> torch.Tensor:
        raise NotImplementedError("NnUNetStyleDataset does not support single label loading")

    @override
    def load(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Load and process a single sample following nnU-Net's pipeline.
        Matches generate_train_batch from data_loader_3d.py:10-73
        """
        # Load original data
        image, label = self._annotations.dataset()[idx]
        annotation = self._annotations[idx]
        roi_shape = self._annotations.roi_shape(percentile=self._percentile)

        # Determine if this sample needs foreground (batch-based)
        force_fg = self._should_oversample_foreground()
        self._batch_counter += 1

        # Get foreground locations if needed
        class_locations = self._get_foreground_locations(idx) if force_fg else None

        # Get bbox (can be negative or exceed bounds)
        bbox_lbs, bbox_ubs = self._get_bbox(annotation, roi_shape, force_fg, class_locations)

        # Crop and pad with appropriate values
        # Image: pad with 0 (nnU-Net line 50)
        cropped_image = self._crop_and_pad(image, bbox_lbs, bbox_ubs, pad_value=0)
        # Segmentation: pad with ignore_label -1 (nnU-Net line 51)
        cropped_label = self._crop_and_pad(label, bbox_lbs, bbox_ubs, pad_value=self._ignore_label)

        return cropped_image, cropped_label
