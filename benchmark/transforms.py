"""
MIPCandy Transform Module - nnUNet-compatible data augmentation using MONAI.

This module provides nnUNet-style transforms built on top of MONAI's transform infrastructure.
Only implements transforms that MONAI doesn't provide natively.
"""
from __future__ import annotations

from typing import Hashable, Sequence

import numpy as np
import torch
from monai.config import KeysCollection
from monai.transforms import (
    Compose,
    MapTransform,
    OneOf,
    RandAdjustContrastd,
    RandAffined,
    RandFlipd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandScaleIntensityd,
    RandSimulateLowResolutiond,
    Randomizable,
    Transform,
)
from scipy.ndimage import label as scipy_label
from skimage.morphology import ball, disk
from torch.nn.functional import conv2d, conv3d, interpolate, pad


# =============================================================================
# nnUNet-specific Scalar Sampling
# =============================================================================
class BGContrast:
    """nnUNet-style contrast/gamma sampling - biased towards values around 1."""

    def __init__(self, value_range: tuple[float, float]) -> None:
        self._range: tuple[float, float] = value_range

    def __call__(self) -> float:
        if np.random.random() < 0.5 and self._range[0] < 1:
            return float(np.random.uniform(self._range[0], 1))
        return float(np.random.uniform(max(self._range[0], 1), self._range[1]))


# =============================================================================
# Transforms MONAI doesn't have (nnUNet-specific)
# =============================================================================
class DownsampleSegForDS(Transform):
    """Downsample segmentation for deep supervision - produces list of tensors."""

    def __init__(self, scales: Sequence[float | Sequence[float]]) -> None:
        self._scales: list = list(scales)

    def __call__(self, seg: torch.Tensor) -> list[torch.Tensor]:
        results = []
        for s in self._scales:
            if not isinstance(s, (tuple, list)):
                s = [s] * (seg.ndim - 1)
            if all(i == 1 for i in s):
                results.append(seg)
            else:
                new_shape = [round(dim * scale) for dim, scale in zip(seg.shape[1:], s)]
                results.append(interpolate(seg[None].float(), new_shape, mode="nearest-exact")[0].to(seg.dtype))
        return results


class DownsampleSegForDSd(MapTransform):
    """Dictionary version of DownsampleSegForDS."""

    def __init__(self, keys: KeysCollection, scales: Sequence[float | Sequence[float]]) -> None:
        super().__init__(keys, allow_missing_keys=False)
        self._transform = DownsampleSegForDS(scales)

    def __call__(self, data: dict[Hashable, torch.Tensor]) -> dict[Hashable, torch.Tensor]:
        d = dict(data)
        for key in self.keys:
            d[key] = self._transform(d[key])
        return d


class Convert3DTo2D(Transform):
    """Convert 3D data to 2D by merging first spatial dim into channels (for anisotropic data)."""

    def __call__(self, img: torch.Tensor) -> tuple[torch.Tensor, int]:
        nch = img.shape[0]
        return img.reshape(img.shape[0] * img.shape[1], *img.shape[2:]), nch


class Convert2DTo3D(Transform):
    """Convert 2D data back to 3D."""

    def __call__(self, img: torch.Tensor, nch: int) -> torch.Tensor:
        return img.reshape(nch, img.shape[0] // nch, *img.shape[1:])


class Convert3DTo2Dd(MapTransform):
    """Dictionary version - stores channel counts for restoration."""

    def __init__(self, keys: KeysCollection) -> None:
        super().__init__(keys, allow_missing_keys=False)
        self._transform = Convert3DTo2D()

    def __call__(self, data: dict[Hashable, torch.Tensor]) -> dict[Hashable, torch.Tensor]:
        d = dict(data)
        for key in self.keys:
            d[key], d[f"_nch_{key}"] = self._transform(d[key])
        return d


class Convert2DTo3Dd(MapTransform):
    """Dictionary version - restores from stored channel counts."""

    def __init__(self, keys: KeysCollection) -> None:
        super().__init__(keys, allow_missing_keys=False)
        self._transform = Convert2DTo3D()

    def __call__(self, data: dict[Hashable, torch.Tensor]) -> dict[Hashable, torch.Tensor]:
        d = dict(data)
        for key in self.keys:
            nch_key = f"_nch_{key}"
            d[key] = self._transform(d[key], d[nch_key])
            del d[nch_key]
        return d


class ConvertSegToRegions(Transform):
    """Convert segmentation to region-based binary masks."""

    def __init__(self, regions: Sequence[int | Sequence[int]], channel: int = 0) -> None:
        self._regions: list[torch.Tensor] = [
            torch.tensor([r]) if isinstance(r, int) else torch.tensor(r) for r in regions
        ]
        self._channel: int = channel

    def __call__(self, seg: torch.Tensor) -> torch.Tensor:
        output = torch.zeros((len(self._regions), *seg.shape[1:]), dtype=torch.bool, device=seg.device)
        for i, labels in enumerate(self._regions):
            if len(labels) == 1:
                output[i] = seg[self._channel] == labels[0]
            else:
                output[i] = torch.isin(seg[self._channel], labels)
        return output


class ConvertSegToRegionsd(MapTransform):
    """Dictionary version of ConvertSegToRegions."""

    def __init__(self, keys: KeysCollection, regions: Sequence[int | Sequence[int]], channel: int = 0) -> None:
        super().__init__(keys, allow_missing_keys=False)
        self._transform = ConvertSegToRegions(regions, channel)

    def __call__(self, data: dict[Hashable, torch.Tensor]) -> dict[Hashable, torch.Tensor]:
        d = dict(data)
        for key in self.keys:
            d[key] = self._transform(d[key])
        return d


class MoveSegAsOneHotToData(Transform):
    """Move segmentation channel as one-hot encoding to image (for cascade training)."""

    def __init__(self, source_channel: int, labels: Sequence[int], remove_from_seg: bool = True) -> None:
        self._source_channel: int = source_channel
        self._labels: list[int] = list(labels)
        self._remove: bool = remove_from_seg

    def __call__(self, image: torch.Tensor, seg: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        seg_slice = seg[self._source_channel]
        onehot = torch.zeros((len(self._labels), *seg_slice.shape), dtype=image.dtype)
        for i, label in enumerate(self._labels):
            onehot[i][seg_slice == label] = 1
        new_image = torch.cat((image, onehot))
        if self._remove:
            keep = [i for i in range(seg.shape[0]) if i != self._source_channel]
            seg = seg[keep]
        return new_image, seg


class MoveSegAsOneHotToDatad(MapTransform):
    """Dictionary version of MoveSegAsOneHotToData."""

    def __init__(
        self,
        image_key: str,
        seg_key: str,
        source_channel: int,
        labels: Sequence[int],
        remove_from_seg: bool = True,
    ) -> None:
        super().__init__([image_key, seg_key], allow_missing_keys=False)
        self._image_key: str = image_key
        self._seg_key: str = seg_key
        self._transform = MoveSegAsOneHotToData(source_channel, labels, remove_from_seg)

    def __call__(self, data: dict[Hashable, torch.Tensor]) -> dict[Hashable, torch.Tensor]:
        d = dict(data)
        d[self._image_key], d[self._seg_key] = self._transform(d[self._image_key], d[self._seg_key])
        return d


class RemoveLabel(Transform):
    """Replace one label value with another in segmentation."""

    def __init__(self, label: int, set_to: int) -> None:
        self._label: int = label
        self._set_to: int = set_to

    def __call__(self, seg: torch.Tensor) -> torch.Tensor:
        seg = seg.clone()
        seg[seg == self._label] = self._set_to
        return seg


class RemoveLabeld(MapTransform):
    """Dictionary version of RemoveLabel."""

    def __init__(self, keys: KeysCollection, label: int, set_to: int) -> None:
        super().__init__(keys, allow_missing_keys=False)
        self._transform = RemoveLabel(label, set_to)

    def __call__(self, data: dict[Hashable, torch.Tensor]) -> dict[Hashable, torch.Tensor]:
        d = dict(data)
        for key in self.keys:
            d[key] = self._transform(d[key])
        return d


class RandApplyRandomBinaryOperator(Randomizable, Transform):
    """Randomly apply binary morphological operations to one-hot channels."""

    def __init__(
        self,
        channels: Sequence[int],
        prob: float = 0.4,
        strel_size: tuple[int, int] = (1, 8),
        p_per_label: float = 1.0,
    ) -> None:
        self._channels: list[int] = list(channels)
        self._prob: float = prob
        self._strel_size: tuple[int, int] = strel_size
        self._p_per_label: float = p_per_label

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        if self.R.random() > self._prob:
            return img

        channels = self._channels.copy()
        self.R.shuffle(channels)

        for c in channels:
            if self.R.random() > self._p_per_label:
                continue

            size = self.R.randint(self._strel_size[0], self._strel_size[1] + 1)
            op = self.R.choice([_binary_dilation, _binary_erosion, _binary_opening, _binary_closing])

            workon = img[c].to(bool)
            strel = torch.from_numpy(disk(size, dtype=bool) if workon.ndim == 2 else ball(size, dtype=bool))
            result = op(workon, strel)

            added = result & (~workon)
            for oc in self._channels:
                if oc != c:
                    img[oc][added] = 0
            img[c] = result.to(img.dtype)

        return img


class RandApplyRandomBinaryOperatord(MapTransform, Randomizable):
    """Dictionary version of RandApplyRandomBinaryOperator."""

    def __init__(
        self,
        keys: KeysCollection,
        channels: Sequence[int],
        prob: float = 0.4,
        strel_size: tuple[int, int] = (1, 8),
        p_per_label: float = 1.0,
    ) -> None:
        MapTransform.__init__(self, keys, allow_missing_keys=False)
        self._transform = RandApplyRandomBinaryOperator(channels, prob, strel_size, p_per_label)

    def __call__(self, data: dict[Hashable, torch.Tensor]) -> dict[Hashable, torch.Tensor]:
        d = dict(data)
        for key in self.keys:
            d[key] = self._transform(d[key])
        return d


class RandRemoveConnectedComponent(Randomizable, Transform):
    """Randomly remove connected components from one-hot encoding."""

    def __init__(
        self,
        channels: Sequence[int],
        prob: float = 0.2,
        fill_with_other_p: float = 0.0,
        max_coverage: float = 0.15,
        p_per_label: float = 1.0,
    ) -> None:
        self._channels: list[int] = list(channels)
        self._prob: float = prob
        self._fill_p: float = fill_with_other_p
        self._max_coverage: float = max_coverage
        self._p_per_label: float = p_per_label

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        if self.R.random() > self._prob:
            return img

        channels = self._channels.copy()
        self.R.shuffle(channels)

        for c in channels:
            if self.R.random() > self._p_per_label:
                continue

            workon = img[c].to(bool).numpy()
            if not np.any(workon):
                continue

            num_voxels = int(np.prod(workon.shape))
            labeled, num_components = scipy_label(workon)
            if num_components == 0:
                continue

            component_sizes = {i: int((labeled == i).sum()) for i in range(1, num_components + 1)}
            valid = [i for i, size in component_sizes.items() if size < num_voxels * self._max_coverage]

            if valid:
                chosen = self.R.choice(valid)
                mask = labeled == chosen
                img[c][mask] = 0

                if self.R.random() < self._fill_p:
                    others = [i for i in self._channels if i != c]
                    if others:
                        other = self.R.choice(others)
                        img[other][mask] = 1

        return img


class RandRemoveConnectedComponentd(MapTransform, Randomizable):
    """Dictionary version of RandRemoveConnectedComponent."""

    def __init__(
        self,
        keys: KeysCollection,
        channels: Sequence[int],
        prob: float = 0.2,
        fill_with_other_p: float = 0.0,
        max_coverage: float = 0.15,
        p_per_label: float = 1.0,
    ) -> None:
        MapTransform.__init__(self, keys, allow_missing_keys=False)
        self._transform = RandRemoveConnectedComponent(channels, prob, fill_with_other_p, max_coverage, p_per_label)

    def __call__(self, data: dict[Hashable, torch.Tensor]) -> dict[Hashable, torch.Tensor]:
        d = dict(data)
        for key in self.keys:
            d[key] = self._transform(d[key])
        return d


class RandGammad(MapTransform, Randomizable):
    """nnUNet-style gamma transform with invert option and retain_stats."""

    def __init__(
        self,
        keys: KeysCollection,
        prob: float = 0.3,
        gamma: tuple[float, float] = (0.7, 1.5),
        p_invert: float = 0.0,
        p_per_channel: float = 1.0,
        p_retain_stats: float = 1.0,
    ) -> None:
        super().__init__(keys, allow_missing_keys=False)
        self._prob: float = prob
        self._gamma: tuple[float, float] = gamma
        self._p_invert: float = p_invert
        self._p_per_channel: float = p_per_channel
        self._p_retain_stats: float = p_retain_stats

    def __call__(self, data: dict[Hashable, torch.Tensor]) -> dict[Hashable, torch.Tensor]:
        if self.R.random() > self._prob:
            return data

        d = dict(data)
        for key in self.keys:
            img = d[key]
            for c in range(img.shape[0]):
                if self.R.random() > self._p_per_channel:
                    continue

                g = BGContrast(self._gamma)()
                invert = self.R.random() < self._p_invert
                retain = self.R.random() < self._p_retain_stats

                if invert:
                    img[c] *= -1
                if retain:
                    mean, std = img[c].mean(), img[c].std()

                minm = img[c].min()
                rnge = (img[c].max() - minm).clamp(min=1e-7)
                img[c] = torch.pow((img[c] - minm) / rnge, g) * rnge + minm

                if retain:
                    mn_here, std_here = img[c].mean(), img[c].std().clamp(min=1e-7)
                    img[c] = (img[c] - mn_here) * (std / std_here) + mean
                if invert:
                    img[c] *= -1

            d[key] = img
        return d


# =============================================================================
# Binary Morphology Helpers
# =============================================================================
def _binary_dilation(tensor: torch.Tensor, strel: torch.Tensor) -> torch.Tensor:
    tensor_f = tensor.float()
    if tensor.ndim == 2:
        strel_k = strel[None, None].float()
        padded = pad(tensor_f[None, None], [strel.shape[-1] // 2] * 4, mode="constant", value=0)
        out = conv2d(padded, strel_k)
    else:
        strel_k = strel[None, None].float()
        padded = pad(tensor_f[None, None], [strel.shape[-1] // 2] * 6, mode="constant", value=0)
        out = conv3d(padded, strel_k)
    return (out > 0).squeeze(0).squeeze(0)


def _binary_erosion(tensor: torch.Tensor, strel: torch.Tensor) -> torch.Tensor:
    return ~_binary_dilation(~tensor, strel)


def _binary_opening(tensor: torch.Tensor, strel: torch.Tensor) -> torch.Tensor:
    return _binary_dilation(_binary_erosion(tensor, strel), strel)


def _binary_closing(tensor: torch.Tensor, strel: torch.Tensor) -> torch.Tensor:
    return _binary_erosion(_binary_dilation(tensor, strel), strel)


# =============================================================================
# Factory Functions - nnUNet-style Pipelines using MONAI
# =============================================================================
def training_transforms(
    keys: tuple[str, str] = ("image", "label"),
    patch_size: tuple[int, ...] = (128, 128, 128),
    rotation: tuple[float, float] = (-30 / 360 * 2 * np.pi, 30 / 360 * 2 * np.pi),
    scale: tuple[float, float] = (0.7, 1.4),
    mirror_axes: tuple[int, ...] | None = (0, 1, 2),
    do_dummy_2d: bool = False,
    deep_supervision_scales: Sequence[float] | None = None,
    is_cascaded: bool = False,
    foreground_labels: Sequence[int] | None = None,
    regions: Sequence[int | Sequence[int]] | None = None,
    ignore_label: int | None = None,
) -> Compose:
    """
    Create nnUNet-style training transforms using MONAI infrastructure.

    Args:
        keys: (image_key, label_key) for dictionary transforms
        patch_size: spatial size of output patches
        rotation: (min, max) rotation in radians
        scale: (min, max) scale factors
        mirror_axes: axes to randomly flip, None to disable
        do_dummy_2d: use pseudo-2D augmentation for anisotropic data
        deep_supervision_scales: scales for deep supervision downsampling
        is_cascaded: enable cascade training transforms
        foreground_labels: labels for cascade one-hot encoding
        regions: region definitions for region-based training
        ignore_label: label to treat as ignore

    Returns:
        Composed MONAI transforms
    """
    image_key, label_key = keys
    transforms: list = []

    # Pseudo-2D for anisotropic data
    if do_dummy_2d:
        transforms.append(Convert3DTo2Dd(keys=[image_key, label_key]))

    # Spatial transforms (rotation, scaling) - using MONAI RandAffine
    transforms.append(
        RandAffined(
            keys=[image_key, label_key],
            prob=0.2,
            rotate_range=[rotation] * 3 if len(patch_size) == 3 else [rotation],
            scale_range=[(s - 1, s - 1) for s in scale],  # MONAI uses additive range
            mode=["bilinear", "nearest"],
            padding_mode="zeros",
        )
    )

    if do_dummy_2d:
        transforms.append(Convert2DTo3Dd(keys=[image_key, label_key]))

    # Intensity transforms - MONAI versions
    transforms.append(RandGaussianNoised(keys=[image_key], prob=0.1, mean=0.0, std=0.1))
    transforms.append(RandGaussianSmoothd(keys=[image_key], prob=0.2, sigma_x=(0.5, 1.0), sigma_y=(0.5, 1.0), sigma_z=(0.5, 1.0)))
    transforms.append(RandScaleIntensityd(keys=[image_key], prob=0.15, factors=0.25))  # multiplicative brightness
    transforms.append(RandAdjustContrastd(keys=[image_key], prob=0.15, gamma=(0.75, 1.25)))
    transforms.append(RandSimulateLowResolutiond(keys=[image_key], prob=0.25, zoom_range=(0.5, 1.0)))

    # Gamma transforms (nnUNet-specific with invert option)
    transforms.append(RandGammad(keys=[image_key], prob=0.1, gamma=(0.7, 1.5), p_invert=1.0, p_retain_stats=1.0))
    transforms.append(RandGammad(keys=[image_key], prob=0.3, gamma=(0.7, 1.5), p_invert=0.0, p_retain_stats=1.0))

    # Mirror/Flip
    if mirror_axes:
        for axis in mirror_axes:
            transforms.append(RandFlipd(keys=[image_key, label_key], prob=0.5, spatial_axis=axis))

    # Remove invalid labels
    transforms.append(RemoveLabeld(keys=[label_key], label=-1, set_to=0))

    # Cascade training
    if is_cascaded and foreground_labels:
        transforms.append(
            MoveSegAsOneHotToDatad(
                image_key=image_key,
                seg_key=label_key,
                source_channel=1,
                labels=foreground_labels,
                remove_from_seg=True,
            )
        )
        cascade_channels = list(range(-len(foreground_labels), 0))
        transforms.append(
            RandApplyRandomBinaryOperatord(keys=[image_key], channels=cascade_channels, prob=0.4, strel_size=(1, 8))
        )
        transforms.append(
            RandRemoveConnectedComponentd(keys=[image_key], channels=cascade_channels, prob=0.2, max_coverage=0.15)
        )

    # Region-based training
    if regions:
        region_list = list(regions) + ([ignore_label] if ignore_label is not None else [])
        transforms.append(ConvertSegToRegionsd(keys=[label_key], regions=region_list, channel=0))

    # Deep supervision
    if deep_supervision_scales:
        transforms.append(DownsampleSegForDSd(keys=[label_key], scales=deep_supervision_scales))

    return Compose(transforms)


def validation_transforms(
    keys: tuple[str, str] = ("image", "label"),
    deep_supervision_scales: Sequence[float] | None = None,
    is_cascaded: bool = False,
    foreground_labels: Sequence[int] | None = None,
    regions: Sequence[int | Sequence[int]] | None = None,
    ignore_label: int | None = None,
) -> Compose:
    """
    Create nnUNet-style validation transforms using MONAI infrastructure.

    Args:
        keys: (image_key, label_key) for dictionary transforms
        deep_supervision_scales: scales for deep supervision downsampling
        is_cascaded: enable cascade training transforms
        foreground_labels: labels for cascade one-hot encoding
        regions: region definitions for region-based training
        ignore_label: label to treat as ignore

    Returns:
        Composed MONAI transforms
    """
    image_key, label_key = keys
    transforms: list = []

    transforms.append(RemoveLabeld(keys=[label_key], label=-1, set_to=0))

    if is_cascaded and foreground_labels:
        transforms.append(
            MoveSegAsOneHotToDatad(
                image_key=image_key,
                seg_key=label_key,
                source_channel=1,
                labels=foreground_labels,
                remove_from_seg=True,
            )
        )

    if regions:
        region_list = list(regions) + ([ignore_label] if ignore_label is not None else [])
        transforms.append(ConvertSegToRegionsd(keys=[label_key], regions=region_list, channel=0))

    if deep_supervision_scales:
        transforms.append(DownsampleSegForDSd(keys=[label_key], scales=deep_supervision_scales))

    return Compose(transforms)


# =============================================================================
# Re-export MONAI transforms for convenience
# =============================================================================
__all__ = [
    # MONAI re-exports
    "Compose",
    "OneOf",
    "RandAffined",
    "RandFlipd",
    "RandGaussianNoised",
    "RandGaussianSmoothd",
    "RandScaleIntensityd",
    "RandAdjustContrastd",
    "RandSimulateLowResolutiond",
    # nnUNet-specific
    "BGContrast",
    "DownsampleSegForDS",
    "DownsampleSegForDSd",
    "Convert3DTo2D",
    "Convert3DTo2Dd",
    "Convert2DTo3D",
    "Convert2DTo3Dd",
    "ConvertSegToRegions",
    "ConvertSegToRegionsd",
    "MoveSegAsOneHotToData",
    "MoveSegAsOneHotToDatad",
    "RemoveLabel",
    "RemoveLabeld",
    "RandGammad",
    "RandApplyRandomBinaryOperator",
    "RandApplyRandomBinaryOperatord",
    "RandRemoveConnectedComponent",
    "RandRemoveConnectedComponentd",
    # Factory functions
    "training_transforms",
    "validation_transforms",
]
