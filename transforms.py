"""
nnUNet-style data augmentation using MONAI transforms.

This module implements the default nnUNet data augmentation pipeline.

Usage:
    # Option 1: Intensity-only augmentation (for image_transform parameter)
    from mipcandy_bundles.transforms import build_intensity_transforms
    dataset = NNUNetDataset(folder, image_transform=build_intensity_transforms())

    # Option 2: Full augmentation with synchronized spatial transforms
    from mipcandy_bundles.transforms import build_nnunet_transforms
    transform = build_nnunet_transforms()
    # Use with custom dataset that applies dict-based transforms
"""

import numpy as np
import torch
from monai.transforms import (
    Compose,
    RandRotated,
    RandZoomd,
    RandFlipd,
    RandGaussianNoise,
    RandGaussianSmooth,
    RandAdjustContrast,
    RandScaleIntensity,
    Transform,
    MapTransform,
)
from torch import nn


class RandGamma(Transform):
    """Random gamma correction with optional image inversion."""

    def __init__(
        self,
        gamma_range: tuple[float, float] = (0.7, 1.5),
        prob: float = 0.3,
        invert_image: bool = False,
        retain_stats: bool = True,
    ) -> None:
        super().__init__()
        self.gamma_range: tuple[float, float] = gamma_range
        self.prob: float = prob
        self.invert_image: bool = invert_image
        self.retain_stats: bool = retain_stats

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() > self.prob:
            return x

        gamma = torch.empty(1).uniform_(self.gamma_range[0], self.gamma_range[1]).item()

        if self.retain_stats:
            mean_orig = x.mean()
            std_orig = x.std()

        if self.invert_image:
            x = -x

        x_min = x.min()
        x_range = x.max() - x_min
        if x_range > 1e-8:
            x_norm = (x - x_min) / x_range
            x_gamma = torch.pow(x_norm + 1e-8, gamma)
            x = x_gamma * x_range + x_min

        if self.invert_image:
            x = -x

        if self.retain_stats:
            std_new = x.std()
            if std_new > 1e-8:
                x = (x - x.mean()) / std_new * std_orig + mean_orig

        return x


class SimulateLowResolution(Transform):
    """Simulate low resolution by downsampling and upsampling."""

    def __init__(
        self,
        scale_range: tuple[float, float] = (0.5, 1.0),
        prob: float = 0.25,
    ) -> None:
        super().__init__()
        self.scale_range: tuple[float, float] = scale_range
        self.prob: float = prob

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() > self.prob:
            return x

        scale = torch.empty(1).uniform_(self.scale_range[0], self.scale_range[1]).item()
        if scale >= 0.99:
            return x

        original_shape = x.shape

        if x.ndim == 3:  # CHW
            new_shape = [max(1, int(s * scale)) for s in x.shape[1:]]
            x_down = torch.nn.functional.interpolate(
                x.unsqueeze(0), size=new_shape, mode='bilinear', align_corners=False
            )
            x_up = torch.nn.functional.interpolate(
                x_down, size=original_shape[1:], mode='bilinear', align_corners=False
            )
        elif x.ndim == 4:  # CDHW
            new_shape = [max(1, int(s * scale)) for s in x.shape[1:]]
            x_down = torch.nn.functional.interpolate(
                x.unsqueeze(0), size=new_shape, mode='trilinear', align_corners=False
            )
            x_up = torch.nn.functional.interpolate(
                x_down, size=original_shape[1:], mode='trilinear', align_corners=False
            )
        else:
            return x

        return x_up.squeeze(0)


def build_intensity_transforms(num_dims: int = 3) -> Compose:
    """
    Build nnUNet-style intensity augmentation pipeline.

    This includes all intensity transforms that only affect the image,
    not requiring synchronized application to labels.

    Use this as the `image_transform` parameter in NNUNetDataset.

    Args:
        num_dims: Number of spatial dimensions (2 or 3).

    Returns:
        MONAI Compose transform for intensity augmentation.

    nnUNet default parameters:
        - Gaussian noise: p=0.1, std=0.1
        - Gaussian blur: p=0.2, sigma=(0.5, 1.0)
        - Brightness: p=0.15, factor=(-0.25, 0.25)
        - Contrast: p=0.15, gamma=(0.75, 1.25)
        - Low resolution: p=0.25, scale=(0.5, 1.0)
        - Gamma (inverted): p=0.1, gamma=(0.7, 1.5)
        - Gamma: p=0.3, gamma=(0.7, 1.5)
    """
    transforms = [
        # Gaussian noise: p=0.1, variance=0-0.1
        RandGaussianNoise(prob=0.1, mean=0.0, std=0.1),

        # Gaussian blur: p=0.2, sigma=0.5-1.0
        RandGaussianSmooth(
            prob=0.2,
            sigma_x=(0.5, 1.0),
            sigma_y=(0.5, 1.0),
            sigma_z=(0.5, 1.0) if num_dims == 3 else (0.0, 0.0),
        ),

        # Brightness (multiplicative): p=0.15, range=0.75-1.25
        RandScaleIntensity(factors=(-0.25, 0.25), prob=0.15),

        # Contrast: p=0.15, range=0.75-1.25
        RandAdjustContrast(prob=0.15, gamma=(0.75, 1.25)),

        # Simulate low resolution: p=0.25, scale=0.5-1.0
        SimulateLowResolution(scale_range=(0.5, 1.0), prob=0.25),

        # Gamma with inversion: p=0.1
        RandGamma(gamma_range=(0.7, 1.5), prob=0.1, invert_image=True, retain_stats=True),

        # Gamma without inversion: p=0.3
        RandGamma(gamma_range=(0.7, 1.5), prob=0.3, invert_image=False, retain_stats=True),
    ]

    return Compose(transforms)


def build_spatial_transforms(
    num_dims: int = 3,
    rotation_range: tuple[float, float] | None = None,
    anisotropic: bool = False,
) -> Compose:
    """
    Build nnUNet-style spatial augmentation pipeline (dict-based).

    This returns dict-based transforms that synchronize spatial
    transformations between image and label.

    Args:
        num_dims: Number of spatial dimensions (2 or 3).
        rotation_range: Custom rotation range in radians. If None, auto-configured.
        anisotropic: Whether data is anisotropic (affects rotation range).

    Returns:
        MONAI Compose transform for spatial augmentation (dict-based).

    nnUNet default parameters:
        - Rotation: p=0.2
            - 2D isotropic: (-180, 180) degrees
            - 2D anisotropic: (-15, 15) degrees
            - 3D isotropic: (-30, 30) degrees
            - 3D anisotropic: (-180, 180) degrees
        - Scaling: p=0.2, range=(0.7, 1.4)
        - Mirror: p=0.5 per axis
    """
    keys = ["image", "label"]

    # Configure rotation range
    if rotation_range is None:
        if num_dims == 2:
            if anisotropic:
                rotation_range = (-15 / 360 * 2 * np.pi, 15 / 360 * 2 * np.pi)
            else:
                rotation_range = (-np.pi, np.pi)
        else:  # 3D
            if anisotropic:
                rotation_range = (-np.pi, np.pi)
            else:
                rotation_range = (-30 / 360 * 2 * np.pi, 30 / 360 * 2 * np.pi)

    spatial_axes = (0, 1) if num_dims == 2 else (0, 1, 2)

    transforms = [
        # Rotation: p=0.2
        RandRotated(
            keys=keys,
            range_x=rotation_range,
            range_y=rotation_range if num_dims == 3 else 0,
            range_z=rotation_range if num_dims == 3 else 0,
            prob=0.2,
            mode=["bilinear", "nearest"],
            padding_mode="zeros",
        ),

        # Scaling: p=0.2, range=0.7-1.4
        RandZoomd(
            keys=keys,
            min_zoom=0.7,
            max_zoom=1.4,
            prob=0.2,
            mode=["trilinear" if num_dims == 3 else "bilinear", "nearest"],
            padding_mode="constant",
            keep_size=True,
        ),
    ]

    # Mirror/Flip: p=0.5 per axis
    for axis in spatial_axes:
        transforms.append(RandFlipd(keys=keys, prob=0.5, spatial_axis=axis))

    return Compose(transforms)


def build_nnunet_transforms(
    num_dims: int = 3,
    rotation_range: tuple[float, float] | None = None,
    anisotropic: bool = False,
) -> Compose:
    """
    Build complete nnUNet-style augmentation pipeline (dict-based).

    This combines spatial and intensity transforms. Input should be
    a dict with "image" and "label" keys.

    Args:
        num_dims: Number of spatial dimensions (2 or 3).
        rotation_range: Custom rotation range in radians.
        anisotropic: Whether data is anisotropic.

    Returns:
        MONAI Compose transform for full nnUNet augmentation.

    Example:
        transform = build_nnunet_transforms(num_dims=3)
        data = {"image": image_tensor, "label": label_tensor}
        augmented = transform(data)
    """
    spatial = build_spatial_transforms(
        num_dims=num_dims,
        rotation_range=rotation_range,
        anisotropic=anisotropic,
    )

    # Wrap intensity transforms to only apply to "image" key
    class IntensityWrapper(MapTransform):
        def __init__(self, transform: Transform) -> None:
            super().__init__(keys=["image"])
            self.transform: Transform = transform

        def __call__(self, data: dict) -> dict:
            d = dict(data)
            d["image"] = self.transform(d["image"])
            return d

    intensity_list = [
        RandGaussianNoise(prob=0.1, mean=0.0, std=0.1),
        RandGaussianSmooth(
            prob=0.2,
            sigma_x=(0.5, 1.0),
            sigma_y=(0.5, 1.0),
            sigma_z=(0.5, 1.0) if num_dims == 3 else (0.0, 0.0),
        ),
        RandScaleIntensity(factors=(-0.25, 0.25), prob=0.15),
        RandAdjustContrast(prob=0.15, gamma=(0.75, 1.25)),
        SimulateLowResolution(scale_range=(0.5, 1.0), prob=0.25),
        RandGamma(gamma_range=(0.7, 1.5), prob=0.1, invert_image=True, retain_stats=True),
        RandGamma(gamma_range=(0.7, 1.5), prob=0.3, invert_image=False, retain_stats=True),
    ]

    all_transforms = list(spatial.transforms)
    for t in intensity_list:
        all_transforms.append(IntensityWrapper(t))

    return Compose(all_transforms)
