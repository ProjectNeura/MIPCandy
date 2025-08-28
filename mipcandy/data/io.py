from math import floor
from os import PathLike

import SimpleITK as SpITK
import torch

from mipcandy.data.geometric import ensure_num_dimensions


def resample_to_isotropic(image: SpITK.Image, *, target_iso: float | None = None,
                          interpolator: int = SpITK.sitkBSpline) -> SpITK.Image:
    dim = image.GetDimension()
    old_spacing = image.GetSpacing()
    old_size = image.GetSize()
    origin = image.GetOrigin()
    direction = image.GetDirection()
    if target_iso is None:
        target_iso = min(old_spacing)
    new_spacing = (target_iso,) * dim
    new_size = (max(1, floor(old_spacing[i] * (old_size[i] - 1) / new_spacing[i] + 1)) for i in range(dim))
    return SpITK.Resample(
        image, new_size, SpITK.Transform(), interpolator, origin, new_spacing, direction, 0, image.GetPixelID()
    )


def load_image(path: str | PathLike[str], *, is_label: bool = False, align_spacing: bool = False,
               device: torch.device | str = "cpu") -> torch.Tensor:
    file = SpITK.ReadImage(path)
    if align_spacing:
        file = resample_to_isotropic(file, interpolator=SpITK.sitkNearestNeighbor if is_label else SpITK.sitkBSpline)
    img = torch.tensor(SpITK.GetArrayFromImage(file), dtype=torch.float, device=device)
    if path.endswith(".nii.gz") or path.endswith(".nii") or path.endswith(".mha"):
        return ensure_num_dimensions(img, 4)
    if path.endswith(".png") or path.endswith(".jpg") or path.endswith(".jpeg"):
        return ensure_num_dimensions(img, 3)
    raise NotImplementedError(f"Unsupported file type: {path}")


def save_image(image: torch.Tensor, path: str | PathLike[str]) -> None:
    SpITK.WriteImage(SpITK.GetImageFromArray(image.detach().cpu().numpy()), path)
