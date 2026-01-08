from math import floor
from os import PathLike

import SimpleITK as SpITK
import torch
from safetensors.torch import save_file, load_file

from mipcandy.data.convertion import auto_convert
from mipcandy.data.geometric import ensure_num_dimensions
from mipcandy.types import Device


def fast_save(x: torch.Tensor, path: str | PathLike[str]) -> None:
    save_file({"payload": x if x.is_contiguous() else x.contiguous()}, path)


def fast_load(path: str | PathLike[str], *, device: Device = "cpu") -> torch.Tensor:
    return load_file(path, device)["payload"]


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
    new_size = tuple(max(1, floor(old_spacing[i] * (old_size[i] - 1) / new_spacing[i] + 1)) for i in range(dim))
    return SpITK.Resample(
        image, new_size, SpITK.Transform(), interpolator, origin, new_spacing, direction, 0, image.GetPixelID()
    )


def load_image(path: str | PathLike[str], *, is_label: bool = False, align_spacing: bool = False,
               device: Device = "cpu") -> torch.Tensor:
    file = SpITK.ReadImage(path)
    if align_spacing:
        file = resample_to_isotropic(file, interpolator=SpITK.sitkNearestNeighbor if is_label else SpITK.sitkBSpline)
    img = torch.tensor(SpITK.GetArrayFromImage(file), dtype=torch.float, device=device)
    if path.endswith(".nii.gz") or path.endswith(".nii") or path.endswith(".mha"):
        img = ensure_num_dimensions(img, 4, append_before=False).permute(3, 0, 1, 2)
        return img.squeeze(1) if img.shape[1] == 1 else img
    if path.endswith(".png") or path.endswith(".jpg") or path.endswith(".jpeg"):
        return ensure_num_dimensions(img, 3, append_before=False).permute(2, 0, 1)
    raise NotImplementedError(f"Unsupported file type: {path}")


def save_image(image: torch.Tensor, path: str | PathLike[str]) -> None:
    if path.endswith(".nii.gz") or path.endswith(".nii") or path.endswith(".mha"):
        image = ensure_num_dimensions(image, 4).permute(1, 2, 3, 0)
        return SpITK.WriteImage(SpITK.GetImageFromArray(image.detach().cpu().numpy(), isVector=False), path)
    if path.endswith(".png") or path.endswith(".jpg") or path.endswith(".jpeg"):
        image = auto_convert(ensure_num_dimensions(image, 3)).to(torch.uint8).permute(1, 2, 0)
        return SpITK.WriteImage(SpITK.GetImageFromArray(image.detach().cpu().numpy(), isVector=True), path)
    raise NotImplementedError(f"Unsupported file type: {path}")
