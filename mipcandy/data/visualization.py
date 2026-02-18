from importlib.util import find_spec
from math import ceil
from multiprocessing import get_context
from os import PathLike
from typing import Literal
from warnings import warn

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn

from mipcandy.common import ColorizeLabel
from mipcandy.data.convertion import auto_convert
from mipcandy.data.geometric import ensure_num_dimensions


def visualize2d(image: torch.Tensor, *, title: str | None = None, cmap: str | None = None, is_label: bool = False,
                blocking: bool = False, screenshot_as: str | PathLike[str] | None = None) -> None:
    image = image.detach().cpu()
    if image.ndim < 2:
        raise ValueError(f"`image` must have at least 2 dimensions, got {image.shape}")
    if image.ndim > 3:
        image = ensure_num_dimensions(image, 3)
    if image.ndim == 3:
        if image.shape[0] == 1:
            image = image.squeeze(0)
        else:
            image = image.permute(1, 2, 0)
    image = auto_convert(image)
    if not cmap:
        cmap = "jet" if is_label else "gray"
    plt.imshow(image.numpy(), cmap, vmin=0, vmax=255)
    plt.title(title)
    plt.axis("off")
    if screenshot_as:
        plt.savefig(screenshot_as)
        if blocking:
            plt.close()
            return
    plt.show(block=blocking)


def _visualize3d_with_pyvista(image: np.ndarray, title: str | None, cmap: str,
                              screenshot_as: str | PathLike[str] | None) -> None:
    from pyvista import Plotter
    p = Plotter(title=title, off_screen=bool(screenshot_as))
    p.add_volume(image, cmap=cmap)
    if screenshot_as:
        p.screenshot(screenshot_as)
    else:
        p.show()


__LABEL_COLORMAP: list[str] = [
    "#ffffff", "#2e4057", "#7a0f1c", "#004f4f", "#9a7b00", "#2c2f38", "#5c136f", "#113f2e", "#8a3b12", "#2b1a6f",
    "#4a5a1a", "#006b6e", "#3b1f14", "#0a2c66", "#5a0f3c", "#0f5c3a"
]


def visualize3d(image: torch.Tensor, *, title: str | None = None, cmap: str | list[str] | None = None,
                max_volume: int = 1e6, is_label: bool = False,
                backend: Literal["auto", "matplotlib", "pyvista"] = "auto", blocking: bool = False,
                screenshot_as: str | PathLike[str] | None = None) -> None:
    image = image.detach().cpu()
    if image.ndim < 3:
        raise ValueError(f"`image` must have at least 3 dimensions, got {image.shape}")
    if image.ndim > 3:
        image = ensure_num_dimensions(image, 3)
    d, h, w = image.shape
    total = d * h * w
    ratio = int(ceil((total / max_volume) ** (1 / 3))) if total > max_volume else 1
    if ratio > 1:
        image = ensure_num_dimensions(nn.functional.avg_pool3d(
            ensure_num_dimensions(image, 5).float(), kernel_size=ratio, stride=ratio, ceil_mode=True
        ), 3).to(image.dtype)
    if backend == "auto":
        backend = "pyvista" if find_spec("pyvista") else "matplotlib"
    if is_label:
        max_id = image.max()
        if max_id > 1 and torch.is_floating_point(image):
            raise ValueError(f"Label must be class ids that are in [0, 1] or of integer type, got {image.dtype}")
        if not cmap:
            cmap = __LABEL_COLORMAP[:max_id + 1] if backend == "pyvista" and max_id < len(__LABEL_COLORMAP) else "jet"
    elif not cmap:
        cmap = "gray"
    image = image.numpy()
    match backend:
        case "matplotlib":
            warn("Using Matplotlib for 3D visualization is inefficient and inaccurate, consider using PyVista")
            face_colors = getattr(plt.cm, cmap)(image)
            face_colors[..., 3] = image * (image > 0)
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")
            ax.voxels(image, facecolors=face_colors)
            ax.set_title(title)
            if screenshot_as:
                fig.savefig(screenshot_as)
                if blocking:
                    plt.close()
                    return
            plt.show(block=blocking)
        case "pyvista":
            image = image.transpose(1, 2, 0)
            if blocking:
                return _visualize3d_with_pyvista(image, title, cmap, screenshot_as)
            ctx = get_context("spawn")
            return ctx.Process(target=_visualize3d_with_pyvista, args=(image, title, cmap, screenshot_as),
                               daemon=False).start()


def overlay(image: torch.Tensor, label: torch.Tensor, *, max_label_opacity: float = .5,
            label_colorizer: ColorizeLabel | None = ColorizeLabel(batch=False)) -> torch.Tensor:
    """
    :param image: base image
    :param label: label (class ids) to overlay on top of `image`
    :param max_label_opacity: maximum opacity of the label
    :param label_colorizer: colorizer for the label, defaults to `ColorizeLabel()`
    """
    if image.ndim < 2 or label.ndim < 2:
        raise ValueError("Only 2D images can be overlaid")
    image = ensure_num_dimensions(image, 3)
    label = ensure_num_dimensions(label, 2)
    image = auto_convert(image)
    if image.shape[0] == 1:
        image = image.repeat(3, 1, 1)
    image_c, image_shape = image.shape[0], image.shape[1:]
    label_shape = label.shape
    if image_shape != label_shape:
        raise ValueError(f"Unmatched shapes {image_shape} and {label_shape}")
    alpha = (label > 0).int()
    if label_colorizer:
        label = label_colorizer(label)
        if label.shape[0] == 4:
            alpha = label[-1]
            label = label[:-1]
    elif label.shape[0] == 1:
        label = label.repeat(3, 1, 1)
    if not (image_c == label.shape[0] == 3):
        raise ValueError("Unsupported number of channels")
    if alpha.max() > 0:
        alpha = alpha * max_label_opacity / alpha.max()
    return image * (1 - alpha) + label * alpha
