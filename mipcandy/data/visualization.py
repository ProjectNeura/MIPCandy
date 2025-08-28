from importlib.util import find_spec
from math import ceil
from multiprocessing import get_context
from os import PathLike
from typing import Literal

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn

from mipcandy.common import ColorizeLabel
from mipcandy.data.geometric import ensure_num_dimensions


def visualize2d(image: torch.Tensor, *, title: str | None = None, cmap: str = "gray",
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
    plt.imshow(image.numpy(), cmap=cmap)
    plt.title(title)
    plt.axis("off")
    if screenshot_as:
        plt.savefig(screenshot_as)
        if blocking:
            plt.close()
            return
    plt.show(block=blocking)


def _visualize3d_with_pyvista(image: np.ndarray, title: str | None, cmap: str,
                              screenshot_as: str | PathLike[str] | None, auto_close: bool) -> None:
    from pyvista import Plotter
    p = Plotter(title=title)
    p.add_volume(image, cmap=cmap)
    if screenshot_as:
        p.show(screenshot=screenshot_as, auto_close=auto_close)
    else:
        p.show()


def visualize3d(image: torch.Tensor, *, title: str | None = None, cmap: str = "gray", max_volume: int = 5e5,
                backend: Literal["auto", "matplotlib", "pyvista"] = "auto", blocking: bool = False,
                screenshot_as: str | PathLike[str] | None = None) -> None:
    image = image.detach().float().cpu()
    if image.ndim < 3:
        raise ValueError(f"`image` must have at least 3 dimensions, got {image.shape}")
    if image.ndim > 4:
        image = ensure_num_dimensions(image, 4)
    if image.ndim == 4 and image.shape[0] == 1:
        image = image.squeeze(0)
    d, h, w = image.shape
    total = d * h * w
    ds = int(ceil((total / max_volume) ** (1 / 3))) if total > max_volume else 1
    if ds > 1:
        image = ensure_num_dimensions(nn.functional.avg_pool3d(ensure_num_dimensions(image, 5), kernel_size=ds,
                                                               stride=ds, ceil_mode=True), 3)
    image /= image.max()
    image = image.numpy()
    if backend == "auto":
        backend = "pyvista" if find_spec("pyvista") else "matplotlib"
    match backend:
        case "matplotlib":
            face_colors = getattr(plt.cm, cmap)(image)
            face_colors[..., 3] = image
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")
            ax.voxels(image, facecolors=face_colors)
            ax.set_title(title)
            if screenshot_as:
                fig.savefig(screenshot_as)
                if blocking:
                    return
            plt.show(block=blocking)
        case "pyvista":
            if blocking:
                return _visualize3d_with_pyvista(image, title, cmap, screenshot_as, blocking)
            ctx = get_context("spawn")
            return ctx.Process(target=_visualize3d_with_pyvista, args=(image, title, cmap, screenshot_as, blocking),
                               daemon=False).start()


def overlay(image: torch.Tensor, label: torch.Tensor, *, max_label_opacity: float = .8,
            label_colorizer: ColorizeLabel | None = ColorizeLabel()) -> torch.Tensor:
    if image.ndim < 2 or label.ndim < 2:
        raise ValueError("Only 2D images can be overlaid")
    image = ensure_num_dimensions(image, 3)
    label = ensure_num_dimensions(label, 2)
    if image.shape[0] == 1:
        if 0 <= image.min() < image.max() <= 1:
            image *= 255
        image = image.repeat(3, 1, 1)
    image_c, image_shape = image.shape[0], image.shape[1:]
    label_shape = label.shape
    if image_shape != label_shape:
        raise ValueError(f"Unmatched shapes {image_shape} and {label_shape}")
    alpha = torch.ones(label_shape, device=image.device).unsqueeze(0)
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
        alpha = alpha / alpha.max() * max_label_opacity
    return (image * (1 - alpha) + label * alpha).round().int()
