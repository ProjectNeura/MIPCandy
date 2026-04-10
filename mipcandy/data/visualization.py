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
    if title:
        plt.title(title)
    plt.axis("off")
    if screenshot_as:
        plt.savefig(screenshot_as)
        if blocking:
            plt.close()
            return
    plt.show(block=blocking)
    plt.close()


def _visualize3d_with_pyvista(image: np.ndarray, title: str | None, cmap: str,
                              screenshot_as: str | PathLike[str] | None) -> None:
    from pyvista import Plotter
    p = Plotter(title=title, off_screen=bool(screenshot_as))
    p.add_volume(image, cmap=cmap)
    if screenshot_as:
        p.screenshot(str(screenshot_as))
    else:
        p.show()


def _resolve_plotly_colorscale(cmap: str | list[str]) -> str | list:
    if isinstance(cmap, list):
        if len(cmap) == 1:
            return [[0.0, cmap[0]], [1.0, cmap[0]]]
        return [[i / (len(cmap) - 1), c] for i, c in enumerate(cmap)]
    mapping = {
        "gray": "Gray",
        "binary": "Greys",
        "jet": "Jet",
        "viridis": "Viridis",
        "plasma": "Plasma",
        "inferno": "Inferno",
        "magma": "Magma",
        "cividis": "Cividis",
    }
    return mapping.get(cmap, cmap)


def _visualize3d_labels_with_plotly_mesh(image: np.ndarray, *, title: str | None, cmap: str | list[str],
                                         screenshot_as: str | PathLike[str] | None, show: bool) -> None:
    from plotly import graph_objects as go
    if not find_spec("skimage"):
        raise ImportError("`skimage` is required for 3D label visualization, install it with `pip install skimage`")
    from skimage.measure import marching_cubes
    traces = []
    max_id = int(image.max())
    for cls in range(1, max_id + 1):
        mask = (image == cls)
        if not np.any(mask):
            continue
        mask_f = mask.astype(np.float32)
        if mask_f.sum() < 4:
            continue
        try:
            verts, faces, _, _ = marching_cubes(mask_f, level=.5)
        except ValueError:
            continue
        z, y, x = verts[:, 0], verts[:, 1], verts[:, 2]
        i, j, k = faces[:, 0], faces[:, 1], faces[:, 2]
        color = (
            cmap[min(cls, len(cmap) - 1)]
            if isinstance(cmap, list)
            else None
        )
        traces.append(go.Mesh3d(x=x, y=y, z=z, i=i, j=j, k=k, color=color, opacity=0.55, name=f"class {cls}",
                                showscale=False, flatshading=True))
    fig = go.Figure(data=traces)
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="W",
            yaxis_title="H",
            zaxis_title="D",
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, t=40 if title else 0, b=0),
    )
    if screenshot_as:
        path = str(screenshot_as)
        if not path.endswith(".html"):
            path += ".html"
        fig.write_html(str(path))
    if show:
        fig.show()


def _visualize3d_scalar_with_plotly_volume(image: np.ndarray, *, title: str | None, cmap: str | list[str],
                                           screenshot_as: str | PathLike[str] | None, show: bool) -> None:
    import plotly.graph_objects as go
    d, h, w = image.shape
    z, y, x = np.mgrid[0:d, 0:h, 0:w]
    values = image.astype(np.float32)
    colorscale = _resolve_plotly_colorscale(cmap)
    vmin = float(values.min())
    vmax = float(values.max())
    if vmax <= vmin:
        vmax = vmin + 1e-6
    nz = values[values > 0]
    isomin = float(np.percentile(nz, 10)) if nz.size > 0 else vmin
    fig = go.Figure(data=[go.Volume(
        x=x.ravel(), y=y.ravel(), z=z.ravel(), value=values.ravel(), isomin=isomin, isomax=vmax, opacity=.08,
        surface_count=12, caps=dict(x_show=False, y_show=False, z_show=False), colorscale=colorscale
    )])
    fig.update_layout(title=title, scene=dict(xaxis_title="W", yaxis_title="H", zaxis_title="D", aspectmode="data"),
                      margin=dict(l=0, r=0, t=40 if title else 0, b=0))
    if screenshot_as:
        path = str(screenshot_as)
        if not path.endswith(".html"):
            path += ".html"
        fig.write_html(str(path))
    if show:
        fig.show()


__LABEL_COLORMAP: list[str] = [
    "#ffffff", "#2e4057", "#7a0f1c", "#004f4f", "#9a7b00", "#2c2f38", "#5c136f", "#113f2e", "#8a3b12", "#2b1a6f",
    "#4a5a1a", "#006b6e", "#3b1f14", "#0a2c66", "#5a0f3c", "#0f5c3a"
]


def visualize3d(image: torch.Tensor, *, title: str | None = None, cmap: str | list[str] | None = None,
                max_volume: int = int(1e6), is_label: bool = False,
                backend: Literal["auto", "matplotlib", "pyvista", "plotly"] = "auto", blocking: bool = False,
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
        backend = "plotly" if find_spec("plotly") else ("pyvista" if find_spec("pyvista") else "matplotlib")
    if is_label:
        max_id = image.max()
        if max_id > 1 and torch.is_floating_point(image):
            raise ValueError(f"Label must be class ids that are in [0, 1] or of integer type, got {image.dtype}")
        if not cmap:
            cmap = __LABEL_COLORMAP[:max_id + 1] if backend == "pyvista" and max_id < len(__LABEL_COLORMAP) else "jet"
    elif not cmap:
        cmap = "binary"
    image = image.numpy()
    match backend:
        case "matplotlib":
            warn("Using Matplotlib for 3D visualization is inefficient and inaccurate, consider using PyVista")
            face_colors = getattr(plt.cm, cmap)(image)
            face_colors[..., 3] = image * (image > 0)
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")
            ax.voxels(image, facecolors=face_colors)
            if title:
                ax.set_title(title)
            if screenshot_as:
                fig.savefig(screenshot_as)
                if blocking:
                    plt.close()
                    return
            plt.show(block=blocking)
            plt.close()
        case "pyvista":
            image = image.transpose(1, 2, 0)
            if blocking:
                _visualize3d_with_pyvista(image, title, cmap, screenshot_as)
                return
            ctx = get_context("spawn")
            ctx.Process(target=_visualize3d_with_pyvista, args=(image, title, cmap, screenshot_as),
                        daemon=False).start()
        case "plotly":
            if is_label:
                _visualize3d_labels_with_plotly_mesh(image, title=title, cmap=cmap, screenshot_as=screenshot_as,
                    show=not (blocking and screenshot_as))
            else:
                _visualize3d_scalar_with_plotly_volume(image, title=title, cmap=cmap, screenshot_as=screenshot_as,
                    show=not (blocking and screenshot_as))
        case _:
            raise ValueError(f"Unsupported backend: {backend}")


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
