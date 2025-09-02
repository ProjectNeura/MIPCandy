from importlib.util import find_spec
from math import ceil
from multiprocessing import get_context
from os import PathLike
from typing import Literal

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn

from mipcandy.common import Normalize, ColorizeLabel
from mipcandy.data.geometric import ensure_num_dimensions


def auto_convert(image: torch.Tensor) -> torch.Tensor:
    # Convert to float first to avoid UInt16 min/max issues
    image = image.float()
    return (image * 255 if 0 <= image.min() < image.max() <= 1 else Normalize(domain=(0, 255))(image)).int()


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
    image = auto_convert(image)
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
                              screenshot_as: str | PathLike[str] | None, auto_close: bool) -> None:
    from pyvista import Plotter
    p = Plotter(title=title)
    p.add_volume(image, cmap=cmap)
    if screenshot_as:
        p.show(screenshot=screenshot_as, auto_close=auto_close)
    else:
        p.show()


def visualize3d(image: torch.Tensor, *, title: str | None = None, cmap: str = "gray", max_volume: int = 1e6,
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
    ratio = int(ceil((total / max_volume) ** (1 / 3))) if total > max_volume else 1
    if ratio > 1:
        image = ensure_num_dimensions(nn.functional.avg_pool3d(ensure_num_dimensions(image, 5), kernel_size=ratio,
                                                               stride=ratio, ceil_mode=True), 3)
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


def prepare_for_display(image: torch.Tensor, mode="auto", channels=None) -> np.ndarray:
    """
    Convert an arbitrary channel image to a displayable numpy array (H,W), (H,W,3) or (H,W,4)
    mode: "auto" | "select" | "pca" | "all"
    channels: specify the channel indices to use (for mode="select")
    """
    c, h, w = image.shape

    if c == 1:
        return image.squeeze(0).cpu().numpy()  # (H,W)
    elif c == 3:
        return image.permute(1,2,0).cpu().numpy()  # (H,W,3)
    elif c == 4 and mode=="auto":
        # default to RGBA
        return image.permute(1,2,0).cpu().numpy()
    elif c > 4:
        if mode == "select" and channels:
            return image[channels,:,:].permute(1,2,0).cpu().numpy()
        elif mode == "pca":
            # simple PCA to 3D
            flat = image.view(c, -1).T  # (H*W, C)
            from sklearn.decomposition import PCA
            rgb = PCA(n_components=3).fit_transform(flat)
            return rgb.reshape(h,w,3)
        elif mode == "all":
            # return (C,H,W), upper layer is responsible for subplot
            return image.cpu().numpy()
        else:
            raise ValueError(f"Cannot auto-display {c}-channel image. "
                             f"Use mode='select' or 'pca'.")


def overlay(image: torch.Tensor, label: torch.Tensor, *,
            max_label_opacity: float = .5,
            label_colorizer: ColorizeLabel | None = ColorizeLabel(),
            image_mode: str = "auto",
            channels: list[int] | None = None) -> torch.Tensor:
    """
    Overlay a 2D image with a label, returning a CHW (torch) tensor for later visualization.
    - image: (C,H,W) or (H,W) torch.Tensor
    - label: (H,W) or (1,H,W) torch.Tensor
    - image_mode: "auto" | "select" | "pca"
    - channels: if select mode, specify three channel indices
    """
    if image.ndim < 2 or label.ndim < 2:
        raise ValueError("Only 2D images can be overlaid")

    # Unify shapes
    image = ensure_num_dimensions(image, 3).detach().cpu().float()  # (C,H,W)
    label = ensure_num_dimensions(label, 2).detach().cpu().long()  # (H,W)

    C, H, W = image.shape
    if label.shape != (H, W):
        raise ValueError(f"Unmatched shapes {(H, W)} and {tuple(label.shape)}")

    # ---- Channel handling for images: convert any number of channels to 3 (RGB) ----
    if C == 1:
        image3 = image.repeat(3, 1, 1)
    elif C == 3:
        image3 = image
    elif C == 4:
        if image_mode == "select" and channels:
            assert len(channels) == 3, "select mode requires 3 channel indices"
            image3 = image[channels, :, :]
        else:
            # 4 channels are commonly RGBA or 4 modalities: default to first three
            image3 = image[:3, :, :]
    else:  # C > 4
        if image_mode == "select" and channels:
            assert len(channels) == 3, "select mode requires 3 channel indices"
            image3 = image[channels, :, :]
        elif image_mode == "pca":
            # PCA to 3 channels
            flat = image.view(C, -1).T  # (H*W, C)
            from sklearn.decomposition import PCA
            rgb = PCA(n_components=3).fit_transform(flat.numpy())
            image3 = torch.from_numpy(rgb.T).view(3, H, W).float()
        else:
            # safe fallback: take first three and hint user to use select/pca
            image3 = image[:3, :, :]

    # Convert to 0..255 int domain
    image3 = auto_convert(image3).float()  # (3,H,W), 0..255

    # ---- Label color and alpha ----
    alpha = (label > 0).float()  # (H,W)
    if label_colorizer:
        lab = label_colorizer(label)  # expected (3,H,W) or (4,H,W)
        if lab.shape[0] == 4:
            a = lab[-1].float()
            # Normalize to [0,1]
            if a.max() > 0:
                alpha = a / a.max()
            lab = lab[:3]
    else:
        lab = label.float().unsqueeze(0).repeat(3, 1, 1)  # single channel copied to 3

    lab = auto_convert(lab).float()  # (3,H,W), 0..255

    # Alpha scaling to desired opacity
    if alpha.max() > 0:
        alpha = alpha * (max_label_opacity / alpha.max())  # (H,W), 0..max

    # Blending (broadcasting: 3×H×W and H×W)
    blended = image3 * (1 - alpha) + lab * alpha  # (3,H,W), float 0..255
    return blended.int()  # still returns CHW; visualize2d will handle HWC conversion and imshow

