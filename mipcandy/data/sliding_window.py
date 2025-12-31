from math import log10
from os import PathLike, makedirs, listdir
from typing import override, Literal

import torch
from rich.console import Console
from rich.progress import Progress

from mipcandy.common import Pad2d, Pad3d
from mipcandy.data.dataset import UnsupervisedDataset, SupervisedDataset, MergedDataset, PathBasedUnsupervisedDataset, \
    TensorLoader
from mipcandy.data.io import fast_save
from mipcandy.data.transform import JointTransform
from mipcandy.types import Shape, Transform, Device


def do_sliding_window(x: torch.Tensor, window_shape: Shape, *, overlap: float = .5) -> list[torch.Tensor]:
    stride = tuple(int(s * (1 + overlap)) for s in window_shape)
    ndim = len(stride)
    if ndim not in (2, 3):
        raise ValueError(f"Window shape must be 2D or 3D, got {ndim}D")
    if ndim == 2:
        x = Pad2d(stride, batch=False)(x)
        x = x.unfold(1, window_shape[0], stride[0]).unfold(2, window_shape[1], stride[1])
        c, n_h, n_w, win_h, win_w = x.shape
        x = x.permute(1, 2, 0, 3, 4).reshape(n_h * n_w, c, win_h, win_w)
        return [x[i] for i in range(x.shape[0])]
    x = Pad3d(stride, batch=False)(x)
    x = x.unfold(1, window_shape[0], stride[0]).unfold(2, window_shape[1], stride[1]).unfold(3, window_shape[2],
                                                                                             stride[2])
    c, n_d, n_h, n_w, win_d, win_h, win_w = x.shape
    x = x.permute(1, 2, 3, 0, 4, 5, 6).reshape(n_d * n_h * n_w, c, win_d, win_h, win_w)
    return [x[i] for i in range(x.shape[0])]


def revert_sliding_window(windows: list[torch.Tensor], *, overlap: float = .5) -> torch.Tensor:
    first_window = windows[0]
    ndim = first_window.ndim - 1
    if ndim not in (2, 3):
        raise ValueError(f"Windows must be 2D or 3D (excluding channel dim), got {ndim}D")
    window_shape = first_window.shape[1:]
    c = first_window.shape[0]
    stride = tuple(int(w * (1 - overlap)) for w in window_shape)
    num_windows = len(windows)
    if ndim == 2:
        h_win, w_win = window_shape
        import math
        grid_size = math.isqrt(num_windows)
        n_h = n_w = grid_size
        while n_h * n_w < num_windows:
            n_w += 1
        if n_h * n_w > num_windows:
            for nh in range(1, num_windows + 1):
                if num_windows % nh == 0:
                    n_h = nh
                    n_w = num_windows // nh
                    break
        out_h = (n_h - 1) * stride[0] + h_win
        out_w = (n_w - 1) * stride[1] + w_win
        output = torch.zeros(1, c, out_h, out_w, device=first_window.device, dtype=first_window.dtype)
        weights = torch.zeros(1, 1, out_h, out_w, device=first_window.device, dtype=first_window.dtype)
        idx = 0
        for i in range(n_h):
            for j in range(n_w):
                if idx >= num_windows:
                    break
                h_start = i * stride[0]
                w_start = j * stride[1]
                output[0, :, h_start:h_start + h_win, w_start:w_start + w_win] += windows[idx]
                weights[0, 0, h_start:h_start + h_win, w_start:w_start + w_win] += 1
                idx += 1
        return output / weights.clamp(min=1)
    else:
        d_win, h_win, w_win = window_shape
        import math
        grid_size = round(num_windows ** (1 / 3))
        n_d = n_h = n_w = grid_size
        while n_d * n_h * n_w < num_windows:
            n_w += 1
            if n_d * n_h * n_w < num_windows:
                n_h += 1
            if n_d * n_h * n_w < num_windows:
                n_d += 1
        for nd in range(1, num_windows + 1):
            if num_windows % nd == 0:
                remaining = num_windows // nd
                for nh in range(1, remaining + 1):
                    if remaining % nh == 0:
                        n_d = nd
                        n_h = nh
                        n_w = remaining // nh
                        break
                break
        out_d = (n_d - 1) * stride[0] + d_win
        out_h = (n_h - 1) * stride[1] + h_win
        out_w = (n_w - 1) * stride[2] + w_win
        output = torch.zeros(1, c, out_d, out_h, out_w, device=first_window.device, dtype=first_window.dtype)
        weights = torch.zeros(1, 1, out_d, out_h, out_w, device=first_window.device, dtype=first_window.dtype)
        idx = 0
        for i in range(n_d):
            for j in range(n_h):
                for k in range(n_w):
                    if idx >= num_windows:
                        break
                    d_start = i * stride[0]
                    h_start = j * stride[1]
                    w_start = k * stride[2]
                    output[0, :, d_start:d_start + d_win, h_start:h_start + h_win, w_start:w_start + w_win] += windows[
                        idx]
                    weights[0, 0, d_start:d_start + d_win, h_start:h_start + h_win, w_start:w_start + w_win] += 1
                    idx += 1
        return output / weights.clamp(min=1)


def _slide(supervised: bool, dataset: UnsupervisedDataset | SupervisedDataset, output_folder: str | PathLike[str],
           window_shape: Shape, *, overlap: float = .5, console: Console = Console()) -> None:
    makedirs(f"{output_folder}/images", exist_ok=True)
    makedirs(f"{output_folder}/labels", exist_ok=True)
    ind = int(log10(len(dataset))) + 1
    with Progress(console=console) as progress:
        task = progress.add_task("Sliding dataset...", total=len(dataset))
        for i, case in enumerate(dataset):
            image = case[0] if supervised else case
            progress.update(task, description=f"Sliding dataset {tuple(image.shape)}...")
            windows = do_sliding_window(image, window_shape, overlap=overlap)
            jnd = int(log10(len(windows))) + 1
            for j, window in enumerate(windows):
                fast_save(window, f"{output_folder}/images/{str(i).zfill(ind)}_{str(j).zfill(jnd)}.pt")
            if supervised:
                label = case[1]
                windows = do_sliding_window(label, window_shape, overlap=overlap)
                for j, window in enumerate(windows):
                    fast_save(window, f"{output_folder}/labels/{str(i).zfill(ind)}_{str(j).zfill(jnd)}.pt")
            progress.update(task, advance=1, description=f"Sliding dataset ({i + 1}/{len(dataset)})...")


def slide_dataset(dataset: UnsupervisedDataset | SupervisedDataset, output_folder: str | PathLike[str],
                  window_shape: Shape, *, overlap: float = .5, console: Console = Console()) -> None:
    _slide(isinstance(dataset, SupervisedDataset), dataset, output_folder, window_shape, overlap=overlap,
           console=console)


class UnsupervisedSWDataset(TensorLoader, PathBasedUnsupervisedDataset):
    def __init__(self, folder: str | PathLike[str], *, subfolder: Literal["images", "labels"] = "images",
                 transform: Transform | None = None, device: Device = "cpu") -> None:
        super().__init__(sorted(listdir(f"{folder}/{subfolder}")), transform=transform, device=device)
        self._folder: str = folder
        self._subfolder: Literal["images", "labels"] = subfolder

    @override
    def load(self, idx: int) -> torch.Tensor:
        return self.do_load(f"{self._folder}/{self._subfolder}/{self._images[idx]}",
                            is_label=self._subfolder == "labels", device=self._device)


class SupervisedSWDataset(TensorLoader, MergedDataset):
    def __init__(self, folder: str | PathLike[str], *, transform: JointTransform | None = None,
                 device: Device = "cpu") -> None:
        super().__init__(UnsupervisedSWDataset(folder, device=device),
                         UnsupervisedSWDataset(folder, subfolder="labels", device=device),
                         transform=transform, device=device)
