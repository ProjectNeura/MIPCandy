from ast import literal_eval
from dataclasses import dataclass
from functools import reduce
from math import log10
from operator import mul
from os import PathLike, makedirs, listdir
from typing import override, Literal

import torch
from rich.console import Console
from rich.progress import Progress

from mipcandy.common import Pad2d, Pad3d, Restore2d, Restore3d
from mipcandy.data.dataset import UnsupervisedDataset, SupervisedDataset, MergedDataset, PathBasedUnsupervisedDataset, \
    TensorLoader
from mipcandy.data.io import fast_save
from mipcandy.data.transform import JointTransform
from mipcandy.types import Shape, Transform, Device


def do_sliding_window(x: torch.Tensor, window_shape: Shape, *, overlap: float = .5) -> tuple[
    list[torch.Tensor], Shape, Pad2d | Pad3d]:
    stride = tuple(int(s * (1 - overlap)) for s in window_shape)
    ndim = len(stride)
    if ndim not in (2, 3):
        raise ValueError(f"Window shape must be 2D or 3D, got {ndim}D")
    if ndim == 2:
        pad = Pad2d(window_shape, batch=False)
        x = pad(x)
        x = x.unfold(1, window_shape[0], stride[0]).unfold(2, window_shape[1], stride[1])
        c, n_h, n_w, win_h, win_w = x.shape
        x = x.permute(1, 2, 0, 3, 4).reshape(n_h * n_w, c, win_h, win_w)
        return [x[i] for i in range(x.shape[0])], (n_h, n_w), pad
    pad = Pad3d(window_shape, batch=False)
    x = pad(x)
    x = x.unfold(1, window_shape[0], stride[0]).unfold(2, window_shape[1], stride[1]).unfold(
        3, window_shape[2], stride[2])
    c, n_d, n_h, n_w, win_d, win_h, win_w = x.shape
    x = x.permute(1, 2, 3, 0, 4, 5, 6).reshape(n_d * n_h * n_w, c, win_d, win_h, win_w)
    return [x[i] for i in range(x.shape[0])], (n_d, n_h, n_w), pad


def revert_sliding_window(windows: list[torch.Tensor], layout: Shape, pad: Pad2d | Pad3d, *,
                          overlap: float = .5) -> torch.Tensor:
    first_window = windows[0]
    ndim = first_window.ndim - 1
    if ndim not in (2, 3):
        raise ValueError(f"Windows must be 2D or 3D (excluding channel dim), got {ndim}D")
    window_shape = first_window.shape[1:]
    c = first_window.shape[0]
    stride = tuple(int(w * (1 - overlap)) for w in window_shape)
    canvas = torch.stack(windows)
    if ndim == 2:
        h_win, w_win = window_shape
        n_h, n_w = layout
        out_h = (n_h - 1) * stride[0] + h_win
        out_w = (n_w - 1) * stride[1] + w_win
        output = torch.zeros(c, out_h, out_w, device=first_window.device, dtype=first_window.dtype)
        weights = torch.zeros(1, out_h, out_w, device=first_window.device, dtype=first_window.dtype)
        canvas = canvas[:n_h * n_w].reshape(n_h, n_w, c, h_win, w_win)
        for i in range(n_h):
            for j in range(n_w):
                h_start = i * stride[0]
                w_start = j * stride[1]
                output[:, h_start:h_start + h_win, w_start:w_start + w_win] += canvas[i, j]
                weights[0, h_start:h_start + h_win, w_start:w_start + w_win] += 1
        return Restore2d(pad)(output / weights.clamp(min=1))
    d_win, h_win, w_win = window_shape
    n_d, n_h, n_w = layout
    out_d = (n_d - 1) * stride[0] + d_win
    out_h = (n_h - 1) * stride[1] + h_win
    out_w = (n_w - 1) * stride[2] + w_win
    output = torch.zeros(c, out_d, out_h, out_w, device=first_window.device, dtype=first_window.dtype)
    weights = torch.zeros(1, out_d, out_h, out_w, device=first_window.device, dtype=first_window.dtype)
    canvas = canvas[:n_d * n_h * n_w].reshape(n_d, n_h, n_w, c, d_win, h_win, w_win)
    for i in range(n_d):
        for j in range(n_h):
            for k in range(n_w):
                d_start = i * stride[0]
                h_start = j * stride[1]
                w_start = k * stride[2]
                output[:, d_start:d_start + d_win, h_start:h_start + h_win, w_start:w_start + w_win] += canvas[i, j, k]
                weights[0, d_start:d_start + d_win, h_start:h_start + h_win, w_start:w_start + w_win] += 1
    return Restore3d(pad)(output / weights.clamp(min=1))


def _slide(supervised: bool, dataset: UnsupervisedDataset | SupervisedDataset, output_folder: str | PathLike[str],
           window_shape: Shape, *, overlap: float = .5, console: Console = Console()) -> None:
    makedirs(f"{output_folder}/images", exist_ok=True)
    if supervised:
        makedirs(f"{output_folder}/labels", exist_ok=True)
    makedirs(f"{output_folder}/paddings", exist_ok=True)
    ind = int(log10(len(dataset))) + 1
    with Progress(console=console) as progress:
        task = progress.add_task("Sliding dataset...", total=len(dataset))
        for i, case in enumerate(dataset):
            image = case[0] if supervised else case
            progress.update(task, description=f"Sliding dataset {tuple(image.shape)}...")
            windows, layout, pad = do_sliding_window(image, window_shape, overlap=overlap)
            torch.save(pad, f"{output_folder}/paddings/{str(i).zfill(ind)}.pt")
            jnd = int(log10(len(windows))) + 1
            for j, window in enumerate(windows):
                path = f"{output_folder}/images/{str(i).zfill(ind)}_{str(j).zfill(jnd)}"
                fast_save(window, f"{path}_{layout}.pt" if j == 0 else f"{path}.pt")
            if supervised:
                label = case[1]
                windows, layout, _ = do_sliding_window(label, window_shape, overlap=overlap)
                for j, window in enumerate(windows):
                    path = f"{output_folder}/labels/{str(i).zfill(ind)}_{str(j).zfill(jnd)}"
                    fast_save(window, f"{path}_{layout}.pt" if j == 0 else f"{path}.pt")
            progress.update(task, advance=1, description=f"Sliding dataset ({i + 1}/{len(dataset)})...")


def slide_dataset(dataset: UnsupervisedDataset | SupervisedDataset, output_folder: str | PathLike[str],
                  window_shape: Shape, *, overlap: float = .5, console: Console = Console()) -> None:
    _slide(isinstance(dataset, SupervisedDataset), dataset, output_folder, window_shape, overlap=overlap,
           console=console)


@dataclass
class SWCase(object):
    window_indices: list[int]
    layout: Shape | None


class UnsupervisedSWDataset(TensorLoader, PathBasedUnsupervisedDataset):
    def __init__(self, folder: str | PathLike[str], *, subfolder: Literal["images", "labels"] = "images",
                 transform: Transform | None = None, device: Device = "cpu") -> None:
        super().__init__(sorted(listdir(f"{folder}/{subfolder}")), transform=transform, device=device)
        self._folder: str = folder
        self._subfolder: Literal["images", "labels"] = subfolder
        self._groups: list[SWCase] = []
        self._paddings: list[str] = sorted(listdir(f"{folder}/paddings"))
        for idx, filename in enumerate(self._images):
            meta = filename[:filename.rfind(".")].split("_")
            case_id = int(meta[0])
            if case_id >= len(self._groups):
                if case_id != len(self._groups):
                    raise ValueError(f"Mismatched case id {case_id}")
                self._groups.append(SWCase([], None))
            self._groups[case_id].window_indices.append(idx)
            if len(meta) == 3:
                if self._groups[case_id].layout:
                    raise ValueError(f"Duplicated layout specification for case {case_id}")
                self._groups[case_id].layout = literal_eval(meta[2])
        for idx, case in enumerate(self._groups):
            windows, layout = case.window_indices, case.layout
            if not layout:
                raise ValueError(f"Layout not specified for case {idx}")
            if len(windows) != reduce(mul, layout):
                raise ValueError(f"Mismatched number of windows {len(windows)} and layout {layout} for case {idx}")

    @override
    def load(self, idx: int) -> torch.Tensor:
        return self.do_load(f"{self._folder}/{self._subfolder}/{self._images[idx]}",
                            is_label=self._subfolder == "labels", device=self._device)

    def case(self, case_idx: int) -> tuple[list[torch.Tensor], Shape, Pad2d | Pad3d]:
        case = self._groups[case_idx]
        windows = [self[idx] for idx in case.window_indices]
        return windows, case.layout, torch.load(f"{self._folder}/paddings/{self._paddings[case_idx]}",
                                                weights_only=False)


class SupervisedSWDataset(TensorLoader, MergedDataset, SupervisedDataset[UnsupervisedSWDataset]):
    def __init__(self, folder: str | PathLike[str], *, transform: JointTransform | None = None,
                 device: Device = "cpu") -> None:
        MergedDataset.__init__(self, UnsupervisedSWDataset(folder, device=device),
                               UnsupervisedSWDataset(folder, subfolder="labels", device=device),
                               transform=transform, device=device)

    def images(self) -> UnsupervisedSWDataset:
        return self._images

    def labels(self) -> UnsupervisedSWDataset:
        return self._labels
