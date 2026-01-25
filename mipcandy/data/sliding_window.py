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
from torch import nn

from mipcandy.data.dataset import UnsupervisedDataset, SupervisedDataset, MergedDataset, PathBasedUnsupervisedDataset, \
    TensorLoader
from mipcandy.data.io import fast_save
from mipcandy.data.transform import JointTransform
from mipcandy.types import Shape, Transform, Device


def do_sliding_window(x: torch.Tensor, window_shape: Shape, *, overlap: float = .5) -> tuple[
    torch.Tensor, Shape, Shape]:
    stride = tuple(int(s * (1 - overlap)) for s in window_shape)
    ndim = len(stride)
    if ndim not in (2, 3):
        raise ValueError(f"Window shape must be 2D or 3D, got {ndim}D")
    original_shape = tuple(x.shape[1:])
    padded_shape = []
    for i, size in enumerate(original_shape):
        if size <= window_shape[i]:
            padded_shape.append(window_shape[i])
        else:
            excess = (size - window_shape[i]) % stride[i]
            padded_shape.append(size if excess == 0 else (size + stride[i] - excess))
    padding_values = []
    for i in range(ndim - 1, -1, -1):
        pad_total = padded_shape[i] - original_shape[i]
        pad_before = pad_total // 2
        pad_after = pad_total - pad_before
        padding_values.extend([pad_before, pad_after])
    x = nn.functional.pad(x, padding_values, mode="constant", value=0)
    if ndim == 2:
        x = x.unfold(1, window_shape[0], stride[0]).unfold(2, window_shape[1], stride[1])
        c, n_h, n_w, win_h, win_w = x.shape
        x = x.permute(1, 2, 0, 3, 4).reshape(n_h * n_w, c, win_h, win_w)
        return x, (n_h, n_w), (original_shape[0], original_shape[1])
    x = x.unfold(1, window_shape[0], stride[0]).unfold(2, window_shape[1], stride[1]).unfold(
        3, window_shape[2], stride[2])
    c, n_d, n_h, n_w, win_d, win_h, win_w = x.shape
    x = x.permute(1, 2, 3, 0, 4, 5, 6).reshape(n_d * n_h * n_w, c, win_d, win_h, win_w)
    return x, (n_d, n_h, n_w), (original_shape[0], original_shape[1], original_shape[2])


def revert_sliding_window(windows: torch.Tensor, layout: Shape, original_shape: Shape, *,
                          overlap: float = .5) -> torch.Tensor:
    first_window = windows[0]
    ndim = first_window.ndim - 1
    if ndim not in (2, 3):
        raise ValueError(f"Windows must be 2D or 3D (excluding channel dim), got {ndim}D")
    window_shape = first_window.shape[1:]
    c = first_window.shape[0]
    stride = tuple(int(w * (1 - overlap)) for w in window_shape)
    if ndim == 2:
        h_win, w_win = window_shape
        n_h, n_w = layout
        out_h = (n_h - 1) * stride[0] + h_win
        out_w = (n_w - 1) * stride[1] + w_win
        windows_flat = windows[:n_h * n_w].view(n_h * n_w, c * h_win * w_win)
        output = nn.functional.fold(
            windows_flat.transpose(0, 1),
            output_size=(out_h, out_w),
            kernel_size=(h_win, w_win),
            stride=stride
        )
        weights = nn.functional.fold(
            torch.ones(c * h_win * w_win, n_h * n_w, device=first_window.device, dtype=torch.uint8),
            output_size=(out_h, out_w),
            kernel_size=(h_win, w_win),
            stride=stride
        ).sum(dim=0, keepdim=True)
        output /= weights.clamp(min=1)
        pad_h = out_h - original_shape[0]
        pad_w = out_w - original_shape[1]
        h_start = pad_h // 2
        w_start = pad_w // 2
        return output[:, h_start:h_start + original_shape[0], w_start:w_start + original_shape[1]]
    d_win, h_win, w_win = window_shape
    n_d, n_h, n_w = layout
    out_d = (n_d - 1) * stride[0] + d_win
    out_h = (n_h - 1) * stride[1] + h_win
    out_w = (n_w - 1) * stride[2] + w_win
    output = torch.zeros(c, out_d, out_h, out_w, device=first_window.device, dtype=first_window.dtype)
    weights = torch.zeros(1, out_d, out_h, out_w, device=first_window.device, dtype=torch.uint8)
    windows = windows[:n_d * n_h * n_w].view(n_d, n_h, n_w, c, d_win, h_win, w_win)
    for i in range(n_d):
        d_start = i * stride[0]
        d_slice = slice(d_start, d_start + d_win)
        for j in range(n_h):
            h_start = j * stride[1]
            h_slice = slice(h_start, h_start + h_win)
            for k in range(n_w):
                w_start = k * stride[2]
                w_slice = slice(w_start, w_start + w_win)
                output[:, d_slice, h_slice, w_slice] += windows[i, j, k]
                weights[0, d_slice, h_slice, w_slice] += 1
    output /= weights.clamp(min=1)
    pad_d = out_d - original_shape[0]
    pad_h = out_h - original_shape[1]
    pad_w = out_w - original_shape[2]
    d_start = pad_d // 2
    h_start = pad_h // 2
    w_start = pad_w // 2
    return output[:, d_start:d_start + original_shape[0], h_start:h_start + original_shape[1],
    w_start:w_start + original_shape[2]]


def _slide_internal(image: torch.Tensor, window_shape: Shape, overlap: float, i: int, ind: int, output_folder: str, *,
                    is_label: bool = False) -> None:
    windows, layout, original_shape = do_sliding_window(image, window_shape, overlap=overlap)
    jnd = int(log10(windows.shape[0])) + 1
    for j in range(windows.shape[0]):
        path = f"{output_folder}/{"labels" if is_label else "images"}/{str(i).zfill(ind)}_{str(j).zfill(jnd)}"
        fast_save(windows[j], f"{path}_{layout}_{original_shape}.pt" if j == 0 else f"{path}.pt")


def _slide(supervised: bool, dataset: UnsupervisedDataset | SupervisedDataset, output_folder: str | PathLike[str],
           window_shape: Shape, *, overlap: float = .5, console: Console = Console()) -> None:
    makedirs(f"{output_folder}/images", exist_ok=True)
    if supervised:
        makedirs(f"{output_folder}/labels", exist_ok=True)
    ind = int(log10(len(dataset))) + 1
    with Progress(console=console) as progress:
        task = progress.add_task("Sliding dataset...", total=len(dataset))
        for i, case in enumerate(dataset):
            image = case[0] if supervised else case
            progress.update(task, description=f"Sliding dataset {tuple(image.shape)}...")
            _slide_internal(image, window_shape, overlap, i, ind, output_folder)
            if supervised:
                label = case[1]
                _slide_internal(label, window_shape, overlap, i, ind, output_folder, is_label=True)
            progress.update(task, advance=1, description=f"Sliding dataset ({i + 1}/{len(dataset)})...")


def slide_dataset(dataset: UnsupervisedDataset | SupervisedDataset, output_folder: str | PathLike[str],
                  window_shape: Shape, *, overlap: float = .5, console: Console = Console()) -> None:
    _slide(isinstance(dataset, SupervisedDataset), dataset, output_folder, window_shape, overlap=overlap,
           console=console)


@dataclass
class SWCase(object):
    window_indices: list[int]
    layout: Shape | None
    original_shape: Shape | None


class UnsupervisedSWDataset(TensorLoader, PathBasedUnsupervisedDataset):
    def __init__(self, folder: str | PathLike[str], *, subfolder: Literal["images", "labels"] = "images",
                 transform: Transform | None = None, device: Device = "cpu") -> None:
        super().__init__(sorted(listdir(f"{folder}/{subfolder}")), transform=transform, device=device)
        self._folder: str = folder
        self._subfolder: Literal["images", "labels"] = subfolder
        self._groups: list[SWCase] = []
        for idx, filename in enumerate(self._images):
            meta = filename[:filename.rfind(".")].split("_")
            case_id = int(meta[0])
            if case_id >= len(self._groups):
                if case_id != len(self._groups):
                    raise ValueError(f"Mismatched case id {case_id}")
                self._groups.append(SWCase([], None, None))
            self._groups[case_id].window_indices.append(idx)
            if len(meta) == 4:
                if self._groups[case_id].layout:
                    raise ValueError(f"Duplicated layout specification for case {case_id}")
                self._groups[case_id].layout = literal_eval(meta[2])
                if self._groups[case_id].original_shape:
                    raise ValueError(f"Duplicated original shape specification for case {case_id}")
                self._groups[case_id].original_shape = literal_eval(meta[3])
        for idx, case in enumerate(self._groups):
            windows, layout, original_shape = case.window_indices, case.layout, case.original_shape
            if not layout:
                raise ValueError(f"Layout not specified for case {idx}")
            if not original_shape:
                raise ValueError(f"Original shape not specified for case {idx}")
            if len(windows) != reduce(mul, layout):
                raise ValueError(f"Mismatched number of windows {len(windows)} and layout {layout} for case {idx}")

    @override
    def load(self, idx: int) -> torch.Tensor:
        return self.do_load(f"{self._folder}/{self._subfolder}/{self._images[idx]}",
                            is_label=self._subfolder == "labels", device=self._device)

    def case_meta(self, case_idx: int) -> tuple[int, Shape, Shape]:
        case = self._groups[case_idx]
        return len(case.window_indices), case.layout, case.original_shape

    def case(self, case_idx: int, *, part: slice | None = None) -> torch.Tensor:
        indices = self._groups[case_idx].window_indices
        return torch.stack([self[idx] for idx in (indices[part] if part else indices)])


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
