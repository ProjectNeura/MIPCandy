from dataclasses import dataclass
from inspect import stack
from os import PathLike
from time import time
from typing import Sequence, override

import torch
from psutil import cpu_percent, virtual_memory

from mipcandy.types import Device


@dataclass
class ProfilerFrame(object):
    stack: str
    cpu: float
    mem: float
    gpu: list[float] | None = None
    gpu_mem: list[float] | None = None

    @override
    def __str__(self) -> str:
        r = f"[{self.stack}] CPU: {self.cpu:.2f}% @ Memory: {self.mem:.2f}%\n"
        if self.gpu and self.gpu_mem:
            for i, gpu in enumerate(self.gpu):
                r += f"\tGPU {i}: {gpu:.2f}% @ Memory: {self.gpu_mem[i]:.2f}%\n"
        return r

    def export(self, duration: float) -> str:
        return f"{duration:.2f}s\t{self}"


class _LineBreak(object):
    def __init__(self, message: str) -> None:
        self.message: str = message

    @override
    def __str__(self) -> str:
        return f"<{self.message}>\n"

    def export(self, duration: float) -> str:
        return f"{duration:.2f}s\t{self}"


class Profiler(object):
    def __init__(self, title: str, save_as: str | PathLike[str], *, gpus: Sequence[Device] = ()) -> None:
        self.title: str = title
        self.save_as: str = save_as
        self.total_mem: float = self.get_total_mem()
        self.has_gpu: bool = len(gpus) > 0
        self._gpus: Sequence[Device] = gpus
        self.total_gpu_mem: list[float] = [self.get_total_gpu_mem(device) for device in gpus]
        with open(save_as, "w") as f:
            f.write(f"# {title}\nTotal memory: {self.total_mem}, Total GPU memory: {self.total_gpu_mem}\n\n")
        self._t0: float = time()

    @staticmethod
    def get_cpu_usage() -> float:
        return cpu_percent()

    def get_mem_usage(self) -> float:
        return 100 * virtual_memory().used / self.total_mem

    @staticmethod
    def get_total_mem() -> float:
        return virtual_memory().total

    @staticmethod
    def get_gpu_usage(device: Device) -> float:
        return torch.cuda.utilization(device)

    def get_gpu_mem_usage(self, device: Device) -> float:
        return 100 * torch.cuda.memory_allocated(device) / self.total_gpu_mem[self._gpus.index(device)]

    @staticmethod
    def get_total_gpu_mem(device: Device) -> float:
        return torch.cuda.get_device_properties(device).total_memory

    def _save(self, obj: ProfilerFrame | _LineBreak) -> None:
        with open(self.save_as, "a") as f:
            t = time()
            f.write(f"{obj.export(t - self._t0)}\n")
            self._t0 = t

    def record(self, *, stack_trace_offset: int = 1) -> ProfilerFrame:
        frame = ProfilerFrame(" -> ".join([f"{f.function}:{f.lineno}" for f in stack()[stack_trace_offset::-1]]),
                              self.get_cpu_usage(), self.get_mem_usage())
        if self.has_gpu:
            frame.gpu = [torch.cuda.utilization(device) for device in self._gpus]
            frame.gpu_mem = [self.get_gpu_mem_usage(device) for device in self._gpus]
        self._save(frame)
        return frame

    def line_break(self, message: str) -> None:
        self._save(_LineBreak(message))
