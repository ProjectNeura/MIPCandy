from abc import ABCMeta, abstractmethod
from os import PathLike
from typing import Any, Generator, Self, override

import torch
from safetensors.torch import save_model, load_model
from torch import nn

from mipcandy.types import Device, AmbiguousShape


def batch_int_multiply(f: float, *n: int) -> Generator[int, None, None]:
    for i in n:
        r = i * f
        if not r.is_integer():
            raise ValueError(f"Inequivalent conversion")
        yield int(r)


def batch_int_divide(f: float, *n: int) -> Generator[int, None, None]:
    return batch_int_multiply(1 / f, *n)


class LayerT(object):
    def __init__(self, m: type[nn.Module], **kwargs) -> None:
        self.m: type[nn.Module] = m
        self.kwargs: dict[str, Any] = kwargs

    def update(self, *, must_exist: bool = True, inplace: bool = False, **kwargs) -> Self:
        if not inplace:
            return self.copy().update(must_exist=must_exist, inplace=True, **kwargs)
        for k, v in kwargs.items():
            if not must_exist or k in self.kwargs:
                self.kwargs[k] = v
        return self

    def assemble(self, *args, **kwargs) -> nn.Module:
        self_kwargs = self.kwargs.copy()
        for k, v in self_kwargs.items():
            if isinstance(v, str) and v in kwargs:
                self_kwargs[k] = kwargs.pop(v)
        return self.m(*args, **self_kwargs, **kwargs)

    def copy(self) -> Self:
        return self.__class__(self.m, **self.kwargs)


class HasDevice(object):
    def __init__(self, device: Device) -> None:
        self._device: Device = device

    def device(self, *, device: Device | None = None) -> None | Device:
        if device is None:
            return self._device
        self._device = device


def auto_device() -> Device:
    if torch.cuda.is_available():
        return f"cuda:{max(range(torch.cuda.device_count()),
                           key=lambda i: torch.cuda.memory_reserved(i) - torch.cuda.memory_allocated(i))}"
    if torch.mps.is_available():
        return "mps"
    return "cpu"


class WithPaddingModule(HasDevice):
    def __init__(self, device: Device) -> None:
        super().__init__(device)
        self._padding_module: nn.Module | None = None
        self._restoring_module: nn.Module | None = None
        self._padding_module_built: bool = False

    def build_padding_module(self) -> nn.Module | None:
        return None

    def build_restoring_module(self, padding_module: nn.Module | None) -> nn.Module | None:
        return None

    def _lazy_load_padding_module(self) -> None:
        if self._padding_module_built:
            return
        self._padding_module = self.build_padding_module()
        if self._padding_module:
            self._padding_module = self._padding_module.to(self._device)
        self._restoring_module = self.build_restoring_module(self._padding_module)
        if self._restoring_module:
            self._restoring_module = self._restoring_module.to(self._device)
        self._padding_module_built = True

    def get_padding_module(self) -> nn.Module | None:
        self._lazy_load_padding_module()
        return self._padding_module

    def get_restoring_module(self) -> nn.Module | None:
        self._lazy_load_padding_module()
        return self._restoring_module


class WithCheckpoint(object, metaclass=ABCMeta):
    @abstractmethod
    def load_checkpoint(self, model: nn.Module, path: str | PathLike[str]) -> nn.Module:
        raise NotImplementedError

    @abstractmethod
    def save_checkpoint(self, model: nn.Module, path: str | PathLike[str]) -> None:
        raise NotImplementedError


class WithNetwork(WithCheckpoint, HasDevice, metaclass=ABCMeta):
    def __init__(self, device: Device) -> None:
        super().__init__(device)

    @override
    def load_checkpoint(self, model: nn.Module, path: str | PathLike[str]) -> nn.Module:
        load_model(model, path)
        return model

    @override
    def save_checkpoint(self, model: nn.Module, path: str | PathLike[str]) -> None:
        save_model(model, path)

    @abstractmethod
    def build_network(self, example_shape: AmbiguousShape) -> nn.Module:
        raise NotImplementedError

    @staticmethod
    def compile_model(model: nn.Module) -> nn.Module:
        return torch.compile(model)

    def build_network_from_checkpoint(self, example_shape: AmbiguousShape, path: str | PathLike[str],
                                      compile_model: bool) -> nn.Module:
        """
        Internally exposed interface for overriding. Use `load_model()` instead.
        """
        model = self.build_network(example_shape)
        return self.load_checkpoint(self.compile_model(model) if compile_model else model, path)

    def load_model(self, example_shape: AmbiguousShape, compile_model: bool, *,
                   path: str | PathLike[str] | None = None) -> nn.Module:
        if path:
            return self.build_network_from_checkpoint(example_shape, path, compile_model).to(self._device)
        model = self.build_network(example_shape).to(self._device)
        return self.compile_model(model) if compile_model else model

    def save_model(self, model: nn.Module, path: str | PathLike[str]) -> None:
        model.compile()
        self.save_checkpoint(model, path)
