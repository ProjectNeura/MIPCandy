from typing import Any, Generator, Self

from torch import nn

from mipcandy.types import Device


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

    def update(self, *, must_exist: bool = True, **kwargs) -> Self:
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


class HasDevice(object):
    def __init__(self, device: Device) -> None:
        self._device: Device = device


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
