from typing import override

from torch import optim


class AbsoluteLinearLR(optim.lr_scheduler.LRScheduler):
    """
    lr = kx + b
    """
    def __init__(self, optimizer: optim.Optimizer, k: float, b: float, *, min_lr: float = 1e-6,
                 restart: bool = False, last_epoch: int = -1) -> None:
        self._k: float = k
        self._b: float = b
        if min_lr < 0:
            raise ValueError(f"`min_lr` must be positive, but got {min_lr}")
        self._min_lr: float = min_lr
        self._restart: bool = restart
        self._restart_step: int = 0
        super().__init__(optimizer, last_epoch)

    def _interp(self, epoch: int) -> float:
        epoch -= self._restart_step
        r = self._k * epoch + self._b
        if r < self._min_lr:
            if self._restart:
                self._restart_step = epoch
                return self._interp(epoch)
            return self._min_lr
        return r

    @override
    def get_lr(self) -> list[float]:
        target = self._interp(self.last_epoch)
        return [target for _ in self.optimizer.param_groups]


class PolyLRScheduler(optim.lr_scheduler.LRScheduler):
    def __init__(self, optimizer: optim.Optimizer, initial_lr: float, max_steps: int, *, exponent: float = .9,
                 last_epoch: int = -1) -> None:
        self._initial_lr: float = initial_lr
        self._max_steps: int = max_steps
        self._exponent: float = exponent
        super().__init__(optimizer, last_epoch)

    def _interp(self, epoch: int) -> float:
        return self._initial_lr * (1 - epoch / self._max_steps) ** self._exponent

    @override
    def get_lr(self) -> list[float]:
        target = self._interp(self.last_epoch)
        return [target for _ in self.optimizer.param_groups]
