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

    def _interp(self, step: int) -> float:
        step -= self._restart_step
        r = self._k * step + self._b
        if r < self._min_lr:
            if self._restart:
                self._restart_step = step
                return self._interp(step)
            return self._min_lr
        return r

    @override
    def get_lr(self) -> list[float]:
        target = self._interp(self.last_epoch)
        return [target for _ in self.optimizer.param_groups]
