from typing import override

from wandb import init, Run

from mipcandy.frontend.prototype import Frontend
from mipcandy.types import Settings


class WandBFrontend(Frontend):
    def __init__(self, secrets: Settings) -> None:
        super().__init__(secrets)
        self._entity: str = self.require_nonempty_secret("wandb_entity", require_type=str)
        self._project: str = self.require_nonempty_secret("wandb_project", require_type=str)
        self._run: Run | None = None

    @override
    def on_experiment_created(self, experiment_id: str, trainer: str, model: str, note: str, num_macs: float,
                              num_params: float, num_epochs: int, early_stop_tolerance: int) -> None:
        self._run = init(entity=self._entity, project=self._project, config={
            "experiment_id": experiment_id, "trainer": trainer, "model": model, "note": note, "num_macs": num_macs,
            "num_params": num_params, "num_epochs": num_epochs
        })

    @override
    def on_experiment_updated(self, experiment_id: str, epoch: int, metrics: dict[str, list[float]],
                              early_stop_tolerance: int) -> None:
        if self._run:
            self._run.log(metrics)

    @override
    def on_experiment_completed(self, experiment_id: str) -> None:
        if not self._run:
            raise RuntimeError("Experiment has not been created")
        self._run.finish()
