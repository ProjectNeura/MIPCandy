from os import PathLike
from os.path import abspath, exists
from typing import override

from yaml import load, SafeLoader, dump, SafeDumper

from mipcandy.types import Secret, Secrets


_DEFAULT_SECRETS_PATH: str = f"{abspath(__file__)[:-12]}secrets.yml"


def load_secrets(*, path: str | PathLike[str] = _DEFAULT_SECRETS_PATH) -> Secrets:
    if not exists(path):
        with open(path, "w") as f:
            f.write("# fill in your secrets here, do not commit this file\n")
    with open(path) as f:
        secrets = load(f.read(), SafeLoader)
        if secrets is None:
            return {}
        if not isinstance(secrets, dict):
            raise ValueError(f"Invalid secrets file: {path}")
        return secrets


def save_secrets(secrets: Secrets, *, path: str | PathLike[str] = _DEFAULT_SECRETS_PATH) -> None:
    with open(path, "w") as f:
        f.write("# fill in your secrets here, do not commit this file\n")
        dump(secrets, f, SafeDumper)


class Frontend(object):
    def __init__(self, secrets: Secrets) -> None:
        self._secrets: Secrets = secrets

    def require_nonempty_secret(self, entry: str, *, require_type: type | None = None) -> Secret:
        if entry not in self._secrets:
            raise ValueError(f"Missing secret {entry}")
        secret = self._secrets[entry]
        if require_type is None or isinstance(secret, require_type):
            return secret
        raise ValueError(f"Invalid secret type {type(secret)}, {require_type} expected")

    def on_experiment_created(self, experiment_id: str, trainer: str, model: str, note: str, num_params: float,
                              num_macs: float, num_epochs: int, early_stop_tolerance: int) -> None:
        ...

    def on_experiment_updated(self, experiment_id: str, epoch: int, metrics: dict[str, list[float]],
                              early_stop_tolerance: int) -> None:
        ...

    def on_experiment_completed(self, experiment_id: str) -> None:
        ...

    def on_experiment_interrupted(self, experiment_id: str, error: Exception) -> None:
        ...


def create_hybrid_frontend(*frontends: Frontend) -> type[Frontend]:
    class HybridFrontend(Frontend):
        def __init__(self, secrets: Secrets) -> None:
            super().__init__(secrets)

        @override
        def on_experiment_created(self, experiment_id: str, trainer: str, model: str, note: str, num_macs: float,
                                  num_params: float, num_epochs: int, early_stop_tolerance: int) -> None:
            for frontend in frontends:
                frontend.on_experiment_created(experiment_id, trainer, model, note, num_macs, num_params, num_epochs,
                                               early_stop_tolerance)

        @override
        def on_experiment_updated(self, experiment_id: str, epoch: int, metrics: dict[str, list[float]],
                                  early_stop_tolerance: int) -> None:
            for frontend in frontends:
                frontend.on_experiment_updated(experiment_id, epoch, metrics, early_stop_tolerance)

        @override
        def on_experiment_completed(self, experiment_id: str) -> None:
            for frontend in frontends:
                frontend.on_experiment_completed(experiment_id)

        @override
        def on_experiment_interrupted(self, experiment_id: str, error: Exception) -> None:
            for frontend in frontends:
                frontend.on_experiment_interrupted(experiment_id, error)

    return HybridFrontend
