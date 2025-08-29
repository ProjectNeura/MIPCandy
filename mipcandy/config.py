from os import PathLike
from os.path import abspath, exists

from yaml import load, SafeLoader, dump, SafeDumper

from mipcandy.types import Settings

_DIR: str = abspath(__file__)[:-9]
_DEFAULT_SETTINGS_PATH: str = f"{_DIR}settings.yml"
_DEFAULT_SECRETS_PATH: str = f"{_DIR}secrets.yml"


def _load(path: str | PathLike[str], *, hint: str = "fill in your settings here") -> Settings:
    if not exists(path):
        with open(path, "w") as f:
            f.write(f"# {hint}\n")
    with open(path) as f:
        settings = load(f.read(), SafeLoader)
        if settings is None:
            return {}
        if not isinstance(settings, dict):
            raise ValueError(f"Invalid settings file: {path}")
        return settings


def _save(settings: Settings, path: str | PathLike[str], *, hint: str = "fill in your settings here") -> None:
    with open(path, "w") as f:
        f.write(f"# {hint}\n")
        dump(settings, f, SafeDumper)


def load_settings(*, path: str | PathLike[str] = _DEFAULT_SETTINGS_PATH) -> Settings:
    return _load(path)


def save_settings(settings: Settings, *, path: str | PathLike[str] = _DEFAULT_SETTINGS_PATH) -> None:
    _save(settings, path)


def load_secrets(*, path: str | PathLike[str] = _DEFAULT_SECRETS_PATH) -> Settings:
    return _load(path, hint="fill in your secrets here, do not commit this file")


def save_secrets(secrets: Settings, *, path: str | PathLike[str] = _DEFAULT_SECRETS_PATH) -> None:
    _save(secrets, path, hint="fill in your secrets here, do not commit this file")
