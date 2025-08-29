from os import PathLike
from os.path import abspath, exists

from yaml import load, SafeLoader, dump, SafeDumper

from mipcandy.types import Secrets

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
