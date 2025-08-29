from typing import Literal

from mipcandy.config import load_settings, save_settings, load_secrets, save_secrets


def config(target: Literal["setting", "secret"], key: str, value: str) -> None:
    match target:
        case "setting":
            settings = load_settings()
            settings[key] = value
            save_settings(settings)
        case "secret":
            secrets = load_secrets()
            secrets[key] = value
            save_secrets(secrets)
