from os import PathLike
from typing import Any

from mipcandy import Device, Frontend


class UnitTest(object):
    def __init__(self, input_folder: str | PathLike[str], output_folder: str | PathLike[str], num_epochs: int,
                 device: Device, frontend: type[Frontend]) -> None:
        self.input_folder: str = input_folder
        self.output_folder: str = output_folder
        self.num_epochs: int = num_epochs
        self.device: Device = device
        self.frontend: type[Frontend] = frontend

    def set_up(self) -> None:
        pass

    def execute(self) -> None:
        pass

    def clean_up(self) -> None:
        pass

    def run(self) -> tuple[bool, Exception | None]:
        try:
            self.set_up()
            self.execute()
        except Exception as e:
            try:
                self.clean_up()
            except Exception as e2:
                print(f"Failed to clean up after exception: {e2}")
            return False, e
        return True, None

    def __setitem__(self, key: str, value: Any) -> None:
        setattr(self, "_x_" + key, value)

    def __getitem__(self, item: str) -> Any:
        return getattr(self, "_x_" + item)
