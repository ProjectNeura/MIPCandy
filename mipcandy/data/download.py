from os import PathLike
from zipfile import ZipFile

from requests import get
from rich.progress import track


def download_dataset(name: str, to: str | PathLike[str], *, endpoint: str = "cds.projectneura.org") -> None:
    to_zip = f"{to}.zip"
    with get(f"https://{endpoint}/{name}.zip", stream=True) as response:
        response.raise_for_status()
        with open(to_zip, "wb") as f:
            for chunk in track(response.iter_content(chunk_size=8192), description="Downloading..."):
                f.write(chunk)
    print(f"Dataset downloaded successfully as {to_zip}")
    print("Unzipping...")
    with ZipFile(to_zip, "r") as zip_ref:
        zip_ref.extractall(to)
    print(f"Dataset extracted successfully to {to_zip}")
