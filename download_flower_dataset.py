from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import requests

URL = "https://drive.google.com/file/d/1OsiBs2udl32-1CqTXCitmov4NQCYdA9g/view?usp=share_link"


def download_flowers_dataset(destination: Path):
    response = requests.get(URL)
    response.raise_for_status()
    with TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        temp_zip = temp_dir / "flowers.zip"
        print(response.content.decode("utf-8"))
        temp_zip.write_bytes(response.content)
        with ZipFile(temp_zip) as temp_zip_file:
            temp_zip_file.extractall(destination)


if __name__ == "__main__":
    dataset_folder = Path(__file__).parent / "datasets"
    dataset_folder.mkdir(exist_ok=True)
    download_flowers_dataset(dataset_folder / "flowers")
