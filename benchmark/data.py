from os import makedirs
from typing import override, Literal

from rich.progress import Progress

from benchmark.prototype import UnitTest
from mipcandy import NNUNetDataset, visualize3d, JointTransform, inspect, RandomROIDataset


class DataTest(UnitTest):
    dataset: str = "AbdomenCT-1K-ss1"
    transform: JointTransform | None = None

    @override
    def set_up(self) -> None:
        self["dataset"] = NNUNetDataset(f"{self.input_folder}/{DataTest.dataset}", transform=self.transform,
                                        device=self.device)
        self["dataset"].preload(f"{self.input_folder}/{DataTest.dataset}/preloaded")


class FoldedDataTest(DataTest):
    fold: Literal[0, 1, 2, 3, 4, "all"] = 0

    @override
    def set_up(self) -> None:
        super().set_up()
        self["train_dataset"], self["val_dataset"] = self["dataset"].fold(fold=self.fold)


class RandomROIDatasetTest(DataTest):
    @override
    def execute(self) -> None:
        annotations = inspect(self["dataset"])
        dataset = RandomROIDataset(annotations, 2)
        print(dataset.roi_shape())
        o = f"{self.output_folder}/RandomROIPreviews"
        makedirs(o, exist_ok=True)
        makedirs(f"{o}/images", exist_ok=True)
        makedirs(f"{o}/labels", exist_ok=True)
        makedirs(f"{o}/imageROIs", exist_ok=True)
        makedirs(f"{o}/labelROIs", exist_ok=True)
        with Progress() as progress:
            task = progress.add_task("Generating Previews...", total=len(dataset))
            for idx, (image_roi, label_roi) in enumerate(dataset):
                image, label = self["dataset"][idx]
                visualize3d(image, title="image raw", screenshot_as=f"{o}/images/{idx}.png")
                visualize3d(label.int(), title="label raw", is_label=True, screenshot_as=f"{o}/labels/{idx}.png")
                visualize3d(image_roi, title="image roi", screenshot_as=f"{o}/imageROIs/{idx}.png")
                visualize3d(label_roi.int(), title="label roi", is_label=True, screenshot_as=f"{o}/labelROIs/{idx}.png")
                progress.update(task, advance=1)
