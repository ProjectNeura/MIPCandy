from os import makedirs
from time import time
from typing import override, Literal

from rich.progress import Progress

from benchmark.prototype import UnitTest
from mipcandy import NNUNetDataset, do_sliding_window, visualize3d, revert_sliding_window, JointTransform, inspect, \
    RandomROIDataset


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


class SlidingWindowTest(DataTest):
    @override
    def execute(self) -> None:
        image, _ = self["dataset"][0]
        print(image.shape)
        visualize3d(image, title="raw")
        t0 = time()
        windows, layout, pad = do_sliding_window(image, (128, 128, 128))
        print(f"took {time() - t0:.2f}s")
        print(windows[0].shape, layout)
        t0 = time()
        recon = revert_sliding_window(windows, layout, pad)
        print(f"took {time() - t0:.2f}s")
        print(recon.shape)
        visualize3d(recon, title="reconstructed")


class RandomROIDatasetTest(DataTest):
    @override
    def execute(self) -> None:
        annotations = inspect(self["dataset"])
        dataset = RandomROIDataset(annotations)
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
