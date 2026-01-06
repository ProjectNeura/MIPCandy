from typing import override, Literal

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
        windows, layout, pad = do_sliding_window(image, (128, 128, 128))
        print(windows[0].shape, layout)
        visualize3d(windows[0], title="first window")
        recon = revert_sliding_window(windows, layout, pad)
        print(recon.shape)
        visualize3d(recon, title="reconstructed")


class RandomROIDatasetTest(DataTest):
    @override
    def execute(self) -> None:
        annotations = inspect(self["dataset"])
        dataset = RandomROIDataset(annotations)
        print(len(dataset))
        image, label = self["dataset"][0]
        image_roi, label_roi = dataset[0]
        visualize3d(image, title="image raw")
        visualize3d(label, title="label raw")
        visualize3d(image_roi, title="image roi")
        visualize3d(label_roi, title="label roi")
