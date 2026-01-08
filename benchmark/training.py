from os import removedirs
from os.path import exists
from typing import override

from monai.transforms import Resized
from torch.utils.data import DataLoader

from benchmark.data import DataTest, FoldedDataTest
from benchmark.unet import UNetTrainer, UNetSlidingTrainer
from mipcandy import SegmentationTrainer, slide_dataset, Shape, SupervisedSWDataset, JointTransform, inspect, \
    ROIDataset, PadTo, MONAITransform, SimpleDataset


class TrainingTest(FoldedDataTest):
    trainer: type[SegmentationTrainer] = UNetTrainer
    resize: Shape = (256, 256, 256)
    num_classes: int = 5

    @override
    def set_up(self) -> None:
        self.transform = JointTransform(transform=Resized(("image", "label"), self.resize))
        super().set_up()
        train_dataloader = DataLoader(self["train_dataset"], batch_size=2, shuffle=True)
        val_dataloader = DataLoader(self["val_dataset"], batch_size=1, shuffle=False)
        self["trainer"] = self.trainer(self.output_folder, train_dataloader, val_dataloader, recoverable=False,
                                       device=self.device)
        self["trainer"].num_classes = self.num_classes
        self["trainer"].set_frontend(self.frontend)

    @override
    def execute(self) -> None:
        self["trainer"].train(self.num_epochs, note=f"Training test {self.resize}")

    @override
    def clean_up(self) -> None:
        removedirs(self["trainer"].experiment_folder())


class SlidingTrainingTest(FoldedDataTest):
    trainer: type[SegmentationTrainer] = UNetSlidingTrainer
    window_shape: Shape = (128, 128, 128)
    num_classes: int = 5
    overlap: float = .1

    @override
    def set_up(self) -> None:
        super().set_up()
        val_dataset = self["val_dataset"]
        if not exists(f"{self.output_folder}/val_slided"):
            slide_dataset(val_dataset, f"{self.output_folder}/val_slided", self.window_shape, overlap=self.overlap)
        slided_val_dataset = SupervisedSWDataset(f"{self.output_folder}/val_slided")
        train_dataset = self["train_dataset"]
        train_dataset.transform(
            transform=JointTransform(transform=MONAITransform(PadTo(self.window_shape, batch=False)))
        )
        annotations = inspect(train_dataset)
        annotations.set_roi_shape(self.window_shape)
        train_dataset = ROIDataset(annotations)
        train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False)
        self["trainer"] = self.trainer(self.output_folder, train_dataloader, val_dataloader, recoverable=False,
                                       device=self.device)
        self["trainer"].num_classes = self.num_classes
        self["trainer"].set_validation_dataset(SimpleDataset(f"{self.input_folder}/{DataTest.dataset}/labelsTr"))
        self["trainer"].set_slided_validation_dataset(slided_val_dataset)
        self["trainer"].overlap = self.overlap
        self["trainer"].set_frontend(self.frontend)

    @override
    def execute(self) -> None:
        self["trainer"].train(self.num_epochs, note="Training test with sliding window")

    @override
    def clean_up(self) -> None:
        removedirs(self["trainer"].experiment_folder())
