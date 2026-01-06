from os.path import exists
from typing import override

from monai.transforms import Resized
from torch.utils.data import DataLoader

from benchmark.data import FoldedDataTest
from benchmark.unet import UNetTrainer, UNetSlidingTrainer
from mipcandy import Trainer, slide_dataset, Shape, SupervisedSWDataset, JointTransform, inspect, ROIDataset, PadTo, \
    MONAITransform


class TrainingTest(FoldedDataTest):
    trainer: type[Trainer] = UNetTrainer
    resize: Shape = (256, 256, 256)

    @override
    def set_up(self) -> None:
        self.transform = JointTransform(transform=Resized(("image", "label"), self.resize))
        super().set_up()
        train_dataloader = DataLoader(self["train_dataset"], batch_size=2, shuffle=True, pin_memory=True)
        val_dataloader = DataLoader(self["val_dataset"], batch_size=1, shuffle=False, pin_memory=True)
        self["trainer"] = self.trainer(self.output_folder, train_dataloader, val_dataloader, recoverable=False,
                                       device=self.device)

    @override
    def execute(self) -> None:
        self["trainer"].train(self.num_epochs, note=f"Training test {self.resize}")


class SlidingTrainingTest(FoldedDataTest):
    trainer: type[Trainer] = UNetSlidingTrainer
    window_shape: Shape = (128, 128, 128)

    @override
    def set_up(self) -> None:
        super().set_up()
        val_dataset = self["val_dataset"]
        if not exists(f"{self.output_folder}/val_slided"):
            slide_dataset(val_dataset, f"{self.output_folder}/val_slided", self.window_shape)
        slided_val_dataset = SupervisedSWDataset(f"{self.output_folder}/val_slided")
        train_dataset = self["train_dataset"]
        train_dataset.transform(transform=JointTransform(transform=MONAITransform(PadTo(self.window_shape))))
        annotations = inspect(train_dataset)
        annotations.set_roi_shape(self.window_shape)
        train_dataset = ROIDataset(annotations)
        train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True, pin_memory=True)
        val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, pin_memory=True)
        self["trainer"] = self.trainer(self.output_folder, train_dataloader, val_dataloader, recoverable=False,
                                       device=self.device)
        self["trainer"].set_slided_validation_dataset(slided_val_dataset)

    @override
    def execute(self) -> None:
        self["trainer"].train(self.num_epochs, note="Training test")
