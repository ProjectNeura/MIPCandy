from os import removedirs
from os.path import exists
from typing import override

from monai.transforms import Resized
from torch.utils.data import DataLoader

from benchmark.data import DataTest, FoldedDataTest
from benchmark.unet import UNetTrainer
from mipcandy import SegmentationTrainer, Shape, JointTransform, inspect, load_inspection_annotations, \
    RandomROIDataset, CTNormalize


class TrainingTest(DataTest):
    trainer: type[SegmentationTrainer] = UNetTrainer
    resize: Shape = (128, 128, 128)
    num_classes: int = 5
    _continue: str | None = None  # internal flag for continued training

    def set_up_datasets(self) -> None:
        super().set_up()
        path = f"{self.output_folder}/training_test.json"
        self["dataset"].device(device="cpu")
        if exists(path):
            annotations = load_inspection_annotations(path, self["dataset"])
        else:
            annotations = inspect(self["dataset"])
            annotations.save(path)
        dataset = RandomROIDataset(annotations, 2, num_patches_per_case=3)
        dataset.roi_shape(roi_shape=(80, 160, 160))
        self["annotations"] = annotations
        self["train_dataset"], self["val_dataset"] = dataset.fold(fold=0)

    @override
    def set_up(self) -> None:
        self.set_up_datasets()
        train, val = self["train_dataset"], self["val_dataset"]
        val.preload(f"{self.output_folder}/valPreloaded")
        ct_normalization = CTNormalize(*self["annotations"].intensity_stats())
        train.set_transform(JointTransform(image_only=ct_normalization))
        val.set_transform(JointTransform(image_only=ct_normalization))
        train_dataloader = DataLoader(train, batch_size=2, shuffle=True, pin_memory=True, prefetch_factor=2,
                                      num_workers=2, persistent_workers=True)
        val_dataloader = DataLoader(val, batch_size=1, shuffle=False, pin_memory=True)
        trainer = self.trainer(self.output_folder, train_dataloader, val_dataloader, device=self.device)
        trainer.num_classes = self.num_classes
        trainer.set_frontend(self.frontend)
        self["trainer"] = trainer

    @override
    def execute(self) -> None:
        if not self._continue:
            return self["trainer"].train(self.num_epochs, note=f"Training test {self.resize}", compile_model=True,
                                         val_score_prediction=False)
        self["trainer"].recover_from(self._continue)
        return self["trainer"].continue_training(self.num_epochs)

    @override
    def clean_up(self) -> None:
        removedirs(self["trainer"].experiment_folder())


class ResizeTrainingTest(FoldedDataTest):
    trainer: type[SegmentationTrainer] = UNetTrainer
    resize: Shape = (256, 256, 256)
    num_classes: int = 5

    @override
    def set_up(self) -> None:
        self.transform = JointTransform(transform=Resized(("image", "label"), self.resize))
        super().set_up()
        train_dataloader = DataLoader(self["train_dataset"], batch_size=2, shuffle=True)
        val_dataloader = DataLoader(self["val_dataset"], batch_size=1, shuffle=False)
        trainer = self.trainer(self.output_folder, train_dataloader, val_dataloader, recoverable=False,
                               profiler=True, device=self.device)
        trainer.num_classes = self.num_classes
        trainer.set_frontend(self.frontend)
        self["trainer"] = trainer

    @override
    def execute(self) -> None:
        self["trainer"].train(self.num_epochs, note=f"Resize Training test {self.resize}")

    @override
    def clean_up(self) -> None:
        removedirs(self["trainer"].experiment_folder())
