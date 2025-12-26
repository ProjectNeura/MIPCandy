from argparse import ArgumentParser
from os import PathLike
from os.path import exists

import torch
from monai.transforms import Resized
from torch.utils.data import DataLoader
from torchvision.transforms import Compose

from mipcandy import Device, auto_device, download_dataset, NNUNetDataset, inspect, InspectionAnnotations, \
    load_inspection_annotations, ROIDataset, JointTransform
from transforms import build_nnunet_transforms
from unet import UNetTrainer, UNetSlidingTrainer

BENCHMARK_DATASET: str = "nnunet_datasets/AbdomenCT-1K-ss1"
BENCHMARK_NUM_CLASSES: int = 5


def inspect_dataset(dataset: NNUNetDataset, output_folder: str | PathLike[str]) -> InspectionAnnotations:
    if exists(f"{output_folder}/annotations.json"):
        return load_inspection_annotations(f"{output_folder}/annotations.json", dataset)
    annotations = inspect(dataset)
    annotations.save(f"{output_folder}/annotations.json")
    return annotations


def full(input_folder: str | PathLike[str], output_folder: str | PathLike[str], *, num_epochs: int = 100,
         device: Device | None = None) -> None:
    if not device:
        device = auto_device()
    if not exists(f"{input_folder}/dataset"):
        download_dataset(f"nnunet_datasets/{BENCHMARK_DATASET}", f"{input_folder}/dataset")
    dataset = NNUNetDataset(f"{input_folder}/dataset", transform=JointTransform(
        transform=build_nnunet_transforms()), device="cuda")
    train, val = dataset.fold()
    train_loader = DataLoader(train, batch_size=1, shuffle=True)
    val_loader = DataLoader(val, batch_size=1, shuffle=False)
    getattr(torch, "_dynamo").config.automatic_dynamic_shapes = True
    trainer = UNetSlidingTrainer(output_folder, train_loader, val_loader, recoverable=False, device=device)
    trainer.num_classes = BENCHMARK_NUM_CLASSES
    trainer.train(num_epochs, note="MIP Candy Benchmark - full size")


def resize(size: int, input_folder: str | PathLike[str], output_folder: str | PathLike[str], *, num_epochs: int = 100,
           device: Device | None = None) -> None:
    if not device:
        device = auto_device()
    if not exists(f"{input_folder}/dataset"):
        download_dataset(f"nnunet_datasets/{BENCHMARK_DATASET}", f"{input_folder}/dataset")
    dataset = NNUNetDataset(f"{input_folder}/dataset", transform=JointTransform(transform=Compose([
        Resized(("image", "label"), (size, size, size)), build_nnunet_transforms()
    ])), device=device)
    train, val = dataset.fold()
    train_loader = DataLoader(train, batch_size=2, shuffle=True)
    val_loader = DataLoader(val, batch_size=1, shuffle=False)
    trainer = UNetTrainer(output_folder, train_loader, val_loader, recoverable=False, device=device)
    trainer.num_classes = BENCHMARK_NUM_CLASSES
    trainer.train(num_epochs, note=f"MIP Candy Benchmark - resize{size}")


def resize128(input_folder: str | PathLike[str], output_folder: str | PathLike[str], *, num_epochs: int = 100,
              device: Device | None = None) -> None:
    resize(128, input_folder, output_folder, num_epochs=num_epochs, device=device)


def resize256(input_folder: str | PathLike[str], output_folder: str | PathLike[str], *, num_epochs: int = 100,
              device: Device | None = None) -> None:
    resize(256, input_folder, output_folder, num_epochs=num_epochs, device=device)


def roi(input_folder: str | PathLike[str], output_folder: str | PathLike[str], *, num_epochs: int = 100,
        device: Device | None = None) -> None:
    if not device:
        device = auto_device()
    if not exists(f"{input_folder}/dataset"):
        download_dataset(f"nnunet_datasets/{BENCHMARK_DATASET}", f"{input_folder}/dataset")
    dataset = NNUNetDataset(f"{input_folder}/dataset")
    annotations = inspect_dataset(dataset, output_folder)
    dataset = ROIDataset(annotations)
    train, val = dataset.fold()
    train_loader = DataLoader(train, batch_size=2, shuffle=True)
    val_loader = DataLoader(val, batch_size=1, shuffle=False)
    trainer = UNetTrainer(output_folder, train_loader, val_loader, recoverable=False, device=device)
    trainer.num_classes = BENCHMARK_NUM_CLASSES
    trainer.train(num_epochs, note=f"MIP Candy Benchmark - roi")


if __name__ == "__main__":
    parser = ArgumentParser(prog="MIP Candy Benchmark", description="MIP Candy Benchmark",
                            epilog="GitHub: https://github.com/ProjectNeura/MIPCandy")
    parser.add_argument("test", choices=("full", "resize128", "resize256", "roi"))
    parser.add_argument("-i", "--input-folder")
    parser.add_argument("-o", "--output-folder")
    parser.add_argument("--num-epochs", type=int, default=100)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    test = locals()[args.test]
    test(args.input_folder, args.output_folder, num_epochs=args.num_epochs, device=args.device)
