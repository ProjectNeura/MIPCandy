from argparse import ArgumentParser
from os import PathLike
from os.path import exists

import torch
from monai.transforms import Resized
from torch.utils.data import DataLoader
from torchvision.transforms import Compose

from mipcandy import Device, auto_device, download_dataset, NNUNetDataset, inspect, InspectionAnnotations, \
    load_inspection_annotations, ROIDataset, JointTransform, RandomROIDataset, Frontend
from mipcandy.frontend.notion_fe import NotionFrontend
from mipcandy.frontend.wandb_fe import WandBFrontend
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
         device: Device | None = None, frontend: type[Frontend] = Frontend) -> None:
    if not device:
        device = auto_device()
    if not exists(f"{input_folder}/dataset"):
        download_dataset(f"nnunet_datasets/{BENCHMARK_DATASET}", f"{input_folder}/dataset")
    dataset = NNUNetDataset(f"{input_folder}/dataset", device=device)
    train, val = dataset.fold()
    annotations = inspect(train)
    annotations.set_roi_shape((32, 224, 224))
    train = RandomROIDataset(annotations)
    train._transform = JointTransform(transform=build_nnunet_transforms())
    train_loader = DataLoader(train, batch_size=2, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val, batch_size=1, shuffle=False)
    getattr(torch, "_dynamo").config.automatic_dynamic_shapes = True
    trainer = UNetSlidingTrainer(output_folder, train_loader, val_loader, recoverable=False, device=device)
    trainer.num_classes = BENCHMARK_NUM_CLASSES
    trainer.set_frontend(frontend)
    trainer.train(num_epochs, note="MIP Candy Benchmark - full size")


def resize(size: int, input_folder: str | PathLike[str], output_folder: str | PathLike[str], *, num_epochs: int = 100,
           device: Device | None = None, frontend: type[Frontend] = Frontend) -> None:
    if not device:
        device = auto_device()
    if not exists(f"{input_folder}/dataset"):
        download_dataset(f"nnunet_datasets/{BENCHMARK_DATASET}", f"{input_folder}/dataset")
    dataset = NNUNetDataset(f"{input_folder}/dataset", transform=JointTransform(transform=Compose([
        Resized(("image", "label"), (size, size, size)), build_nnunet_transforms()
    ])), device=device)
    train, val = dataset.fold()
    train_loader = DataLoader(train, batch_size=2, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val, batch_size=1, shuffle=False)
    trainer = UNetTrainer(output_folder, train_loader, val_loader, recoverable=False, device=device)
    trainer.num_classes = BENCHMARK_NUM_CLASSES
    trainer.set_frontend(frontend)
    trainer.train(num_epochs, note=f"MIP Candy Benchmark - resize{size}")


def resize128(input_folder: str | PathLike[str], output_folder: str | PathLike[str], *, num_epochs: int = 100,
              device: Device | None = None, frontend: type[Frontend] = Frontend) -> None:
    resize(128, input_folder, output_folder, num_epochs=num_epochs, device=device)


def resize256(input_folder: str | PathLike[str], output_folder: str | PathLike[str], *, num_epochs: int = 100,
              device: Device | None = None, frontend: type[Frontend] = Frontend) -> None:
    resize(256, input_folder, output_folder, num_epochs=num_epochs, device=device)


def roi(input_folder: str | PathLike[str], output_folder: str | PathLike[str], *, num_epochs: int = 100,
        device: Device | None = None, frontend: type[Frontend] = Frontend) -> None:
    if not device:
        device = auto_device()
    if not exists(f"{input_folder}/dataset"):
        download_dataset(f"nnunet_datasets/{BENCHMARK_DATASET}", f"{input_folder}/dataset")
    dataset = NNUNetDataset(f"{input_folder}/dataset")
    annotations = inspect_dataset(dataset, output_folder)
    dataset = ROIDataset(annotations)
    train, val = dataset.fold()
    train_loader = DataLoader(train, batch_size=2, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val, batch_size=1, shuffle=False)
    trainer = UNetTrainer(output_folder, train_loader, val_loader, recoverable=False, device=device)
    trainer.num_classes = BENCHMARK_NUM_CLASSES
    trainer.set_frontend(frontend)
    trainer.train(num_epochs, note=f"MIP Candy Benchmark - roi")


if __name__ == "__main__":
    parser = ArgumentParser(prog="MIP Candy Benchmark", description="MIP Candy Benchmark",
                            epilog="GitHub: https://github.com/ProjectNeura/MIPCandy")
    parser.add_argument("test", choices=("full", "resize128", "resize256", "roi"))
    parser.add_argument("-i", "--input-folder")
    parser.add_argument("-o", "--output-folder")
    parser.add_argument("--num-epochs", type=int, default=100)
    parser.add_argument("--device", default=None)
    parser.add_argument("--front-end", choices=(None, "n", "w"), default=None)
    args = parser.parse_args()
    test = locals()[args.test]
    frontend = {None: Frontend, "n": NotionFrontend, "w": WandBFrontend}[args.front_end]
    test(args.input_folder, args.output_folder, num_epochs=args.num_epochs, device=args.device, frontend=frontend)
