from argparse import ArgumentParser
from os import PathLike
from os.path import exists

from mipcandy_bundles.unet import UNetTrainer
from torch.utils.data import DataLoader

from mipcandy import Device, auto_device, download_dataset, NNUNetDataset


def main(input_folder: str | PathLike[str], output_folder: str | PathLike[str], *, num_epochs: int = 100,
         device: Device | None = None) -> None:
    if not device:
        device = auto_device()
    if not exists(f"{input_folder}/dataset"):
        download_dataset("nnunet_datasets/AbdomenCT-1K-ss1", f"{input_folder}/dataset")
    dataset = NNUNetDataset(f"{input_folder}/dataset")
    train, val = dataset.fold()
    train_loader = DataLoader(train, batch_size=2, shuffle=True)
    val_loader = DataLoader(val, batch_size=1, shuffle=False)
    trainer = UNetTrainer(output_folder, train_loader, val_loader, recoverable=False, device=device)
    trainer.train(num_epochs, note="MIP Candy Benchmark")


if __name__ == "__main__":
    parser = ArgumentParser(prog="MIP Candy Benchmark", description="MIP Candy Benchmark",
                            epilog="GitHub: https://github.com/ProjectNeura/MIPCandy")
    parser.add_argument("-i", "--input-folder")
    parser.add_argument("-o", "--output-folder")
    parser.add_argument("--num-epochs", type=int, default=100)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    main(args.input_folder, args.output_folder, num_epochs=args.num_epochs, device=args.device)
