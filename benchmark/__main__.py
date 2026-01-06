from argparse import ArgumentParser
from os.path import exists

from benchmark.data import DataTest, SlidingWindowTest
from benchmark.training import TrainingTest, SlidingTrainingTest
from mipcandy import auto_device, download_dataset, Frontend, NotionFrontend, WandBFrontend

BENCHMARK_DATASET: str = "AbdomenCT-1K-ss1"

if __name__ == "__main__":
    tests = {
        "SlidingWindow": SlidingWindowTest,
        "Training": TrainingTest,
        "SlidingTraining": SlidingTrainingTest
    }
    parser = ArgumentParser(prog="MIP Candy Benchmark", description="MIP Candy Benchmark",
                            epilog="GitHub: https://github.com/ProjectNeura/MIPCandy")
    parser.add_argument("test", choices=tests.keys())
    parser.add_argument("-i", "--input-folder")
    parser.add_argument("-o", "--output-folder")
    parser.add_argument("--num-epochs", type=int, default=100)
    parser.add_argument("--device", default=None)
    parser.add_argument("--front-end", choices=(None, "n", "w"), default=None)
    args = parser.parse_args()
    DataTest.dataset = BENCHMARK_DATASET
    test = tests[args.test](
        args.input_folder, args.output_folder, args.num_epochs, args.device if args.device else auto_device(), {
            None: Frontend, "n": NotionFrontend, "w": WandBFrontend
        }[args.front_end]
    )
    if not exists(f"{args.input_folder}/{BENCHMARK_DATASET}"):
        download_dataset(f"nnunet_datasets/{BENCHMARK_DATASET}", f"{args.input_folder}/{BENCHMARK_DATASET}")
    stat, err = test.run()
    if not stat:
        raise err
