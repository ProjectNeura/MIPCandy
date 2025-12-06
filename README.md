from torch import nn

# MIP Candy: A Candy for Medical Image Processing

![GitHub code size in bytes](https://img.shields.io/github/languages/code-size/ProjectNeura/MIPCandy)
![PyPI](https://img.shields.io/pypi/v/mipcandy)
![GitHub Release](https://img.shields.io/github/v/release/ProjectNeura/MIPCandy)
![GitHub Release Date - Published_At](https://img.shields.io/github/release-date/ProjectNeura/MIPCandy)

MIP Candy is Project Neura's next-generation infrastructure framework for medical image processing. It integrates a
handful number of common network architectures with their corresponding training, inference, and evaluation pipelines
that are out-of-the-box ready to use. Additionally, it also provides adapters to popular frontend dashboards such as
Notion, WandB, and TensorBoard.

:link: [Home](https://mipcandy.projectneura.org)

:link: [Docs](https://mipcandy-docs.projectneura.org)

## Key Features

<details>
<summary>Easily adapt to your network architecture</summary>
All it takes to train, infer with, or evaluate your PyTorch model is overriding one method.

```python
from typing import override

from torch import nn
from mipcandy import SegmentationTrainer


class MyTrainer(SegmentationTrainer):
    @override
    def build_network(self, example_shape: tuple[int, ...]) -> nn.Module:
        ...
```

</details>

<details>
<summary>Satisfying command-line UI design</summary>
<img src="home/assets/cmd-ui.png" alt="cmd-ui"/>
</details>

<details>
<summary>Built-in 2D and 3D visualization for intuitive understanding</summary>
</details>

<details>
<summary>Continue training after interruption</summary>
</details>

## Installation

Note that MIP Candy requires **Python >= 3.12**.

```shell
pip install "mipcandy[standard]"
```