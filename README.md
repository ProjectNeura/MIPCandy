from torch import nn

# MIP Candy: A Candy for Medical Image Processing

![GitHub code size in bytes](https://img.shields.io/github/languages/code-size/ProjectNeura/MIPCandy)
![PyPI](https://img.shields.io/pypi/v/mipcandy)
![GitHub Release](https://img.shields.io/github/v/release/ProjectNeura/MIPCandy)
![GitHub Release Date - Published_At](https://img.shields.io/github/release-date/ProjectNeura/MIPCandy)

![poster](home/assets/poster.png)

MIP Candy is Project Neura's next-generation infrastructure framework for medical image processing. It integrates a
handful number of common network architectures with their corresponding training, inference, and evaluation pipelines
that are out-of-the-box ready to use. Additionally, it also provides adapters to popular frontend dashboards such as
Notion, WandB, and TensorBoard.

:link: [Home](https://mipcandy.projectneura.org)

:link: [Docs](https://mipcandy-docs.projectneura.org)

## Key Features

Why MIP Candy? :thinking:

<details>
<summary>Easy adaptation to fit your needs</summary>
We provide tons of easy-to-use techniques for training that seamlessly support your customized experiments.

- Sliding window
- ROI inspection
- ROI cropping to align dataset shape (100% or 33% foreground)
- Automatic padding
- ...

You only need to override one method to create a trainer for your network architecture.

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
<img src="home/assets/cli-ui.png" alt="cmd-ui"/>
</details>

<details>
<summary>Built-in 2D and 3D visualization for intuitive understanding</summary>
<img src="home/assets/visualization.png" alt="visualization"/>
</details>

<details>
<summary>High availability with interruption tolerance</summary>
Interrupted experiments can be resumed with ease.
<img src="home/assets/recovery.png" alt="recovery"/>
</details>

<details>
<summary>Support of various frontend platforms for remote monitoring</summary>

MIP Candy Supports [Notion](https://mipcandy-projectneura.notion.site), WandB, and TensorBoard.

<img src="home/assets/notion.png" alt="notion"/>
</details>

## Installation

Note that MIP Candy requires **Python >= 3.12**.

```shell
pip install "mipcandy[standard]"
```