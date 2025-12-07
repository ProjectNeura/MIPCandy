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
We provide tons of easy-to-use techniques for training.

- Sliding window
- ROI inspection
- ROI cropping to align dataset shape (100% or 33% foreground)
- Automatic padding
- ...
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
<summary>Continue training after interruption</summary>
<img src="home/assets/recovery.png" alt="recovery"/>
</details>

<details>
<summary>Support various frontend platforms for remote monitoring</summary>
MIP Candy Supports Notion, WandB, and TensorBoard.
<iframe src="https://mipcandy-projectneura.notion.site" width="100%" height="600" frameborder="0" allowfullscreen></iframe>
</details>

## Installation

Note that MIP Candy requires **Python >= 3.12**.

```shell
pip install "mipcandy[standard]"
```