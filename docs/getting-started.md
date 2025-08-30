# Getting Started with MIPCandy

MIPCandy is a PyTorch-based framework for medical image processing. This guide will get you up and running quickly.

## Installation

### Requirements
- Python >= 3.12
- PyTorch >= 2.0

### Install MIPCandy

```bash
# Standard installation with 3D visualization
pip install "mipcandy[standard]"

# Basic installation
pip install mipcandy

# Development installation
git clone https://github.com/ProjectNeura/MIPCandy.git
cd MIPCandy
pip install -e ".[standard]"
```

## Core Concepts

### Medical Image Processing
MIPCandy is designed specifically for medical imaging tasks:
- **Medical Image Formats**: NIfTI, DICOM, MetaImage, NRRD via SimpleITK
- **2D/3D Processing**: Handle both slice-based and volumetric data
- **Specialized Metrics**: Dice coefficient, IoU, medical-specific accuracy measures

### Framework Components
- **Data Module**: Loading, preprocessing, visualization
- **Training Module**: Complete training pipelines with experiment tracking
- **Inference Module**: Model deployment and prediction
- **Evaluation Module**: Comprehensive evaluation metrics
- **Frontend Module**: Integration with Notion, WandB, TensorBoard

## Quick Example

```python
import torch
import mipcandy as mc
from torch.utils.data import DataLoader

# Load medical image
image = mc.load_image("path/to/scan.nii.gz")
print(f"Loaded: {image.shape}, {image.dtype}")

# Create dataset
images = [torch.randn(1, 64, 64, 64) for _ in range(10)]
dataset = mc.DatasetFromMemory(images)
dataloader = DataLoader(dataset, batch_size=2)

# Calculate medical metrics
mask = torch.randint(0, 2, (64, 64, 64)).bool()
label = torch.randint(0, 2, (64, 64, 64)).bool()
dice = mc.dice_similarity_coefficient_binary(mask, label)
print(f"Dice coefficient: {dice.item():.4f}")
```

## Next Steps

- Read the [Tutorial](tutorial.md) for complete workflows
- Check the [API Reference](api.md) for detailed documentation
- Explore examples in the GitHub repository

## Common Issues

### Memory Issues
For large medical volumes, use sliding window processing:
```python
# Use SlidingTrainer for large volumes
trainer = mc.SlidingTrainer(model, dataloader)
```

### Path Issues on Windows
Always use forward slashes or raw strings:
```python
# Good
image = mc.load_image("D:/data/scan.nii.gz")
image = mc.load_image(r"D:\data\scan.nii.gz")
```

## Support

- **GitHub Issues**: [Report bugs](https://github.com/ProjectNeura/MIPCandy/issues)
- **Discussions**: [Ask questions](https://github.com/ProjectNeura/MIPCandy/discussions)(Currently NOT Available)