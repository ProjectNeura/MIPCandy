# MIP Candy: A Candy for Medical Image Processing

![GitHub code size in bytes](https://img.shields.io/github/languages/code-size/ProjectNeura/MIPCandy)
![PyPI](https://img.shields.io/pypi/v/mipcandy)
![GitHub Release](https://img.shields.io/github/v/release/ProjectNeura/MIPCandy)
![GitHub Release Date - Published_At](https://img.shields.io/github/release-date/ProjectNeura/MIPCandy)

MIP Candy is Project Neura's next-generation infrastructure framework for medical image processing. It integrates common network architectures with their corresponding training, inference, and evaluation pipelines that are out-of-the-box ready to use. Additionally, it provides adapters to popular frontend dashboards such as Notion, WandB, and TensorBoard.

🏠 [Home](https://mipcandy.projectneura.org) | 📚 [Documentation](https://mipcandy-docs.projectneura.org)

## ✨ Features

- **🔧 Out-of-the-box Training Pipeline**: Complete training workflows for medical image processing tasks
- **📊 Data Handling**: Support for various medical image formats via SimpleITK
- **🎯 Specialized Architectures**: Pre-configured models for segmentation and other medical tasks  
- **📈 Experiment Tracking**: Integration with Notion, WandB, and TensorBoard
- **🔍 Comprehensive Metrics**: Medical imaging metrics (Dice, IoU, precision, recall)
- **🖼️ Visualization Tools**: 2D/3D medical image visualization
- **⚡ Sliding Window Inference**: Efficient processing of large medical volumes

## 🚀 Quick Start

### Installation

**Requirements**: Python >= 3.12

```bash
# Standard installation with 3D visualization support
pip install "mipcandy[standard]"

# Basic installation
pip install mipcandy

# Development installation
git clone https://github.com/ProjectNeura/MIPCandy.git
cd MIPCandy
pip install -e ".[standard]"
```

### Basic Usage

```python
import torch
import mipcandy as mc
from torch.utils.data import DataLoader

# Load and preprocess medical images
image = mc.load_image("path/to/image.nii.gz")
label = mc.load_image("path/to/label.nii.gz", is_label=True)
print(f"Loaded image: {image.shape}, dtype: {image.dtype}")

# Create a dataset from memory
images = [torch.randn(1, 64, 64, 64) for _ in range(10)]
dataset = mc.DatasetFromMemory(images)
dataloader = DataLoader(dataset, batch_size=2)

# Set up training
class MySegmentationTrainer(mc.SegmentationTrainer):
    num_classes = 2  # Set as class attribute
    
    def __init__(self, model, dataloader):
        super().__init__(model, dataloader)

# Create a simple model and train
model = torch.nn.Sequential(
    torch.nn.Conv3d(1, 32, 3, padding=1),
    torch.nn.ReLU(),
    torch.nn.Conv3d(32, 2, 1)
)

# Note: Full training setup requires more configuration
# trainer = MySegmentationTrainer(model, dataloader)
# trainer.fit(num_epochs=100)

# Test metrics with binary masks
mask = torch.randint(0, 2, (2, 64, 64)).bool()
label_mask = torch.randint(0, 2, (2, 64, 64)).bool()
dice_score = mc.dice_similarity_coefficient_binary(mask, label_mask)
print(f"Dice score: {dice_score.mean().item():.4f}")
```

## 🏗️ Architecture

MIP Candy is organized into several key modules:

- **`mipcandy.data`**: Data loading, preprocessing, and visualization
- **`mipcandy.training`**: Training loops and experiment management  
- **`mipcandy.inference`**: Model inference and prediction utilities
- **`mipcandy.evaluation`**: Evaluation metrics and result analysis
- **`mipcandy.frontend`**: Dashboard integrations (Notion, WandB)
- **`mipcandy.preset`**: Pre-configured trainers for common tasks
- **`mipcandy.common`**: Core modules and optimization utilities

## 📖 Examples

### Medical Image Segmentation

```python
import mipcandy as mc
from torch import nn

# Define your segmentation model
model = nn.Sequential(
    # Your segmentation architecture here
)

# Create segmentation trainer
class OrganSegmentationTrainer(mc.SegmentationTrainer):
    num_classes = 3  # background + 2 organs
    
    def __init__(self, model, dataloader):
        super().__init__(model, dataloader)

# Load your data
dataset = mc.NNUNetDataset("path/to/nnunet/data")
dataloader = mc.DataLoader(dataset, batch_size=2)

# Train
trainer = OrganSegmentationTrainer(model, dataloader)
trainer.fit(num_epochs=200)
```

### Custom Dataset Integration

```python
import mipcandy as mc

class MyMedicalDataset(mc.SupervisedDataset):
    def __init__(self, data_dir, transform=None):
        super().__init__()
        self.data_dir = data_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def load(self, idx):
        image = self.do_load(self.image_paths[idx])
        label = self.do_load(self.label_paths[idx], is_label=True)
        
        if self.transform:
            image, label = self.transform(image, label)
            
        return image, label
```

## 🎯 Key Components

### Data Handling
- **Dataset Classes**: `SupervisedDataset`, `UnsupervisedDataset`, `NNUNetDataset`
- **Image I/O**: Support for NIfTI, DICOM, and other medical formats
- **Preprocessing**: Resampling, normalization, geometric transforms
- **Visualization**: 2D slice viewing and 3D volume rendering

### Training Framework  
- **Trainers**: `Trainer`, `SlidingTrainer`, `SegmentationTrainer`
- **Optimization**: Custom loss functions and learning rate schedulers
- **Experiment Tracking**: Automatic logging and visualization

### Inference & Evaluation
- **Predictor**: Flexible inference with various input formats
- **Metrics**: Dice coefficient, IoU, accuracy, precision, recall
- **Evaluator**: Comprehensive model evaluation

## 🛠️ Development

```bash
# Clone the repository
git clone https://github.com/ProjectNeura/MIPCandy.git
cd MIPCandy

# Install in development mode
pip install -e ".[standard]"

# Run tests (if available)
python -m pytest

# Build documentation
# (Documentation build instructions)
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

MIP Candy is developed by [Project Neura](https://projectneura.org). We thank the medical imaging community for their continued support and feedback.