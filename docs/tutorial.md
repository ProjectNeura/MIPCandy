# MIPCandy Tutorial

This tutorial covers common medical image processing workflows using MIPCandy.

## Medical Image Segmentation

### 1. Prepare Your Data

```python
import torch
import mipcandy as mc
from torch.utils.data import DataLoader

# Load medical images
image = mc.load_image("data/patient001_scan.nii.gz")
label = mc.load_image("data/patient001_label.nii.gz", is_label=True)

print(f"Image shape: {image.shape}")  # e.g., (1, 512, 512, 256)
print(f"Label shape: {label.shape}")  # matching shape
```

### 2. Create Custom Dataset

```python
class MedicalSegmentationDataset(mc.SupervisedDataset):
    def __init__(self, data_dir, image_files, label_files):
        super().__init__()
        self.data_dir = data_dir
        self.image_files = image_files
        self.label_files = label_files
        
    def __len__(self):
        return len(self.image_files)
    
    def load(self, idx):
        image_path = f"{self.data_dir}/{self.image_files[idx]}"
        label_path = f"{self.data_dir}/{self.label_files[idx]}"
        
        image = self.do_load(image_path)
        label = self.do_load(label_path, is_label=True)
        
        return image, label

# Create dataset
image_files = ["scan001.nii.gz", "scan002.nii.gz", "scan003.nii.gz"]
label_files = ["label001.nii.gz", "label002.nii.gz", "label003.nii.gz"]
dataset = MedicalSegmentationDataset("data/", image_files, label_files)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
```

### 3. Define Segmentation Model

```python
import torch.nn as nn

class Simple3DUNet(nn.Module):
    def __init__(self, in_channels=1, num_classes=2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.decoder = nn.Sequential(
            nn.Conv3d(64, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, num_classes, 1)
        )
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

model = Simple3DUNet(in_channels=1, num_classes=2)
```

### 4. Set Up Training

```python
class OrganSegmentationTrainer(mc.SegmentationTrainer):
    num_classes = 2  # background + organ
    
    def __init__(self, model, dataloader):
        super().__init__(model, dataloader)

# Create trainer
trainer = OrganSegmentationTrainer(model, dataloader)

# Start training
trainer.fit(num_epochs=50)
```

### 5. Evaluate Results

```python
# Load test data
test_image = mc.load_image("test/patient_test.nii.gz")
test_label = mc.load_image("test/patient_test_label.nii.gz", is_label=True)

# Make prediction
model.eval()
with torch.no_grad():
    prediction = model(test_image.unsqueeze(0))
    prediction = torch.sigmoid(prediction).squeeze(0)

# Calculate metrics
pred_binary = (prediction > 0.5).bool()
label_binary = test_label.bool()

dice = mc.dice_similarity_coefficient_binary(pred_binary, label_binary)
iou = mc.iou_binary(pred_binary, label_binary)
accuracy = mc.accuracy_binary(pred_binary, label_binary)

print(f"Dice coefficient: {dice.item():.4f}")
print(f"IoU: {iou.item():.4f}")
print(f"Accuracy: {accuracy.item():.4f}")
```

## Working with Large Volumes

For large medical images that don't fit in memory:

```python
class LargeVolumeTrainer(mc.SlidingTrainer):
    def __init__(self, model, dataloader):
        super().__init__(model, dataloader)
        # Sliding window parameters are handled automatically

# Use for large 3D volumes
large_trainer = LargeVolumeTrainer(model, dataloader)
large_trainer.fit(num_epochs=100)
```

## Visualization

```python
# Visualize 2D slice
slice_2d = image[0, :, :, 128]  # Get middle slice
mc.visualize2d(slice_2d, title="CT Scan Slice")

# Visualize 3D volume
mc.visualize3d(image, title="3D Volume")

# Overlay prediction on original image
overlay = mc.overlay(test_image, prediction > 0.5)
mc.visualize2d(overlay[0, :, :, 128], title="Prediction Overlay")
```

## Frontend Integration

### Setting up WandB Integration

1. Create `secrets.yml`:
```yaml
wandb:
  api_key: "your_wandb_api_key"
  project: "medical_segmentation"
  entity: "your_username"
```

2. Use in training:
```python
# Load secrets and create frontend
secrets = mc.load_secrets()
frontend = mc.create_hybrid_frontend(secrets)

# Training with logging
class LoggingTrainer(mc.SegmentationTrainer):
    def __init__(self, model, dataloader):
        super().__init__(model, dataloader)
        self.frontend = frontend

trainer = LoggingTrainer(model, dataloader)
trainer.fit(num_epochs=100)  # Automatically logs metrics
```

## Data Preprocessing Best Practices

### Resampling to Isotropic

```python
# Resample to 1mm isotropic voxels
resampled_image = mc.resample_to_isotropic(image, target_spacing=(1.0, 1.0, 1.0))
```

### Normalization

```python
# Z-score normalization
normalized = (image - image.mean()) / image.std()

# Min-max normalization
normalized = (image - image.min()) / (image.max() - image.min())
```

### Data Augmentation

```python
from torchvision import transforms

transform = transforms.Compose([
    # Add your medical-specific transforms here
    transforms.RandomRotation(degrees=10),
    transforms.RandomHorizontalFlip(p=0.5)
])

class AugmentedDataset(mc.SupervisedDataset):
    def __init__(self, base_dataset, transform=None):
        super().__init__()
        self.base_dataset = base_dataset
        self.transform = transform
    
    def __len__(self):
        return len(self.base_dataset)
    
    def load(self, idx):
        image, label = self.base_dataset.load(idx)
        
        if self.transform:
            # Apply same transform to both image and label
            seed = torch.randint(0, 2**32, (1,)).item()
            torch.manual_seed(seed)
            image = self.transform(image)
            torch.manual_seed(seed)
            label = self.transform(label)
            
        return image, label
```

## Common Patterns

### Multi-class Segmentation

```python
class MultiClassSegmentationTrainer(mc.SegmentationTrainer):
    num_classes = 4  # background + 3 organs
    
    def build_criterion(self):
        return mc.DiceBCELossWithLogits(self.num_classes)

# Use multiclass metrics
pred_multiclass = torch.argmax(prediction, dim=1)
label_multiclass = label.long()
dice_multiclass = mc.dice_similarity_coefficient_multiclass(
    pred_multiclass, label_multiclass
)
```

### Custom Loss Functions

```python
class CustomTrainer(mc.SegmentationTrainer):
    def build_criterion(self):
        return nn.CrossEntropyLoss()
    
    def build_optimizer(self, params):
        return torch.optim.AdamW(params, lr=1e-3, weight_decay=1e-4)
    
    def build_scheduler(self, optimizer, num_epochs):
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)
```

## Next Steps

- Explore the [API Reference](api.md) for detailed function documentation
- Check the GitHub repository for more examples
- Join our community discussions for help and tips