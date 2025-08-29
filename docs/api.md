# MIPCandy API Reference

Complete API documentation for all MIPCandy modules and functions.

## Data Module (`mipcandy.data`)

### Dataset Classes

#### `SupervisedDataset`
Abstract base class for supervised learning datasets.

```python
class SupervisedDataset(mipcandy.data.SupervisedDataset):
    def load(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Load image and label pair at given index."""
        pass
    
    def __len__(self) -> int:
        """Return dataset size."""
        pass
```

#### `UnsupervisedDataset`
Abstract base class for unsupervised learning datasets.

```python
class UnsupervisedDataset(mipcandy.data.UnsupervisedDataset):
    def load(self, idx: int) -> torch.Tensor:
        """Load image at given index."""
        pass
```

#### `DatasetFromMemory`
Dataset that stores data in memory.

```python
DatasetFromMemory(
    images: Sequence[torch.Tensor],
    device: torch.device | str = "cpu"
)
```

- **images**: List of tensor images to store in memory
- **device**: Target device for tensors

#### `NNUNetDataset`
Dataset compatible with nnU-Net data format.

```python
NNUNetDataset(data_path: str | PathLike[str])
```

### Data I/O Functions

#### `load_image()`
Load medical images from file.

```python
load_image(
    path: str | PathLike[str],
    *,
    is_label: bool = False,
    device: torch.device | str = "cpu",
    **kwargs
) -> torch.Tensor
```

- **path**: Path to medical image file
- **is_label**: Whether image is a segmentation label
- **device**: Target device for tensor
- **Returns**: Loaded image as tensor

#### `save_image()`
Save tensor as medical image.

```python
save_image(
    tensor: torch.Tensor,
    path: str | PathLike[str],
    **kwargs
)
```

### Geometric Operations

#### `ensure_num_dimensions()`
Ensure tensor has specific number of dimensions.

```python
ensure_num_dimensions(
    tensor: torch.Tensor,
    ndim: int
) -> torch.Tensor
```

#### `orthographic_views()`
Generate orthographic projections of 3D volume.

```python
orthographic_views(volume: torch.Tensor) -> tuple[torch.Tensor, ...]
```

#### `resample_to_isotropic()`
Resample image to isotropic spacing.

```python
resample_to_isotropic(
    image: torch.Tensor,
    target_spacing: tuple[float, float, float] = (1.0, 1.0, 1.0)
) -> torch.Tensor
```

### Visualization

#### `visualize2d()`
Display 2D image or slice.

```python
visualize2d(
    image: torch.Tensor,
    *,
    title: str = "",
    blocking: bool = False,
    screenshot_as: str | None = None
)
```

#### `visualize3d()`
Display 3D volume interactively.

```python
visualize3d(
    volume: torch.Tensor,
    *,
    title: str = "",
    blocking: bool = False,
    screenshot_as: str | None = None
)
```

#### `overlay()`
Create overlay of image and mask.

```python
overlay(
    image: torch.Tensor,
    mask: torch.Tensor
) -> torch.Tensor
```

## Training Module (`mipcandy.training`)

### Trainer Classes

#### `Trainer`
Base class for training neural networks.

```python
class CustomTrainer(mipcandy.training.Trainer):
    def build_criterion(self) -> nn.Module:
        """Build loss function."""
        pass
    
    def build_optimizer(self, params) -> optim.Optimizer:
        """Build optimizer."""
        pass
    
    def build_scheduler(self, optimizer, num_epochs) -> optim.lr_scheduler.LRScheduler:
        """Build learning rate scheduler."""
        pass
    
    def backward(self, images, labels, toolbox) -> tuple[float, dict]:
        """Perform backward pass."""
        pass
```

#### `SlidingTrainer`
Trainer for sliding window inference on large volumes.

```python
SlidingTrainer(model: nn.Module, dataloader: DataLoader)
```

#### `SegmentationTrainer`
Pre-configured trainer for segmentation tasks.

```python
class MySegmentationTrainer(mipcandy.preset.SegmentationTrainer):
    num_classes: int = 1  # Set number of classes
```

### Training Components

#### `TrainerToolbox`
Container for training components.

```python
TrainerToolbox(
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler.LRScheduler,
    criterion: nn.Module
)
```

## Metrics Module (`mipcandy.metrics`)

### Binary Metrics

#### `dice_similarity_coefficient_binary()`
Calculate Dice coefficient for binary masks.

```python
dice_similarity_coefficient_binary(
    mask: torch.Tensor,
    label: torch.Tensor,
    *,
    if_empty: float = 1.0
) -> torch.Tensor
```

- **mask**: Predicted binary mask (bool tensor)
- **label**: Ground truth binary mask (bool tensor)
- **if_empty**: Value to return if both masks are empty
- **Returns**: Dice coefficient(s)

#### `iou_binary()`
Calculate Intersection over Union for binary masks.

```python
iou_binary(
    mask: torch.Tensor,
    label: torch.Tensor,
    *,
    if_empty: float = 1.0
) -> torch.Tensor
```

#### `accuracy_binary()`
Calculate accuracy for binary predictions.

```python
accuracy_binary(
    mask: torch.Tensor,
    label: torch.Tensor
) -> torch.Tensor
```

#### `precision_binary()` / `recall_binary()`
Calculate precision and recall for binary predictions.

```python
precision_binary(mask: torch.Tensor, label: torch.Tensor) -> torch.Tensor
recall_binary(mask: torch.Tensor, label: torch.Tensor) -> torch.Tensor
```

### Multi-class Metrics

#### `dice_similarity_coefficient_multiclass()`
Calculate Dice coefficient for multi-class segmentation.

```python
dice_similarity_coefficient_multiclass(
    pred: torch.Tensor,
    label: torch.Tensor,
    num_classes: int
) -> torch.Tensor
```

#### Other Multi-class Functions
- `iou_multiclass()`
- `accuracy_multiclass()`
- `precision_multiclass()`
- `recall_multiclass()`

### Utility Functions

#### `do_reduction()`
Apply reduction to tensor.

```python
do_reduction(
    x: torch.Tensor,
    method: Literal["mean", "median", "sum", "none"] = "mean"
) -> torch.Tensor
```

## Frontend Module (`mipcandy.frontend`)

### Frontend Integration

#### `load_secrets()`
Load configuration from secrets file.

```python
load_secrets(
    *,
    path: str | PathLike[str] = "secrets.yml"
) -> dict
```

#### `create_hybrid_frontend()`
Create frontend integrations from secrets.

```python
create_hybrid_frontend(secrets: dict) -> Frontend
```

#### `NotionFrontend`
Integration with Notion databases.

```python
NotionFrontend(secrets: dict)
```

## Common Module (`mipcandy.common`)

### Loss Functions

#### `DiceBCELossWithLogits`
Combined Dice and Binary Cross Entropy loss.

```python
DiceBCELossWithLogits(num_classes: int = 1)
```

### Learning Rate Schedulers

#### `AbsoluteLinearLR`
Linear learning rate scheduler with absolute step size.

```python
AbsoluteLinearLR(
    optimizer: optim.Optimizer,
    step_size: float,
    max_lr: float
)
```

### Padding Modules

#### `Pad2d` / `Pad3d`
Padding modules for 2D and 3D tensors.

```python
Pad2d(padding: int | tuple)
Pad3d(padding: int | tuple)
```

## Layer Module (`mipcandy.layer`)

### Utility Functions

#### `batch_int_multiply()` / `batch_int_divide()`
Batch operations for integer tensors.

```python
batch_int_multiply(tensor: torch.Tensor, factor: int) -> torch.Tensor
batch_int_divide(tensor: torch.Tensor, divisor: int) -> torch.Tensor
```

### Type Definitions

#### `LayerT`
Type alias for neural network layers.

#### `HasDevice`
Mixin class for device management.

#### `WithPaddingModule`
Base class for modules with padding.

## Evaluation Module (`mipcandy.evaluation`)

### Evaluator

#### `Evaluator`
Base class for model evaluation.

```python
evaluator = Evaluator(model, dataset)
results = evaluator.evaluate()
```

### Evaluation Types

#### `EvalCase`
Single evaluation case container.

#### `EvalResult`
Evaluation results container.

## Utilities

### `sanity_check()`
Perform model sanity checks.

```python
sanity_check(
    model: nn.Module,
    input_shape: tuple[int, ...],
    device: torch.device = torch.device("cpu")
)
```

### `num_trainable_params()`
Count trainable parameters in model.

```python
num_trainable_params(model: nn.Module) -> int
```

## Type Definitions (`mipcandy.types`)

- **`Secret`**: Type for configuration secrets
- **`Secrets`**: Dictionary of secrets
- **`Params`**: Type for model parameters  
- **`Transform`**: Type for data transforms
- **`SupportedPredictant`**: Supported prediction input types
- **`Colormap`**: Type for visualization colormaps

---

*For usage examples, see the [Tutorial](tutorial.md) and [Getting Started](getting-started.md) guides.*