# MIP Candy: 医学图像处理的甜品

![GitHub code size in bytes](https://img.shields.io/github/languages/code-size/ProjectNeura/MIPCandy)
![PyPI](https://img.shields.io/pypi/v/mipcandy)
![GitHub Release](https://img.shields.io/github/v/release/ProjectNeura/MIPCandy)
![GitHub Release Date - Published_At](https://img.shields.io/github/release-date/ProjectNeura/MIPCandy)

MIP Candy 是 Project Neura 开发的下一代医学图像处理基础设施框架。它集成了常用的网络架构及其对应的训练、推理和评估流水线，开箱即用。此外，它还提供了与流行前端仪表板（如 Notion、WandB 和 TensorBoard）的适配器。

🏠 [主页](https://mipcandy.projectneura.org) | 📚 [文档](https://mipcandy-docs.projectneura.org) | 🇺🇸 [English](README.md)

## ✨ 特性

- **🔧 开箱即用的训练流水线**: 为医学图像处理任务提供完整的训练工作流
- **📊 数据处理**: 通过 SimpleITK 支持各种医学图像格式
- **🎯 专用架构**: 为分割和其他医学任务预配置的模型
- **📈 实验跟踪**: 与 Notion、WandB 和 TensorBoard 的集成
- **🔍 综合指标**: 医学影像指标（Dice、IoU、精确度、召回率）
- **🖼️ 可视化工具**: 2D/3D 医学图像可视化
- **⚡ 滑动窗口推理**: 高效处理大型医学体积数据

## 🚀 快速开始

### 安装

**系统要求**: Python >= 3.12

```bash
# 标准安装（包含3D可视化支持）
pip install "mipcandy[standard]"

# 基础安装
pip install mipcandy

# 开发安装
git clone https://github.com/ProjectNeura/MIPCandy.git
cd MIPCandy
pip install -e ".[standard]"
```

### 基本使用

```python
import torch
import mipcandy as mc
from torch.utils.data import DataLoader

# 加载和预处理医学图像
image = mc.load_image("path/to/image.nii.gz")
label = mc.load_image("path/to/label.nii.gz", is_label=True)
print(f"已加载图像: {image.shape}, 数据类型: {image.dtype}")

# 创建内存数据集
images = [torch.randn(1, 64, 64, 64) for _ in range(10)]
dataset = mc.DatasetFromMemory(images)
dataloader = DataLoader(dataset, batch_size=2)

# 设置训练
class MySegmentationTrainer(mc.SegmentationTrainer):
    num_classes = 2  # 设置为类属性
    
    def __init__(self, model, dataloader):
        super().__init__(model, dataloader)

# 创建简单模型
model = torch.nn.Sequential(
    torch.nn.Conv3d(1, 32, 3, padding=1),
    torch.nn.ReLU(),
    torch.nn.Conv3d(32, 2, 1)
)

# 注意：完整训练设置需要更多配置
# trainer = MySegmentationTrainer(model, dataloader)
# trainer.fit(num_epochs=100)

# 测试指标计算
mask = torch.randint(0, 2, (2, 64, 64)).bool()
label_mask = torch.randint(0, 2, (2, 64, 64)).bool()
dice_score = mc.dice_similarity_coefficient_binary(mask, label_mask)
print(f"Dice系数: {dice_score.mean().item():.4f}")
```

## 🏗️ 架构

MIP Candy 组织为几个关键模块：

- **`mipcandy.data`**: 数据加载、预处理和可视化
- **`mipcandy.training`**: 训练循环和实验管理
- **`mipcandy.inference`**: 模型推理和预测工具
- **`mipcandy.evaluation`**: 评估指标和结果分析
- **`mipcandy.frontend`**: 仪表板集成（Notion、WandB）
- **`mipcandy.preset`**: 常见任务的预配置训练器
- **`mipcandy.common`**: 核心模块和优化工具

## 📖 使用示例

### 医学图像分割

```python
import mipcandy as mc
from torch import nn

# 定义分割模型
model = nn.Sequential(
    # 在此处添加您的分割架构
)

# 创建分割训练器
class OrganSegmentationTrainer(mc.SegmentationTrainer):
    num_classes = 3  # 背景 + 2个器官
    
    def __init__(self, model, dataloader):
        super().__init__(model, dataloader)

# 加载数据
dataset = mc.NNUNetDataset("path/to/nnunet/data")
dataloader = mc.DataLoader(dataset, batch_size=2)

# 训练
trainer = OrganSegmentationTrainer(model, dataloader)
trainer.fit(num_epochs=200)
```

### 自定义数据集集成

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

### 前端集成配置

```python
import mipcandy as mc

# 配置WandB集成
secrets = mc.load_secrets()  # 从 secrets.yml 加载
frontend = mc.create_hybrid_frontend(secrets)

# 在训练器中使用前端
class MyTrainer(mc.SegmentationTrainer):
    def __init__(self, model, dataloader):
        super().__init__(model, dataloader)
        self.frontend = frontend
```

### 滑动窗口推理

```python
import mipcandy as mc

# 对大体积医学图像使用滑动窗口
class MyLargeVolumeTrainer(mc.SlidingTrainer):
    def __init__(self, model, dataloader):
        super().__init__(model, dataloader)
        # 滑动窗口会自动处理大体积数据

trainer = MyLargeVolumeTrainer(model, dataloader)
trainer.fit(num_epochs=100)
```

## 🎯 核心组件

### 数据处理
- **数据集类**: `SupervisedDataset`、`UnsupervisedDataset`、`NNUNetDataset`
- **图像I/O**: 支持 NIfTI、DICOM 和其他医学格式
- **预处理**: 重采样、标准化、几何变换
- **可视化**: 2D 切片查看和 3D 体积渲染

### 训练框架
- **训练器**: `Trainer`、`SlidingTrainer`、`SegmentationTrainer`
- **优化**: 自定义损失函数和学习率调度器
- **实验跟踪**: 自动日志记录和可视化

### 推理与评估
- **预测器**: 支持多种输入格式的灵活推理
- **指标**: Dice 系数、IoU、准确度、精确度、召回率
- **评估器**: 全面的模型评估

### 可视化工具
- **2D 可视化**: 医学图像切片的交互式显示
- **3D 可视化**: 体积数据的三维渲染
- **叠加显示**: 图像和标签的叠加可视化

## 🔧 高级配置

### 自定义损失函数

```python
import mipcandy as mc
from torch import nn

class MyCustomTrainer(mc.SegmentationTrainer):
    def build_criterion(self):
        return mc.DiceBCELossWithLogits(self.num_classes)
    
    def build_optimizer(self, params):
        return torch.optim.AdamW(params, lr=1e-3)
    
    def build_scheduler(self, optimizer, num_epochs):
        return mc.AbsoluteLinearLR(optimizer, -8e-6, 1e-2)
```

### 实验管理

```python
# 创建 secrets.yml 文件配置前端集成
# notion:
#   token: "your_notion_token"
#   database_id: "your_database_id"
# wandb:
#   api_key: "your_wandb_key"
#   project: "medical_segmentation"

# 在训练过程中自动记录指标和可视化结果
trainer.fit(num_epochs=100)  # 自动保存预览图和指标
```

## 📊 支持的医学图像格式

- **NIfTI** (.nii, .nii.gz): 神经影像格式
- **DICOM** (.dcm): 医学数字影像和通信标准
- **MetaImage** (.mhd, .mha): ITK 元图像格式
- **NRRD** (.nrrd): 近原始栅格数据
- **其他**: 通过 SimpleITK 支持的格式

## 🛠️ 开发指南

```bash
# 克隆仓库
git clone https://github.com/ProjectNeura/MIPCandy.git
cd MIPCandy

# 开发模式安装
pip install -e ".[standard]"

# 运行测试（如果可用）
python -m pytest

# 构建文档
# （文档构建说明）
```

## 🤝 贡献

我们欢迎贡献！请查看我们的[贡献指南](CONTRIBUTING.md)了解详情。

### 开发规范
- 遵循 PEP 8 代码风格
- 为新功能添加单元测试
- 更新相关文档
- 提交前运行代码检查

## 📄 许可证

本项目基于 Apache License 2.0 许可证 - 详情请见 [LICENSE](LICENSE) 文件。

## 🙏 致谢

MIP Candy 由 [Project Neura](https://projectneura.org) 开发。我们感谢医学影像社区的持续支持和反馈。

## 📞 支持与社区

- **问题报告**: [GitHub Issues](https://github.com/ProjectNeura/MIPCandy/issues)
- **功能请求**: [GitHub Discussions](https://github.com/ProjectNeura/MIPCandy/discussions)
- **文档**: [官方文档](https://mipcandy-docs.projectneura.org)
- **邮箱**: central@projectneura.org

## 🔗 相关资源

- [Project Neura 官网](https://projectneura.org)
- [医学图像处理教程](https://mipcandy-docs.projectneura.org/tutorials)
- [API 参考文档](https://mipcandy-docs.projectneura.org/api)
- [示例代码库](https://github.com/ProjectNeura/MIPCandy-Examples)