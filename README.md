# VLM-VLA Agent：多模态变化检测与动作预测系统

> 基于 CLIP 和 Qwen 的遥感影像变化检测与动作预测系统

[快速开始](./快速开始.md) | [项目指南](./项目指南.md)

---

## 📋 项目概述

VLM-VLA Agent 是一个先进的多模态 AI 系统，结合了视觉语言模型（VLM）和视觉语言动作（VLA）架构，用于：

- **变化检测**：使用 CLIP 视觉编码器检测卫星影像中的时序变化
- **语义理解**：使用 Qwen2.5 LLM 理解变化描述
- **动作预测**：预测变化区域的边界框和像素级坐标
- **高效训练**：使用 LoRA 微调和 4 位量化优化

### 核心特性

- ✅ **完全离线**：所有模型本地加载，无需联网
- ✅ **自适应环境**：自动识别 Kaggle 或本地环境
- ✅ **高效微调**：LoRA + 4位量化
- ✅ **灵活数据处理**：支持 Arrow/原始图像目录/JSON标注自动检测
- ✅ **生产级代码**：完整的错误处理、日志记录和训练容错
- ✅ **一键部署**：Kaggle 部署指南
- ✅ **多分割支持**：自动检测 train/val/test 数据集分割

---

## 🏗️ 系统架构

```
输入 (时序影像 + 文本描述)
    ↓
[CLIP 视觉编码器] → 视觉特征 (冻结)
    ↓
[投影层] → LLM 嵌入空间
    ↓
[Qwen2.5 LLM + LoRA] → 语言理解
    ↓
[特征融合] → 组合表示
    ↓
[动作头 MLP] → 预测动作向量 [cx, cy, scale]
```

### 关键组件

| 组件 | 详情 | 状态 |
|------|------|------|
| **视觉编码器** | CLIP ViT-B/32 (冻结) | ✅ |
| **语言模型** | Qwen2.5-0.5B (4位量化) | ✅ |
| **微调方法** | LoRA (rank=8) | ✅ |
| **动作输出** | 3D向量: [中心x, 中心y, 尺度] | ✅ |
| **数据集** | LEVIR-CC (遥感变化检测) | ✅ |

---

## 📁 项目结构

```
VLM_Agent_Project/
├── src/                        # 核心源代码
│   ├── __init__.py             # 包初始化
│   ├── config.py               # 配置管理（路径映射、超参数）
│   ├── dataset.py              # Arrow 格式数据加载器
│   ├── model.py                # VLM-VLA 模型架构
│   ├── train.py                # 训练循环与验证
│   └── utils.py                # 工具函数库
├── requirements.txt            # Python 依赖（带注释说明）
├── README.md                   # 本文件（项目概览）
├── 快速开始.md                 # 快速部署指南
├── 项目指南.md                 # 详细技术文档
├── KAGGLE部署清单.md           # Kaggle 完整部署清单 ⭐
├── check_setup.py              # 项目验证脚本
├── kaggle_launch.py            # Kaggle 启动脚本
└── kaggle_quick_start.py       # Kaggle 一键启动脚本 ⭐
```

**文档说明**：
- 📖 **README.md** - 快速了解项目（你现在所在的文档）
- 🚀 **快速开始.md** - 详细的本地/Kaggle部署步骤
- 📚 **项目指南.md** - 深入的技术细节和架构说明
- ⭐ **KAGGLE部署清单.md** - Kaggle专用完整部署指南（推荐）

---

## 🚀 快速开始

### 环境要求

**本地环境：**
- Python 3.8+
- CUDA 11.8+ (可选，用于 GPU 加速)
- 16+ GB RAM
- 20+ GB 磁盘空间

**Kaggle 环境：**
- GPU T4 x2 (推荐)
- 自动提供 13GB RAM 和 100+ GB 磁盘

### 安装步骤

```bash
# 1. 克隆仓库
git clone https://github.com/Jackzhuang123/VLM_Agent_Project.git
cd VLM_Agent_Project

# 2. 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. 安装依赖
pip install -r requirements.txt

# 4. 验证项目
python check_setup.py

# 5. 开始训练（本地）
python -m src.train
```

**依赖包说明**：
- `torch>=2.1.0` - PyTorch 深度学习框架
- `transformers>=4.37.0` - HuggingFace Transformers（CLIP + Qwen）
- `datasets>=2.14.0` - HuggingFace Datasets（数据加载）
- `peft>=0.7.0` - Parameter-Efficient Fine-Tuning（LoRA）
- `bitsandbytes>=0.41.0` - 4位量化支持
- `accelerate>=0.25.0` - 分布式训练加速
- `einops>=0.7.0` - 张量操作工具
- `scipy>=1.11.0` - 科学计算库
- `Pillow>=10.0.0` - 图像处理
- `numpy>=1.24.0` - 数值计算
- `tqdm` - 进度条显示

### Kaggle 部署（推荐）

1. **创建 Kaggle Notebook**
2. **添加数据集**（在右侧面板 → Input）：
   - ✅ `levir-cc-dateset`
   - ✅ `clip-vit-b32`
   - ✅ `qwen2.5-0.5b`
3. **设置 GPU**：选择 T4 x2
4. **运行以下 Cells**：

```python
# Cell 1: 验证路径
import os
print("=" * 60)
print("📦 检查 Kaggle Input 数据集")
print("=" * 60)
for item in os.listdir("/kaggle/input"):
    print(f"📁 {item}/")
print("=" * 60)

# Cell 2: 克隆仓库并安装依赖
!git clone https://github.com/Jackzhuang123/VLM_Agent_Project.git
%cd VLM_Agent_Project

# 安装所有必需的依赖包
!pip install -q transformers>=4.37.0 \
    datasets>=2.14.0 \
    peft>=0.7.0 \
    bitsandbytes>=0.41.0 \
    accelerate>=0.25.0 \
    pyarrow>=14.0.0 \
    einops>=0.7.0 \
    scipy>=1.11.0 \
    tqdm

# Cell 3: 验证项目设置
!python check_setup.py

# Cell 4: 开始训练
!python -m src.train
```

**重要提示**：
- ⚠️ 确保 GitHub 仓库 URL 正确：`https://github.com/Jackzhuang123/VLM_Agent_Project.git`
- ⚠️ Kaggle 已预装 `torch` 和 `torchvision`，无需重新安装
- ⚠️ `Pillow` 和 `numpy` 也已预装
- ✅ 需安装的依赖包：`transformers`, `datasets`, `peft`, `bitsandbytes`, `accelerate`, `pyarrow`, `einops`, `scipy`, `tqdm`
- 📌 `pyarrow` 用于读取 Arrow 格式数据集文件（如 `levir-cc-train.arrow`）

详细步骤请参考 [快速开始.md](./快速开始.md)

---

## ⚙️ 配置说明

### 关键超参数

编辑 `src/config.py` 进行自定义：

```python
# 训练设置
MAX_EPOCHS = 10              # 训练轮数
BATCH_SIZE = 4               # 批次大小（T4 GPU 限制）
LEARNING_RATE = 1e-4         # 学习率
MAX_GRAD_NORM = 1.0          # 梯度裁剪

# LoRA 配置
LORA_R = 8                   # LoRA 秩
LORA_ALPHA = 16              # 缩放因子
LORA_DROPOUT = 0.05          # Dropout 率

# 模型维度设置（已优化）
VISION_OUTPUT_DIM = 512      # CLIP 输出维度
LLM_HIDDEN_DIM = 1024        # LLM 隐藏维度

# 数据设置
IMAGE_SIZE = 224             # CLIP 输入尺寸
MAX_TEXT_LENGTH = 128        # 文本最大长度
```

### 路径配置与数据支持

系统自动检测环境和数据格式：

**Kaggle 环境**：
- 使用 `/kaggle/input/` 路径
- 自动尝试多个可能的数据集路径

**本地开发**：
- 使用项目根目录下的 `data/Levir-CC-dataset`
- 支持的数据格式：
  - Arrow 格式：`data/Levir-CC-dataset/levir-cc-train.arrow`
  - 原始图像 + JSON：`data/Levir-CC-dataset/images/{train|val|test}/{A|B}` + `LevirCCcaptions.json`
  - 简化结构：`data/Levir-CC-dataset/{train|val|test}/{A|B}`

**JSON 标注支持**：
- 自动加载 `LevirCCcaptions.json` 作为图像标注
- 支持多数组拼接格式

---

## 📊 性能指标

### 内存使用（Kaggle T4 12GB）

| 组件 | 内存占用 |
|------|---------|
| CLIP 模型（冻结） | ~1.5 GB |
| Qwen2.5（4位量化） | ~2.5 GB |
| 批次大小 4 | ~3-4 GB |
| 优化器状态 | ~1-2 GB |
| **总计** | **~8-10 GB** ✅ |

### 训练速度（T4 GPU）

- 每轮次：2-3 分钟
- 10 轮总时间：25-35 分钟
- 每 500 步保存检查点

---

## 🔧 常见问题

### ❓ 错误："Dataset not found" 或 "Directory is neither a Dataset nor DatasetDict"

**原因**: 数据集路径配置错误，或数据集不是Arrow格式

**解决方案**：
```python
# 步骤1: 运行诊断脚本检查数据集结构
!python kaggle_dataset_check.py

# 步骤2: 检查精确路径
import os
for root, dirs, files in os.walk("/kaggle/input"):
    if 'levir' in root.lower():
        print(root)

# 步骤3: 更新 src/config.py 中的路径
# 确保路径指向 LEVIR-CC 根目录，例如:
# KAGGLE_PATHS = {
#     'dataset': '/kaggle/input/levir-cc-dateset/LEVIR-CC'
# }
```

**注意**: 从v1.1开始，项目支持自动从原始LEVIR-CC文件结构加载数据，无需Arrow格式转换。

### ❓ 错误："CUDA out of memory"

**解决方案**：编辑 `src/config.py`
```python
BATCH_SIZE = 2               # 从 4 减小到 2
GRADIENT_ACCUMULATION_STEPS = 2  # 增加梯度累积
USE_MIXED_PRECISION = False  # 禁用混合精度
```

### ❓ 训练太慢

**解决方案**：
```python
NUM_WORKERS = 8              # 增加数据加载工作进程
```

更多问题请参考 [快速开始.md - 常见问题](./快速开始.md#常见问题)

---

## 📚 文档导航

- **[README.md](./README.md)**（本文件）：项目概述和快速参考
- **[快速开始.md](./快速开始.md)**：详细的安装和部署指南
- **[项目指南.md](./项目指南.md)**：技术细节和模块说明

---

## 🎓 使用流程

### 1. 项目验证（2分钟）
```bash
python check_setup.py
```

### 2. 配置检查（可选）
```bash
python -m src.config  # 查看配置
python -m src.model   # 测试模型加载
```

### 3. 开始训练
```bash
# 本地训练
python -m src.train

# 或在 Kaggle 上部署（推荐）
```

### 4. 监控训练
```python
# 检查输出目录
import os
for item in os.listdir("output"):
    print(item)
```

### 5. 使用检查点
```python
import torch
from src.model import create_model

checkpoint = torch.load("checkpoint_best.pt")
model = create_model()
model.load_state_dict(checkpoint['model_state_dict'])
```

---

## 📖 技术参考

### 核心论文

- [CLIP: Learning Transferable Visual Models](https://arxiv.org/abs/2103.14030)
- [Qwen: A Large Language Model](https://arxiv.org/abs/2309.16609)
- [LoRA: Low-Rank Adaptation](https://arxiv.org/abs/2106.09685)
- [LEVIR-CC Dataset](https://github.com/S2-looking/LEVIR-CC)

### 相关资源

- HuggingFace Transformers
- HuggingFace Datasets
- PyTorch 文档
- PEFT 文档

---

## 🤝 贡献和支持

### 需要帮助？

1. 查看 [快速开始.md](./快速开始.md) 中的常见问题
2. 查看 [项目指南.md](./项目指南.md) 中的技术细节
3. 运行 `python check_setup.py` 诊断问题
4. 查看源代码中的详细注释

### 贡献指南

欢迎提交 Issue 和 Pull Request！

---

## 📜 许可证

MIT License - 可自由用于研究和商业项目

---

## 🎉 开始使用

准备好了吗？从这里开始：

```bash
# 快速验证
python check_setup.py

# 查看配置
python -m src.config

# 开始训练
python -m src.train
```

**或者在 Kaggle 上一键部署** → [快速开始.md](./快速开始.md)

---

**项目状态**: ✅ 生产就绪
**版本**: 1.0
**更新日期**: 2024年1月

---

*VLM-VLA Agent - 为高效的多模态学习而设计* 🚀

