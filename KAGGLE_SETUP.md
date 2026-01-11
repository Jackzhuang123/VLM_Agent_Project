# Kaggle 完整设置指南

> **重要**: 请严格按照以下步骤操作，确保使用最新代码

---

## 🚨 Step 0: 清理旧代码（重要！）

```python
import os
import shutil

# 完全清理工作目录
work_dir = "/kaggle/working"
for item in os.listdir(work_dir):
    item_path = os.path.join(work_dir, item)
    if os.path.isdir(item_path):
        print(f"删除目录: {item}")
        shutil.rmtree(item_path)
    else:
        print(f"删除文件: {item}")
        os.remove(item_path)

print("\n✅ 工作目录已清理")
```

---

## 📥 Step 1: 克隆最新代码

```bash
%%bash
cd /kaggle/working
git clone https://github.com/Jackzhuang123/VLM_Agent_Project.git
cd VLM_Agent_Project
echo "✅ 当前目录: $(pwd)"
echo "✅ 文件列表:"
ls -la
```

**验证**: 确保输出显示 `/kaggle/working/VLM_Agent_Project`

---

## 📦 Step 2: 安装依赖

```bash
%%bash
cd /kaggle/working/VLM_Agent_Project
pip install -q transformers>=4.37.0 \
    datasets>=2.14.0 \
    peft>=0.7.0 \
    bitsandbytes>=0.41.0 \
    accelerate>=0.25.0 \
    pyarrow>=14.0.0 \
    einops>=0.7.0 \
    tqdm

echo "✅ 依赖安装完成"
```

---

## 🔍 Step 3: 验证代码版本（重要！）

```python
import sys
sys.path.insert(0, '/kaggle/working/VLM_Agent_Project')

# 检查关键函数是否包含最新修复
with open('/kaggle/working/VLM_Agent_Project/src/dataset.py', 'r') as f:
    content = f.read()

# 检查是否包含关键修复
checks = {
    'formatted_as(None)': 'formatted_as(None)' in content,
    'HuggingFace Image bytes': "'bytes' in image_data" in content,
    'Arrow文件加载': 'arrow_files = list(dataset_path.glob' in content,
}

print("代码版本检查:")
for check_name, passed in checks.items():
    status = "✅" if passed else "❌"
    print(f"  {status} {check_name}")

if all(checks.values()):
    print("\n✅ 代码版本正确，包含所有最新修复！")
else:
    print("\n❌ 代码版本过旧，请重新克隆！")
    print("   运行: !rm -rf /kaggle/working/VLM_Agent_Project")
    print("   然后重新执行 Step 1")
```

---

## 🚀 Step 4: 开始训练

```python
import sys
import os

# 确保使用正确的项目路径
project_path = '/kaggle/working/VLM_Agent_Project'
if project_path not in sys.path:
    sys.path.insert(0, project_path)

# 切换到项目目录
os.chdir(project_path)
print(f"当前工作目录: {os.getcwd()}")

# 开始训练
!python -m src.train
```

---

## 🐛 Step 5: 如果还是失败

运行调试脚本：

```python
import sys
sys.path.insert(0, '/kaggle/working/VLM_Agent_Project')

!python /kaggle/working/VLM_Agent_Project/debug_dataset.py
```

---

## ⚠️ 常见错误

### 错误1: 多层嵌套目录

**症状**:
```
/kaggle/working/VLM_Agent_Project/VLM_Agent_Project/VLM_Agent_Project/...
```

**解决**:
```bash
# 删除所有嵌套目录
!rm -rf /kaggle/working/*
# 重新克隆
!cd /kaggle/working && git clone https://github.com/Jackzhuang123/VLM_Agent_Project.git
```

### 错误2: 旧代码版本

**症状**:
```
FileNotFoundError: /root/.cache/huggingface/.../Levir-CC-dataset.zip
```

**解决**:
1. 运行 Step 3 验证代码版本
2. 如果检查失败，重新克隆（Step 0 + Step 1）

### 错误3: Python路径问题

**症状**:
```
ModuleNotFoundError: No module named 'src'
```

**解决**:
```python
import sys
import os
sys.path.insert(0, '/kaggle/working/VLM_Agent_Project')
os.chdir('/kaggle/working/VLM_Agent_Project')
```

---

## 📊 预期成功输出

```
============================================================
加载 LEVIR-CC 数据集
============================================================
数据集路径: /kaggle/input/levir-cc-dateset/LEVIR-CC
⚠️  Arrow格式加载失败: Directory ... is neither a Dataset nor DatasetDict
🔄 尝试从原始文件结构加载...
✅ 找到 Arrow 文件: levir-cc-train.arrow
✅ 使用 datasets 库加载成功: 22326 个样本
✅ 从 Arrow 文件加载成功，共 22326 个样本

============================================================
Dataset Structure Inspection
============================================================
Dataset size: 22326
Sample keys: ['A', 'B', 'captions', ...]
Sample data types:
  A: dict with 'bytes' (image data)    ← 关键！
  B: dict with 'bytes' (image data)    ← 关键！

Detected keys:
  Image 1 key: A
  Image 2 key: B
  Caption key: captions
  BBox key: bbox

✅ Dataset initialized with 20093 samples

Train/Val split完成...
开始训练...
```

---

## 🎯 关键修复说明

最新代码包含以下修复：

1. **Arrow文件直接加载** (`load_raw_levir_cc_dataset`)
   - 自动检测 `.arrow` 文件
   - 使用 `datasets.load_dataset('arrow', ...)` 加载

2. **HuggingFace Image格式支持** (`_load_image`)
   - 支持 `{'bytes': ...}` 字典格式
   - 直接从内存bytes加载图像

3. **禁用自动格式化** (`formatted_as(None)`)
   - 在 `_inspect_data_structure` 中使用
   - 在 `__getitem__` 中使用
   - 避免尝试从不存在的路径加载

---

## 💡 快速命令（一键设置）

将以下代码复制到一个 Kaggle Cell 中运行：

```python
# 完整设置脚本
import os, sys, shutil

# 1. 清理
for item in os.listdir("/kaggle/working"):
    path = f"/kaggle/working/{item}"
    (shutil.rmtree if os.path.isdir(path) else os.remove)(path)

# 2. 克隆
!cd /kaggle/working && git clone https://github.com/Jackzhuang123/VLM_Agent_Project.git

# 3. 安装依赖
!pip install -q transformers datasets peft bitsandbytes accelerate pyarrow einops tqdm

# 4. 设置路径
sys.path.insert(0, '/kaggle/working/VLM_Agent_Project')
os.chdir('/kaggle/working/VLM_Agent_Project')

# 5. 验证
with open('src/dataset.py') as f:
    assert 'formatted_as(None)' in f.read(), "❌ 代码版本错误！"
print("✅ 设置完成！")

# 6. 开始训练
!python -m src.train
```

---

**最后更新**: 2024-01-11
**Git Commit**: `8715398` (fix: 支持HuggingFace Image格式的bytes数据加载)

