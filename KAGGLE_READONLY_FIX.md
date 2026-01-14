# Kaggle 只读文件系统修复

## 问题描述

在 Kaggle 环境中运行训练时，遇到以下错误：

```
OSError: [Errno 30] Read-only file system: '/kaggle/input/levir-cc-dataset/LEVIR-CC/train/tmplphzukly'
```

## 根本原因

1. **Kaggle 的文件系统权限**
   - `/kaggle/input/` 是只读的（数据集输入目录）
   - 只有 `/kaggle/working/` 是可写的

2. **HuggingFace train_test_split 的行为**
   - 调用 `dataset.train_test_split()` 时，HF 尝试在数据集所在目录创建临时文件
   - 对于位于 `/kaggle/input/` 中的数据集，这会失败
   - 同样，`dataset.select()` 也可能需要在原目录创建缓存

## 解决方案

### 1. 设置缓存目录

在 `create_dataloaders()` 中，首先设置 HF_DATASETS_CACHE 到可写目录：

```python
cache_dir = os.path.join(Config.WORKING_DIR, '.cache')
os.makedirs(cache_dir, exist_ok=True)
os.environ['HF_DATASETS_CACHE'] = cache_dir
```

### 2. 手动数据集分割

不使用 HF 的 `train_test_split()`，而是手动进行分割：

```python
import numpy as np

n_samples = len(dataset_split)
indices = np.arange(n_samples)

# 设置随机种子
np.random.seed(seed)
np.random.shuffle(indices)

# 分割
split_point = int(n_samples * (1 - test_split))
train_indices = sorted(indices[:split_point].tolist())
test_indices = sorted(indices[split_point:].tolist())
```

### 3. 处理 select() 失败

如果 `dataset.select()` 也失败，使用索引包装对象：

```python
class IndexedDataset:
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices
        self._len = len(indices)

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]
```

这个包装对象按需访问底层数据集，避免创建新文件。

### 4. 适配 LevirCCActionDataset

修改 `_disable_auto_decoding()` 以识别和跳过 `IndexedDataset` 对象：

```python
if hasattr(self.dataset, '__class__') and self.dataset.__class__.__name__ == 'IndexedDataset':
    print(f"ℹ️  检测到 IndexedDataset 包装对象，跳过自动解码禁用")
    return
```

## 修改文件

### src/dataset.py

**函数：** `create_dataloaders()`
- 添加缓存目录设置
- 实现手动数据集分割
- 添加 `IndexedDataset` 降级方案
- 改进错误处理和日志

**函数：** `_disable_auto_decoding()`
- 添加对 `IndexedDataset` 的检测
- 跳过不需要解码禁用的包装对象

## 工作流程

```
尝试 train_test_split()
       ↓
   [成功] → 创建分割数据集
       ↓
   [OSError: Read-only file system]
       ↓
   尝试 select()
       ↓
   [成功] → 使用 select 创建子集
       ↓
   [OSError: Read-only file system]
       ↓
   创建 IndexedDataset 包装对象
       ↓
   成功（按需访问）
```

## 特点

✅ **多层级降级方案**
- 最优：使用 HF 原生方法（有缓存优化）
- 次优：使用 select()（较轻）
- 降级：使用索引包装（最轻）

✅ **兼容所有环境**
- 本地开发：自动使用最优方案
- Kaggle：自动降级到适合方案
- 其他受限环境：自动适配

✅ **无数据复制**
- IndexedDataset 只存储索引
- 实际数据不被复制到内存
- 节省内存和时间

✅ **性能影响最小**
- 分割操作快速
- 访问时间复杂度 O(1)
- 无缓存开销

## 测试验证

运行以下命令在 Kaggle 环境验证：

```python
python -m src.train
```

预期输出：
```
🔄 正在分割数据集 (train: 90.0%, test: 10.0%)...
✅ 数据集分割完成: 训练集 20093 个样本，测试集 2233 个样本
```

## 注意事项

1. **种子重现性**
   - 使用 numpy 种子确保每次运行结果相同
   - 与 sklearn 的行为一致

2. **索引顺序**
   - 训练和测试索引都排序，确保数据连贯性
   - 在 IndexedDataset 中也保持这个顺序

3. **错误处理**
   - 捕获 OSError 和 "No space left" 错误
   - 提供清晰的错误消息
   - 自动降级到下一个方案

## 相关环境变量

如果需要手动指定缓存目录，可以设置：

```bash
export HF_DATASETS_CACHE=/path/to/writable/cache
```

或在代码中：

```python
os.environ['HF_DATASETS_CACHE'] = '/path/to/writable/cache'
```

## 其他可能的问题

如果仍然遇到文件系统错误，检查：

1. `/kaggle/working/` 是否有足够的磁盘空间
2. 文件权限是否正确
3. 数据集大小是否超过可用空间

可以通过以下命令检查空间：

```bash
df -h /kaggle/working/

