# Kaggle 只读文件系统问题修复 - 快速总结

## 问题
```
OSError: [Errno 30] Read-only file system: '/kaggle/input/levir-cc-dataset/LEVIR-CC/train/tmp...'
```

## 原因
Kaggle `/kaggle/input/` 是只读的，但 HuggingFace 的 `train_test_split()` 和 `select()` 尝试在该目录创建临时文件

## 解决方案

### 核心修改

**文件：** `src/dataset.py` → `create_dataloaders()` 函数

#### 1. 设置可写缓存目录
```python
cache_dir = os.path.join(Config.WORKING_DIR, '.cache')
os.makedirs(cache_dir, exist_ok=True)
os.environ['HF_DATASETS_CACHE'] = cache_dir
```

#### 2. 手动数据集分割
```python
n_samples = len(dataset_split)
indices = np.arange(n_samples)
np.random.seed(seed)
np.random.shuffle(indices)

split_point = int(n_samples * (1 - test_split))
train_indices = sorted(indices[:split_point].tolist())
test_indices = sorted(indices[split_point:].tolist())
```

#### 3. 降级方案 - IndexedDataset 包装
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

当 `select()` 也失败时使用此包装，避免创建任何临时文件。

#### 4. 适配 LevirCCActionDataset
在 `_disable_auto_decoding()` 中添加：
```python
if self.dataset.__class__.__name__ == 'IndexedDataset':
    print(f"ℹ️  检测到 IndexedDataset 包装对象，跳过自动解码禁用")
    return
```

## 关键特点

| 特点 | 效果 |
|------|------|
| 多层级降级 | 最优→次优→降级，确保在任何环境都能工作 |
| 无数据复制 | IndexedDataset 只存储索引，节省内存 |
| 快速访问 | O(1) 访问时间 |
| 完全兼容 | 支持所有现有代码，无破坏性改动 |

## 修改影响

✅ **完全向后兼容**
- 本地开发：使用 HF 原生方法
- Kaggle：自动降级到合适方案
- 其他环境：自动适配

✅ **无性能下降**
- 分割快速（O(n)）
- 访问高效（O(1)）
- 无额外内存开销

✅ **改进了鲁棒性**
- 处理只读文件系统
- 处理磁盘空间不足
- 清晰的错误提示

## 测试验证

在 Kaggle 运行：
```bash
python -m src.train
```

预期看到：
```
🔄 正在分割数据集 (train: 90.0%, test: 10.0%)...
✅ 数据集分割完成: 训练集 20093 个样本，测试集 2233 个样本
```

## 相关文件

- 详细说明：`KAGGLE_READONLY_FIX.md`
- 数据集加载说明：`DATASET_LOADING_FIX.md`
- 快速参考：`QUICK_FIX_REFERENCE.md`

## 环境变量

可选：手动指定缓存目录
```bash
export HF_DATASETS_CACHE=/path/to/writable/cache

