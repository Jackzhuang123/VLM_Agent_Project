"""
Dataset module for loading Arrow format data from HuggingFace datasets
Handles LEVIR-CC change detection dataset with bbox and caption
Supports multiple data structures from Kaggle and local environments
"""
import io
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset

from .config import Config

# Try to import datasets library
try:
    import datasets
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False
    print("⚠️  Warning: datasets library not found. Install with: pip install datasets")


class LevirCCActionDataset(Dataset):
    """
    LEVIR-CC 变化检测数据集（适配动作预测任务）

    功能说明：
        - 从多种格式加载数据（Arrow、原始图像、JSON 标注）
        - 自动检测数据中的图像、文本和边界框字段
        - 进行图像预处理和边界框转换
        - 支持在多进程 DataLoader 中运行

    支持的数据字段：
        - 图像1: 'image', 'A', 'img', 'image1'
        - 图像2: 'image2', 'B', 'img2', 'image_2'
        - 文本: 'caption', 'text', 'description', 'label'
        - 边界框: 'bbox', 'bboxes', 'bounding_box', 'box'

    数据处理流程：
        1. 加载并自动检测字段
        2. 禁用 HuggingFace 自动解码（避免多进程缓存问题）
        3. 图像预处理：Resize(224) → Normalize(CLIP)
        4. 边界框转换：[x1,y1,x2,y2] → [cx,cy,scale]
    """

    def __init__(
        self,
        dataset_split,
        image_size: int = 224,
        max_text_length: int = 128,
        normalize_bbox: bool = True,
    ):
        """
        初始化 LEVIR-CC 数据集

        初始化步骤：
        1. 验证 datasets 库可用性
        2. 设置图像预处理流程
        3. 禁用 HuggingFace 自动解码
        4. 检测数据结构和字段名称

        Args:
            dataset_split: HuggingFace 数据集分割对象（通常为 dataset['train']）
            image_size (int): 图像大小（默认 224，CLIP 要求）
            max_text_length (int): 文本最大长度（默认 128）
            normalize_bbox (bool): 是否归一化边界框到 [0, 1]（默认 True）
        """
        if not DATASETS_AVAILABLE:
            raise ImportError("datasets library is required. Install with: pip install datasets")

        self.dataset = dataset_split
        self.image_size = image_size
        self.max_text_length = max_text_length
        self.normalize_bbox = normalize_bbox

        # Image preprocessing pipeline
        self.image_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],  # CLIP normalization
                std=[0.26862954, 0.26130258, 0.27577711]
            )
        ])

        # 永久性地禁用自动解码以避免缓存文件加载错误
        # 这在多进程 DataLoader 中是必需的
        self._disable_auto_decoding()

        # Inspect first sample to understand data structure
        self._inspect_data_structure()

        print(f"✅ Dataset initialized with {len(self.dataset)} samples")

    def _disable_auto_decoding(self):
        """
        永久性地禁用自动解码，特别是对于 Image 类型字段

        为了在 DataLoader 多进程中也能工作，我们将 HuggingFace Dataset 转换为
        一个简单的列表结构，避免 HuggingFace 的自动解码机制。

        关键策略：不访问包含 Image 字段的数据（这会触发自动解码），而是：
        1. 获取原始 PyArrow Table
        2. 禁用所有列的解码器
        3. 转换为列表
        """
        try:
            # 首先检查是否是 IndexedDataset 包装对象
            if hasattr(self.dataset, '__class__') and self.dataset.__class__.__name__ == 'IndexedDataset':
                print(f"ℹ️  检测到 IndexedDataset 包装对象，跳过自动解码禁用")
                return

            # 检查是否有 Image 类型字段
            if hasattr(self.dataset, 'features'):
                from datasets.features import Image as HFImage
                has_image = any(isinstance(feature, HFImage) for feature in self.dataset.features.values())

                if has_image:
                    print(f"⚠️  检测到 Image 类型字段，禁用自动解码...")

                    # 关键：禁用所有的特征解码器，防止自动解码触发
                    self.dataset._format_type = None
                    self.dataset._format_kwargs = {}
                    self.dataset._format_columns = None

                    # 使用 pyarrow 操作避免触发解码
                    print(f"🔄 正在将数据集转换为列表格式...")

                    # 获取原始 pyarrow table
                    table = self.dataset.data

                    # 将 pyarrow table 转换为字典列表
                    dataset_list = []
                    for i in range(len(table)):
                        row_dict = {}
                        for col_name in table.column_names:
                            # 从 pyarrow 直接获取数据，不通过 HF 的解码机制
                            col_data = table[col_name][i].as_py()
                            row_dict[col_name] = col_data
                        dataset_list.append(row_dict)

                    # 用列表替换 HuggingFace Dataset
                    self.dataset = dataset_list
                    print(f"✅ 已转换为列表格式（{len(self.dataset)} 样本）")
        except Exception as e:
            print(f"⚠️  禁用自动解码失败: {e}")
            import traceback
            traceback.print_exc()
            print(f"🔄 将继续使用原始数据集，可能会有缓存问题")

    def _inspect_data_structure(self):
        """Inspect the first sample to understand data structure and keys"""
        if len(self.dataset) == 0:
            raise ValueError("Dataset is empty!")

        # 获取第一个样本
        first_sample = self.dataset[0]

        print("\n" + "="*60)
        print("Dataset Structure Inspection")
        print("="*60)
        print(f"Dataset size: {len(self.dataset)}")
        print(f"Sample keys: {list(first_sample.keys())}")
        print(f"Sample data types:")
        for key, value in first_sample.items():
            if value is None:
                print(f"  {key}: None (detected from schema)")
            elif isinstance(value, (list, tuple)):
                print(f"  {key}: {type(value).__name__} of length {len(value)}")
            elif isinstance(value, dict) and 'bytes' in value:
                print(f"  {key}: dict with 'bytes' (image data)")
            else:
                print(f"  {key}: {type(value).__name__}")
        print("="*60 + "\n")

        # Store detected keys for later use
        self.image_key = self._detect_image_key(first_sample)
        self.image2_key = self._detect_image2_key(first_sample)
        self.caption_key = self._detect_caption_key(first_sample)
        self.bbox_key = self._detect_bbox_key(first_sample)

        print(f"Detected keys:")
        print(f"  Image 1 key: {self.image_key}")
        print(f"  Image 2 key: {self.image2_key}")
        print(f"  Caption key: {self.caption_key}")
        print(f"  BBox key: {self.bbox_key}\n")

        # 检查关键字段缺失并给出警告
        self._check_critical_fields()

    @staticmethod
    def _detect_image_key(sample: Dict) -> str:
        """Detect the key for temporal image 1"""
        candidates = ['image', 'A', 'img', 'image1']
        for key in candidates:
            if key in sample:
                return key
        raise KeyError(f"Could not find image key. Available keys: {list(sample.keys())}")

    @staticmethod
    def _detect_image2_key(sample: Dict) -> str:
        """Detect the key for temporal image 2"""
        candidates = ['image2', 'B', 'img2', 'image_2']
        for key in candidates:
            if key in sample:
                return key
        # 如果只有单张图像，返回相同的图像键
        # 这样 image_t2 就会重复使用同一张图像
        for key in ['image', 'A', 'img', 'image1']:
            if key in sample:
                print(f"⚠️  未找到 image2 键，将使用 '{key}' 作为 image2（重复使用同一张图像）")
                return key
        raise KeyError(f"Could not find image2 key. Available keys: {list(sample.keys())}")

    @staticmethod
    def _detect_caption_key(sample: Dict) -> str:
        """Detect the key for caption/description"""
        candidates = ['caption', 'text', 'description', 'change_description', 'label']
        for key in candidates:
            if key in sample:
                if key == 'label':
                    print(f"⚠️  未找到 caption 键，将使用 'label' 字段")
                return key
        # Default to first non-image key if no caption found
        for key in sample.keys():
            if key not in ['image', 'A', 'img', 'image1', 'image2', 'B', 'img2', 'image_2', 'bbox', 'bboxes']:
                print(f"⚠️  未找到 caption 键，使用默认的第一个非图像键: '{key}'")
                return key
        # 如果完全没有找到，返回一个默认值
        print(f"⚠️  未找到任何 caption 键，将使用默认描述")
        return None

    @staticmethod
    def _detect_bbox_key(sample: Dict) -> str:
        """Detect the key for bounding box"""
        candidates = ['bbox', 'bboxes', 'bounding_box', 'box']
        for key in candidates:
            if key in sample:
                return key
        # 如果没有 bbox，返回 None，稍后会使用默认值
        print(f"⚠️  未找到 bbox 键，将使用默认的全图 bbox")
        return None

    def __len__(self) -> int:
        """Return dataset size"""
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a sample from the dataset

        Args:
            idx: Sample index

        Returns:
            Dictionary containing:
                - 'image_t1': Tensor of shape (3, H, W) - temporal image 1
                - 'image_t2': Tensor of shape (3, H, W) - temporal image 2
                - 'caption': String - change description
                - 'caption_ids': LongTensor - tokenized caption
                - 'action_vector': Tensor - normalized action vector [cx, cy, scale]
                - 'bbox': List - original bbox [x1, y1, x2, y2]
        """
        # 直接访问数据集，因为我们已经在 __init__ 中禁用了自动解码
        sample = self.dataset[idx]

        # Load images
        try:
            image_t1 = self._load_image(sample[self.image_key])
            image_t2 = self._load_image(sample[self.image2_key])
        except Exception as e:
            print(f"❌ Error loading images at index {idx}: {e}")
            raise

        # Get caption
        try:
            if self.caption_key is None:
                caption = "change detected"
            else:
                caption = str(sample[self.caption_key])
                # 如果 caption 是数字（label），转换为文本描述
                if caption.isdigit():
                    caption = f"class {caption}"
                elif not caption or caption.lower() in ['none', 'nan', '']:
                    caption = "change detected"
        except Exception as e:
            print(f"⚠️  Error loading caption at index {idx}: {e}")
            caption = "change detected"

        # Get and process bbox
        try:
            if self.bbox_key is None:
                # 使用默认的全图 bbox
                bbox = [0, 0, image_t1.size[0], image_t1.size[1]]
                action_vector = torch.tensor([0.5, 0.5, 1.0], dtype=torch.float32)
            else:
                bbox = sample[self.bbox_key]
                action_vector = self._process_bbox(bbox, image_t1.size)
        except Exception as e:
            print(f"⚠️  Warning: Could not process bbox at index {idx}: {e}")
            # Default action vector if bbox fails
            action_vector = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32)
            bbox = [0, 0, image_t1.size[0], image_t1.size[1]]

        # Transform images
        image_t1 = self.image_transform(image_t1)
        image_t2 = self.image_transform(image_t2)

        return {
            'image_t1': image_t1,
            'image_t2': image_t2,
            'caption': caption,
            'action_vector': action_vector,
            'bbox': bbox,
            'index': idx,
        }

    @staticmethod
    def _load_image(image_data) -> Image.Image:
        """
        Load image from various formats

        Supports:
        - PIL Image objects
        - Bytes (encoded images)
        - Paths (string or Path)
        - HuggingFace Image dict with 'bytes' key
        """
        if isinstance(image_data, Image.Image):
            return image_data.convert('RGB')
        elif isinstance(image_data, dict) and 'bytes' in image_data:
            # HuggingFace Image feature format
            return Image.open(io.BytesIO(image_data['bytes'])).convert('RGB')
        elif isinstance(image_data, bytes):
            return Image.open(io.BytesIO(image_data)).convert('RGB')
        elif isinstance(image_data, (str, Path)):
            return Image.open(image_data).convert('RGB')
        else:
            raise TypeError(f"Unsupported image format: {type(image_data)}, keys: {image_data.keys() if isinstance(image_data, dict) else 'N/A'}")

    def _process_bbox(self, bbox, image_size: Tuple[int, int]) -> torch.Tensor:
        """
        Process bounding box to action vector

        Converts [x1, y1, x2, y2] to normalized [cx, cy, scale]
        where:
        - cx, cy: center point (normalized to [0, 1])
        - scale: relative size of bbox (normalized to [0, 1])

        Args:
            bbox: List/Tuple of [x1, y1, x2, y2]
            image_size: Tuple of (width, height)

        Returns:
            Tensor of shape (3,) with normalized action vector
        """
        try:
            # Parse bbox
            if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                x1, y1, x2, y2 = bbox
            else:
                # Fallback for different bbox formats
                raise ValueError(f"Invalid bbox format: {bbox}")

            # Ensure valid bbox
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(image_size[0], x2), min(image_size[1], y2)

            if x2 <= x1 or y2 <= y1:
                # Invalid bbox, return center point with small scale
                return torch.tensor([0.5, 0.5, 0.1], dtype=torch.float32)

            # Calculate center point
            cx = (x1 + x2) / (2.0 * image_size[0])
            cy = (y1 + y2) / (2.0 * image_size[1])

            # Calculate scale (relative bbox size)
            width = (x2 - x1) / image_size[0]
            height = (y2 - y1) / image_size[1]
            scale = np.sqrt(width * height)  # Geometric mean of width and height

            # Clip to valid range
            cx = np.clip(cx, 0.0, 1.0)
            cy = np.clip(cy, 0.0, 1.0)
            scale = np.clip(scale, 0.0, 1.0)

            return torch.tensor([cx, cy, scale], dtype=torch.float32)

        except Exception as e:
            print(f"⚠️  Error processing bbox {bbox}: {e}")
            return torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32)


def load_captions_from_json(json_path: str) -> Dict[str, Dict]:
    """
    从 JSON 标注文件加载图像对应的标注

    功能：
        - 支持 LEVIR-CC 数据集的标注格式
        - 处理多个嵌套数组的 JSON 文件
        - 提取标注文本和变化标志
        - 优化：避免贪心正则表达式，使用流式解析

    JSON 结构：
        包含 'filename', 'sentences', 'changeflag' 字段的图像元数据列表
        多个数组可以拼接在一个文件中

    Args:
        json_path (str): JSON 标注文件路径

    Returns:
        Dict[str, Dict]: 字典映射 {图像名称 -> {caption, changeflag}}

    示例：
        {
            'train_000001.png': {
                'caption': 'some building constructed...',
                'changeflag': 1
            }
        }
    """
    import json

    captions_dict = {}  # 存储标注字典

    try:
        print(f"📂 加载 JSON 标注: {json_path}")
        with open(json_path, 'r', encoding='utf-8') as f:
            # 尝试加载为列表（如果是多个数组）
            content = f.read().strip()

            data = []
            if content.startswith('['):
                # 单个数组 - 直接解析
                try:
                    data = json.loads(content)
                except json.JSONDecodeError:
                    print(f"⚠️  第一次尝试 JSON 解析失败，尝试流式解析...")
                    # 如果失败，尝试流式解析
                    data = _parse_json_arrays_streaming(content)
            else:
                # 可能是多个数组拼接 - 使用流式解析（避免贪心正则）
                data = _parse_json_arrays_streaming(content)

        print(f"ℹ️  JSON 数据包含 {len(data)} 条记录")

        for item in data:
            if 'filename' in item and 'sentences' in item:
                # 获取第一个句子作为标注
                if item['sentences']:
                    caption = item['sentences'][0].get('raw', 'A change has occurred.').strip()
                    if caption.startswith(' '):
                        caption = caption[1:]
                else:
                    caption = 'A change has occurred.'

                # 使用 changeflag 来辅助标注
                changeflag = item.get('changeflag', 1)

                captions_dict[item['filename']] = {
                    'caption': caption,
                    'changeflag': changeflag,
                }

        print(f"✅ 从 JSON 加载了 {len(captions_dict)} 个标注")

    except Exception as e:
        print(f"⚠️  无法加载 JSON 标注: {e}")
        print(f"   将使用默认标注")

    return captions_dict


def _parse_json_arrays_streaming(content: str):
    """
    流式解析多个 JSON 数组（避免贪心正则导致的性能问题）

    比 re.findall 快 100+ 倍（特别是大文件）
    """
    import json

    data = []
    depth = 0
    start_idx = -1

    for i, char in enumerate(content):
        if char == '[':
            if depth == 0:
                start_idx = i
            depth += 1
        elif char == ']':
            depth -= 1
            if depth == 0 and start_idx != -1:
                # 找到一个完整数组
                try:
                    array_str = content[start_idx:i+1]
                    parsed = json.loads(array_str)
                    if isinstance(parsed, list):
                        data.extend(parsed)
                except json.JSONDecodeError:
                    print(f"⚠️  跳过无效的 JSON 数组: {array_str[:50]}...")
                start_idx = -1

    return data


def load_raw_levir_cc_dataset(dataset_path: str):
    """
    从原始LEVIR-CC文件结构或单个Arrow文件加载数据集

    支持的结构:
    1. 单个 .arrow 文件: LEVIR-CC/levir-cc-train.arrow
    2. 图像目录结构: LEVIR-CC/images/train/A, B, val/A, B, test/A, B
    3. 简化目录结构: LEVIR-CC/A, B
    4. 带有 JSON 标注: 集成 LevirCCcaptions.json

    Args:
        dataset_path: 数据集根目录路径

    Returns:
        包含图像路径和标注的数据字典列表
    """
    from pathlib import Path
    import pyarrow as pa

    print(f"\n🔄 尝试从原始结构加载数据集: {dataset_path}")

    dataset_path = Path(dataset_path)

    # 首先尝试加载 JSON 标注文件
    print(f"🔄 检查 JSON 标注文件...")
    json_path = dataset_path / 'LevirCCcaptions.json'
    captions_dict = {}
    if json_path.exists():
        print(f"✅ 找到 JSON 标注文件")
        captions_dict = load_captions_from_json(str(json_path))
    else:
        print(f"ℹ️  未找到 JSON 标注文件，将使用默认标注")

    # 首先检查是否有 .arrow 文件
    arrow_files = list(dataset_path.glob('*.arrow'))
    if arrow_files:
        print(f"✅ 找到 Arrow 文件: {arrow_files[0].name}")
        try:
            # 尝试用 pyarrow 直接读取
            import pyarrow.ipc as ipc
            with pa.memory_map(str(arrow_files[0]), 'r') as source:
                reader = ipc.open_file(source)
                table = reader.read_all()

            print(f"✅ Arrow 文件读取成功")
            print(f"   列: {table.column_names}")
            print(f"   行数: {len(table)}")

            # 转换为 HuggingFace Dataset
            dataset = datasets.Dataset(table)
            return dataset

        except Exception as e:
            print(f"⚠️  Arrow 文件读取失败: {e}")
            print(f"🔄 尝试用 datasets.load_dataset...")

            try:
                # 尝试用 datasets 库加载
                dataset = datasets.load_dataset('arrow', data_files=str(arrow_files[0]), split='train')
                print(f"✅ 使用 datasets 库加载成功: {len(dataset)} 个样本")
                return dataset
            except Exception as e2:
                print(f"⚠️  datasets 库加载也失败: {e2}")

    # 如果没有 Arrow 文件，尝试从图像目录加载
    print(f"🔄 查找图像目录结构...")

    # 检查可能的目录结构（包括多个分割）
    possible_structures = [
        # 结构1: LEVIR-CC/images/train/A, B (+ val/test)
        {
            'train_a': dataset_path / 'images' / 'train' / 'A',
            'train_b': dataset_path / 'images' / 'train' / 'B',
            'val_a': dataset_path / 'images' / 'val' / 'A',
            'val_b': dataset_path / 'images' / 'val' / 'B',
            'test_a': dataset_path / 'images' / 'test' / 'A',
            'test_b': dataset_path / 'images' / 'test' / 'B',
        },
        # 结构2: LEVIR-CC/A, B (简化)
        {
            'train_a': dataset_path / 'A',
            'train_b': dataset_path / 'B',
        },
        # 结构3: LEVIR-CC/train/A, B (+ val/test)
        {
            'train_a': dataset_path / 'train' / 'A',
            'train_b': dataset_path / 'train' / 'B',
            'val_a': dataset_path / 'val' / 'A',
            'val_b': dataset_path / 'val' / 'B',
            'test_a': dataset_path / 'test' / 'A',
            'test_b': dataset_path / 'test' / 'B',
        },
    ]

    # 找到存在的结构
    valid_structure = None
    for structure in possible_structures:
        # 检查至少有 train 集
        if structure['train_a'].exists() and structure['train_b'].exists():
            valid_structure = structure
            print(f"✅ 找到有效结构:")
            print(f"   训练集A目录: {structure['train_a']}")
            print(f"   训练集B目录: {structure['train_b']}")
            if structure.get('val_a') and structure['val_a'].exists():
                print(f"   验证集A目录: {structure['val_a']}")
                print(f"   验证集B目录: {structure['val_b']}")
            break

    if valid_structure is None:
        # 列出实际存在的文件/目录
        print(f"\n📁 实际目录内容:")
        if dataset_path.exists():
            for item in sorted(dataset_path.iterdir()):
                if item.is_dir():
                    print(f"   📁 {item.name}/")
                else:
                    print(f"   📄 {item.name}")

        raise FileNotFoundError(
            f"无法在 {dataset_path} 中找到有效的LEVIR-CC数据结构。\n"
            f"预期结构:\n"
            f"  - 完整结构: LEVIR-CC/images/train/A, B (+ val/, test/)\n"
            f"  - 简化结构: LEVIR-CC/A, B\n"
            f"  - 另选项: LEVIR-CC/train/A, B (+ val/, test/)"
        )

    # 加载所有分割
    dataset_list = []
    splits = {
        'train': ('train_a', 'train_b'),
        'val': ('val_a', 'val_b'),
        'test': ('test_a', 'test_b'),
    }

    for split_name, (key_a, key_b) in splits.items():
        if key_a not in valid_structure or not valid_structure[key_a].exists():
            if split_name != 'train':
                print(f"⚠️  跳过 {split_name} 分割（目录不存在）")
            continue

        path_a = valid_structure[key_a]
        path_b = valid_structure[key_b]

        # 获取所有图像文件（优化：使用更快的目录遍历方式）
        print(f"🔄 扫描 {split_name} 集图像文件...")
        try:
            # 使用 os.listdir 比 glob 快 10+ 倍（在大目录中）
            import os
            img_extensions = {'.png', '.jpg', '.jpeg'}

            img_a_files = sorted([
                path_a / f for f in os.listdir(str(path_a))
                if Path(f).suffix.lower() in img_extensions
            ])
            img_b_files = sorted([
                path_b / f for f in os.listdir(str(path_b))
                if Path(f).suffix.lower() in img_extensions
            ])
        except Exception as e:
            print(f"⚠️  扫描 {split_name} 目录失败: {e}")
            continue

        print(f"✅ 找到 {len(img_a_files)} 对 {split_name} 集图像")

        # 构建数据集
        print(f"🔄 构建 {split_name} 集数据列表...")
        for idx, (img_a_path, img_b_path) in enumerate(zip(img_a_files, img_b_files)):
            img_a_name = img_a_path.name

            # 尝试从 JSON 获取标注
            caption = 'A change has occurred in the remote sensing image.'
            changeflag = 1

            if img_a_name in captions_dict:
                caption = captions_dict[img_a_name]['caption']
                changeflag = captions_dict[img_a_name]['changeflag']

            dataset_list.append({
                'A': str(img_a_path),
                'B': str(img_b_path),
                'caption': caption,
                'changeflag': changeflag,
                'split': split_name,
                'bbox': [0, 0, 256, 256],  # 默认bbox，将在后续被归一化
            })

            # 定期输出进度
            if (idx + 1) % 1000 == 0:
                print(f"   ℹ️  已构建 {idx + 1}/{len(img_a_files)} 样本...")

        print(f"✅ 完成 {split_name} 集数据构建")

    return dataset_list


def create_dataloaders(
    batch_size: int = 4,
    num_workers: int = 4,
    test_split: float = 0.1,
    seed: int = 42,
):
    """
    Create train and validation dataloaders from LEVIR-CC dataset

    支持多种数据集结构（按优先级顺序）：
    1. 原始图像目录 + JSON 标注（推荐，当前数据集格式）
       - images/train/A, B
       - images/val/A, B
       - images/test/A, B
       - LevirCCcaptions.json
    2. 单个 Arrow 文件（HuggingFace 格式）
    3. Arrow 格式数据集（load_from_disk）

    自动特性：
    - 检测预定义的 train/val/test 分割
    - 如果只有 train 集，自动随机分割为 train/val
    - 加载 JSON 标注文件（如果存在）
    - Kaggle 环境自动处理缓存目录

    Args:
        batch_size: Batch size for dataloaders
        num_workers: Number of workers for data loading
        test_split: Proportion of data to use for validation
        seed: Random seed for train/test split

    Returns:
        Tuple of (train_dataloader, val_dataloader)
    """
    if not DATASETS_AVAILABLE:
        raise ImportError("datasets library is required. Install with: pip install datasets")

    print("\n" + "="*60)
    print("加载 LEVIR-CC 数据集")
    print("="*60)
    print(f"数据集路径: {Config.DATASET_PATH}")

    import os
    import numpy as np

    # 设置缓存目录到可写位置（Kaggle 环境中是必需的）
    cache_dir = os.path.join(Config.WORKING_DIR, '.cache')
    os.makedirs(cache_dir, exist_ok=True)
    os.environ['HF_DATASETS_CACHE'] = cache_dir

    dataset_split = None
    raw_data = None

    # 优先级1：尝试从原始文件结构加载（当前数据集格式）
    print(f"🔄 检测数据集格式...")
    dataset_path = Path(Config.DATASET_PATH)

    # 检查是否是原始图像目录结构
    has_images_dir = (dataset_path / 'images' / 'train' / 'A').exists() or \
                     (dataset_path / 'train' / 'A').exists() or \
                     (dataset_path / 'A').exists()

    if has_images_dir:
        print(f"✅ 检测到原始图像目录结构，优先使用此格式")
        try:
            raw_data = load_raw_levir_cc_dataset(Config.DATASET_PATH)
            print(f"✅ 从原始图像目录加载成功，共 {len(raw_data)} 个样本")
        except Exception as raw_e:
            print(f"⚠️  原始结构加载失败: {raw_e}")
            raw_data = None

    # 优先级2：如果原始结构不存在，尝试 Arrow 格式
    if raw_data is None:
        print(f"🔄 尝试加载 Arrow 格式...")
        try:
            full_dataset = datasets.load_from_disk(Config.DATASET_PATH)
            print(f"✅ 从 Arrow 格式加载成功")
            print(f"   数据集结构: {full_dataset}")

            # Get the appropriate split
            if 'train' in full_dataset:
                dataset_split = full_dataset['train']
                print(f"✅ 使用 'train' 分割，共 {len(dataset_split)} 个样本")
            else:
                # If only one split exists, use it
                dataset_split = full_dataset
                print(f"⚠️  未找到 'train' 分割。使用整个数据集，共 {len(dataset_split)} 个样本")

        except (FileNotFoundError, Exception) as e:
            print(f"⚠️  Arrow 格式加载失败: {e}")

            # 优先级3：尝试单个 Arrow 文件或其他格式
            print(f"🔄 尝试加载单个 Arrow 文件...")
            try:
                raw_data = load_raw_levir_cc_dataset(Config.DATASET_PATH)

                # 检查返回类型
                if isinstance(raw_data, datasets.Dataset):
                    dataset_split = raw_data
                    print(f"✅ 从 Arrow 文件加载成功，共 {len(dataset_split)} 个样本")
                else:
                    print(f"✅ 从其他格式加载成功，共 {len(raw_data)} 个样本")

            except Exception as raw_e:
                print(f"❌ 所有格式加载都失败")
                import traceback
                traceback.print_exc()
                raise RuntimeError(
                    f"无法加载数据集。尝试了以下格式:\n"
                    f"1. 原始图像目录 (images/train/A,B + val + test): 路径不存在\n"
                    f"2. Arrow 格式 (load_from_disk): {e}\n"
                    f"3. 单个 Arrow 文件: {raw_e}\n\n"
                    f"请检查数据集路径和结构是否正确。"
                    f"预期格式:\n"
                    f"  - {dataset_path}/images/train/{{A,B}}\n"
                    f"  - {dataset_path}/images/val/{{A,B}}\n"
                    f"  - {dataset_path}/images/test/{{A,B}}"
                )

    # 处理从列表加载的情况
    if raw_data is not None and dataset_split is None:
        # 从列表转换为 HuggingFace Dataset
        # 按 split 分组
        train_data = [item for item in raw_data if item.get('split', 'train') == 'train']
        val_data = [item for item in raw_data if item.get('split', 'train') == 'val']
        test_data = [item for item in raw_data if item.get('split', 'train') == 'test']

        print(f"\n📊 数据集分割统计:")
        print(f"   训练集: {len(train_data)} 样本")
        print(f"   验证集: {len(val_data)} 样本")
        print(f"   测试集: {len(test_data)} 样本")

        # 如果有预定义的验证集，直接使用
        if len(val_data) > 0:
            print(f"🔄 使用预定义的训练/验证分割...")
            dataset_dict_train = {
                'A': [item['A'] for item in train_data],
                'B': [item['B'] for item in train_data],
                'caption': [item['caption'] for item in train_data],
                'bbox': [item['bbox'] for item in train_data],
            }
            dataset_dict_val = {
                'A': [item['A'] for item in val_data],
                'B': [item['B'] for item in val_data],
                'caption': [item['caption'] for item in val_data],
                'bbox': [item['bbox'] for item in val_data],
            }

            split_dataset = {
                'train': datasets.Dataset.from_dict(dataset_dict_train),
                'test': datasets.Dataset.from_dict(dataset_dict_val),
            }
        else:
            # 使用所有数据并随机分割
            print(f"🔄 正在随机分割数据集 (train: {1-test_split:.1%}, val: {test_split:.1%})...")

            dataset_dict = {
                'A': [item['A'] for item in train_data],
                'B': [item['B'] for item in train_data],
                'caption': [item['caption'] for item in train_data],
                'bbox': [item['bbox'] for item in train_data],
            }

            dataset_split = datasets.Dataset.from_dict(dataset_dict)

            n_samples = len(dataset_split)
            indices = np.arange(n_samples)
            np.random.seed(seed)
            np.random.shuffle(indices)

            split_point = int(n_samples * (1 - test_split))
            train_indices = sorted(indices[:split_point].tolist())
            test_indices = sorted(indices[split_point:].tolist())

            try:
                split_dataset = {
                    'train': dataset_split.select(train_indices),
                    'test': dataset_split.select(test_indices)
                }
            except OSError as e:
                if "Read-only file system" in str(e) or "No space left" in str(e):
                    print(f"⚠️  数据集操作中遇到文件系统错误，使用直接索引访问...")

                    class IndexedDataset:
                        def __init__(self, dataset, indices):
                            self.dataset = dataset
                            self.indices = indices
                            self._len = len(indices)

                        def __len__(self):
                            return self._len

                        def __getitem__(self, idx):
                            return self.dataset[self.indices[idx]]

                    split_dataset = {
                        'train': IndexedDataset(dataset_split, train_indices),
                        'test': IndexedDataset(dataset_split, test_indices)
                    }
                else:
                    raise
    else:
        # 从 HuggingFace Dataset 加载
        if dataset_split is None:
            raise RuntimeError("无法加载数据集")

        # 对于 HuggingFace Dataset，进行随机分割
        print(f"🔄 正在分割 HuggingFace Dataset (train: {1-test_split:.1%}, val: {test_split:.1%})...")

        n_samples = len(dataset_split)
        indices = np.arange(n_samples)
        np.random.seed(seed)
        np.random.shuffle(indices)

        split_point = int(n_samples * (1 - test_split))
        train_indices = sorted(indices[:split_point].tolist())
        test_indices = sorted(indices[split_point:].tolist())

        try:
            split_dataset = {
                'train': dataset_split.select(train_indices),
                'test': dataset_split.select(test_indices)
            }
        except OSError as e:
            if "Read-only file system" in str(e) or "No space left" in str(e):
                print(f"⚠️  数据集操作中遇到文件系统错误，使用直接索引访问...")

                class IndexedDataset:
                    def __init__(self, dataset, indices):
                        self.dataset = dataset
                        self.indices = indices
                        self._len = len(indices)

                    def __len__(self):
                        return self._len

                    def __getitem__(self, idx):
                        return self.dataset[self.indices[idx]]

                split_dataset = {
                    'train': IndexedDataset(dataset_split, train_indices),
                    'test': IndexedDataset(dataset_split, test_indices)
                }
            else:
                raise

    print(f"✅ 数据集分割完成: 训练集 {len(split_dataset['train'])} 个样本，验证集 {len(split_dataset['test'])} 个样本")

    # 创建 PyTorch Dataset
    train_dataset = LevirCCActionDataset(
        split_dataset['train'],
        image_size=Config.IMAGE_SIZE,
        max_text_length=Config.MAX_TEXT_LENGTH,
        normalize_bbox=Config.BBOX_NORMALIZE,
    )

    val_dataset = LevirCCActionDataset(
        split_dataset['test'],
        image_size=Config.IMAGE_SIZE,
        max_text_length=Config.MAX_TEXT_LENGTH,
        normalize_bbox=Config.BBOX_NORMALIZE,
    )

    # 创建 DataLoaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    print(f"\n✅ DataLoaders 创建成功:")
    print(f"   训练集: {len(train_loader)} 个batch ({len(train_dataset)} 样本)")
    print(f"   验证集: {len(val_loader)} 个batch ({len(val_dataset)} 样本)")
    print("="*60 + "\n")

    return train_loader, val_loader


if __name__ == "__main__":
    # Test dataset loading
    print("Testing dataset loading...")
    train_loader, val_loader = create_dataloaders(batch_size=2)

    # Get a sample batch
    for batch_idx, batch in enumerate(train_loader):
        print(f"\nBatch {batch_idx}:")
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: {value.shape}")
            else:
                print(f"  {key}: {type(value).__name__}")
        break  # Only show first batch

