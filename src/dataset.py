"""
Dataset module for loading Arrow format data from HuggingFace datasets
Handles LEVIR-CC change detection dataset with bbox and caption
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
    LEVIR-CC change detection dataset adapted for action prediction

    Loads data from HuggingFace Arrow format (.arrow files)
    Expected dataset structure:
    - 'image' or 'A': Temporal image 1
    - 'image2' or 'B': Temporal image 2
    - 'caption': Change description text
    - 'bbox': Bounding box [x1, y1, x2, y2] of change region
    """

    def __init__(
        self,
        dataset_split,
        image_size: int = 224,
        max_text_length: int = 128,
        normalize_bbox: bool = True,
    ):
        """
        Initialize LEVIR-CC dataset

        Args:
            dataset_split: HuggingFace dataset split (usually dataset['train'])
            image_size: Target size for image resizing
            max_text_length: Maximum length for tokenized text
            normalize_bbox: Whether to normalize bbox to [0, 1] range
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

        # Inspect first sample to understand data structure
        self._inspect_data_structure()

        print(f"✅ Dataset initialized with {len(self.dataset)} samples")

    def _inspect_data_structure(self):
        """Inspect the first sample to understand data structure and keys"""
        if len(self.dataset) == 0:
            raise ValueError("Dataset is empty!")

        first_sample = self.dataset[0]

        print("\n" + "="*60)
        print("Dataset Structure Inspection")
        print("="*60)
        print(f"Dataset size: {len(self.dataset)}")
        print(f"Sample keys: {list(first_sample.keys())}")
        print(f"Sample data types:")
        for key, value in first_sample.items():
            if isinstance(value, (list, tuple)):
                print(f"  {key}: {type(value).__name__} of length {len(value)}")
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
        raise KeyError(f"Could not find image2 key. Available keys: {list(sample.keys())}")

    @staticmethod
    def _detect_caption_key(sample: Dict) -> str:
        """Detect the key for caption/description"""
        candidates = ['caption', 'text', 'description', 'change_description']
        for key in candidates:
            if key in sample:
                return key
        # Default to first non-image key if no caption found
        for key in sample.keys():
            if key not in ['image', 'A', 'img', 'image1', 'image2', 'B', 'img2', 'image_2', 'bbox', 'bboxes']:
                return key
        raise KeyError(f"Could not find caption key. Available keys: {list(sample.keys())}")

    @staticmethod
    def _detect_bbox_key(sample: Dict) -> str:
        """Detect the key for bounding box"""
        candidates = ['bbox', 'bboxes', 'bounding_box', 'box']
        for key in candidates:
            if key in sample:
                return key
        raise KeyError(f"Could not find bbox key. Available keys: {list(sample.keys())}")

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
            caption = str(sample[self.caption_key])
            if not caption or caption.lower() in ['none', 'nan', '']:
                caption = "change detected"
        except Exception as e:
            print(f"⚠️  Error loading caption at index {idx}: {e}")
            caption = "change detected"

        # Get and process bbox
        try:
            bbox = sample[self.bbox_key]
            action_vector = self._process_bbox(bbox, image_t1.size)
        except Exception as e:
            print(f"⚠️  Warning: Could not process bbox at index {idx}: {e}")
            # Default action vector if bbox fails
            action_vector = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32)
            bbox = [0, 0, 224, 224]

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
        """
        if isinstance(image_data, Image.Image):
            return image_data.convert('RGB')
        elif isinstance(image_data, bytes):
            return Image.open(io.BytesIO(image_data)).convert('RGB')
        elif isinstance(image_data, (str, Path)):
            return Image.open(image_data).convert('RGB')
        else:
            raise TypeError(f"Unsupported image format: {type(image_data)}")

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


def create_dataloaders(
    batch_size: int = 4,
    num_workers: int = 4,
    test_split: float = 0.1,
    seed: int = 42,
):
    """
    Create train and validation dataloaders from LEVIR-CC dataset

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
    print("Loading LEVIR-CC Dataset from Arrow Format")
    print("="*60)
    print(f"Dataset path: {Config.DATASET_PATH}")

    try:
        # Load dataset from disk (Arrow format)
        full_dataset = datasets.load_from_disk(Config.DATASET_PATH)
        print(f"✅ Dataset loaded successfully")
        print(f"   Dataset structure: {full_dataset}")

        # Get the appropriate split
        if 'train' in full_dataset:
            dataset_split = full_dataset['train']
            print(f"✅ Using 'train' split with {len(dataset_split)} samples")
        else:
            # If only one split exists, use it
            dataset_split = full_dataset
            print(f"⚠️  No 'train' split found. Using entire dataset with {len(dataset_split)} samples")

    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        raise

    # Create train/val split
    split_dataset = dataset_split.train_test_split(
        test_size=test_split,
        seed=seed
    )

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

    # Create dataloaders
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

    print(f"\n✅ Dataloaders created:")
    print(f"   Train: {len(train_loader)} batches ({len(train_dataset)} samples)")
    print(f"   Val: {len(val_loader)} batches ({len(val_dataset)} samples)")
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

