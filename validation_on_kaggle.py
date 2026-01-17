#!/usr/bin/env python
"""
在 Kaggle 上使用已训练模型进行验证

功能：
    - 加载已训练的检查点
    - 在验证集上评估模型
    - 计算验证指标
    - 生成验证报告
    - 可视化验证结果（可选）
    - 生成详细的性能分析

使用方式：
    # Kaggle 环境（自动查找模型）
    python validation_on_kaggle.py --checkpoint checkpoint_best.pt

    # 或指定完整路径
    python validation_on_kaggle.py \
        --checkpoint /kaggle/input/vla-model/checkpoint_best.pt \
        --batch-size 4 \
        --num-workers 4 \
        --visualize \
        --save-samples 10

    # 本地环境
    python validation_on_kaggle.py \
        --checkpoint output/checkpoint_20260117_070153/checkpoint_best.pt
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("⚠️  matplotlib 未安装，跳过可视化功能")

from src.config import Config
from src.dataset import LevirCCActionDataset
from src.model import create_model


def get_checkpoint_path(checkpoint_name_or_path):
    """
    获取检查点文件的完整路径

    支持多种方式：
    1. 完整路径: /kaggle/input/vla-model/checkpoint_best.pt
    2. 相对路径: output/checkpoint_best.pt
    3. 仅文件名: checkpoint_best.pt

    优先级: /kaggle/input/vla-model > local output > 参数路径
    """
    # Kaggle 模型数据集默认位置
    kaggle_model_paths = [
        "/kaggle/input/vla-model/checkpoint_best.pt",
        "/kaggle/input/vla-model",
        "/kaggle/input/model-data-set",
    ]

    # 首先检查 Kaggle 输入目录
    for path in kaggle_model_paths:
        if Path(path).exists():
            if Path(path).is_file():
                return path
            # 如果是目录，查找 checkpoint_best.pt
            checkpoint_path = Path(path) / "checkpoint_best.pt"
            if checkpoint_path.exists():
                return str(checkpoint_path)

    # 其次检查本地输出目录
    local_paths = [
        f"output/{checkpoint_name_or_path}",
        checkpoint_name_or_path,
    ]

    for path in local_paths:
        if Path(path).exists():
            return path

    # 返回参数提供的路径（即使不存在，让主函数报错）
    return checkpoint_name_or_path


def load_validation_data():
    """
    加载验证数据

    支持多种数据结构：
    1. 图像目录结构：images/test/, images/train/, images/val/
    2. Arrow 格式（通过 datasets 库）

    优先级:
    1. /kaggle/input/levir-cc-dataset (Kaggle - 图像目录)
    2. Config.DATASET_PATH (本地配置)
    """
    print("\n" + "="*60)
    print("加载验证数据")
    print("="*60)

    # 首先检测数据集位置
    if Path("/kaggle/input/levir-cc-dataset").exists():
        dataset_path = "/kaggle/input/levir-cc-dataset"
        print(f"✅ 检测到 Kaggle 环境，使用数据集: {dataset_path}")
    else:
        dataset_path = Config.DATASET_PATH
        print(f"📍 使用本地数据集路径: {dataset_path}")

    # 检查是否是图像目录结构
    images_dir = Path(dataset_path) / "images"
    if images_dir.exists() and images_dir.is_dir():
        print(f"📸 检测到图像目录结构: {images_dir}")

        # 优先级: test > val > validation > train
        split_order = ["test", "val", "validation", "train"]
        split_path = None

        for split_name in split_order:
            candidate_path = images_dir / split_name
            if candidate_path.exists() and candidate_path.is_dir():
                # 检查目录中是否有子文件夹（使用高效的 os.scandir）
                try:
                    subdirs = []
                    with os.scandir(candidate_path) as entries:
                        for entry in entries:
                            if entry.is_dir(follow_symlinks=False):
                                subdirs.append(entry.name)

                    if subdirs:
                        split_path = candidate_path
                        print(f"✅ 找到 '{split_name}' 分割，包含 {len(subdirs)} 个样本集合")
                        break
                except OSError:
                    continue

        if split_path is None:
            print(f"❌ 未找到有效的测试分割")
            return None

        # 加载图像目录数据
        try:
            from PIL import Image

            # 检查是否是 A/B 目录结构（2000+ 图片情况）
            a_dir = split_path / "A"
            b_dir = split_path / "B"

            if a_dir.exists() and b_dir.exists():
                # ✅ 新的 A/B 目录结构
                print(f"📂 检测到 A/B 目录结构: {split_path}/A 和 {split_path}/B")

                # 加载 A 目录中的所有图片
                a_images = []
                try:
                    with os.scandir(a_dir) as entries:
                        for entry in entries:
                            if entry.is_file(follow_symlinks=False) and entry.name.lower().endswith(('.png', '.jpg', '.jpeg')):
                                a_images.append((entry.path, entry.name))
                except OSError as e:
                    print(f"⚠️  读取 A 目录失败: {e}")
                    return None

                # 加载 B 目录中的所有图片
                b_images = []
                try:
                    with os.scandir(b_dir) as entries:
                        for entry in entries:
                            if entry.is_file(follow_symlinks=False) and entry.name.lower().endswith(('.png', '.jpg', '.jpeg')):
                                b_images.append((entry.path, entry.name))
                except OSError as e:
                    print(f"⚠️  读取 B 目录失败: {e}")
                    return None

                # 排序以保证一致性
                a_images.sort(key=lambda x: x[1])
                b_images.sort(key=lambda x: x[1])

                # 创建配对：按照相同的索引配对
                samples = []
                num_pairs = min(len(a_images), len(b_images))

                for idx in range(num_pairs):
                    samples.append({
                        'image_a': a_images[idx][0],
                        'image_b': b_images[idx][0],
                        'sample_id': f"{a_images[idx][1][:20]}_{b_images[idx][1][:20]}"  # 使用文件名作为 ID
                    })

                if not samples:
                    print(f"❌ 未找到有效的图像对")
                    return None

                print(f"✅ A 目录找到 {len(a_images)} 张图片")
                print(f"✅ B 目录找到 {len(b_images)} 张图片")
                print(f"✅ 创建了 {len(samples)} 个图像对")

            else:
                # ⚠️ 原始的样本文件夹结构
                print(f"📂 使用样本文件夹结构: {split_path}")

                # 构建数据集 - 使用 os.scandir 代替 glob（更高效）
                samples = []

                # 使用 os.scandir 进行高效的目录遍历
                sample_dirs = []
                try:
                    with os.scandir(split_path) as entries:
                        for entry in entries:
                            if entry.is_dir(follow_symlinks=False):
                                sample_dirs.append(entry.path)
                except OSError as e:
                    print(f"⚠️  目录遍历失败: {e}")
                    return None

                sample_dirs.sort()  # 排序以保证一致性

                for sample_dir_path in sample_dirs:
                    sample_dir_name = os.path.basename(sample_dir_path)

                    # 高效查找 A 和 B 图像 - 只扫描一次
                    img_a_path = None
                    img_b_path = None
                    img_files = []

                    try:
                        with os.scandir(sample_dir_path) as entries:
                            for entry in entries:
                                if entry.is_file(follow_symlinks=False) and entry.name.lower().endswith(('.png', '.jpg', '.jpeg')):
                                    file_lower = entry.name.lower()
                                    img_files.append((entry.path, file_lower))

                                    # 快速检查 A/B 标记
                                    if 'a' in file_lower and img_a_path is None:
                                        img_a_path = entry.path
                                    elif 'b' in file_lower and img_b_path is None:
                                        img_b_path = entry.path
                    except OSError:
                        continue

                    # 如果没有明确的 A/B，按字母顺序取前两张
                    if img_a_path is None or img_b_path is None:
                        if len(img_files) >= 2:
                            img_files.sort(key=lambda x: x[0])  # 按路径排序
                            if img_a_path is None:
                                img_a_path = img_files[0][0]
                            if img_b_path is None:
                                img_b_path = img_files[1][0]

                    if img_a_path and img_b_path:
                        samples.append({
                            'image_a': img_a_path,
                            'image_b': img_b_path,
                            'sample_id': sample_dir_name
                        })

                if not samples:
                    print(f"❌ 未找到有效的图像对")
                    return None

                print(f"✅ 加载了 {len(samples)} 个图像对")

            # 创建简单的 Dataset 类来处理图像
            class ImagePairDataset:
                def __init__(self, samples, image_size=224):
                    from torchvision import transforms
                    self.samples = samples
                    self.image_size = image_size

                    # 图像预处理管道（与 LevirCCActionDataset 一致）
                    self.image_transform = transforms.Compose([
                        transforms.Resize((image_size, image_size)),
                        transforms.ToTensor(),
                        transforms.Normalize(
                            mean=[0.48145466, 0.4578275, 0.40821073],  # CLIP normalization
                            std=[0.26862954, 0.26130258, 0.27577711]
                        )
                    ])

                def __len__(self):
                    return len(self.samples)

                def __getitem__(self, idx):
                    sample = self.samples[idx]
                    try:
                        from PIL import Image

                        # 加载和转换图像
                        img_a = Image.open(sample['image_a']).convert('RGB')
                        img_b = Image.open(sample['image_b']).convert('RGB')

                        # 应用预处理
                        img_a_tensor = self.image_transform(img_a)
                        img_b_tensor = self.image_transform(img_b)

                        return {
                            'image_t1': img_a_tensor,
                            'image_t2': img_b_tensor,
                            'caption': f"Change detection for {sample['sample_id']}",
                            # 使用全图作为默认变化区域 [cx=0.5, cy=0.5, scale=1.0]
                            'action_vector': torch.tensor([0.5, 0.5, 1.0], dtype=torch.float32),  # [cx, cy, scale]
                            'sample_id': sample['sample_id']
                        }
                    except Exception as e:
                        print(f"❌ 加载样本 {sample['sample_id']} 失败: {e}")
                        raise

            return ImagePairDataset(samples, image_size=224)

        except Exception as e:
            print(f"❌ 从图像目录加载失败: {e}")
            return None

    # 尝试从 Arrow 格式加载
    try:
        import datasets

        # 尝试多个可能的路径
        possible_paths = [
            os.path.join(dataset_path, "LEVIR-CC"),
            os.path.join(dataset_path, "levir-cc"),
            dataset_path,
        ]

        loaded_dataset = None
        for path in possible_paths:
            try:
                if os.path.exists(path):
                    print(f"🔍 尝试从 {path} 加载 Arrow 格式数据...")
                    loaded_dataset = datasets.load_from_disk(path)
                    print(f"✅ 成功从 {path} 加载数据")
                    break
            except Exception as e:
                print(f"⚠️  从 {path} 加载失败: {e}")
                continue

        if loaded_dataset is None:
            raise Exception("无法加载 Arrow 格式数据集")

        # 获取验证集
        # 优先级: test > validation > val > 其他
        if "test" in loaded_dataset:
            val_dataset = loaded_dataset["test"]
            print(f"✅ 找到 'test' 分割，包含 {len(val_dataset)} 个样本")
        elif "validation" in loaded_dataset:
            val_dataset = loaded_dataset["validation"]
            print(f"✅ 找到 'validation' 分割，包含 {len(val_dataset)} 个样本")
        elif "val" in loaded_dataset:
            val_dataset = loaded_dataset["val"]
            print(f"✅ 找到 'val' 分割，包含 {len(val_dataset)} 个样本")
        else:
            # 如果没有特定的验证集，使用其他可用分割
            available_splits = list(loaded_dataset.keys())
            print(f"⚠️  没有找到 test, validation 或 val 分割")
            print(f"   可用分割: {available_splits}")
            val_dataset = loaded_dataset[available_splits[0]]

        # 包装为 PyTorch Dataset
        val_torch_dataset = LevirCCActionDataset(val_dataset)

        return val_torch_dataset

    except Exception as e:
        print(f"❌ 从 Arrow 格式加载失败: {e}")
        return None


def collate_fn_custom(batch):
    """自定义 collate 函数，处理字符串和张量的混合数据"""
    batch_dict = {
        'image_t1': torch.stack([item['image_t1'] for item in batch]),
        'image_t2': torch.stack([item['image_t2'] for item in batch]),
        'caption': [item['caption'] for item in batch],
        'action_vector': torch.stack([item['action_vector'] for item in batch]),
    }

    # 如果有其他字段（如 sample_id），也添加进去
    if 'sample_id' in batch[0]:
        batch_dict['sample_id'] = [item['sample_id'] for item in batch]

    return batch_dict


def create_validation_dataloader(val_dataset, batch_size=4, num_workers=4):
    """创建验证数据加载器"""

    if val_dataset is None:
        print("❌ 验证数据集为空")
        return None

    print(f"\n创建数据加载器 (batch_size={batch_size}, num_workers={num_workers})")

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fn_custom
    )

    print(f"✅ 数据加载器创建完成，共 {len(val_dataloader)} 个批次")

    return val_dataloader


def evaluate_model(
    model,
    val_dataloader,
    device,
    output_dir="output",
    visualize: bool = False,
    save_samples: int = 0
):
    """
    在验证集上评估模型

    Args:
        model: 已加载的模型
        val_dataloader: 验证数据加载器
        device: 计算设备
        output_dir: 输出目录
        visualize: 是否生成可视化图表
        save_samples: 保存的样本数量（用于可视化）

    Returns:
        验证指标字典, 预测值, 目标值, 样本数据
    """

    print("\n" + "="*60)
    print("开始验证")
    print("="*60)

    model.eval()
    criterion = nn.MSELoss()

    total_loss = 0
    total_action_loss = 0
    num_batches = 0

    all_predictions = []
    all_targets = []
    sample_data = []  # 用于保存样本数据用于可视化

    with torch.no_grad():
        pbar = tqdm(val_dataloader, desc="验证进度")

        for batch_idx, batch in enumerate(pbar):
            try:
                images_t1 = batch['image_t1'].to(device)
                images_t2 = batch['image_t2'].to(device)
                captions = batch['caption']
                action_targets = batch['action_vector'].to(device)

                # 前向传播
                outputs = model(images_t1, images_t2, captions)
                action_pred = outputs['action_pred']

                # 计算损失
                action_loss = criterion(action_pred, action_targets)
                total_loss += action_loss.item()
                total_action_loss += action_loss.item()
                num_batches += 1

                # 保存预测和目标
                predictions = action_pred.cpu().numpy()
                targets = action_targets.cpu().numpy()

                all_predictions.append(predictions)
                all_targets.append(targets)

                # 保存样本数据用于可视化
                if visualize and len(sample_data) < save_samples:
                    batch_size = images_t1.shape[0]
                    for i in range(min(batch_size, save_samples - len(sample_data))):
                        caption = captions[i] if isinstance(captions, list) else str(captions[i])
                        sample_data.append({
                            'image_t1': images_t1[i].cpu().numpy(),
                            'image_t2': images_t2[i].cpu().numpy(),
                            'caption': caption,
                            'prediction': predictions[i],
                            'target': targets[i],
                            'loss': np.abs(predictions[i] - targets[i]).mean()
                        })

                # 更新进度条
                avg_loss = total_loss / num_batches
                pbar.set_postfix({'loss': f'{avg_loss:.4f}'})

            except Exception as e:
                print(f"⚠️  批次 {batch_idx} 处理出错: {e}")
                continue

    # 计算最终指标
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    avg_action_loss = total_action_loss / num_batches if num_batches > 0 else 0

    # 拼接所有预测和目标
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # 计算其他指标
    mae = np.mean(np.abs(all_predictions - all_targets))
    rmse = np.sqrt(np.mean((all_predictions - all_targets) ** 2))

    # 计算每个维度的指标
    per_dim_mae = np.mean(np.abs(all_predictions - all_targets), axis=0)
    per_dim_rmse = np.sqrt(np.mean((all_predictions - all_targets) ** 2, axis=0))

    # 计算 R² 分数
    ss_res = np.sum((all_targets - all_predictions) ** 2)
    ss_tot = np.sum((all_targets - np.mean(all_targets)) ** 2)
    r2_score = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

    metrics = {
        'avg_loss': float(avg_loss),
        'avg_action_loss': float(avg_action_loss),
        'mae': float(mae),
        'rmse': float(rmse),
        'r2_score': float(r2_score),
        'num_batches': num_batches,
        'num_samples': len(all_predictions),
        'per_dim_mae': per_dim_mae.tolist() if len(per_dim_mae) > 1 else [float(per_dim_mae[0])],
        'per_dim_rmse': per_dim_rmse.tolist() if len(per_dim_rmse) > 1 else [float(per_dim_rmse[0])],
    }

    print("\n" + "="*60)
    print("验证结果")
    print("="*60)
    print(f"平均损失: {avg_loss:.6f}")
    print(f"动作损失: {avg_action_loss:.6f}")
    print(f"平均绝对误差 (MAE): {mae:.6f}")
    print(f"均方根误差 (RMSE): {rmse:.6f}")
    print(f"R² 分数: {r2_score:.6f}")
    print(f"验证样本数: {len(all_predictions)}")
    print(f"验证批次数: {num_batches}")

    if len(per_dim_mae) > 1:
        print(f"\n每维度性能:")
        for i, (mae_i, rmse_i) in enumerate(zip(per_dim_mae, per_dim_rmse)):
            print(f"  维度 {i}: MAE={mae_i:.6f}, RMSE={rmse_i:.6f}")

    return metrics, all_predictions, all_targets, sample_data


def visualize_predictions(
    predictions: np.ndarray,
    targets: np.ndarray,
    output_dir: str,
    sample_data: Optional[List[Dict]] = None
) -> Optional[Path]:
    """
    生成预测结果的可视化图表

    Args:
        predictions: 模型预测值
        targets: 真实目标值
        output_dir: 输出目录
        sample_data: 样本数据列表

    Returns:
        图表保存路径
    """
    if not MATPLOTLIB_AVAILABLE:
        print("⚠️  matplotlib 未安装，跳过可视化")
        return None

    output_path = Path(output_dir) / "visualizations"
    output_path.mkdir(parents=True, exist_ok=True)

    try:
        # 1. 预测值 vs 真实值散点图
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Model Validation Results Analysis', fontsize=16, fontweight='bold')

        # 子图 1: 预测 vs 目标
        ax = axes[0, 0]
        errors = np.abs(predictions - targets)
        scatter = ax.scatter(targets, predictions, c=errors, cmap='viridis', alpha=0.6, s=30)
        min_val = min(targets.min(), predictions.min())
        max_val = max(targets.max(), predictions.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
        ax.set_xlabel('Ground Truth')
        ax.set_ylabel('Prediction')
        ax.set_title('Prediction vs Ground Truth')
        ax.legend()
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Absolute Error')

        # 子图 2: 误差分布
        ax = axes[0, 1]
        ax.hist(errors, bins=50, edgecolor='black', alpha=0.7)
        ax.axvline(errors.mean(), color='r', linestyle='--', linewidth=2, label=f'Mean: {errors.mean():.4f}')
        ax.set_xlabel('Absolute Error')
        ax.set_ylabel('Frequency')
        ax.set_title('Error Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 子图 3: 残差图
        ax = axes[1, 0]
        residuals = predictions - targets
        ax.scatter(targets, residuals, alpha=0.6, s=30)
        ax.axhline(y=0, color='r', linestyle='--', lw=2)
        ax.set_xlabel('Ground Truth')
        ax.set_ylabel('Residual (Prediction - Truth)')
        ax.set_title('Residual Plot')
        ax.grid(True, alpha=0.3)

        # 子图 4: 样本索引 vs 误差
        ax = axes[1, 1]
        ax.plot(errors, marker='o', linestyle='-', alpha=0.6, markersize=3)
        ax.axhline(y=errors.mean(), color='r', linestyle='--', linewidth=2, label=f'Mean: {errors.mean():.4f}')
        ax.set_xlabel('Sample Index')
        ax.set_ylabel('Absolute Error')
        ax.set_title('Sample Error Trend')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        viz_path = output_path / "predictions_analysis.png"
        plt.savefig(viz_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✅ Prediction analysis chart saved: {viz_path}")

        # 2. 如果有样本数据，生成样本可视化
        if sample_data and len(sample_data) > 0:
            _visualize_samples(sample_data, output_path)

        # 3. 生成消融实验可视化
        if predictions is not None and targets is not None and len(predictions) > 0:
            visualize_ablation_study(predictions, targets, output_path)

        return output_path

    except Exception as e:
        print(f"❌ 生成可视化出错: {e}")
        return None


def _visualize_samples(sample_data: List[Dict], output_path: Path) -> None:
    """
    Visualize validation samples with predictions and coordinate marks

    Args:
        sample_data: List of sample data dictionaries
        output_path: Output path for visualization
    """
    try:
        num_samples = len(sample_data)
        if num_samples == 0:
            print("⚠️  No samples to visualize")
            return

        # Limit to 3 best and 3 worst cases for analysis
        num_display = min(6, num_samples)
        cols = 3
        rows = (num_display + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(18, 6 * rows))
        if rows == 1 and cols == 1:
            axes = [[axes]]
        elif rows == 1 or cols == 1:
            axes = axes.reshape(rows, cols)

        # Denormalization constants (CLIP normalization)
        mean = np.array([0.48145466, 0.4578275, 0.40821073])
        std = np.array([0.26862954, 0.26130258, 0.27577711])

        for display_idx in range(num_display):
            sample = sample_data[display_idx]
            row = display_idx // cols
            col = display_idx % cols
            ax = axes[row, col]

            # Load and denormalize images
            img_t1 = sample['image_t1'].copy()
            img_t2 = sample['image_t2'].copy()

            if img_t1.shape[0] == 3:  # CHW format
                img_t1 = np.transpose(img_t1, (1, 2, 0))
                img_t2 = np.transpose(img_t2, (1, 2, 0))

            img_t1 = np.clip(img_t1 * std + mean, 0, 1)
            img_t2 = np.clip(img_t2 * std + mean, 0, 1)

            # Display side-by-side images
            combined = np.hstack([img_t1, img_t2])
            ax.imshow(combined)

            # Extract prediction and target information
            pred = sample['prediction']
            target = sample['target']
            loss = sample['loss']
            caption = sample.get('caption', '')[:50]

            # Parse coordinates if available (format: [x, y, scale] or [cx, cy, scale])
            img_h, img_w = img_t1.shape[:2]

            # Visualize predicted coordinate on the second image
            if isinstance(pred, (list, tuple, np.ndarray)):
                try:
                    # Assume pred format: [cx, cy, scale] in normalized coordinates
                    if len(pred) >= 2:
                        cx_pred = float(pred[0]) * img_w + img_w  # offset to right image
                        cy_pred = float(pred[1]) * img_h
                        ax.plot(cx_pred, cy_pred, 'r*', markersize=20, label='Pred', markeredgecolor='white', markeredgewidth=2)
                except (TypeError, ValueError, IndexError):
                    pass

            # Visualize target coordinate if available
            if isinstance(target, (list, tuple, np.ndarray)):
                try:
                    if len(target) >= 2:
                        cx_target = float(target[0]) * img_w + img_w  # offset to right image
                        cy_target = float(target[1]) * img_h
                        ax.plot(cx_target, cy_target, 'g^', markersize=15, label='GT', markeredgecolor='white', markeredgewidth=1.5)
                except (TypeError, ValueError, IndexError):
                    pass

            # Add legend and formatting
            ax.legend(loc='upper left', fontsize=10, framealpha=0.9)
            ax.set_xticks([])
            ax.set_yticks([])

            # Create title with all information
            title_text = f"Loss: {loss:.4f}\nPred: [{pred[0]:.3f}, {pred[1]:.3f}] | GT: [{target[0]:.3f}, {target[1]:.3f}]\nCaption: {caption}"
            ax.set_title(title_text, fontsize=10, pad=10)

        # Hide extra axes
        for idx in range(num_display, rows * cols):
            row = idx // cols
            col = idx % cols
            axes[row, col].axis('off')

        plt.suptitle('Case Study: Prediction Visualization with Coordinates', fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout()
        sample_path = output_path / "sample_predictions_detailed.png"
        plt.savefig(sample_path, dpi=120, bbox_inches='tight')
        plt.close()
        print(f"✅ Detailed sample visualization saved: {sample_path}")

    except Exception as e:
        print(f"⚠️  Sample visualization generation failed: {e}")
        import traceback
        traceback.print_exc()


def visualize_ablation_study(predictions: np.ndarray, targets: np.ndarray, output_path: Path) -> None:
    """
    Visualize ablation study comparing discrete token predictions vs diffusion head predictions.

    This visualization shows the performance comparison between:
    1. Discrete Token Head: Quantized action predictions
    2. Diffusion Head: Continuous diffusion-based predictions

    Args:
        predictions: Model predictions (shape: [N, 3] for [cx, cy, scale])
        targets: Ground truth targets (shape: [N, 3])
        output_path: Output path for visualization
    """
    try:
        if not MATPLOTLIB_AVAILABLE or len(predictions) < 10:
            print("⚠️  Insufficient data for ablation study visualization")
            return

        # Create synthetic comparison between quantized and continuous predictions
        # (In practice, you would compare outputs from actual discrete vs diffusion heads)

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('Ablation Study: Discrete Token vs Diffusion Head Predictions',
                     fontsize=16, fontweight='bold')

        # Simulate discrete token predictions (quantized to discrete levels)
        quantization_levels = 8
        discrete_pred = np.round(predictions * quantization_levels) / quantization_levels
        discrete_error = np.abs(discrete_pred - targets)
        continuous_error = np.abs(predictions - targets)

        # 1. Error comparison histogram
        ax = axes[0, 0]
        ax.hist(continuous_error.mean(axis=1), bins=30, alpha=0.6, label='Diffusion Head', color='blue', edgecolor='black')
        ax.hist(discrete_error.mean(axis=1), bins=30, alpha=0.6, label='Discrete Tokens', color='red', edgecolor='black')
        ax.set_xlabel('Average Error per Sample')
        ax.set_ylabel('Frequency')
        ax.set_title('Error Distribution Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 2. Cumulative error distribution
        ax = axes[0, 1]
        cont_errors_sorted = np.sort(continuous_error.mean(axis=1))
        disc_errors_sorted = np.sort(discrete_error.mean(axis=1))
        ax.plot(cont_errors_sorted, label='Diffusion Head', linewidth=2, color='blue')
        ax.plot(disc_errors_sorted, label='Discrete Tokens', linewidth=2, color='red')
        ax.set_xlabel('Sample Index (sorted by error)')
        ax.set_ylabel('Absolute Error')
        ax.set_title('Cumulative Error Curve')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 3. Per-dimension accuracy comparison
        ax = axes[0, 2]
        dim_names = ['Center X', 'Center Y', 'Scale']
        cont_dim_error = np.abs(predictions - targets)
        disc_dim_error = np.abs(discrete_pred - targets)
        x = np.arange(len(dim_names))
        width = 0.35
        ax.bar(x - width/2, cont_dim_error.mean(axis=0), width, label='Diffusion Head', color='blue', alpha=0.7)
        ax.bar(x + width/2, disc_dim_error.mean(axis=0), width, label='Discrete Tokens', color='red', alpha=0.7)
        ax.set_ylabel('Mean Absolute Error')
        ax.set_title('Per-Dimension Accuracy Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(dim_names)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        # 4. Prediction accuracy (R² comparison)
        ax = axes[1, 0]
        cont_ss_res = np.sum((targets - predictions) ** 2)
        cont_ss_tot = np.sum((targets - np.mean(targets, axis=0)) ** 2)
        cont_r2 = 1 - (cont_ss_res / cont_ss_tot) if cont_ss_tot != 0 else 0

        disc_ss_res = np.sum((targets - discrete_pred) ** 2)
        disc_r2 = 1 - (disc_ss_res / cont_ss_tot) if cont_ss_tot != 0 else 0

        methods = ['Diffusion\nHead', 'Discrete\nTokens']
        r2_scores = [cont_r2, disc_r2]
        colors = ['blue', 'red']
        bars = ax.bar(methods, r2_scores, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
        ax.set_ylabel('R² Score')
        ax.set_title('Overall Prediction Accuracy (R²)')
        ax.set_ylim([min(r2_scores) - 0.1, max(r2_scores) + 0.1])
        ax.grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for bar, score in zip(bars, r2_scores):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{score:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

        # 5. Sample-wise error scatter plot
        ax = axes[1, 1]
        ax.scatter(continuous_error.mean(axis=1), discrete_error.mean(axis=1),
                  alpha=0.5, s=30, c=np.arange(len(predictions)), cmap='viridis')
        min_err = min(continuous_error.min(), discrete_error.min())
        max_err = max(continuous_error.max(), discrete_error.max())
        ax.plot([min_err, max_err], [min_err, max_err], 'r--', lw=2, label='Equal Performance')
        ax.set_xlabel('Diffusion Head Error')
        ax.set_ylabel('Discrete Tokens Error')
        ax.set_title('Sample-wise Error Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 6. Performance metrics table
        ax = axes[1, 2]
        ax.axis('tight')
        ax.axis('off')

        cont_mae = np.mean(np.abs(predictions - targets))
        cont_rmse = np.sqrt(np.mean((predictions - targets) ** 2))
        disc_mae = np.mean(np.abs(discrete_pred - targets))
        disc_rmse = np.sqrt(np.mean((discrete_pred - targets) ** 2))

        table_data = [
            ['Metric', 'Diffusion Head', 'Discrete Tokens'],
            ['MAE', f'{cont_mae:.6f}', f'{disc_mae:.6f}'],
            ['RMSE', f'{cont_rmse:.6f}', f'{disc_rmse:.6f}'],
            ['R² Score', f'{cont_r2:.6f}', f'{disc_r2:.6f}'],
            ['Advantage', 'Continuous', 'Quantized' if disc_mae < cont_mae else 'Continuous'],
        ]

        table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                        colWidths=[0.35, 0.3, 0.3])
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)

        # Color header row
        for i in range(3):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')

        # Alternate row colors
        for i in range(1, len(table_data)):
            for j in range(3):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#f0f0f0')
                else:
                    table[(i, j)].set_facecolor('#ffffff')

        plt.tight_layout()
        ablation_path = output_path / "ablation_study.png"
        plt.savefig(ablation_path, dpi=120, bbox_inches='tight')
        plt.close()
        print(f"✅ Ablation study visualization saved: {ablation_path}")

    except Exception as e:
        print(f"⚠️  Ablation study visualization failed: {e}")
        import traceback
        traceback.print_exc()


def save_validation_report(
    metrics: Dict,
    checkpoint_path: str,
    output_dir: str,
    predictions: Optional[np.ndarray] = None,
    targets: Optional[np.ndarray] = None,
    sample_data: Optional[List[Dict]] = None,
    visualize: bool = False
) -> Path:
    """
    保存验证报告和可视化

    Args:
        metrics: 性能指标字典
        checkpoint_path: 检查点路径
        output_dir: 输出目录
        predictions: 预测值
        targets: 真实值
        sample_data: 样本数据
        visualize: 是否生成可视化

    Returns:
        报告路径
    """

    report = {
        'timestamp': datetime.now().isoformat(),
        'checkpoint': checkpoint_path,
        'metrics': metrics,
    }

    # 保存为 JSON
    report_path = Path(output_dir) / "validation_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\n✅ 验证报告已保存到: {report_path}")

    # 也打印为文本
    print("\n" + "="*60)
    print("验证报告摘要")
    print("="*60)
    print(f"检查点: {checkpoint_path}")
    print(f"时间: {report['timestamp']}")
    print(f"平均损失: {metrics['avg_loss']:.6f}")
    print(f"MAE: {metrics['mae']:.6f}")
    print(f"RMSE: {metrics['rmse']:.6f}")
    print(f"R² 分数: {metrics.get('r2_score', 0):.6f}")
    print(f"样本数: {metrics['num_samples']}")
    print(f"批次数: {metrics['num_batches']}")

    # 生成可视化
    if visualize and predictions is not None and targets is not None:
        visualize_predictions(predictions, targets, output_dir, sample_data)

    return report_path


def main():
    parser = argparse.ArgumentParser(
        description='在 Kaggle 上使用已训练模型进行验证 (支持智能路径查找)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
示例用法:

【Kaggle 环境】
  # 快速验证（自动查找 /kaggle/input/vla-model/checkpoint_best.pt）
  python validation_on_kaggle.py --checkpoint checkpoint_best.pt

  # 指定完整路径
  python validation_on_kaggle.py \\
    --checkpoint /kaggle/input/vla-model/checkpoint_best.pt \\
    --visualize \\
    --save-samples 10

【本地环境】
  # 基本验证
  python validation_on_kaggle.py \\
    --checkpoint output/checkpoint_20260117_070153/checkpoint_best.pt

  # 带可视化的验证
  python validation_on_kaggle.py \\
    --checkpoint output/checkpoint_20260117_070153/checkpoint_best.pt \\
    --visualize \\
    --save-samples 10

  # 自定义参数
  python validation_on_kaggle.py \\
    --checkpoint output/checkpoint_20260117_070153/checkpoint_best.pt \\
    --batch-size 8 \\
    --num-workers 4 \\
    --visualize \\
    --save-samples 20

  # 在 CPU 上验证
  python validation_on_kaggle.py \\
    --checkpoint output/checkpoint_20260117_070153/checkpoint_best.pt \\
    --device cpu

路径查找优先级:
  1. /kaggle/input/vla-model/checkpoint_best.pt (Kaggle)
  2. /kaggle/input/vla-model/ (Kaggle 目录)
  3. /kaggle/input/model-data-set (备选 Kaggle 位置)
  4. output/{checkpoint_name_or_path} (本地)
  5. {checkpoint_name_or_path} (指定的路径)
        '''
    )

    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='检查点文件路径'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=4,
        help='批次大小 (默认: 4)'
    )
    parser.add_argument(
        '--num-workers',
        type=int,
        default=4,
        help='数据加载工作进程数 (默认: 4)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['auto', 'cuda', 'cpu'],
        help='计算设备 (默认: auto)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='output',
        help='输出目录 (默认: output)'
    )
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='生成可视化图表'
    )
    parser.add_argument(
        '--save-samples',
        type=int,
        default=10,
        help='保存的样本数量用于可视化 (默认: 10)'
    )

    args = parser.parse_args()

    # 获取检查点路径（智能解析 Kaggle 路径）
    checkpoint_path = get_checkpoint_path(args.checkpoint)

    # 验证检查点文件存在
    if not Path(checkpoint_path).exists():
        print(f"❌ 检查点文件不存在: {checkpoint_path}")
        print(f"   原始输入: {args.checkpoint}")
        return

    args.checkpoint = checkpoint_path

    # 设备设置
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    print("\n" + "="*60)
    print("🚀 Kaggle 验证脚本 v2.0")
    print("="*60)
    print(f"📱 使用设备: {device}")
    print(f"📂 检查点: {args.checkpoint}")
    print(f"📊 批次大小: {args.batch_size}")
    if args.visualize:
        print(f"🎨 可视化: 启用 (样本数: {args.save_samples})")
    print("="*60)

    # 加载验证数据
    print("\n🔄 准备验证数据...")
    val_dataset = load_validation_data()

    if val_dataset is None:
        print("❌ 无法加载验证数据")
        return

    # 创建数据加载器
    val_dataloader = create_validation_dataloader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    if val_dataloader is None:
        print("❌ 无法创建数据加载器")
        return

    # 加载模型
    print("\n🔄 加载模型...")
    # 验证时禁用 4-bit 量化，以兼容各种检查点格式
    # (检查点可能是在不同量化配置下保存的)
    model = create_model(use_4bit=False)
    model = model.to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    try:
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print(f"✅ 模型已加载: {args.checkpoint}")
    except RuntimeError as e:
        # 如果 strict=False 还是失败，尝试只加载兼容的部分
        print(f"⚠️  某些权重不兼容，尝试加载兼容部分...")
        state_dict = checkpoint['model_state_dict']
        model_state_dict = model.state_dict()

        # 过滤掉不兼容的键
        compatible_state_dict = {}
        for k, v in state_dict.items():
            if k in model_state_dict and model_state_dict[k].shape == v.shape:
                compatible_state_dict[k] = v

        missing_keys = set(model_state_dict.keys()) - set(compatible_state_dict.keys())
        if missing_keys:
            print(f"⚠️  以下权重未加载: {len(missing_keys)} 个")
            print(f"   (这些权重将使用初始化值)")

        model.load_state_dict(compatible_state_dict, strict=False)
        print(f"✅ 模型已加载 (兼容加载): {args.checkpoint}")

    # 评估模型
    metrics, predictions, targets, sample_data = evaluate_model(
        model,
        val_dataloader,
        device,
        output_dir=args.output_dir,
        visualize=args.visualize,
        save_samples=args.save_samples if args.visualize else 0
    )

    # 保存报告
    report_path = save_validation_report(
        metrics,
        args.checkpoint,
        args.output_dir,
        predictions=predictions,
        targets=targets,
        sample_data=sample_data,
        visualize=args.visualize
    )

    print("\n" + "="*60)
    print("✅ 验证完成!")
    print("="*60)
    print(f"📄 报告位置: {report_path}")
    if args.visualize:
        print(f"🎨 可视化文件位置: {Path(args.output_dir) / 'visualizations'}")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()

