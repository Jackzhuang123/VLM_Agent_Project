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
    python validation_on_kaggle.py \
        --checkpoint output/checkpoint_20260117_070153/checkpoint_best.pt \
        --batch-size 4 \
        --num-workers 4 \
        --visualize \
        --save-samples 10
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


def load_validation_data():
    """
    加载验证数据

    支持两种数据结构：
    1. Arrow 格式（通过 datasets 库）
    2. 图像目录 + JSON 标注（本地结构）
    """
    print("\n" + "="*60)
    print("加载验证数据")
    print("="*60)

    try:
        import datasets

        # 尝试从 Arrow 格式加载
        dataset_path = Config.DATASET_PATH

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
                    print(f"🔍 尝试从 {path} 加载数据...")
                    loaded_dataset = datasets.load_from_disk(path)
                    print(f"✅ 成功从 {path} 加载数据")
                    break
            except Exception as e:
                print(f"⚠️  从 {path} 加载失败: {e}")
                continue

        if loaded_dataset is None:
            raise Exception("无法加载数据集")

        # 获取验证集
        if "validation" in loaded_dataset:
            val_dataset = loaded_dataset["validation"]
            print(f"✅ 找到 'validation' 分割，包含 {len(val_dataset)} 个样本")
        elif "val" in loaded_dataset:
            val_dataset = loaded_dataset["val"]
            print(f"✅ 找到 'val' 分割，包含 {len(val_dataset)} 个样本")
        else:
            # 如果没有特定的验证集，使用测试集或全部数据
            available_splits = list(loaded_dataset.keys())
            print(f"⚠️  没有找到 validation 或 val 分割")
            print(f"   可用分割: {available_splits}")
            val_dataset = loaded_dataset[available_splits[0]]

        # 包装为 PyTorch Dataset
        val_torch_dataset = LevirCCActionDataset(val_dataset)

        return val_torch_dataset

    except Exception as e:
        print(f"❌ 从 Arrow 格式加载失败: {e}")
        print("⚠️  尝试从图像目录加载...")

        # 如果 Arrow 加载失败，返回 None，后续可以手动处理
        return None


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
        pin_memory=torch.cuda.is_available()
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
                        sample_data.append({
                            'image_t1': images_t1[i].cpu().numpy(),
                            'image_t2': images_t2[i].cpu().numpy(),
                            'caption': captions[i] if isinstance(captions, list) else captions[i].item(),
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
        fig.suptitle('模型验证结果可视化', fontsize=16, fontweight='bold')

        # 子图 1: 预测 vs 目标
        ax = axes[0, 0]
        errors = np.abs(predictions - targets)
        scatter = ax.scatter(targets, predictions, c=errors, cmap='viridis', alpha=0.6, s=30)
        min_val = min(targets.min(), predictions.min())
        max_val = max(targets.max(), predictions.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='完美预测')
        ax.set_xlabel('真实值')
        ax.set_ylabel('预测值')
        ax.set_title('预测值 vs 真实值')
        ax.legend()
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('绝对误差')

        # 子图 2: 误差分布
        ax = axes[0, 1]
        ax.hist(errors, bins=50, edgecolor='black', alpha=0.7)
        ax.axvline(errors.mean(), color='r', linestyle='--', linewidth=2, label=f'平均: {errors.mean():.4f}')
        ax.set_xlabel('绝对误差')
        ax.set_ylabel('频次')
        ax.set_title('误差分布')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 子图 3: 残差图
        ax = axes[1, 0]
        residuals = predictions - targets
        ax.scatter(targets, residuals, alpha=0.6, s=30)
        ax.axhline(y=0, color='r', linestyle='--', lw=2)
        ax.set_xlabel('真实值')
        ax.set_ylabel('残差 (预测 - 真实)')
        ax.set_title('残差图')
        ax.grid(True, alpha=0.3)

        # 子图 4: 样本索引 vs 误差
        ax = axes[1, 1]
        ax.plot(errors, marker='o', linestyle='-', alpha=0.6, markersize=3)
        ax.axhline(y=errors.mean(), color='r', linestyle='--', linewidth=2, label=f'平均: {errors.mean():.4f}')
        ax.set_xlabel('样本索引')
        ax.set_ylabel('绝对误差')
        ax.set_title('样本误差趋势')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        viz_path = output_path / "predictions_analysis.png"
        plt.savefig(viz_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✅ 预测分析图表已保存: {viz_path}")

        # 2. 如果有样本数据，生成样本可视化
        if sample_data and len(sample_data) > 0:
            _visualize_samples(sample_data, output_path)

        return output_path

    except Exception as e:
        print(f"❌ 生成可视化出错: {e}")
        return None


def _visualize_samples(sample_data: List[Dict], output_path: Path) -> None:
    """
    可视化验证样本

    Args:
        sample_data: 样本数据列表
        output_path: 输出路径
    """
    try:
        num_samples = len(sample_data)
        cols = min(3, num_samples)
        rows = (num_samples + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
        if rows == 1 and cols == 1:
            axes = [[axes]]
        elif rows == 1 or cols == 1:
            axes = axes.reshape(rows, cols)

        for idx, sample in enumerate(sample_data):
            row = idx // cols
            col = idx % cols
            ax = axes[row, col]

            # 显示两张图像
            img_t1 = sample['image_t1']
            img_t2 = sample['image_t2']

            # 反归一化图像（CLIP normalization）
            mean = np.array([0.48145466, 0.4578275, 0.40821073])
            std = np.array([0.26862954, 0.26130258, 0.27577711])

            if img_t1.shape[0] == 3:  # CHW format
                img_t1 = np.transpose(img_t1, (1, 2, 0))
                img_t2 = np.transpose(img_t2, (1, 2, 0))

            img_t1 = np.clip(img_t1 * std + mean, 0, 1)
            img_t2 = np.clip(img_t2 * std + mean, 0, 1)

            # 并排显示两张图像
            combined = np.hstack([img_t1, img_t2])
            ax.imshow(combined)
            ax.axis('off')

            # 获取预测和真实值
            pred = sample['prediction']
            target = sample['target']
            loss = sample['loss']
            caption = sample.get('caption', '')[:30]  # 截断标题

            # 添加标题
            title = f"损失: {loss:.4f}\n预测: {pred}\n真实: {target}\n文本: {caption}"
            ax.set_title(title, fontsize=9)

        # 隐藏额外的轴
        for idx in range(num_samples, rows * cols):
            row = idx // cols
            col = idx % cols
            axes[row, col].axis('off')

        plt.tight_layout()
        sample_path = output_path / "sample_predictions.png"
        plt.savefig(sample_path, dpi=100, bbox_inches='tight')
        plt.close()
        print(f"✅ 样本可视化已保存: {sample_path}")

    except Exception as e:
        print(f"⚠️  样本可视化生成失败: {e}")


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
        description='在 Kaggle 上使用已训练模型进行验证',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
示例用法:
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

    # 验证检查点文件存在
    if not Path(args.checkpoint).exists():
        print(f"❌ 检查点文件不存在: {args.checkpoint}")
        return

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
    model = create_model()
    model = model.to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✅ 模型已加载: {args.checkpoint}")

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

