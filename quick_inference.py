#!/usr/bin/env python3
"""
快速推理脚本 - 直接运行进行推理预测

使用方法：
    # 单个样本推理
    python quick_inference.py --image1 before.jpg --image2 after.jpg --caption "building change"

    # 使用最优检查点推理
    python quick_inference.py --use-best --image1 before.jpg --image2 after.jpg --caption "change"

    # 显示结果
    python quick_inference.py --visualize --image1 before.jpg --image2 after.jpg --caption "change"
"""

import argparse
import sys
from pathlib import Path

import torch

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from src.config import Config
from src.inference import VLMInference


def find_best_checkpoint():
    """查找最优的检查点"""
    output_dir = Path(Config.OUTPUT_DIR)
    checkpoints = list(output_dir.glob("checkpoint_*/checkpoint_best.pt"))

    if not checkpoints:
        print("❌ 未找到检查点，请先运行训练")
        return None

    # 返回最新的检查点
    latest = sorted(checkpoints, key=lambda x: x.stat().st_mtime)[-1]
    return str(latest)


def main():
    parser = argparse.ArgumentParser(
        description='VLM-VLA Agent 快速推理脚本',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  # 基础推理
  python quick_inference.py --image1 before.jpg --image2 after.jpg --caption "building change"

  # 使用自动找到的最优模型
  python quick_inference.py --use-best --image1 before.jpg --image2 after.jpg --caption "change"

  # 可视化结果
  python quick_inference.py --visualize --image1 before.jpg --image2 after.jpg --caption "change" --output result.jpg
        """
    )

    # 检查点相关参数
    parser.add_argument(
        '--checkpoint',
        type=str,
        help='检查点路径（如果为空则使用 --use-best）'
    )
    parser.add_argument(
        '--use-best',
        action='store_true',
        help='自动使用最优检查点'
    )

    # 输入数据参数
    parser.add_argument(
        '--image1',
        type=str,
        required=True,
        help='时序图像 1 路径'
    )
    parser.add_argument(
        '--image2',
        type=str,
        required=True,
        help='时序图像 2 路径'
    )
    parser.add_argument(
        '--caption',
        type=str,
        required=True,
        help='变化描述文本'
    )

    # 输出参数
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='可视化推理结果'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='输出文件路径（用于可视化）'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='推理设备 (cuda/cpu)'
    )

    args = parser.parse_args()

    # 确定检查点
    if args.checkpoint:
        checkpoint_path = args.checkpoint
    elif args.use_best:
        checkpoint_path = find_best_checkpoint()
        if not checkpoint_path:
            return
    else:
        print("❌ 请指定 --checkpoint 或 --use-best")
        parser.print_help()
        return

    # 检查文件是否存在
    if not Path(args.image1).exists():
        print(f"❌ 图像 1 不存在: {args.image1}")
        return

    if not Path(args.image2).exists():
        print(f"❌ 图像 2 不存在: {args.image2}")
        return

    if not Path(checkpoint_path).exists():
        print(f"❌ 检查点不存在: {checkpoint_path}")
        return

    print("\n" + "="*60)
    print("VLM-VLA Agent 推理")
    print("="*60)
    print(f"检查点: {checkpoint_path}")
    print(f"图像 1: {args.image1}")
    print(f"图像 2: {args.image2}")
    print(f"文本: {args.caption}")
    print(f"设备: {args.device}")
    print("="*60 + "\n")

    # 初始化推理器
    try:
        inference = VLMInference(
            checkpoint_path=checkpoint_path,
            device=args.device
        )
    except Exception as e:
        print(f"❌ 初始化推理器失败: {e}")
        return

    # 执行推理
    print("正在推理...")
    try:
        result = inference.predict(
            args.image1,
            args.image2,
            args.caption
        )
    except Exception as e:
        print(f"❌ 推理失败: {e}")
        import traceback
        traceback.print_exc()
        return

    # 打印结果
    print("\n" + "="*60)
    print("推理结果")
    print("="*60)
    print(f"动作向量: {result['action_pred']}")
    print(f"  中心 X 坐标: {result['cx']:.6f}")
    print(f"  中心 Y 坐标: {result['cy']:.6f}")
    print(f"  尺度因子: {result['scale']:.6f}")
    print("="*60 + "\n")

    # 可视化结果
    if args.visualize:
        output_path = args.output or "inference_result.jpg"
        print(f"正在生成可视化结果...")

        try:
            vis_img = inference.visualize_result(
                args.image1,
                args.image2,
                result,
                output_path=output_path
            )
            print(f"✅ 可视化结果已保存到: {output_path}")
        except Exception as e:
            print(f"⚠️  可视化失败: {e}")

    print("✅ 推理完成！")


if __name__ == "__main__":
    main()

