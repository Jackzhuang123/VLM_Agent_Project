"""
推理脚本 - 使用训练好的 VLM-VLA Agent 模型进行预测

功能：
    1. 加载训练好的模型检查点
    2. 处理输入的图像对和文本
    3. 预测变化检测和动作向量
    4. 支持批处理推理
    5. 支持结果可视化
"""

import os
from pathlib import Path
from typing import Optional, Dict, Tuple, List

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .config import Config
from .model import VLM_ActionAgent
from .dataset import create_image_transform


class VLMInference:
    """
    VLM-VLA Agent 推理器

    用于加载训练好的模型并执行推理
    """

    def __init__(
        self,
        checkpoint_path: str,
        clip_path: str = None,
        llm_path: str = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        初始化推理器

        Args:
            checkpoint_path: 训练好的模型检查点路径
            clip_path: CLIP 模型路径（默认使用 Config 配置）
            llm_path: LLM 模型路径（默认使用 Config 配置）
            device: 推理设备（cuda 或 cpu）
        """
        self.device = torch.device(device)
        self.checkpoint_path = Path(checkpoint_path)

        print(f"\n{'='*60}")
        print("初始化推理器")
        print(f"{'='*60}")
        print(f"设备: {self.device}")
        print(f"检查点: {self.checkpoint_path}")

        # 加载模型
        self._load_model(
            clip_path or Config.CLIP_PATH,
            llm_path or Config.LLM_PATH
        )

        # 加载检查点
        self._load_checkpoint()

        # 设置为推理模式
        self.model.eval()

        # 初始化图像转换
        self.transform = create_image_transform(split='test')

        print(f"{'='*60}\n")

    def _load_model(self, clip_path: str, llm_path: str):
        """加载模型架构"""
        print("加载模型架构...")

        self.model = VLM_ActionAgent(
            clip_path=clip_path,
            llm_path=llm_path,
            freeze_vision=True,
            use_lora=True,
            use_4bit=Config.USE_4BIT,
        )

        self.model.to(self.device)
        print("✅ 模型架构加载完成")

    def _load_checkpoint(self):
        """加载训练好的权重"""
        print(f"加载检查点权重: {self.checkpoint_path}")

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"检查点不存在: {self.checkpoint_path}")

        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)

        # 加载模型权重
        self.model.load_state_dict(checkpoint['model_state_dict'])

        # 打印检查点信息
        print(f"✅ 检查点加载完成")
        print(f"   Epoch: {checkpoint.get('epoch', 'N/A')}")
        print(f"   Global step: {checkpoint.get('global_step', 'N/A')}")
        if 'metrics' in checkpoint and 'val' in checkpoint['metrics']:
            val_loss = checkpoint['metrics']['val']['loss']
            if val_loss:
                print(f"   Val loss: {val_loss[-1]:.6f}")

    def preprocess_image(self, image_path: str) -> torch.Tensor:
        """
        预处理单张图像

        Args:
            image_path: 图像文件路径

        Returns:
            处理后的图像张量 (3, 224, 224)
        """
        # 加载图像
        image = Image.open(image_path).convert('RGB')

        # 应用转换
        image_tensor = self.transform(image)

        return image_tensor

    def preprocess_images(
        self,
        image_t1_path: str,
        image_t2_path: str,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        预处理图像对

        Args:
            image_t1_path: 时序图像 1 的路径
            image_t2_path: 时序图像 2 的路径

        Returns:
            (image_t1, image_t2) 元组，形状为 (3, 224, 224)
        """
        image_t1 = self.preprocess_image(image_t1_path)
        image_t2 = self.preprocess_image(image_t2_path)

        return image_t1, image_t2

    @torch.no_grad()
    def predict(
        self,
        image_t1_path: str,
        image_t2_path: str,
        caption: str,
    ) -> Dict[str, any]:
        """
        对单个样本进行推理

        Args:
            image_t1_path: 时序图像 1 的路径
            image_t2_path: 时序图像 2 的路径
            caption: 变化描述文本

        Returns:
            包含以下内容的字典：
            {
                'action_pred': 预测的动作向量 (cx, cy, scale) in [0, 1]³
                'action_normalized': 归一化的动作向量
                'cx': 中心 x 坐标 [0, 1]
                'cy': 中心 y 坐标 [0, 1]
                'scale': 缩放因子 [0, 1]
            }
        """
        # 预处理图像
        image_t1, image_t2 = self.preprocess_images(image_t1_path, image_t2_path)

        # 添加批处理维度
        image_t1 = image_t1.unsqueeze(0).to(self.device)
        image_t2 = image_t2.unsqueeze(0).to(self.device)

        # 前向传播
        outputs = self.model(
            image_t1,
            image_t2,
            [caption],  # 将文本放在列表中
            action_targets=None,  # 推理时无目标
        )

        # 获取预测
        action_pred = outputs['action_pred'][0]  # (3,)

        return {
            'action_pred': action_pred.cpu().numpy(),
            'cx': float(action_pred[0].item()),
            'cy': float(action_pred[1].item()),
            'scale': float(action_pred[2].item()),
        }

    @torch.no_grad()
    def batch_predict(
        self,
        image_pairs: List[Tuple[str, str]],
        captions: List[str],
    ) -> List[Dict[str, any]]:
        """
        对多个样本进行批处理推理

        Args:
            image_pairs: 图像对列表，每个元素为 (image_t1_path, image_t2_path)
            captions: 对应的文本描述列表

        Returns:
            预测结果列表
        """
        batch_size = len(image_pairs)

        # 预处理所有图像
        images_t1_list = []
        images_t2_list = []

        for image_t1_path, image_t2_path in image_pairs:
            img_t1, img_t2 = self.preprocess_images(image_t1_path, image_t2_path)
            images_t1_list.append(img_t1)
            images_t2_list.append(img_t2)

        # 堆叠为批处理
        images_t1_batch = torch.stack(images_t1_list).to(self.device)
        images_t2_batch = torch.stack(images_t2_list).to(self.device)

        # 前向传播
        outputs = self.model(
            images_t1_batch,
            images_t2_batch,
            captions,
            action_targets=None,
        )

        # 解析预测结果
        action_preds = outputs['action_pred'].cpu().numpy()  # (B, 3)

        results = []
        for i in range(batch_size):
            results.append({
                'action_pred': action_preds[i],
                'cx': float(action_preds[i, 0]),
                'cy': float(action_preds[i, 1]),
                'scale': float(action_preds[i, 2]),
            })

        return results

    def visualize_result(
        self,
        image_t1_path: str,
        image_t2_path: str,
        prediction: Dict,
        output_path: str = None,
    ) -> np.ndarray:
        """
        可视化推理结果

        Args:
            image_t1_path: 时序图像 1 的路径
            image_t2_path: 时序图像 2 的路径
            prediction: 预测结果字典
            output_path: 输出图像路径（如果为 None 则不保存）

        Returns:
            可视化后的图像
        """
        # 加载原始图像
        img_t1 = cv2.imread(image_t1_path)
        img_t2 = cv2.imread(image_t2_path)

        if img_t1 is None or img_t2 is None:
            print("❌ 无法加载图像")
            return None

        h, w = img_t1.shape[:2]

        # 从预测中获取动作向量
        cx = prediction['cx']
        cy = prediction['cy']
        scale = prediction['scale']

        # 转换为像素坐标
        center_x = int(cx * w)
        center_y = int(cy * h)

        # 绘制中心点和缩放信息
        cv2.circle(img_t2, (center_x, center_y), 5, (0, 255, 0), -1)
        cv2.circle(img_t2, (center_x, center_y), int(50 * scale), (0, 255, 0), 2)

        # 添加文本信息
        text = f"cx={cx:.3f}, cy={cy:.3f}, scale={scale:.3f}"
        cv2.putText(
            img_t2,
            text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        # 拼接两张图像
        vis_img = np.hstack([img_t1, img_t2])

        # 保存结果
        if output_path:
            cv2.imwrite(output_path, vis_img)
            print(f"✅ 可视化结果保存到: {output_path}")

        return vis_img


def main():
    """示例：如何使用推理器"""

    # 查找最新的检查点
    output_dir = Path(Config.OUTPUT_DIR)
    checkpoints = list(output_dir.glob("checkpoint_*/checkpoint_best.pt"))

    if not checkpoints:
        print("❌ 未找到检查点，请先运行训练")
        return

    # 使用最新的检查点
    checkpoint_path = sorted(checkpoints, key=lambda x: x.stat().st_mtime)[-1]
    print(f"找到检查点: {checkpoint_path}")

    # 初始化推理器
    inference = VLMInference(checkpoint_path=str(checkpoint_path))

    # 示例 1：单个推理
    print("\n" + "="*60)
    print("示例 1：单个样本推理")
    print("="*60)

    # 这是示例路径，实际使用时需要替换为真实的图像路径
    image_t1_path = "path/to/image_t1.png"
    image_t2_path = "path/to/image_t2.png"
    caption = "Building construction started in this area"

    if Path(image_t1_path).exists() and Path(image_t2_path).exists():
        result = inference.predict(image_t1_path, image_t2_path, caption)
        print(f"✅ 推理结果:")
        print(f"   动作向量: {result['action_pred']}")
        print(f"   中心 x: {result['cx']:.4f}")
        print(f"   中心 y: {result['cy']:.4f}")
        print(f"   缩放: {result['scale']:.4f}")

        # 可视化结果
        vis_img = inference.visualize_result(
            image_t1_path,
            image_t2_path,
            result,
            output_path="inference_result.jpg"
        )
    else:
        print("⚠️  示例图像路径不存在，跳过推理")

    # 示例 2：批处理推理
    print("\n" + "="*60)
    print("示例 2：批处理推理")
    print("="*60)

    # 准备多个样本（这是示例，实际使用时需要真实数据）
    image_pairs = [
        ("path/to/img1_t1.png", "path/to/img1_t2.png"),
        ("path/to/img2_t1.png", "path/to/img2_t2.png"),
    ]
    captions = [
        "Changes in building structure",
        "Road expansion detected",
    ]

    valid_pairs = [
        (p1, p2) for p1, p2 in image_pairs
        if Path(p1).exists() and Path(p2).exists()
    ]

    if valid_pairs:
        results = inference.batch_predict(valid_pairs, captions[:len(valid_pairs)])
        print(f"✅ 批处理推理完成，得到 {len(results)} 个结果")
        for i, result in enumerate(results):
            print(f"   样本 {i+1}: {result['action_pred']}")
    else:
        print("⚠️  未找到有效的图像对")


if __name__ == "__main__":
    main()

