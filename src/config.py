"""
VLM-VLA Agent 配置模块
自动检测 Kaggle 环境并设置适当的路径

功能说明：
    - 自动识别运行环境（Kaggle 或本地开发）
    - 设置相应的数据集和模型路径
    - 定义所有训练超参数
    - 支持混合精度训练和 4 位量化
    - 支持 LoRA 微调配置
"""

import os
from pathlib import Path


class Config:
    """
    VLM-VLA Agent 配置类

    该类包含所有与训练相关的配置项：
    - 环境检测：自动识别 Kaggle 或本地环境
    - 路径配置：数据集、模型、输出目录
    - 超参数：学习率、批次大小等
    - 模型参数：CLIP、LLM、LoRA 配置
    """

    # 检测是否在 Kaggle 环境中运行
    # Kaggle 环境下，/kaggle/input 目录存在
    IS_KAGGLE = os.path.exists("/kaggle/input")

    if IS_KAGGLE:
        # Kaggle 环境路径
        # 支持多种可能的数据集路径结构
        DATASET_PATH = "/kaggle/input/levir-cc-dataset"
        # 如果上面路径不存在，尝试其他可能
        if not os.path.exists(DATASET_PATH):
            # 检查其他可能的路径
            possible_paths = [
                "/kaggle/input/levir-cc-dataset/LEVIR-CC",
                "/kaggle/input/levir-cc/LEVIR-CC",
                "/kaggle/input/levir-cc-dataset",
                "/kaggle/input/levir-cc",
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    DATASET_PATH = path
                    break

        CLIP_PATH = "/kaggle/input/model-data-set/models/clip-vit-b32"
        LLM_PATH = "/kaggle/input/model-data-set/models/qwen2.5-0.5b"
        OUTPUT_DIR = "/kaggle/working/output"
        WORKING_DIR = "/kaggle/working"
    else:
        # 本地开发路径
        PROJECT_ROOT = Path(__file__).parent.parent
        DATASET_PATH = str(PROJECT_ROOT / "data" / "Levir-CC-dataset")
        CLIP_PATH = str(PROJECT_ROOT / "models" / "clip-vit-b32")
        LLM_PATH = str(PROJECT_ROOT / "models" / "qwen2.5-0.5b")
        OUTPUT_DIR = str(PROJECT_ROOT / "output")
        WORKING_DIR = str(PROJECT_ROOT)

    # 如果输出目录不存在则创建
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ==================== 训练超参数 ====================
    MAX_EPOCHS = 10
    BATCH_SIZE = 4  # 受 T4 GPU 显存限制
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 0.01
    WARMUP_STEPS = 500
    MAX_GRAD_NORM = 1.0

    # ==================== 模型超参数 ====================
    # 视觉模块 (CLIP)
    VISION_MODEL_NAME = "openai/clip-vit-base-patch32"
    VISION_HIDDEN_DIM = 768  # CLIP ViT-B/32 隐藏层维度
    VISION_OUTPUT_DIM = 768  # CLIP ViT-B/32 输出维度（池化后）

    # 语言模型 (Qwen)
    LLM_MODEL_NAME = "Qwen/Qwen2.5-0.5B"
    LLM_HIDDEN_DIM = 896  # Qwen2.5-0.5B 隐藏维度
    LLM_NUM_LAYERS = 24

    # LoRA 配置
    LORA_R = 8
    LORA_ALPHA = 16
    LORA_DROPOUT = 0.05
    LORA_TARGET_MODULES = ["q_proj", "v_proj"]

    # 4位量化配置 (BitsAndBytes)
    USE_4BIT = True
    COMPUTE_DTYPE = "bfloat16"  # bfloat16 或 float16

    # ==================== 数据超参数 ====================
    IMAGE_SIZE = 224  # CLIP 输入尺寸
    MAX_TEXT_LENGTH = 128
    BBOX_NORMALIZE = True  # 将边界框归一化到 [0, 1]

    # ==================== 训练设置 ====================
    SAVE_INTERVAL = 500  # 每 500 步保存模型
    EVAL_INTERVAL = 1000  # 每 1000 步评估一次
    LOG_INTERVAL = 100  # 每 100 步记录指标
    NUM_WORKERS = 4  # DataLoader 工作进程数

    # 混合精度训练
    USE_MIXED_PRECISION = True
    MIXED_PRECISION_DTYPE = "bfloat16"

    # 梯度累积
    GRADIENT_ACCUMULATION_STEPS = 1

    # ==================== 调试 ====================
    DEBUG = False
    SEED = 42

    @classmethod
    def print_config(cls):
        """打印当前配置"""
        print("\n" + "="*60)
        print("VLM-VLA 配置")
        print("="*60)
        print(f"环境: {'Kaggle' if cls.IS_KAGGLE else '本地开发'}")
        print(f"数据集路径: {cls.DATASET_PATH}")
        print(f"CLIP 模型路径: {cls.CLIP_PATH}")
        print(f"LLM 模型路径: {cls.LLM_PATH}")
        print(f"输出目录: {cls.OUTPUT_DIR}")
        print(f"\n训练设置:")
        print(f"  最大轮数: {cls.MAX_EPOCHS}")
        print(f"  批次大小: {cls.BATCH_SIZE}")
        print(f"  学习率: {cls.LEARNING_RATE}")
        print(f"  混合精度: {cls.USE_MIXED_PRECISION}")
        print("="*60 + "\n")

    @classmethod
    def verify_paths(cls):
        """
        验证所有必需的路径是否存在（用于 Kaggle 环境）
        如果所有路径都存在则返回 True，否则返回 False
        """
        if not cls.IS_KAGGLE:
            print("⚠️  本地开发模式 - 跳过路径验证")
            return True

        print("\n" + "="*60)
        print("路径验证 (Kaggle 环境)")
        print("="*60)

        paths_to_check = {
            "数据集": cls.DATASET_PATH,
            "CLIP 模型": cls.CLIP_PATH,
            "LLM 模型": cls.LLM_PATH,
            "输出目录": cls.OUTPUT_DIR,
        }

        all_valid = True
        for name, path in paths_to_check.items():
            exists = os.path.exists(path)
            status = "✅" if exists else "❌"
            print(f"{status} {name}: {path}")
            if not exists:
                all_valid = False

        print("="*60 + "\n")
        return all_valid


if __name__ == "__main__":
    Config.print_config()
    Config.verify_paths()

