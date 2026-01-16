"""
VLM-VLA Agent 训练和推理的工具函数库

包含以下功能模块：
    1. 随机种子：设置可复现性的种子
    2. 设备管理：GPU/CPU 设备检测和选择
    3. 模型统计：参数计数和摘要打印
    4. 检查点管理：加载、保存和清理检查点
    5. 内存监控：GPU 内存使用情况跟踪
    6. 路径验证：检查必需的文件和目录
    7. 日志记录：设置和管理日志输出
    8. 性能计算：统计指标平均值计算

使用示例：
    >>> from src.utils import set_seed, get_device, print_model_summary
    >>> set_seed(42)
    >>> device = get_device()
    >>> print_model_summary(model)
"""

import json
import os
import random
from pathlib import Path
from typing import Dict

import numpy as np
import torch


def set_seed(seed: int = 42):
    """
    设置随机种子以确保可复现性

    Args:
        seed: 随机种子值
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # 确保可复现的行为
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"✅ 随机种子已设置为 {seed}")


def get_device() -> torch.device:
    """
    获取最佳可用设备 (GPU 或 CPU)

    Returns:
        torch.device 对象
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"✅ 使用 CUDA 设备: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("⚠️  使用 CPU (较慢，但仍可工作)")

    return device


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """
    统计模型中可训练和冻结的参数数量

    Args:
        model: PyTorch 模型

    Returns:
        包含参数统计的字典
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params

    return {
        'total': total_params,
        'trainable': trainable_params,
        'frozen': frozen_params,
        'trainable_percentage': 100.0 * trainable_params / total_params if total_params > 0 else 0,
    }


def print_model_summary(model: torch.nn.Module):
    """
    打印模型参数摘要

    Args:
        model: PyTorch 模型
    """
    stats = count_parameters(model)

    print("\n" + "="*60)
    print("模型参数摘要")
    print("="*60)
    print(f"总参数数:          {stats['total']:>15,}")
    print(f"可训练参数数:      {stats['trainable']:>15,}")
    print(f"冻结参数数:        {stats['frozen']:>15,}")
    print(f"可训练比例:        {stats['trainable_percentage']:>14.2f}%")
    print("="*60 + "\n")


def load_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer = None,
    device: torch.device = None,
) -> Dict:
    """
    加载训练检查点

    Args:
        checkpoint_path: 检查点文件路径
        model: 要加载状态字典的模型
        optimizer: 可选的优化器，用于加载状态字典
        device: 加载检查点的设备

    Returns:
        检查点字典
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"📥 正在从以下位置加载检查点: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # 加载模型状态
    model.load_state_dict(checkpoint['model_state_dict'])
    print("✅ 模型状态已加载")

    # 如果提供则加载优化器状态
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("✅ 优化器状态已加载")

    # 提取元数据
    epoch = checkpoint.get('epoch', 0)
    global_step = checkpoint.get('global_step', 0)

    print(f"   轮次: {epoch}")
    print(f"   全局步数: {global_step}")

    return checkpoint


def save_config_as_json(config, save_path: str):
    """
    将配置保存为 JSON 以确保可复现性

    Args:
        config: 配置对象或字典
        save_path: 保存 JSON 的路径
    """
    config_dict = {}

    if hasattr(config, '__dict__'):
        # 如果是对象，提取属性
        for key in dir(config):
            if not key.startswith('_') and not callable(getattr(config, key)):
                value = getattr(config, key)
                # 只保存可序列化的类型
                if isinstance(value, (str, int, float, bool, list, dict)):
                    config_dict[key] = value
    else:
        # 如果已经是字典
        config_dict = config

    with open(save_path, 'w') as f:
        json.dump(config_dict, f, indent=2)

    print(f"✅ 配置已保存到 {save_path}")


def create_output_directory(base_path: str = None) -> Path:
    """
    创建带时间戳的输出目录

    Args:
        base_path: 输出目录的基础路径

    Returns:
        输出目录的 Path 对象
    """
    from datetime import datetime

    if base_path is None:
        base_path = "./output"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(base_path) / f"run_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"📁 输出目录已创建: {output_dir}")
    return output_dir


def format_time(seconds: float) -> str:
    """
    将秒数格式化为人类可读的时间

    Args:
        seconds: 以秒为单位的时间

    Returns:
        格式化的时间字符串
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    if hours > 0:
        return f"{hours}小时 {minutes}分钟 {secs}秒"
    elif minutes > 0:
        return f"{minutes}分钟 {secs}秒"
    else:
        return f"{secs}秒"


def get_gpu_memory_info() -> Dict[str, float]:
    """
    获取 GPU 内存使用信息

    Returns:
        包含内存统计信息的字典（以 GB 为单位）
    """
    if not torch.cuda.is_available():
        return {}

    # 总内存
    total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9

    # 已分配内存
    allocated_memory = torch.cuda.memory_allocated(0) / 1e9

    # 保留内存
    reserved_memory = torch.cuda.memory_reserved(0) / 1e9

    # 空闲内存
    free_memory = total_memory - allocated_memory

    return {
        'total_gb': round(total_memory, 2),
        'allocated_gb': round(allocated_memory, 2),
        'reserved_gb': round(reserved_memory, 2),
        'free_gb': round(free_memory, 2),
        'allocated_percentage': round(100.0 * allocated_memory / total_memory, 2),
    }


def print_gpu_memory():
    """打印 GPU 内存使用情况"""
    memory_info = get_gpu_memory_info()

    if not memory_info:
        print("⚠️  GPU 不可用")
        return

    print("\n" + "="*60)
    print("GPU 内存使用")
    print("="*60)
    print(f"总计:      {memory_info['total_gb']:>6.2f} GB")
    print(f"已分配:    {memory_info['allocated_gb']:>6.2f} GB ({memory_info['allocated_percentage']:>5.1f}%)")
    print(f"保留:      {memory_info['reserved_gb']:>6.2f} GB")
    print(f"空闲:      {memory_info['free_gb']:>6.2f} GB")
    print("="*60 + "\n")


def cleanup_old_checkpoints(checkpoint_dir: Path, keep_best: bool = True, max_keep: int = 3):
    """
    清理旧的检查点，只保留最近的几个

    Args:
        checkpoint_dir: 包含检查点的目录
        keep_best: 是否始终保留最佳检查点
        max_keep: 要保留的最大检查点数量
    """
    checkpoint_dir = Path(checkpoint_dir)

    if not checkpoint_dir.exists():
        return

    # 获取所有检查点文件
    checkpoint_files = sorted(
        checkpoint_dir.glob("checkpoint_step_*.pt"),
        key=lambda x: x.stat().st_mtime,  # 按修改时间排序
    )

    # 保留最佳检查点
    keep_files = set()
    if keep_best and (checkpoint_dir / "checkpoint_best.pt").exists():
        keep_files.add("checkpoint_best.pt")

    # 保留最近的几个
    for cp_file in checkpoint_files[-max_keep:]:
        keep_files.add(cp_file.name)

    # 删除其他文件
    deleted_count = 0
    for cp_file in checkpoint_files:
        if cp_file.name not in keep_files:
            cp_file.unlink()
            deleted_count += 1

    if deleted_count > 0:
        print(f"🧹 已删除 {deleted_count} 个旧检查点")


def validate_paths(paths: Dict[str, str]) -> bool:
    """
    验证所有必需的路径是否存在

    Args:
        paths: 路径名称和路径的字典

    Returns:
        如果所有路径都存在则返回 True，否则返回 False
    """
    print("\n" + "="*60)
    print("路径验证")
    print("="*60)

    all_valid = True
    for name, path in paths.items():
        exists = os.path.exists(path)
        status = "✅" if exists else "❌"
        print(f"{status} {name}: {path}")
        if not exists:
            all_valid = False

    print("="*60 + "\n")
    return all_valid


class AverageMeter:
    """
    计算和存储指标的当前值和平均值

    用于训练过程中追踪指标，例如损失、准确率等。

    示例：
        >>> meter = AverageMeter("损失", fmt=":6.4f")
        >>> for batch_loss in losses:
        ...     meter.update(batch_loss, n=batch_size)
        >>> print(meter)  # 输出: 损失  1.2345 (1.2000)
    """

    def __init__(self, name: str, fmt: str = ":.4f"):
        """
        初始化 AverageMeter

        Args:
            name (str): 指标名称（用于打印）
            fmt (str): 数字格式化字符串（默认 4 位小数）
        """
        self.name = name  # 指标名称
        self.fmt = fmt  # 格式化格式
        self.reset()  # 初始化计数器

    def reset(self):
        """重置所有计数器"""
        self.val = 0  # 当前值
        self.avg = 0  # 平均值
        self.sum = 0  # 累积和
        self.count = 0  # 样本计数

    def update(self, val, n: int = 1):
        """
        使用新值更新平均值

        Args:
            val (float): 新的值
            n (int): 值的数量（用于加权平均）
        """
        self.val = val  # 更新当前值
        self.sum += val * n  # 累加加权值
        self.count += n  # 更新计数
        self.avg = self.sum / self.count if self.count > 0 else 0  # 计算平均值

    def __str__(self):
        """返回格式化的字符串表示"""
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(name=self.name, val=self.val, avg=self.avg)


def setup_logging(log_file: str = None):
    """
    设置日志记录配置

    Args:
        log_file: 可选的日志文件路径
    """
    import logging

    log_format = "%(asctime)s - %(levelname)s - %(message)s"

    # 创建日志记录器
    logger = logging.getLogger("VLM_VLA")
    logger.setLevel(logging.DEBUG)

    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(log_format)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # 文件处理器
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(log_format)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger


if __name__ == "__main__":
    # 测试工具函数
    print("测试工具函数...\n")

    set_seed(42)
    device = get_device()

    print_gpu_memory()

    # 测试 AverageMeter
    meter = AverageMeter("损失")
    for i in range(10):
        meter.update(1.0 - 0.1 * i, n=1)
    print(f"\n{meter}")

