#!/usr/bin/env python3
"""
项目设置验证脚本
检查所有文件是否就位且依赖项已安装
"""

import os
import sys


def check_project_structure():
    """检查所有必需的项目文件是否存在"""
    print("\n" + "="*60)
    print("📁 项目结构检查")
    print("="*60)

    required_files = {
        "源代码": [
            "src/__init__.py",
            "src/config.py",
            "src/dataset.py",
            "src/model.py",
            "src/train.py",
            "src/utils.py",
        ],
        "配置和文档": [
            "requirements.txt",
            "README.md",
            "快速开始.md",
            "kaggle_launch.py",
        ],
    }

    all_present = True
    for category, files in required_files.items():
        print(f"\n{category}:")
        for file in files:
            exists = os.path.exists(file)
            status = "✅" if exists else "❌"
            print(f"  {status} {file}")
            if not exists:
                all_present = False

    print("\n" + "="*60)
    return all_present


def check_dependencies():
    """检查是否安装了必需的 Python 包"""
    print("\n" + "="*60)
    print("📦 依赖项检查")
    print("="*60)

    dependencies = {
        "torch": "PyTorch (深度学习框架)",
        "transformers": "HuggingFace Transformers",
        "datasets": "HuggingFace Datasets",
        "peft": "参数高效微调",
        "bitsandbytes": "量化支持",
        "accelerate": "分布式训练支持",
        "PIL": "图像处理",
        "numpy": "数值计算",
    }

    all_installed = True
    for package, description in dependencies.items():
        try:
            __import__(package)
            status = "✅"
            version = ""
            try:
                mod = __import__(package)
                if hasattr(mod, '__version__'):
                    version = f" ({mod.__version__})"
            except:
                pass
            print(f"  {status} {package:<15} {description}{version}")
        except ImportError:
            print(f"  ❌ {package:<15} {description} - 未安装")
            all_installed = False

    print("\n" + "="*60)
    return all_installed


def check_configuration():
    """检查配置文件是否有效"""
    print("\n" + "="*60)
    print("⚙️  配置检查")
    print("="*60)

    try:
        from src.config import Config

        print("\n环境检测:")
        print(f"  {'Kaggle' if Config.IS_KAGGLE else '本地开发'}")

        print("\n关键路径:")
        paths = {
            "数据集": Config.DATASET_PATH,
            "CLIP 模型": Config.CLIP_PATH,
            "LLM 模型": Config.LLM_PATH,
            "输出": Config.OUTPUT_DIR,
        }

        all_valid = True
        for name, path in paths.items():
            exists = os.path.exists(path)
            status = "✅" if exists else "❌"
            print(f"  {status} {name:<15} {path}")
            if not exists and Config.IS_KAGGLE:
                all_valid = False

        print("\n训练超参数:")
        print(f"  最大轮数:       {Config.MAX_EPOCHS}")
        print(f"  批次大小:       {Config.BATCH_SIZE}")
        print(f"  学习率:         {Config.LEARNING_RATE}")
        print(f"  混合精度:       {Config.USE_MIXED_PRECISION}")

        print("\n" + "="*60)
        return all_valid

    except Exception as e:
        print(f"  ❌ 加载配置时出错: {e}")
        print("\n" + "="*60)
        return False


def check_pytorch():
    """检查 PyTorch 和 GPU 设置"""
    print("\n" + "="*60)
    print("🔧 PyTorch 和 GPU 检查")
    print("="*60)

    try:
        import torch

        print(f"\nPyTorch 版本: {torch.__version__}")
        print(f"CUDA 可用: {'✅ 是' if torch.cuda.is_available() else '❌ 否'}")

        if torch.cuda.is_available():
            print(f"\nGPU 信息:")
            print(f"  设备名称: {torch.cuda.get_device_name(0)}")
            print(f"  设备数量: {torch.cuda.device_count()}")

            # 内存
            props = torch.cuda.get_device_properties(0)
            total_memory_gb = props.total_memory / 1e9
            print(f"  总内存: {total_memory_gb:.2f} GB")

            # 当前使用情况
            allocated = torch.cuda.memory_allocated(0) / 1e9
            reserved = torch.cuda.memory_reserved(0) / 1e9
            print(f"  已分配: {allocated:.2f} GB")
            print(f"  保留: {reserved:.2f} GB")
            print(f"  空闲: {total_memory_gb - allocated:.2f} GB")

            print(f"\n✅ GPU 已准备好进行训练!")
        else:
            print(f"\n⚠️  GPU 不可用 - 训练将使用 CPU (较慢)")

        print("\n" + "="*60)
        return True

    except Exception as e:
        print(f"❌ 检查 PyTorch 时出错: {e}")
        print("\n" + "="*60)
        return False


def print_summary(checks):
    """打印所有检查的摘要"""
    print("\n" + "="*70)
    print(" "*20 + "✅ 设置验证摘要")
    print("="*70 + "\n")

    all_passed = True
    for check_name, passed in checks.items():
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"{status}  {check_name}")
        if not passed:
            all_passed = False

    print("\n" + "="*70)

    if all_passed:
        print("\n🎉 所有检查都通过了！您已准备好进行训练!\n")
        print("下一步:")
        print("  1. 查看 README.md 了解详细文档")
        print("  2. 查看快速开始.md了解部署指南")
        print("  3. 运行: python -m src.train\n")
    else:
        print("\n⚠️  某些检查失败。请修复上述问题。\n")
        print("故障排除:")
        print("  • 对于缺失的文件: 确保您在项目根目录中")
        print("  • 对于缺失的包: 运行 'pip install -r requirements.txt'")
        print("  • 对于路径问题: 编辑 src/config.py 设置正确的路径\n")

    print("="*70 + "\n")

    return all_passed


def main():
    """运行所有检查"""
    print("\n" + "="*70)
    print(" "*15 + "🔍 VLM-VLA 项目设置验证")
    print("="*70)

    checks = {}

    # 运行所有检查
    checks["项目结构"] = check_project_structure()
    checks["依赖项"] = check_dependencies()
    checks["配置"] = check_configuration()
    checks["PyTorch/GPU"] = check_pytorch()

    # 打印摘要
    all_passed = print_summary(checks)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())

