#!/usr/bin/env python3
"""
数据集路径诊断脚本
用于在 Kaggle 上调试数据集加载问题
"""

import os
import sys
from pathlib import Path


def debug_dataset_path():
    """调试数据集路径"""

    # 检查路径
    dataset_path = "/kaggle/input/levir-cc-dateset/LEVIR-CC"

    print("=" * 60)
    print("数据集路径诊断")
    print("=" * 60)

    print(f"\n📁 检查路径: {dataset_path}")
    print(f"   路径存在: {os.path.exists(dataset_path)}")

    if not os.path.exists(dataset_path):
        print("❌ 路径不存在！")

        # 检查上级路径
        parent_path = "/kaggle/input/levir-cc-dateset"
        print(f"\n📁 检查上级路径: {parent_path}")
        print(f"   路径存在: {os.path.exists(parent_path)}")

        if os.path.exists(parent_path):
            print(f"\n   内容:")
            for item in os.listdir(parent_path):
                print(f"     - {item}")

        return 1

    # 列出目录内容
    print(f"\n📂 目录内容:")
    path_obj = Path(dataset_path)

    for item in sorted(path_obj.iterdir()):
        if item.is_dir():
            # 统计子项数量
            try:
                count = len(list(item.iterdir()))
                print(f"   📁 {item.name}/ ({count} 项)")
            except:
                print(f"   📁 {item.name}/")
        else:
            # 显示文件大小
            size_mb = item.stat().st_size / (1024 * 1024)
            print(f"   📄 {item.name} ({size_mb:.2f} MB)")

    # 检查 Arrow 文件
    print(f"\n🔍 查找 Arrow 文件:")
    arrow_files = list(path_obj.glob('*.arrow'))

    if arrow_files:
        print(f"   ✅ 找到 {len(arrow_files)} 个 Arrow 文件:")
        for f in arrow_files:
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"      - {f.name} ({size_mb:.2f} MB)")
    else:
        print(f"   ❌ 未找到 Arrow 文件")

    # 检查图像目录
    print(f"\n🔍 查找图像目录:")
    image_dirs = [
        path_obj / 'images' / 'train' / 'A',
        path_obj / 'images' / 'train' / 'B',
        path_obj / 'A',
        path_obj / 'B',
        path_obj / 'train' / 'A',
        path_obj / 'train' / 'B',
    ]

    found_image_dirs = False
    for dir_path in image_dirs:
        if dir_path.exists():
            try:
                count = len(list(dir_path.glob('*.png')) + list(dir_path.glob('*.jpg')))
                print(f"   ✅ {dir_path.relative_to(path_obj)} ({count} 图像)")
                found_image_dirs = True
            except:
                print(f"   ✅ {dir_path.relative_to(path_obj)}")

    if not found_image_dirs:
        print(f"   ❌ 未找到图像目录")

    # 测试加载
    print(f"\n" + "=" * 60)
    print("测试数据加载")
    print("=" * 60)

    try:
        from src.dataset import load_raw_levir_cc_dataset

        print(f"\n调用 load_raw_levir_cc_dataset('{dataset_path}')...")
        result = load_raw_levir_cc_dataset(dataset_path)

        print(f"\n✅ 加载成功!")
        print(f"   结果类型: {type(result).__name__}")

        if hasattr(result, '__len__'):
            print(f"   样本数量: {len(result)}")

        if hasattr(result, 'column_names'):
            print(f"   列名: {result.column_names}")

        return 0

    except Exception as e:
        print(f"\n❌ 加载失败:")
        print(f"   错误: {e}")

        import traceback
        print(f"\n完整错误信息:")
        traceback.print_exc()

        return 1


if __name__ == "__main__":
    sys.exit(debug_dataset_path())

