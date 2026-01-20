#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CIFAR-10数据集转换脚本（从已解压的文件夹）

功能：
从已解压的 cifar-10-batches-py 文件夹转换为训练所需的目录结构

使用方法：
python convert_cifar10.py

作者：AI Assistant
日期：2025-01-20
"""

import os
import pickle
from PIL import Image


def load_cifar10_batch(file_path):
    """加载CIFAR-10批次文件"""
    with open(file_path, 'rb') as f:
        batch = pickle.load(f, encoding='bytes')
    return batch


def save_images_from_batch(batch, output_dir, batch_name, class_names):
    """从批次数据中保存图像"""
    images = batch[b'data']
    labels = batch[b'labels']
    filenames = batch[b'filenames']

    # CIFAR-10图像是32x32 RGB
    images = images.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)

    print(f"  正在处理批次: {batch_name} ({len(images)}张图像)")

    for i, (img_data, label, filename) in enumerate(zip(images, labels, filenames)):
        # 获取类别名称
        class_name = class_names[label]

        # 创建类别目录
        class_dir = os.path.join(output_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)

        # 保存图像
        img = Image.fromarray(img_data)

        # 生成文件名（保留原始文件名）
        if isinstance(filename, bytes):
            filename = filename.decode('utf-8')
        img_path = os.path.join(class_dir, filename)

        img.save(img_path)

        if (i + 1) % 1000 == 0:
            print(f"    已处理 {i + 1} 张图像...")


def convert_cifar10(input_dir, output_dir):
    """转换CIFAR-10数据集"""
    print("=" * 60)
    print("CIFAR-10数据集转换工具")
    print("=" * 60)
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")

    # 检查输入目录是否存在
    if not os.path.exists(input_dir):
        print(f"\n❌ 错误: 输入目录不存在: {input_dir}")
        return False

    # 检查meta文件
    meta_file = os.path.join(input_dir, "batches.meta")
    if not os.path.exists(meta_file):
        print(f"\n❌ 错误: 找不到 batches.meta 文件")
        print(f"请确认 {input_dir} 是正确的 cifar-10-batches-py 目录")
        return False

    # 加载类别名称
    print("\n加载数据集元信息...")
    with open(meta_file, 'rb') as f:
        meta = pickle.load(f, encoding='bytes')
        class_names = [name.decode('utf-8') for name in meta[b'label_names']]

    print(f"✅ CIFAR-10类别: {class_names}")

    # 创建输出目录
    train_dir = os.path.join(output_dir, "train")
    val_dir = os.path.join(output_dir, "val")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    # 处理训练批次 (data_batch_1 到 data_batch_5)
    print(f"\n正在转换训练集...")
    for i in range(1, 6):
        batch_file = os.path.join(input_dir, f"data_batch_{i}")
        if not os.path.exists(batch_file):
            print(f"⚠️ 警告: 找不到 {batch_file}")
            continue

        batch = load_cifar10_batch(batch_file)
        save_images_from_batch(batch, train_dir, f"data_batch_{i}", class_names)

    # 处理测试批次
    print(f"\n正在转换验证集...")
    test_batch_file = os.path.join(input_dir, "test_batch")
    if os.path.exists(test_batch_file):
        test_batch = load_cifar10_batch(test_batch_file)
        save_images_from_batch(test_batch, val_dir, "test_batch", class_names)
    else:
        print(f"⚠️ 警告: 找不到 {test_batch_file}")

    print("\n" + "=" * 60)
    print("✅ CIFAR-10数据集转换完成！")
    print("=" * 60)
    print(f"\n数据集位置:")
    print(f"  训练集: {train_dir}")
    print(f"  验证集: {val_dir}")
    print(f"\n数据集结构:")
    print(f"  {output_dir}/")
    print(f"    ├── train/")
    for class_name in class_names:
        train_class_dir = os.path.join(train_dir, class_name)
        if os.path.exists(train_class_dir):
            count = len(os.listdir(train_class_dir))
            print(f"    │   ├── {class_name}/  ({count}张图像)")
    print(f"    └── val/")
    for class_name in class_names:
        val_class_dir = os.path.join(val_dir, class_name)
        if os.path.exists(val_class_dir):
            count = len(os.listdir(val_class_dir))
            print(f"        ├── {class_name}/  ({count}张图像)")

    print(f"\n总计:")
    print(f"  训练图像: 50,000张")
    print(f"  验证图像: 10,000张")
    print(f"  图像尺寸: 32x32 RGB")
    print(f"  类别数量: 10")

    return True


def main():
    # 固定的路径配置
    input_dir = r"D:\pythonProject2\datasets\cifar-10-batches-py"
    output_dir = r"D:\pythonProject2\datasets\cifar10"

    print(f"使用预设路径:")
    print(f"  输入: {input_dir}")
    print(f"  输出: {output_dir}")
    print()

    # 转换数据集
    success = convert_cifar10(input_dir, output_dir)

    if success:
        print("\n✅ 数据集已准备完成！")
        print("\n下一步:")
        print("1. 启动前端和后端服务")
        print("2. 在前端选择 'CIFAR-10图像分类数据集'")
        print("3. 开始训练任务")
    else:
        print("\n❌ 数据集转换失败，请检查错误信息")


if __name__ == "__main__":
    main()
