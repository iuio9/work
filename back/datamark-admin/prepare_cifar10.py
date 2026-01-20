#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CIFAR-10数据集准备脚本

功能：
1. 自动下载CIFAR-10数据集
2. 转换为适合训练的目录结构
3. 按照训练集/验证集组织文件

使用方法：
python prepare_cifar10.py --output_dir D:/pythonProject2/datasets/cifar10

作者：AI Assistant
日期：2025-01-20
"""

import argparse
import os
import pickle
import shutil
from pathlib import Path
from PIL import Image
import urllib.request
import tarfile


def download_cifar10(download_dir):
    """下载CIFAR-10数据集"""
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = os.path.join(download_dir, "cifar-10-python.tar.gz")

    if os.path.exists(filename):
        print(f"数据集已存在: {filename}")
        return filename

    print(f"正在下载CIFAR-10数据集...")
    print(f"URL: {url}")
    print(f"目标: {filename}")

    os.makedirs(download_dir, exist_ok=True)

    try:
        urllib.request.urlretrieve(url, filename)
        print(f"✅ 下载完成: {filename}")
        return filename
    except Exception as e:
        print(f"❌ 下载失败: {e}")
        print("请手动从以下地址下载:")
        print("https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz")
        return None


def extract_cifar10(tar_file, extract_dir):
    """解压CIFAR-10数据集"""
    print(f"正在解压数据集...")
    os.makedirs(extract_dir, exist_ok=True)

    with tarfile.open(tar_file, 'r:gz') as tar:
        tar.extractall(extract_dir)

    extracted_folder = os.path.join(extract_dir, "cifar-10-batches-py")
    print(f"✅ 解压完成: {extracted_folder}")
    return extracted_folder


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
            print(f"  已处理 {i + 1} 张图像...")


def prepare_cifar10_dataset(output_dir):
    """准备CIFAR-10数据集"""
    print("=" * 60)
    print("CIFAR-10数据集准备工具")
    print("=" * 60)

    # 创建临时下载目录
    temp_dir = os.path.join(output_dir, "temp")
    os.makedirs(temp_dir, exist_ok=True)

    # 步骤1：下载数据集
    tar_file = download_cifar10(temp_dir)
    if tar_file is None:
        return False

    # 步骤2：解压数据集
    extracted_dir = extract_cifar10(tar_file, temp_dir)

    # 步骤3：加载类别名称
    meta_file = os.path.join(extracted_dir, "batches.meta")
    with open(meta_file, 'rb') as f:
        meta = pickle.load(f, encoding='bytes')
        class_names = [name.decode('utf-8') for name in meta[b'label_names']]

    print(f"\nCIFAR-10类别: {class_names}")

    # 步骤4：创建训练集和验证集目录
    train_dir = os.path.join(output_dir, "train")
    val_dir = os.path.join(output_dir, "val")

    print(f"\n正在转换训练集...")
    # 处理训练批次 (data_batch_1 到 data_batch_5)
    for i in range(1, 6):
        batch_file = os.path.join(extracted_dir, f"data_batch_{i}")
        print(f"处理批次: data_batch_{i}")
        batch = load_cifar10_batch(batch_file)
        save_images_from_batch(batch, train_dir, f"batch_{i}", class_names)

    print(f"\n正在转换验证集...")
    # 处理测试批次
    test_batch_file = os.path.join(extracted_dir, "test_batch")
    test_batch = load_cifar10_batch(test_batch_file)
    save_images_from_batch(test_batch, val_dir, "test", class_names)

    # 步骤5：清理临时文件
    print(f"\n清理临时文件...")
    shutil.rmtree(temp_dir)

    print("\n" + "=" * 60)
    print("✅ CIFAR-10数据集准备完成！")
    print("=" * 60)
    print(f"\n数据集位置:")
    print(f"  训练集: {train_dir}")
    print(f"  验证集: {val_dir}")
    print(f"\n数据集结构:")
    print(f"  {output_dir}/")
    print(f"    ├── train/")
    for class_name in class_names:
        print(f"    │   ├── {class_name}/  (5000张图像)")
    print(f"    └── val/")
    for class_name in class_names:
        print(f"        ├── {class_name}/  (1000张图像)")

    print(f"\n总计:")
    print(f"  训练图像: 50,000张")
    print(f"  验证图像: 10,000张")
    print(f"  图像尺寸: 32x32 RGB")
    print(f"  类别数量: 10")

    return True


def main():
    parser = argparse.ArgumentParser(description='准备CIFAR-10数据集')
    parser.add_argument(
        '--output_dir',
        type=str,
        default='D:/pythonProject2/datasets/cifar10',
        help='输出目录路径（默认: D:/pythonProject2/datasets/cifar10）'
    )

    args = parser.parse_args()

    # 转换路径为绝对路径
    output_dir = os.path.abspath(args.output_dir)

    print(f"输出目录: {output_dir}")

    # 准备数据集
    success = prepare_cifar10_dataset(output_dir)

    if success:
        print("\n✅ 数据集已准备完成，可以在前端选择 'CIFAR-10图像分类数据集' 开始训练！")
    else:
        print("\n❌ 数据集准备失败，请检查错误信息")


if __name__ == "__main__":
    main()
