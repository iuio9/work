#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据集加载测试脚本
测试系统支持的各种数据集格式和加载方式

支持的数据集：
1. CIFAR-10/100 (分类)
2. ImageNet格式 (分类)
3. COCO格式 (检测)
4. Pascal VOC格式 (检测/分割)
5. 自定义图像文件夹

作者：Claude Assistant
日期：2026-01-14
"""

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from pathlib import Path
import json
import sys

print("=" * 70)
print("📦 数据集加载测试")
print("=" * 70)
print()

# ==================== 测试1: CIFAR-10 ====================
print("📌 测试1: CIFAR-10 数据集")
print("-" * 70)

dataset_root = Path("./datasets")
dataset_root.mkdir(exist_ok=True)

try:
    # 数据增强
    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # 加载训练集
    trainset = torchvision.datasets.CIFAR10(
        root=str(dataset_root),
        train=True,
        download=True,
        transform=transform
    )

    # 加载测试集
    testset = torchvision.datasets.CIFAR10(
        root=str(dataset_root),
        train=False,
        download=True,
        transform=transform
    )

    print(f"✅ CIFAR-10 加载成功")
    print(f"   训练集: {len(trainset)} 张")
    print(f"   测试集: {len(testset)} 张")
    print(f"   类别数: {len(trainset.classes)}")
    print(f"   类别名: {trainset.classes}")

    # 测试数据加载器
    trainloader = DataLoader(trainset, batch_size=32, shuffle=True, num_workers=2)
    images, labels = next(iter(trainloader))
    print(f"   批次形状: {images.shape} (NCHW)")
    print(f"   标签形状: {labels.shape}")

except Exception as e:
    print(f"❌ CIFAR-10 加载失败: {e}")

print()

# ==================== 测试2: CIFAR-100 ====================
print("📌 测试2: CIFAR-100 数据集")
print("-" * 70)

try:
    trainset = torchvision.datasets.CIFAR100(
        root=str(dataset_root),
        train=True,
        download=True,
        transform=transform
    )

    print(f"✅ CIFAR-100 加载成功")
    print(f"   训练集: {len(trainset)} 张")
    print(f"   类别数: {len(trainset.classes)}")
    print(f"   前10个类: {trainset.classes[:10]}")

except Exception as e:
    print(f"❌ CIFAR-100 加载失败: {e}")

print()

# ==================== 测试3: ImageNet格式 ====================
print("📌 测试3: ImageNet格式数据集")
print("-" * 70)

imagenet_path = Path("/home/user/datasets/imagenet")
if imagenet_path.exists():
    try:
        transform_imagenet = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225]),
        ])

        dataset = torchvision.datasets.ImageFolder(
            root=str(imagenet_path / "train"),
            transform=transform_imagenet
        )

        print(f"✅ ImageNet格式数据集加载成功")
        print(f"   图片数量: {len(dataset)}")
        print(f"   类别数量: {len(dataset.classes)}")
        print(f"   前5个类: {dataset.classes[:5]}")

    except Exception as e:
        print(f"⚠️  ImageNet加载失败: {e}")
else:
    print(f"⚠️  ImageNet路径不存在: {imagenet_path}")
    print(f"   如果有ImageNet数据集，请放在: /home/user/datasets/imagenet/")
    print(f"   目录结构: imagenet/train/class_name/*.jpg")

print()

# ==================== 测试4: 自定义图像文件夹 ====================
print("📌 测试4: 自定义图像文件夹")
print("-" * 70)

class CustomImageDataset(Dataset):
    """自定义图像数据集（演示）"""

    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform

        # 支持的图像格式
        self.image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']

        # 扫描所有类别文件夹
        self.classes = sorted([d.name for d in self.root_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        # 收集所有图像路径
        self.samples = []
        for cls_name in self.classes:
            cls_dir = self.root_dir / cls_name
            for img_path in cls_dir.iterdir():
                if img_path.suffix.lower() in self.image_extensions:
                    self.samples.append((str(img_path), self.class_to_idx[cls_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label


# 示例：检查是否有自定义数据集
custom_dataset_path = Path("/home/user/datasets/custom_images")
if custom_dataset_path.exists():
    try:
        transform_custom = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

        dataset = CustomImageDataset(
            root_dir=custom_dataset_path,
            transform=transform_custom
        )

        print(f"✅ 自定义数据集加载成功")
        print(f"   图片数量: {len(dataset)}")
        print(f"   类别数量: {len(dataset.classes)}")
        print(f"   类别: {dataset.classes}")

    except Exception as e:
        print(f"⚠️  自定义数据集加载失败: {e}")
else:
    print(f"⚠️  自定义数据集路径不存在: {custom_dataset_path}")
    print(f"   自定义数据集格式示例:")
    print(f"   /home/user/datasets/custom_images/")
    print(f"     ├── class_1/")
    print(f"     │   ├── img1.jpg")
    print(f"     │   └── img2.jpg")
    print(f"     └── class_2/")
    print(f"         ├── img3.jpg")
    print(f"         └── img4.jpg")

print()

# ==================== 测试5: COCO格式 ====================
print("📌 测试5: COCO格式（目标检测）")
print("-" * 70)

coco_path = Path("/home/user/datasets/coco")
if (coco_path / "annotations").exists():
    try:
        # 尝试加载COCO数据集
        from pycocotools.coco import COCO

        ann_file = coco_path / "annotations" / "instances_train2017.json"
        if ann_file.exists():
            coco = COCO(str(ann_file))

            # 获取类别信息
            cats = coco.loadCats(coco.getCatIds())
            cat_names = [cat['name'] for cat in cats]

            print(f"✅ COCO数据集加载成功")
            print(f"   类别数: {len(cat_names)}")
            print(f"   前10个类: {cat_names[:10]}")
            print(f"   图片数: {len(coco.getImgIds())}")
        else:
            print(f"⚠️  COCO标注文件不存在: {ann_file}")

    except ImportError:
        print(f"⚠️  pycocotools未安装")
        print(f"   安装命令: pip install pycocotools")
    except Exception as e:
        print(f"⚠️  COCO加载失败: {e}")
else:
    print(f"⚠️  COCO数据集路径不存在: {coco_path}")
    print(f"   COCO数据集下载: https://cocodataset.org/")
    print(f"   目录结构:")
    print(f"     coco/")
    print(f"       ├── annotations/")
    print(f"       │   └── instances_train2017.json")
    print(f"       └── train2017/")
    print(f"           └── *.jpg")

print()

# ==================== 测试6: Pascal VOC格式 ====================
print("📌 测试6: Pascal VOC格式")
print("-" * 70)

voc_path = Path("/home/user/datasets/VOCdevkit")
if voc_path.exists():
    try:
        dataset = torchvision.datasets.VOCDetection(
            root=str(voc_path.parent),
            year='2012',
            image_set='train',
            download=False
        )

        print(f"✅ Pascal VOC加载成功")
        print(f"   图片数量: {len(dataset)}")

    except Exception as e:
        print(f"⚠️  Pascal VOC加载失败: {e}")
else:
    print(f"⚠️  Pascal VOC路径不存在: {voc_path}")
    print(f"   目录结构:")
    print(f"     VOCdevkit/")
    print(f"       └── VOC2012/")
    print(f"           ├── Annotations/")
    print(f"           ├── ImageSets/")
    print(f"           └── JPEGImages/")

print()

# ==================== 测试7: 数据增强效果 ====================
print("📌 测试7: 数据增强效果")
print("-" * 70)

try:
    # 定义多种数据增强
    augmentations = {
        "基础增强": transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]),
        "强增强": transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
        ]),
        "最小增强": transforms.Compose([
            transforms.ToTensor(),
        ]),
    }

    # 使用CIFAR-10测试
    for aug_name, transform in augmentations.items():
        dataset = torchvision.datasets.CIFAR10(
            root=str(dataset_root),
            train=True,
            download=False,
            transform=transform
        )
        print(f"✅ {aug_name:15s} | 数据集大小: {len(dataset)}")

except Exception as e:
    print(f"❌ 数据增强测试失败: {e}")

print()

# ==================== 测试8: DataLoader性能 ====================
print("📌 测试8: DataLoader性能测试")
print("-" * 70)

try:
    import time

    dataset = torchvision.datasets.CIFAR10(
        root=str(dataset_root),
        train=True,
        download=False,
        transform=transform
    )

    # 测试不同num_workers
    for num_workers in [0, 2, 4]:
        loader = DataLoader(
            dataset,
            batch_size=128,
            shuffle=True,
            num_workers=num_workers
        )

        start_time = time.time()
        for i, (images, labels) in enumerate(loader):
            if i >= 10:  # 只测试10个batch
                break
        elapsed = time.time() - start_time

        print(f"✅ num_workers={num_workers} | 加载10个batch用时: {elapsed:.3f}秒")

except Exception as e:
    print(f"❌ DataLoader测试失败: {e}")

print()

# ==================== 总结 ====================
print("=" * 70)
print("📊 数据集测试总结")
print("=" * 70)
print()

print("✅ 已验证的数据集:")
print("   - CIFAR-10/100 (内置，自动下载)")
print()

print("📝 数据集准备建议:")
print()
print("   1. 快速测试（推荐）:")
print("      - 使用CIFAR-10（50,000训练+10,000测试，32x32）")
print("      - 自动下载，无需手动准备")
print("      - 适合验证训练流程")
print()

print("   2. 实际应用:")
print("      - 分类任务: 准备ImageNet格式数据集")
print("      - 检测任务: 准备COCO或VOC格式")
print("      - 分割任务: 准备带mask的数据集")
print()

print("   3. 自定义数据集:")
print("      - 创建文件夹结构: dataset_name/class_name/*.jpg")
print("      - 确保图片格式统一（JPG/PNG）")
print("      - 每个类别至少100张图片（建议）")
print()

print("💡 推荐的数据集配置:")
print()
print("   训练阶段:")
print("   - Batch Size: 32-64 (根据GPU显存调整)")
print("   - Num Workers: 2-4 (根据CPU核心数)")
print("   - Shuffle: True")
print("   - 数据增强: 随机裁剪、翻转、归一化")
print()

print("   验证阶段:")
print("   - Batch Size: 64-128")
print("   - Shuffle: False")
print("   - 数据增强: 仅Resize和归一化")
print()

print("📝 下一步:")
print("   1. 确保至少有一个可用的数据集（CIFAR-10已自动下载）")
print("   2. 运行环境测试: python test_environment.py")
print("   3. 运行简单训练: python test_simple_training.py")
print("   4. 配置完整的蒸馏训练任务")
print()

print("=" * 70)
print("测试完成！")
print("=" * 70)
