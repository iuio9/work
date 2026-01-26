#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
本地测试版本 - Qwen2.5-VL知识蒸馏训练脚本

用途：脱离前后端系统，独立运行测试训练脚本
适合：验证数据集加载、模型训练流程是否正常

使用方法：
    python test_local.py

作者：AI Assistant
日期：2026-01-26
"""

import os
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

# Qwen2.5-VL相关导入
try:
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    QWEN_AVAILABLE = True
except ImportError:
    QWEN_AVAILABLE = False
    warnings.warn("⚠️ Qwen2_5_VL模型库未安装，将使用模拟模式")

# 小模型相关导入
import torchvision.models as models
from transformers import (
    AutoConfig,
    AutoModelForImageClassification,
    AutoImageProcessor,
)
from peft import LoraConfig, get_peft_model, TaskType


# ==================== 配置类 ====================

class SimpleConfig:
    """简化的训练配置类"""

    def __init__(self):
        # ========== 关键配置（需要修改） ==========
        # 数据集根目录（修改为您的实际路径）
        self.datasets_root = r"D:\pythonProject2\datasets"

        # 数据集ID（子目录名）
        self.dataset_id = "cifar10"

        # 教师模型路径
        self.teacher_path = r"D:\pythonProject2\models\Qwen2___5-VL-3B-Instruct"

        # 输出目录
        self.output_dir = r"D:\pythonProject2\test_output"

        # ========== 任务配置 ==========
        self.task_type = "classification"  # classification/detection/segmentation
        self.num_classes = 10  # CIFAR-10有10个类别
        self.image_size = 224

        # ========== 学生模型配置 ==========
        self.student_model_type = "resnet"  # resnet/vit/yolov8/unet/lstm
        self.student_model_size = "resnet18"  # resnet18/resnet50/vit-base等

        # ========== 训练参数 ==========
        self.epochs = 2  # 测试用，只训练2个epoch
        self.batch_size = 8  # 较小的batch size，降低内存占用
        self.learning_rate = 0.0001

        # ========== LoRA配置 ==========
        self.lora_rank = 8
        self.lora_alpha = 16
        self.lora_dropout = 0.1

        # ========== 知识蒸馏参数 ==========
        self.temperature = 4.0
        self.hard_label_weight = 0.3
        self.soft_label_weight = 0.7
        self.distillation_type = "logit"  # logit/feature/hybrid

        # ========== 设备配置 ==========
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ========== 优化器配置 ==========
        self.optimizer_type = "adamw"
        self.weight_decay = 0.01

        # ========== 其他 ==========
        self.save_interval = 1  # 每个epoch都保存
        self.log_interval = 10  # 每10个batch打印一次


# ==================== 数据集类 ====================

class CIFAR10Dataset(Dataset):
    """
    CIFAR-10数据集加载器

    期望的目录结构：
    dataset_root/cifar10/
      ├── train/
      │   ├── airplane/
      │   ├── automobile/
      │   └── ...
      └── val/
          ├── airplane/
          └── ...
    """

    def __init__(
        self,
        dataset_path: str,
        image_size: int = 224,
        mode: str = 'train'
    ):
        self.dataset_path = dataset_path
        self.image_size = image_size
        self.mode = mode

        # 存储所有图像路径和标签
        self.image_paths = []
        self.labels = []
        self.class_names = []

        # 加载数据集
        if os.path.exists(dataset_path):
            self._load_dataset()
        else:
            print(f"❌ 错误: 数据集路径不存在: {dataset_path}")
            print(f"请先运行 convert_cifar10.py 转换数据集")
            sys.exit(1)

        # 数据增强
        if mode == 'train':
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])

    def _load_dataset(self):
        """从目录结构加载数据集"""
        print(f"📂 加载数据集: {self.dataset_path}")

        # 获取所有类别文件夹
        class_folders = sorted([d for d in os.listdir(self.dataset_path)
                               if os.path.isdir(os.path.join(self.dataset_path, d))])

        if not class_folders:
            print(f"❌ 错误: 在 {self.dataset_path} 中未找到类别文件夹")
            sys.exit(1)

        self.class_names = class_folders
        print(f"✅ 找到 {len(self.class_names)} 个类别: {self.class_names}")

        # 遍历每个类别文件夹
        for class_idx, class_name in enumerate(self.class_names):
            class_dir = os.path.join(self.dataset_path, class_name)

            # 获取该类别下的所有图像文件
            image_files = [f for f in os.listdir(class_dir)
                          if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

            for img_file in image_files:
                img_path = os.path.join(class_dir, img_file)
                self.image_paths.append(img_path)
                self.labels.append(class_idx)

        print(f"✅ 加载完成: 共 {len(self.image_paths)} 张图像")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"⚠️ 加载图像失败 {img_path}: {e}")
            # 生成随机图像作为后备
            image_array = np.random.randint(0, 255, (self.image_size, self.image_size, 3), dtype=np.uint8)
            image = Image.fromarray(image_array)

        # 应用变换
        image = self.transform(image)

        return {'pixel_values': image, 'labels': label}


# ==================== 学生模型加载器 ====================

class StudentModelLoader:
    """学生模型加载器（简化版）"""

    @staticmethod
    def load_resnet(model_size: str, num_classes: int):
        """加载ResNet模型"""
        print(f"📦 加载学生模型: ResNet-{model_size}")

        if model_size == "resnet18":
            model = models.resnet18(pretrained=False)
        elif model_size == "resnet34":
            model = models.resnet34(pretrained=False)
        elif model_size == "resnet50":
            model = models.resnet50(pretrained=False)
        else:
            model = models.resnet18(pretrained=False)

        # 修改最后一层以匹配类别数
        model.fc = nn.Linear(model.fc.in_features, num_classes)

        return model


# ==================== 简化的训练器 ====================

class SimpleTrainer:
    """简化的训练器（不依赖后端API）"""

    def __init__(self, config: SimpleConfig):
        self.config = config

        print("\n" + "=" * 60)
        print("本地测试训练器初始化")
        print("=" * 60)

        # 创建输出目录
        os.makedirs(config.output_dir, exist_ok=True)
        print(f"✅ 输出目录: {config.output_dir}")

        # 加载数据集
        train_path = os.path.join(config.datasets_root, config.dataset_id, "train")
        val_path = os.path.join(config.datasets_root, config.dataset_id, "val")

        print(f"\n训练集路径: {train_path}")
        print(f"验证集路径: {val_path}")

        self.train_dataset = CIFAR10Dataset(train_path, config.image_size, mode='train')
        self.val_dataset = CIFAR10Dataset(val_path, config.image_size, mode='val')

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=0  # Windows上设为0避免multiprocessing问题
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=0
        )

        print(f"\n✅ 训练样本数: {len(self.train_dataset)}")
        print(f"✅ 验证样本数: {len(self.val_dataset)}")
        print(f"✅ 训练批次数: {len(self.train_loader)}")

        # 加载学生模型
        print(f"\n正在加载学生模型...")
        self.student_model = StudentModelLoader.load_resnet(
            config.student_model_size,
            config.num_classes
        ).to(config.device)

        # 优化器
        self.optimizer = torch.optim.AdamW(
            self.student_model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )

        # 损失函数
        self.criterion = nn.CrossEntropyLoss()

        print(f"✅ 设备: {config.device}")
        print(f"✅ 学生模型参数量: {sum(p.numel() for p in self.student_model.parameters()):,}")

    def train_epoch(self, epoch: int):
        """训练一个epoch"""
        self.student_model.train()
        total_loss = 0
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.config.epochs}")

        for batch_idx, batch in enumerate(pbar):
            images = batch['pixel_values'].to(self.config.device)
            labels = batch['labels'].to(self.config.device)

            # 前向传播
            self.optimizer.zero_grad()
            outputs = self.student_model(images)
            loss = self.criterion(outputs, labels)

            # 反向传播
            loss.backward()
            self.optimizer.step()

            # 统计
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # 更新进度条
            if (batch_idx + 1) % self.config.log_interval == 0:
                acc = 100. * correct / total
                avg_loss = total_loss / (batch_idx + 1)
                pbar.set_postfix({
                    'loss': f'{avg_loss:.4f}',
                    'acc': f'{acc:.2f}%'
                })

        # Epoch统计
        avg_loss = total_loss / len(self.train_loader)
        acc = 100. * correct / total

        print(f"\n📊 Epoch {epoch} 训练结果:")
        print(f"   Loss: {avg_loss:.4f}")
        print(f"   Accuracy: {acc:.2f}%")

        return avg_loss, acc

    def validate(self):
        """验证模型"""
        self.student_model.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="验证中"):
                images = batch['pixel_values'].to(self.config.device)
                labels = batch['labels'].to(self.config.device)

                outputs = self.student_model(images)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        avg_loss = total_loss / len(self.val_loader)
        acc = 100. * correct / total

        print(f"\n📊 验证结果:")
        print(f"   Loss: {avg_loss:.4f}")
        print(f"   Accuracy: {acc:.2f}%")

        return avg_loss, acc

    def train(self):
        """完整训练流程"""
        print("\n" + "=" * 60)
        print("开始训练")
        print("=" * 60)

        best_acc = 0
        start_time = time.time()

        for epoch in range(1, self.config.epochs + 1):
            print(f"\n{'=' * 60}")
            print(f"Epoch {epoch}/{self.config.epochs}")
            print('=' * 60)

            # 训练
            train_loss, train_acc = self.train_epoch(epoch)

            # 验证
            val_loss, val_acc = self.validate()

            # 保存最佳模型
            if val_acc > best_acc:
                best_acc = val_acc
                save_path = os.path.join(self.config.output_dir, "best_model.pth")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.student_model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_acc': train_acc,
                    'val_acc': val_acc,
                    'best_acc': best_acc,
                }, save_path)
                print(f"\n💾 保存最佳模型: {save_path} (Acc: {best_acc:.2f}%)")

        # 训练结束
        elapsed_time = time.time() - start_time
        print("\n" + "=" * 60)
        print("✅ 训练完成！")
        print("=" * 60)
        print(f"总耗时: {elapsed_time/60:.2f} 分钟")
        print(f"最佳验证准确率: {best_acc:.2f}%")
        print(f"模型保存位置: {self.config.output_dir}")


# ==================== 主函数 ====================

def main():
    print("\n" + "=" * 60)
    print("Qwen2.5-VL 知识蒸馏训练脚本 - 本地测试版")
    print("=" * 60)

    # 创建配置
    config = SimpleConfig()

    # 打印配置信息
    print("\n📋 训练配置:")
    print(f"   数据集根目录: {config.datasets_root}")
    print(f"   数据集ID: {config.dataset_id}")
    print(f"   教师模型: {config.teacher_path}")
    print(f"   学生模型: {config.student_model_type}-{config.student_model_size}")
    print(f"   任务类型: {config.task_type}")
    print(f"   类别数: {config.num_classes}")
    print(f"   训练轮数: {config.epochs}")
    print(f"   批次大小: {config.batch_size}")
    print(f"   学习率: {config.learning_rate}")
    print(f"   输出目录: {config.output_dir}")

    # 检查关键路径
    print("\n🔍 检查路径...")
    train_path = os.path.join(config.datasets_root, config.dataset_id, "train")
    val_path = os.path.join(config.datasets_root, config.dataset_id, "val")

    if not os.path.exists(train_path):
        print(f"❌ 错误: 训练集路径不存在: {train_path}")
        print(f"请先运行 convert_cifar10.py 转换CIFAR-10数据集")
        return

    if not os.path.exists(val_path):
        print(f"❌ 错误: 验证集路径不存在: {val_path}")
        print(f"请先运行 convert_cifar10.py 转换CIFAR-10数据集")
        return

    print("✅ 所有路径检查通过")

    # 创建训练器
    trainer = SimpleTrainer(config)

    # 开始训练
    trainer.train()

    print("\n✅ 测试运行完成！")
    print("如果一切正常，可以将此脚本集成到前后端系统中。")


if __name__ == "__main__":
    main()
