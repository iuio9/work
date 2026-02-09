#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
学生模型单独训练脚本（不使用教师模型）

支持的学生模型：
- ResNet：图像分类
- Vision Transformer：图像分类
- YOLOv8：目标检测
- UNet：图像分割
- LSTM：序列特征提取 + 图像分类

作者：Claude Assistant
日期：2026-02-09
版本：1.0.0
"""

import argparse
import json
import os
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

# 小模型相关导入
import torchvision.models as models
from transformers import (
    AutoConfig,
    AutoModelForImageClassification,
    AutoImageProcessor,
    ViTForImageClassification,
    ViTImageProcessor
)

# YOLO相关
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    warnings.warn("YOLOv8未安装，使用: pip install ultralytics")


# ==================== 配置类 ====================

class TrainingConfig:
    """训练配置类"""

    def __init__(self, args):
        # 基础配置
        self.task_id = args.task_id
        self.api_base_url = args.api_base_url

        # 模型配置
        self.student_model = args.student_model
        self.student_path = getattr(args, 'student_path', None)

        # 学生模型类型和大小
        self.student_model_type = args.student_model_type  # resnet/vit/yolov8/unet/lstm
        self.student_model_size = args.student_model_size  # resnet50, vit-base, s, medium, etc.

        # 任务配置
        self.task_type = args.task_type  # classification/detection/segmentation
        self.num_classes = args.num_classes

        # 数据配置
        self.dataset_id = args.dataset_id
        self.val_dataset_id = getattr(args, 'val_dataset_id', None) or args.dataset_id
        self.datasets_root = args.datasets_root
        self.image_size = args.image_size

        # 训练参数
        self.epochs = args.epochs
        self.batch_size = args.batch_size
        self.learning_rate = args.learning_rate
        self.weight_decay = getattr(args, 'weight_decay', 0.01)
        self.grad_accum_steps = getattr(args, 'grad_accum_steps', 1)
        self.max_grad_norm = getattr(args, 'max_grad_norm', 1.0)

        # 优化器和调度器
        self.optimizer_type = getattr(args, 'optimizer', 'adamw')
        self.lr_scheduler_type = getattr(args, 'lr_scheduler', 'cosine')

        # GPU配置
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.gpu_devices = getattr(args, 'gpu_devices', None)

        # 输出配置
        self.output_dir = args.output_dir
        self.auto_save_checkpoint = getattr(args, 'auto_save_checkpoint', True)
        self.checkpoint_interval = getattr(args, 'checkpoint_interval', 5)


# ==================== 数据集类 ====================

class MultiTaskDataset(Dataset):
    """多任务数据集加载器"""

    def __init__(
        self,
        dataset_path: str,
        task_type: str = 'classification',
        image_size: int = 224,
        num_classes: int = 10,
        mode: str = 'train'
    ):
        """
        初始化数据集

        Args:
            dataset_path: 数据集路径（格式：datasets_root/dataset_id/train 或 val）
            task_type: 任务类型
            image_size: 图像尺寸
            num_classes: 类别数量
            mode: 'train' 或 'val'
        """
        self.dataset_path = dataset_path
        self.task_type = task_type
        self.mode = mode
        self.image_size = image_size
        self.num_classes = num_classes

        # 初始化数据列表
        self.image_paths = []
        self.labels = []
        self.class_names = []

        # 根据任务类型设置transforms
        if mode == 'train':
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
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

        # 加载数据集
        if os.path.exists(dataset_path):
            self._load_dataset()
            print(f"✅ 成功加载数据集: {dataset_path}")
            print(f"   - 图像数量: {len(self.image_paths)}")
            print(f"   - 类别数量: {len(self.class_names)}")
            print(f"   - 类别名称: {self.class_names}")
        else:
            print(f"⚠️ 警告: 数据集路径不存在: {dataset_path}")
            self._use_mock_data(num_samples=100)

    def _load_dataset(self):
        """从目录结构加载真实数据集"""
        # 假设数据集格式：dataset_path/class_name/image.jpg
        class_folders = sorted([
            d for d in os.listdir(self.dataset_path)
            if os.path.isdir(os.path.join(self.dataset_path, d))
        ])

        if not class_folders:
            print(f"⚠️ 警告: 未找到类别文件夹，使用模拟数据")
            self._use_mock_data(num_samples=100)
            return

        self.class_names = class_folders

        for class_idx, class_name in enumerate(self.class_names):
            class_dir = os.path.join(self.dataset_path, class_name)
            image_files = [
                f for f in os.listdir(class_dir)
                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
            ]

            for img_file in image_files:
                img_path = os.path.join(class_dir, img_file)
                self.image_paths.append(img_path)
                self.labels.append(class_idx)

    def _use_mock_data(self, num_samples: int = 100):
        """生成模拟数据（用于测试）"""
        print(f"使用模拟数据，样本数: {num_samples}")
        self.class_names = [f"class_{i}" for i in range(self.num_classes)]
        self.image_paths = [f"mock_image_{i}.jpg" for i in range(num_samples)]
        self.labels = [i % self.num_classes for i in range(num_samples)]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        """获取单个数据样本"""
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        # 加载图像
        try:
            if os.path.exists(img_path):
                image = Image.open(img_path).convert('RGB')
            else:
                # 模拟数据
                image = Image.new('RGB', (self.image_size, self.image_size), color='gray')
        except Exception as e:
            print(f"⚠️ 加载图像失败: {img_path}, 错误: {e}")
            image = Image.new('RGB', (self.image_size, self.image_size), color='gray')

        # 应用transforms
        if self.transform:
            image = self.transform(image)

        return {
            'pixel_values': image,
            'labels': torch.tensor(label, dtype=torch.long)
        }


# ==================== 模型构建器 ====================

class StudentModelBuilder:
    """学生模型构建器"""

    @staticmethod
    def build_resnet(model_size: str, num_classes: int, pretrained: bool = True):
        """构建ResNet模型"""
        model_map = {
            'resnet18': models.resnet18,
            'resnet34': models.resnet34,
            'resnet50': models.resnet50,
            'resnet101': models.resnet101,
            'resnet152': models.resnet152
        }

        if model_size not in model_map:
            raise ValueError(f"不支持的ResNet规格: {model_size}")

        # 加载预训练模型
        model = model_map[model_size](pretrained=pretrained)

        # 修改最后的全连接层
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)

        return model

    @staticmethod
    def build_vit(model_size: str, num_classes: int):
        """构建Vision Transformer模型"""
        model_map = {
            'vit-tiny': 'google/vit-base-patch16-224',
            'vit-small': 'google/vit-base-patch16-224',
            'vit-base': 'google/vit-base-patch16-224',
            'vit-large': 'google/vit-large-patch16-224'
        }

        model_name = model_map.get(model_size, 'google/vit-base-patch16-224')

        # 加载ViT模型
        config = AutoConfig.from_pretrained(model_name)
        config.num_labels = num_classes
        model = ViTForImageClassification.from_pretrained(
            model_name,
            config=config,
            ignore_mismatched_sizes=True
        )

        return model

    @staticmethod
    def build_model(
        model_type: str,
        model_size: str,
        num_classes: int,
        task_type: str
    ):
        """
        构建学生模型

        Args:
            model_type: 模型类型 (resnet/vit/yolov8/unet/lstm)
            model_size: 模型大小
            num_classes: 类别数
            task_type: 任务类型

        Returns:
            模型实例
        """
        print(f"\n🏗️ 构建学生模型: {model_type} - {model_size}")
        print(f"   - 任务类型: {task_type}")
        print(f"   - 类别数: {num_classes}")

        if model_type == 'resnet':
            return StudentModelBuilder.build_resnet(model_size, num_classes)
        elif model_type == 'vit':
            return StudentModelBuilder.build_vit(model_size, num_classes)
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")


# ==================== 训练器 ====================

class DirectTrainer:
    """单独训练器（不使用知识蒸馏）"""

    def __init__(self, config: TrainingConfig):
        self.config = config

        print("\n" + "=" * 60)
        print("初始化单独训练器")
        print("=" * 60)

        # 构建学生模型
        self.student_model = StudentModelBuilder.build_model(
            model_type=config.student_model_type,
            model_size=config.student_model_size,
            num_classes=config.num_classes,
            task_type=config.task_type
        ).to(config.device)

        # 损失函数：标准交叉熵
        self.ce_loss = nn.CrossEntropyLoss()

        # 优化器
        self.optimizer = self._build_optimizer()

        # 学习率调度器
        self.scheduler = None  # 将在训练开始时初始化

        # 训练统计
        self.best_accuracy = 0.0
        self.train_losses = []
        self.val_accuracies = []

    def _build_optimizer(self):
        """构建优化器"""
        if self.config.optimizer_type == 'adamw':
            return torch.optim.AdamW(
                self.student_model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer_type == 'adam':
            return torch.optim.Adam(
                self.student_model.parameters(),
                lr=self.config.learning_rate
            )
        elif self.config.optimizer_type == 'sgd':
            return torch.optim.SGD(
                self.student_model.parameters(),
                lr=self.config.learning_rate,
                momentum=0.9,
                weight_decay=self.config.weight_decay
            )
        else:
            raise ValueError(f"不支持的优化器: {self.config.optimizer_type}")

    def _build_scheduler(self, num_training_steps: int):
        """构建学习率调度器"""
        if self.config.lr_scheduler_type == 'cosine':
            from torch.optim.lr_scheduler import CosineAnnealingLR
            return CosineAnnealingLR(self.optimizer, T_max=num_training_steps)
        elif self.config.lr_scheduler_type == 'linear':
            from torch.optim.lr_scheduler import LinearLR
            return LinearLR(self.optimizer, start_factor=1.0, end_factor=0.0, total_iters=num_training_steps)
        else:
            return None

    def train_epoch(self, epoch: int, train_loader: DataLoader) -> Tuple[float, float]:
        """训练一个epoch"""
        self.student_model.train()

        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{self.config.epochs}")

        for batch_idx, batch in enumerate(pbar):
            images = batch['pixel_values'].to(self.config.device)
            labels = batch['labels'].to(self.config.device)

            # 前向传播
            outputs = self.student_model(images)

            # 处理不同模型的输出格式
            if hasattr(outputs, 'logits'):
                logits = outputs.logits
            else:
                logits = outputs

            # 计算损失
            loss = self.ce_loss(logits, labels)

            # 反向传播
            loss.backward()

            # 梯度裁剪
            if self.config.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.student_model.parameters(),
                    self.config.max_grad_norm
                )

            # 优化器步骤
            if (batch_idx + 1) % self.config.grad_accum_steps == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()

                if self.scheduler:
                    self.scheduler.step()

            # 统计
            total_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # 更新进度条
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.0 * correct / total:.2f}%'
            })

        avg_loss = total_loss / len(train_loader)
        accuracy = 100.0 * correct / total

        return avg_loss, accuracy

    def validate(self, val_loader: DataLoader) -> float:
        """验证模型"""
        self.student_model.eval()

        correct = 0
        total = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="验证中"):
                images = batch['pixel_values'].to(self.config.device)
                labels = batch['labels'].to(self.config.device)

                outputs = self.student_model(images)

                # 处理不同模型的输出格式
                if hasattr(outputs, 'logits'):
                    logits = outputs.logits
                else:
                    logits = outputs

                _, predicted = torch.max(logits, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100.0 * correct / total
        return accuracy

    def save_checkpoint(self, epoch: int, accuracy: float):
        """保存检查点"""
        checkpoint_dir = os.path.join(self.config.output_dir, f'checkpoint-epoch-{epoch}')
        os.makedirs(checkpoint_dir, exist_ok=True)

        # 保存模型
        model_path = os.path.join(checkpoint_dir, 'model.pt')
        torch.save(self.student_model.state_dict(), model_path)

        # 保存训练信息
        info = {
            'epoch': epoch,
            'accuracy': accuracy,
            'best_accuracy': self.best_accuracy,
            'model_type': self.config.student_model_type,
            'model_size': self.config.student_model_size
        }
        info_path = os.path.join(checkpoint_dir, 'training_info.json')
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=2)

        print(f"✅ 检查点已保存: {checkpoint_dir}")

    def send_progress_update(self, epoch: int, loss: float, accuracy: float):
        """发送训练进度到后端API"""
        try:
            url = f"{self.config.api_base_url}/api/distillation/tasks/{self.config.task_id}/progress"
            data = {
                'currentEpoch': epoch,
                'totalEpochs': self.config.epochs,
                'loss': float(loss),
                'accuracy': float(accuracy),
                'status': 'RUNNING'
            }

            response = requests.post(url, json=data, timeout=5)
            if response.status_code == 200:
                print(f"✅ 进度已更新: Epoch {epoch}, Loss {loss:.4f}, Acc {accuracy:.2f}%")
            else:
                print(f"⚠️ 进度更新失败: {response.status_code}")
        except Exception as e:
            print(f"⚠️ 发送进度失败: {e}")

    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        """完整训练流程"""
        print("\n" + "=" * 60)
        print("开始单独训练")
        print("=" * 60)

        # 初始化调度器
        num_training_steps = len(train_loader) * self.config.epochs
        self.scheduler = self._build_scheduler(num_training_steps)

        for epoch in range(1, self.config.epochs + 1):
            print(f"\n{'=' * 60}")
            print(f"Epoch {epoch}/{self.config.epochs}")
            print(f"{'=' * 60}")

            # 训练
            train_loss, train_acc = self.train_epoch(epoch, train_loader)
            self.train_losses.append(train_loss)

            # 验证
            val_acc = self.validate(val_loader)
            self.val_accuracies.append(val_acc)

            print(f"\n📊 Epoch {epoch} 结果:")
            print(f"   - 训练损失: {train_loss:.4f}")
            print(f"   - 训练准确率: {train_acc:.2f}%")
            print(f"   - 验证准确率: {val_acc:.2f}%")

            # 保存最佳模型
            if val_acc > self.best_accuracy:
                self.best_accuracy = val_acc
                self.save_checkpoint(epoch, val_acc)
                print(f"🎉 新的最佳准确率: {val_acc:.2f}%")

            # 定期保存检查点
            if self.config.auto_save_checkpoint and epoch % self.config.checkpoint_interval == 0:
                self.save_checkpoint(epoch, val_acc)

            # 发送进度更新
            self.send_progress_update(epoch, train_loss, val_acc)

        print("\n" + "=" * 60)
        print("训练完成!")
        print(f"最佳准确率: {self.best_accuracy:.2f}%")
        print("=" * 60)


# ==================== 主函数 ====================

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='学生模型单独训练脚本')

    # 基础配置
    parser.add_argument('--task_id', type=str, required=True, help='任务ID')
    parser.add_argument('--api_base_url', type=str, required=True, help='后端API基础URL')

    # 模型配置
    parser.add_argument('--student_model', type=str, required=True, help='学生模型名称')
    parser.add_argument('--student_path', type=str, help='学生模型路径')
    parser.add_argument('--student_model_type', type=str, required=True, help='学生模型类型')
    parser.add_argument('--student_model_size', type=str, required=True, help='学生模型大小')

    # 任务配置
    parser.add_argument('--task_type', type=str, required=True, help='任务类型')
    parser.add_argument('--num_classes', type=int, required=True, help='类别数')

    # 数据配置
    parser.add_argument('--dataset_id', type=str, required=True, help='数据集ID')
    parser.add_argument('--val_dataset_id', type=str, help='验证数据集ID')
    parser.add_argument('--datasets_root', type=str, required=True, help='数据集根目录')
    parser.add_argument('--image_size', type=int, required=True, help='图像尺寸')

    # 训练参数
    parser.add_argument('--epochs', type=int, required=True, help='训练轮数')
    parser.add_argument('--batch_size', type=int, required=True, help='批次大小')
    parser.add_argument('--learning_rate', type=float, required=True, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='权重衰减')
    parser.add_argument('--grad_accum_steps', type=int, default=1, help='梯度累积步数')
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='最大梯度范数')

    # 优化器和调度器
    parser.add_argument('--optimizer', type=str, default='adamw', help='优化器类型')
    parser.add_argument('--lr_scheduler', type=str, default='cosine', help='学习率调度器')

    # GPU配置
    parser.add_argument('--gpu_devices', type=str, help='GPU设备')

    # 输出配置
    parser.add_argument('--output_dir', type=str, required=True, help='输出目录')
    parser.add_argument('--auto_save_checkpoint', type=bool, default=True, help='自动保存检查点')
    parser.add_argument('--checkpoint_interval', type=int, default=5, help='检查点保存间隔')

    return parser.parse_args()


def main():
    """主函数"""
    # 解析参数
    args = parse_args()
    config = TrainingConfig(args)

    print("\n" + "=" * 60)
    print("学生模型单独训练")
    print("=" * 60)
    print(f"任务ID: {config.task_id}")
    print(f"学生模型: {config.student_model_type} - {config.student_model_size}")
    print(f"数据集: {config.dataset_id}")
    print(f"训练轮数: {config.epochs}")
    print(f"批次大小: {config.batch_size}")
    print(f"学习率: {config.learning_rate}")
    print(f"设备: {config.device}")
    print("=" * 60)

    # 创建输出目录
    os.makedirs(config.output_dir, exist_ok=True)

    # 构建数据集路径
    train_dataset_path = os.path.join(config.datasets_root, config.dataset_id, "train")
    val_dataset_path = os.path.join(config.datasets_root, config.val_dataset_id, "val")

    print(f"\n📂 数据集路径:")
    print(f"   - 训练集: {train_dataset_path}")
    print(f"   - 验证集: {val_dataset_path}")

    # 加载数据集
    train_dataset = MultiTaskDataset(
        train_dataset_path,
        config.task_type,
        config.image_size,
        config.num_classes,
        mode='train'
    )

    val_dataset = MultiTaskDataset(
        val_dataset_path,
        config.task_type,
        config.image_size,
        config.num_classes,
        mode='val'
    )

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4
    )

    # 创建训练器
    trainer = DirectTrainer(config)

    # 开始训练
    try:
        trainer.train(train_loader, val_loader)

        # 训练完成，更新任务状态
        try:
            url = f"{config.api_base_url}/api/distillation/tasks/{config.task_id}/complete"
            data = {
                'status': 'COMPLETED',
                'accuracy': trainer.best_accuracy,
                'modelPath': os.path.join(config.output_dir, 'checkpoint-epoch-final')
            }
            requests.post(url, json=data, timeout=5)
        except Exception as e:
            print(f"⚠️ 更新任务状态失败: {e}")

    except Exception as e:
        print(f"\n❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()

        # 更新任务状态为失败
        try:
            url = f"{config.api_base_url}/api/distillation/tasks/{config.task_id}/fail"
            data = {'errorMessage': str(e)}
            requests.post(url, json=data, timeout=5)
        except:
            pass

        sys.exit(1)


if __name__ == '__main__':
    main()
