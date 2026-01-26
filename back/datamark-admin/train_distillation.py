#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
大小模型协同训练（知识蒸馏）训练脚本

功能：
1. 从命令行参数读取所有训练配置
2. 加载教师模型和学生模型
3. 应用LoRA微调配置
4. 实现知识蒸馏训练循环
5. 定期回调后端API更新训练进度
6. 保存训练checkpoints和最终模型

作者：AI Assistant
日期：2025-01-25
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from transformers import (
    AutoConfig,
    AutoModelForImageClassification,
    AutoImageProcessor,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model, TaskType

# ==================== 配置类 ====================

class TrainingConfig:
    """训练配置类，用于存储所有训练参数"""

    def __init__(self, args):
        # 基础配置
        self.task_id = args.task_id
        self.api_base_url = args.api_base_url

        # 模型配置
        self.teacher_model = args.teacher_model
        self.student_model = args.student_model
        self.teacher_path = args.teacher_path
        self.student_path = args.student_path

        # 数据配置
        self.dataset_id = args.dataset_id
        self.val_dataset_id = args.val_dataset_id
        self.datasets_root = args.datasets_root

        # 训练参数
        self.epochs = args.epochs
        self.batch_size = args.batch_size
        self.learning_rate = args.learning_rate

        # 优化器配置
        self.optimizer = args.optimizer  # adamw, adam, sgd
        self.lr_scheduler = args.lr_scheduler  # cosine, linear, constant
        self.weight_decay = args.weight_decay
        self.grad_accum_steps = args.grad_accum_steps
        self.max_grad_norm = args.max_grad_norm

        # GPU配置
        self.gpu_devices = self._parse_gpu_devices(args.gpu_devices)
        self.auto_save_checkpoint = args.auto_save_checkpoint
        self.checkpoint_interval = args.checkpoint_interval

        # LoRA配置
        self.lora_rank = args.lora_rank
        self.lora_alpha = args.lora_alpha
        self.lora_dropout = args.lora_dropout
        self.lora_target_modules = self._parse_list(args.lora_target_modules)
        self.lora_bias = args.lora_bias

        # 知识蒸馏配置
        self.temperature = args.temperature
        self.hard_label_weight = args.hard_label_weight
        self.soft_label_weight = args.soft_label_weight
        self.distill_loss_type = args.distill_loss_type  # kl_div, mse, cosine

        # 输出配置
        self.output_dir = args.output_dir

    def _parse_gpu_devices(self, gpu_str: str) -> List[int]:
        """解析GPU设备列表"""
        if not gpu_str or gpu_str == "":
            return [0]
        return [int(x.strip()) for x in gpu_str.split(",")]

    def _parse_list(self, list_str: str) -> List[str]:
        """解析逗号分隔的字符串列表"""
        if not list_str or list_str == "":
            return []
        return [x.strip() for x in list_str.split(",")]


# ==================== 数据集类 ====================

class ImageAnnotationDataset(Dataset):
    """
    图像标注数据集
    支持：图像分类（如CIFAR-10）

    期望的目录结构：
    dataset_path/
      ├── class1/
      │   ├── img1.jpg
      │   └── img2.jpg
      ├── class2/
      │   └── ...
    """

    def __init__(
        self,
        dataset_path: str,
        image_processor=None,
        num_samples: int = None,
        image_size: int = 224,
        num_classes: int = None,
        is_training: bool = True
    ):
        self.dataset_path = dataset_path
        self.image_processor = image_processor
        self.image_size = image_size
        self.num_classes = num_classes
        self.is_training = is_training

        # 存储所有图像路径和标签
        self.image_paths = []
        self.labels = []
        self.class_names = []

        # 检查数据集路径是否存在
        if os.path.exists(dataset_path):
            self._load_dataset()
        else:
            print(f"⚠️ 警告: 数据集路径不存在: {dataset_path}")
            print(f"使用模拟数据进行演示")
            self._use_mock_data(num_samples or 1000)

        # 数据增强（训练集）
        if self.is_training:
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
            # 验证集不使用数据增强
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])

    def _load_dataset(self):
        """从目录结构加载真实数据集"""
        print(f"📂 加载数据集: {self.dataset_path}")

        # 获取所有类别文件夹
        class_folders = sorted([d for d in os.listdir(self.dataset_path)
                               if os.path.isdir(os.path.join(self.dataset_path, d))])

        if not class_folders:
            print(f"⚠️ 警告: 在 {self.dataset_path} 中未找到类别文件夹")
            self._use_mock_data(1000)
            return

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

        # 更新类别数
        if self.num_classes is None:
            self.num_classes = len(self.class_names)

    def _use_mock_data(self, num_samples):
        """使用模拟数据（当真实数据不可用时）"""
        print(f"🎭 使用模拟数据: {num_samples} 个样本")
        self.class_names = [f"class_{i}" for i in range(self.num_classes or 10)]

        # 生成模拟路径和标签
        for i in range(num_samples):
            self.image_paths.append(f"mock_image_{i}.jpg")
            self.labels.append(i % len(self.class_names))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        # 尝试加载真实图像
        try:
            if os.path.exists(img_path):
                image = Image.open(img_path).convert('RGB')
            else:
                # 生成随机图像（模拟数据）
                import numpy as np
                image_array = np.random.randint(0, 255, (self.image_size, self.image_size, 3), dtype=np.uint8)
                image = Image.fromarray(image_array)
        except Exception as e:
            print(f"⚠️ 加载图像失败 {img_path}: {e}")
            # 生成随机图像作为后备
            import numpy as np
            image_array = np.random.randint(0, 255, (self.image_size, self.image_size, 3), dtype=np.uint8)
            image = Image.fromarray(image_array)

        # 应用变换
        if self.transform:
            image = self.transform(image)

        return image, label

        # 应用变换
        pixel_values = self.transform(image)

        # 标签（这里用随机标签演示）
        # 实际应该从数据库读取真实标注
        if self.num_classes:
            label = torch.randint(0, self.num_classes, (1,)).item()
        else:
            label = torch.randint(0, 10, (1,)).item()  # 默认10个类别

        return {
            'pixel_values': pixel_values,
            'labels': label
        }


# ==================== 知识蒸馏训练器 ====================

class DistillationTrainer:
    """知识蒸馏训练器"""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = self._setup_device()
        self.tokenizer = None
        self.teacher_model = None
        self.student_model = None
        self.optimizer = None
        self.scheduler = None

    def _setup_device(self) -> torch.device:
        """设置训练设备"""
        if torch.cuda.is_available():
            # 使用配置的第一个GPU
            device_id = self.config.gpu_devices[0]
            device = torch.device(f"cuda:{device_id}")
            print(f"✓ 使用GPU设备: cuda:{device_id}")
        else:
            device = torch.device("cpu")
            print("⚠ 使用CPU训练（建议使用GPU）")
        return device

    def load_models(self):
        """加载教师模型和学生模型（图像分类）"""
        print(f"\n{'='*60}")
        print("📚 加载图像分类模型...")
        print(f"{'='*60}")

        # 加载图像处理器
        print(f"正在加载图像处理器: {self.config.teacher_path}")
        self.image_processor = AutoImageProcessor.from_pretrained(
            self.config.teacher_path,
            trust_remote_code=True
        )

        # 确定类别数（从配置或默认）
        # TODO: 从数据库读取实际的类别数
        num_classes = getattr(self.config, 'num_classes', 10)  # 默认10个类别
        print(f"类别数: {num_classes}")

        # 加载教师模型（图像分类）
        print(f"正在加载教师模型: {self.config.teacher_path}")
        teacher_config = AutoConfig.from_pretrained(self.config.teacher_path)
        teacher_config.num_labels = num_classes

        self.teacher_model = AutoModelForImageClassification.from_pretrained(
            self.config.teacher_path,
            config=teacher_config,
            trust_remote_code=True,
            ignore_mismatched_sizes=True  # 允许类别数不匹配
        )
        self.teacher_model.to(self.device)
        self.teacher_model.eval()  # 教师模型设为评估模式

        # 冻结教师模型参数
        for param in self.teacher_model.parameters():
            param.requires_grad = False

        print(f"✓ 教师模型加载成功，参数量: {sum(p.numel() for p in self.teacher_model.parameters()):,}")

        # 加载学生模型
        print(f"正在加载学生模型: {self.config.student_path or '基于教师模型创建'}")

        if self.config.student_path:
            # 从预训练模型加载
            student_config = AutoConfig.from_pretrained(self.config.student_path)
            student_config.num_labels = num_classes
            self.student_model = AutoModelForImageClassification.from_pretrained(
                self.config.student_path,
                config=student_config,
                ignore_mismatched_sizes=True
            )
        else:
            # 创建更小的学生模型
            student_config = AutoConfig.from_pretrained(self.config.teacher_path)
            student_config.num_labels = num_classes

            # 减小模型尺寸（根据模型类型调整）
            if hasattr(student_config, 'num_hidden_layers'):
                student_config.num_hidden_layers = max(6, student_config.num_hidden_layers // 2)
            if hasattr(student_config, 'hidden_size'):
                student_config.hidden_size = max(384, student_config.hidden_size // 2)
            if hasattr(student_config, 'intermediate_size'):
                student_config.intermediate_size = max(1536, student_config.intermediate_size // 2)

            self.student_model = AutoModelForImageClassification.from_config(student_config)

        # 应用LoRA配置
        self._apply_lora()

        self.student_model.to(self.device)

        print(f"✓ 学生模型加载成功，参数量: {sum(p.numel() for p in self.student_model.parameters()):,}")
        print(f"  可训练参数: {sum(p.numel() for p in self.student_model.parameters() if p.requires_grad):,}")

    def _apply_lora(self):
        """应用LoRA配置到学生模型"""
        print(f"\n🔧 应用LoRA配置...")
        print(f"  Rank: {self.config.lora_rank}")
        print(f"  Alpha: {self.config.lora_alpha}")
        print(f"  Dropout: {self.config.lora_dropout}")
        print(f"  Target Modules: {self.config.lora_target_modules or 'default'}")

        lora_config = LoraConfig(
            task_type=TaskType.IMAGE_CLASSIFICATION,  # 图像分类任务
            inference_mode=False,
            r=self.config.lora_rank,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=self.config.lora_target_modules if self.config.lora_target_modules else None,
            bias=self.config.lora_bias,
        )

        self.student_model = get_peft_model(self.student_model, lora_config)
        self.student_model.print_trainable_parameters()

    def setup_optimizer(self, num_training_steps: int):
        """设置优化器和学习率调度器"""
        print(f"\n⚙️ 设置优化器...")

        # 选择优化器
        if self.config.optimizer == "adamw":
            self.optimizer = torch.optim.AdamW(
                self.student_model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer == "adam":
            self.optimizer = torch.optim.Adam(
                self.student_model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer == "sgd":
            self.optimizer = torch.optim.SGD(
                self.student_model.parameters(),
                lr=self.config.learning_rate,
                momentum=0.9,
                weight_decay=self.config.weight_decay
            )
        else:
            raise ValueError(f"不支持的优化器: {self.config.optimizer}")

        print(f"✓ 优化器: {self.config.optimizer.upper()}")
        print(f"  学习率: {self.config.learning_rate}")
        print(f"  权重衰减: {self.config.weight_decay}")

        # 选择学习率调度器
        num_warmup_steps = int(0.1 * num_training_steps)  # 10% warmup

        if self.config.lr_scheduler == "cosine":
            self.scheduler = get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps
            )
        elif self.config.lr_scheduler == "linear":
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps
            )
        else:  # constant
            self.scheduler = None

        print(f"✓ 学习率调度器: {self.config.lr_scheduler or 'constant'}")

    def compute_distillation_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        计算知识蒸馏损失

        Returns:
            total_loss: 总损失
            loss_dict: 损失详情字典
        """
        # 硬标签损失（真实标签）
        hard_loss = F.cross_entropy(student_logits, labels)

        # 软标签损失（教师输出）
        if self.config.distill_loss_type == "kl_div":
            # KL散度损失
            soft_loss = F.kl_div(
                F.log_softmax(student_logits / self.config.temperature, dim=-1),
                F.softmax(teacher_logits / self.config.temperature, dim=-1),
                reduction='batchmean'
            ) * (self.config.temperature ** 2)
        elif self.config.distill_loss_type == "mse":
            # MSE损失
            soft_loss = F.mse_loss(
                F.softmax(student_logits / self.config.temperature, dim=-1),
                F.softmax(teacher_logits / self.config.temperature, dim=-1)
            )
        else:
            # 默认使用KL散度
            soft_loss = F.kl_div(
                F.log_softmax(student_logits / self.config.temperature, dim=-1),
                F.softmax(teacher_logits / self.config.temperature, dim=-1),
                reduction='batchmean'
            ) * (self.config.temperature ** 2)

        # 组合损失
        total_loss = (
            self.config.hard_label_weight * hard_loss +
            self.config.soft_label_weight * soft_loss
        )

        loss_dict = {
            'total_loss': total_loss.item(),
            'hard_loss': hard_loss.item(),
            'soft_loss': soft_loss.item()
        }

        return total_loss, loss_dict

    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int
    ) -> Dict[str, float]:
        """训练一个epoch（图像分类）"""
        self.student_model.train()

        total_loss = 0
        total_hard_loss = 0
        total_soft_loss = 0
        correct = 0
        total = 0

        num_batches = len(train_loader)

        for batch_idx, batch in enumerate(train_loader):
            # 将数据移到设备
            pixel_values = batch['pixel_values'].to(self.device)
            labels = batch['labels'].to(self.device)

            # 教师模型前向传播（不计算梯度）
            with torch.no_grad():
                teacher_outputs = self.teacher_model(pixel_values=pixel_values)
                teacher_logits = teacher_outputs.logits

            # 学生模型前向传播
            student_outputs = self.student_model(pixel_values=pixel_values)
            student_logits = student_outputs.logits

            # 计算损失
            loss, loss_dict = self.compute_distillation_loss(
                student_logits, teacher_logits, labels
            )

            # 梯度累积
            loss = loss / self.config.grad_accum_steps
            loss.backward()

            # 梯度累积后更新
            if (batch_idx + 1) % self.config.grad_accum_steps == 0:
                # 梯度裁剪
                if self.config.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.student_model.parameters(),
                        self.config.max_grad_norm
                    )

                self.optimizer.step()
                if self.scheduler:
                    self.scheduler.step()
                self.optimizer.zero_grad()

            # 统计
            total_loss += loss_dict['total_loss']
            total_hard_loss += loss_dict['hard_loss']
            total_soft_loss += loss_dict['soft_loss']

            # 计算准确率
            _, predicted = torch.max(student_logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # 打印进度
            if (batch_idx + 1) % 10 == 0:
                avg_loss = total_loss / (batch_idx + 1)
                accuracy = 100.0 * correct / total
                print(f"  Batch [{batch_idx+1}/{num_batches}] "
                      f"Loss: {avg_loss:.4f} | Acc: {accuracy:.2f}%")

        # Epoch统计
        epoch_metrics = {
            'loss': total_loss / num_batches,
            'hard_loss': total_hard_loss / num_batches,
            'soft_loss': total_soft_loss / num_batches,
            'accuracy': 100.0 * correct / total
        }

        return epoch_metrics

    @torch.no_grad()
    def evaluate(self, val_loader: DataLoader) -> Dict[str, float]:
        """评估模型（图像分类）"""
        self.student_model.eval()

        total_loss = 0
        correct = 0
        total = 0

        for batch in val_loader:
            pixel_values = batch['pixel_values'].to(self.device)
            labels = batch['labels'].to(self.device)

            # 学生模型前向传播
            outputs = self.student_model(pixel_values=pixel_values)
            logits = outputs.logits

            # 计算损失
            loss = F.cross_entropy(logits, labels)
            total_loss += loss.item()

            # 计算准确率
            _, predicted = torch.max(logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        metrics = {
            'val_loss': total_loss / len(val_loader),
            'val_accuracy': 100.0 * correct / total
        }

        return metrics

    def save_checkpoint(self, epoch: int, metrics: Dict[str, float]):
        """保存checkpoint"""
        checkpoint_dir = os.path.join(self.config.output_dir, f"checkpoint-epoch-{epoch}")
        os.makedirs(checkpoint_dir, exist_ok=True)

        # 保存模型
        self.student_model.save_pretrained(checkpoint_dir)
        self.image_processor.save_pretrained(checkpoint_dir)

        # 保存训练状态
        state = {
            'epoch': epoch,
            'metrics': metrics,
            'optimizer_state': self.optimizer.state_dict(),
        }

        if self.scheduler:
            state['scheduler_state'] = self.scheduler.state_dict()

        torch.save(state, os.path.join(checkpoint_dir, 'training_state.pt'))

        print(f"✓ Checkpoint已保存: {checkpoint_dir}")

    def update_progress_to_backend(
        self,
        epoch: int,
        metrics: Dict[str, float]
    ):
        """回调后端API更新训练进度"""
        try:
            url = f"{self.config.api_base_url}/model-distillation/tasks/{self.config.task_id}/progress"

            params = {
                'currentEpoch': epoch,
                'accuracy': metrics.get('accuracy', 0),
                'loss': metrics.get('loss', 0)
            }

            response = requests.put(url, params=params, timeout=10)

            if response.status_code == 200:
                print(f"✓ 进度已更新到后端 (Epoch {epoch})")
            else:
                print(f"⚠ 更新进度失败: {response.status_code}")

        except Exception as e:
            print(f"⚠ 更新进度异常: {str(e)}")

    def complete_task(self, final_metrics: Dict[str, float]):
        """标记任务完成"""
        try:
            url = f"{self.config.api_base_url}/model-distillation/tasks/{self.config.task_id}/complete"
            response = requests.post(url, timeout=10)

            if response.status_code == 200:
                print(f"\n✓ 训练任务已完成！")
                print(f"  最终准确率: {final_metrics.get('accuracy', 0):.2f}%")
            else:
                print(f"⚠ 标记完成失败: {response.status_code}")

        except Exception as e:
            print(f"⚠ 标记完成异常: {str(e)}")

    def report_error(self, error_message: str):
        """报告训练错误"""
        try:
            url = f"{self.config.api_base_url}/model-distillation/tasks/{self.config.task_id}/error"
            params = {'errorMessage': error_message}
            requests.put(url, params=params, timeout=10)
        except Exception as e:
            print(f"⚠ 报告错误异常: {str(e)}")

    def train(self):
        """完整的训练流程"""
        print(f"\n{'='*60}")
        print(f"🚀 开始训练任务: {self.config.task_id}")
        print(f"{'='*60}")

        try:
            # 加载模型
            self.load_models()

            # 准备数据集（这里使用示例数据，实际需要替换）
            print(f"\n📊 准备图像数据集...")

            # 根据配置构建数据集路径
            # Windows路径示例: D:/pythonProject2/datasets/cifar10
            # Linux路径示例: /data/datasets/cifar10

            # 构建训练集路径
            train_dataset_path = os.path.join(self.config.datasets_root, self.config.dataset_id, "train")

            # 构建验证集路径
            val_dataset_id = self.config.val_dataset_id or self.config.dataset_id
            val_dataset_path = os.path.join(self.config.datasets_root, val_dataset_id, "val")

            print(f"训练集路径: {train_dataset_path}")
            print(f"验证集路径: {val_dataset_path}")

            # 创建训练集（使用数据增强）
            train_dataset = ImageAnnotationDataset(
                dataset_path=train_dataset_path,
                image_processor=self.image_processor,
                image_size=224,
                is_training=True
            )

            # 创建验证集（不使用数据增强）
            val_dataset = ImageAnnotationDataset(
                dataset_path=val_dataset_path,
                image_processor=self.image_processor,
                image_size=224,
                is_training=False
            )

            train_loader = DataLoader(
                train_dataset,
                batch_size=self.config.batch_size,
                shuffle=True,
                num_workers=2
            )

            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=2
            )

            print(f"✓ 训练集: {len(train_dataset)} 样本")
            print(f"✓ 验证集: {len(val_dataset)} 样本")

            # 设置优化器
            num_training_steps = len(train_loader) * self.config.epochs
            self.setup_optimizer(num_training_steps)

            # 创建输出目录
            os.makedirs(self.config.output_dir, exist_ok=True)

            # 训练循环
            print(f"\n{'='*60}")
            print(f"🏋️ 开始训练 ({self.config.epochs} epochs)")
            print(f"{'='*60}\n")

            best_accuracy = 0

            for epoch in range(1, self.config.epochs + 1):
                print(f"\n📍 Epoch {epoch}/{self.config.epochs}")
                print("-" * 60)

                # 训练
                train_metrics = self.train_epoch(train_loader, epoch)

                # 验证
                val_metrics = self.evaluate(val_loader)

                # 合并指标
                all_metrics = {**train_metrics, **val_metrics}

                # 打印epoch结果
                print(f"\n📈 Epoch {epoch} 结果:")
                print(f"  训练损失: {train_metrics['loss']:.4f}")
                print(f"  训练准确率: {train_metrics['accuracy']:.2f}%")
                print(f"  验证损失: {val_metrics['val_loss']:.4f}")
                print(f"  验证准确率: {val_metrics['val_accuracy']:.2f}%")

                # 更新最佳准确率
                if val_metrics['val_accuracy'] > best_accuracy:
                    best_accuracy = val_metrics['val_accuracy']
                    print(f"  🎯 新的最佳准确率!")

                # 保存checkpoint
                if self.config.auto_save_checkpoint and epoch % self.config.checkpoint_interval == 0:
                    self.save_checkpoint(epoch, all_metrics)

                # 回调后端更新进度
                self.update_progress_to_backend(epoch, all_metrics)

            # 保存最终模型
            print(f"\n💾 保存最终模型...")
            final_model_dir = os.path.join(self.config.output_dir, "final_model")
            self.student_model.save_pretrained(final_model_dir)
            self.image_processor.save_pretrained(final_model_dir)
            print(f"✓ 最终模型已保存: {final_model_dir}")

            # 标记任务完成
            final_metrics = {'accuracy': best_accuracy}
            self.complete_task(final_metrics)

            print(f"\n{'='*60}")
            print(f"🎉 训练完成!")
            print(f"  最佳准确率: {best_accuracy:.2f}%")
            print(f"  模型保存路径: {final_model_dir}")
            print(f"{'='*60}\n")

        except Exception as e:
            error_msg = f"训练失败: {str(e)}"
            print(f"\n❌ {error_msg}")
            import traceback
            traceback.print_exc()
            self.report_error(error_msg)
            sys.exit(1)


# ==================== 主函数 ====================

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="大小模型协同训练脚本")

    # 基础配置
    parser.add_argument("--task_id", type=str, required=True, help="任务ID")
    parser.add_argument("--api_base_url", type=str, required=True, help="后端API地址")

    # 模型配置
    parser.add_argument("--teacher_model", type=str, required=True, help="教师模型名称")
    parser.add_argument("--student_model", type=str, required=True, help="学生模型名称")
    parser.add_argument("--teacher_path", type=str, required=True, help="教师模型路径")
    parser.add_argument("--student_path", type=str, default="", help="学生模型路径")

    # 数据配置
    parser.add_argument("--dataset_id", type=str, required=True, help="数据集ID")
    parser.add_argument("--val_dataset_id", type=str, default="", help="验证数据集ID")

    # 训练参数
    parser.add_argument("--epochs", type=int, default=10, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=16, help="批次大小")
    parser.add_argument("--learning_rate", type=float, default=0.0001, help="学习率")

    # 优化器配置
    parser.add_argument("--optimizer", type=str, default="adamw", help="优化器")
    parser.add_argument("--lr_scheduler", type=str, default="cosine", help="学习率调度器")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="权重衰减")
    parser.add_argument("--grad_accum_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="最大梯度范数")

    # GPU配置
    parser.add_argument("--gpu_devices", type=str, default="0", help="GPU设备")
    parser.add_argument("--auto_save_checkpoint", type=bool, default=True, help="自动保存checkpoint")
    parser.add_argument("--checkpoint_interval", type=int, default=5, help="checkpoint保存间隔")

    # LoRA配置
    parser.add_argument("--lora_rank", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")
    parser.add_argument("--lora_target_modules", type=str, default="", help="LoRA目标模块")
    parser.add_argument("--lora_bias", type=str, default="none", help="LoRA bias")

    # 知识蒸馏配置
    parser.add_argument("--temperature", type=float, default=3.0, help="蒸馏温度")
    parser.add_argument("--hard_label_weight", type=float, default=0.3, help="硬标签权重")
    parser.add_argument("--soft_label_weight", type=float, default=0.7, help="软标签权重")
    parser.add_argument("--distill_loss_type", type=str, default="kl_div", help="蒸馏损失类型")

    # 输出配置
    parser.add_argument("--output_dir", type=str, required=True, help="输出目录")

    # 数据集根目录（新增，由后端配置文件传递）
    parser.add_argument("--datasets_root", type=str, required=True, help="数据集根目录")

    return parser.parse_args()


def main():
    """主函数"""
    # 解析参数
    args = parse_args()

    # 创建配置
    config = TrainingConfig(args)

    # 创建训练器
    trainer = DistillationTrainer(config)

    # 开始训练
    trainer.train()


if __name__ == "__main__":
    main()
