#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen2.5-VL 到多架构小模型的知识蒸馏训练脚本

支持的学生模型：
1. LSTM - 序列特征提取
2. UNet - 图像分割
3. YOLOv8 - 目标检测
4. ResNet - 图像分类
5. Vision Transformer - 图像分类

蒸馏策略：
- 特征蒸馏：提取Qwen2.5-VL的视觉编码器特征
- Logits蒸馏：用于分类任务
- 中间层蒸馏：用于ViT等Transformer架构

作者：Claude Assistant
日期：2026-01-11
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

# Qwen2.5-VL相关导入
try:
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    QWEN_AVAILABLE = True
except ImportError:
    QWEN_AVAILABLE = False
    warnings.warn("Qwen2VL模型库未安装，将使用模拟模式")

# 小模型相关导入
import torchvision.models as models
from transformers import ViTForImageClassification, ViTImageProcessor

# YOLO相关（需要安装ultralytics）
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    warnings.warn("YOLOv8未安装")

# ==================== 配置类 ====================

class TrainingConfig:
    """多模型协同训练配置"""

    def __init__(self, args):
        # 基础配置
        self.task_id = args.task_id
        self.api_base_url = args.api_base_url

        # 教师模型配置（Qwen2.5-VL）
        self.teacher_model_path = args.teacher_model_path
        self.teacher_model_type = "qwen2.5-vl"  # 固定为Qwen2.5-VL

        # 学生模型配置
        self.student_model_type = args.student_model_type  # lstm/unet/yolov8/resnet/vit
        self.student_model_path = args.student_model_path
        self.student_model_size = args.student_model_size  # 如resnet18/resnet50, vit-base等

        # 任务配置
        self.task_type = args.task_type  # classification/detection/segmentation
        self.num_classes = args.num_classes

        # 数据配置
        self.dataset_path = args.dataset_path
        self.val_dataset_path = args.val_dataset_path
        self.image_size = args.image_size

        # 训练超参数
        self.epochs = args.epochs
        self.batch_size = args.batch_size
        self.learning_rate = args.learning_rate
        self.optimizer_type = args.optimizer_type  # adamw/adam/sgd
        self.lr_scheduler = args.lr_scheduler  # cosine/linear/step
        self.weight_decay = args.weight_decay
        self.grad_accum_steps = args.grad_accum_steps
        self.max_grad_norm = args.max_grad_norm

        # 蒸馏配置
        self.distillation_type = args.distillation_type  # feature/logit/layer/hybrid
        self.temperature = args.temperature
        self.alpha = args.alpha  # 硬标签权重
        self.beta = args.beta   # 软标签权重
        self.gamma = args.gamma  # 特征蒸馏权重

        # 特征蒸馏配置
        self.feature_loss_type = args.feature_loss_type  # mse/cosine/attention
        self.align_feature = args.align_feature  # 是否使用投影层对齐特征
        self.feature_dim = args.feature_dim  # 特征对齐维度

        # GPU配置
        self.gpu_devices = self._parse_gpu_devices(args.gpu_devices)
        self.use_amp = args.use_amp  # 混合精度训练

        # 输出配置
        self.output_dir = args.output_dir
        self.checkpoint_interval = args.checkpoint_interval
        self.log_interval = args.log_interval

    def _parse_gpu_devices(self, gpu_str: str) -> List[int]:
        """解析GPU设备列表"""
        if not gpu_str or gpu_str == "":
            return [0]
        return [int(x.strip()) for x in gpu_str.split(",")]


# ==================== 数据集类 ====================

class MultiTaskDataset(Dataset):
    """
    多任务数据集，支持分类、检测、分割任务
    """

    def __init__(
        self,
        dataset_path: str,
        task_type: str,
        image_size: int = 224,
        num_classes: int = 10,
        mode: str = 'train'
    ):
        self.dataset_path = Path(dataset_path)
        self.task_type = task_type
        self.image_size = image_size
        self.num_classes = num_classes
        self.mode = mode

        # 数据增强
        if mode == 'train':
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
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

        # TODO: 从数据库加载真实数据
        self.num_samples = 1000 if mode == 'train' else 200

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # TODO: 加载真实图像和标注
        # 这里使用随机数据作为演示

        # 生成随机图像
        image_array = np.random.randint(0, 255, (self.image_size, self.image_size, 3), dtype=np.uint8)
        image = Image.fromarray(image_array)
        pixel_values = self.transform(image)

        # 根据任务类型生成标签
        if self.task_type == 'classification':
            # 分类：类别标签
            label = torch.randint(0, self.num_classes, (1,)).item()
            return {'pixel_values': pixel_values, 'labels': label}

        elif self.task_type == 'detection':
            # 检测：边界框 + 类别
            num_boxes = np.random.randint(1, 5)
            boxes = torch.rand(num_boxes, 4)  # [x1, y1, x2, y2]
            labels = torch.randint(0, self.num_classes, (num_boxes,))
            return {
                'pixel_values': pixel_values,
                'boxes': boxes,
                'labels': labels
            }

        elif self.task_type == 'segmentation':
            # 分割：像素级标签
            mask = torch.randint(0, self.num_classes, (self.image_size, self.image_size))
            return {'pixel_values': pixel_values, 'mask': mask}

        else:
            raise ValueError(f"不支持的任务类型: {self.task_type}")


# ==================== 模型加载器 ====================

class TeacherModelLoader:
    """Qwen2.5-VL教师模型加载器"""

    @staticmethod
    def load_qwen2vl(model_path: str, device: torch.device):
        """加载Qwen2.5-VL模型"""
        if not QWEN_AVAILABLE:
            print("⚠️  Qwen2VL未安装，使用模拟教师模型")
            return None, None

        print(f"正在加载Qwen2.5-VL教师模型: {model_path}")

        # 加载模型和处理器
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map=device
        )
        processor = AutoProcessor.from_pretrained(model_path)

        model.eval()
        for param in model.parameters():
            param.requires_grad = False

        print(f"✓ Qwen2.5-VL加载成功，参数量: {sum(p.numel() for p in model.parameters()):,}")
        return model, processor


class StudentModelLoader:
    """学生模型加载器"""

    @staticmethod
    def load_model(
        model_type: str,
        model_size: str,
        num_classes: int,
        device: torch.device,
        pretrained: bool = True
    ):
        """根据类型加载学生模型"""

        print(f"\n正在加载学生模型: {model_type}-{model_size}")

        if model_type == 'resnet':
            return StudentModelLoader._load_resnet(model_size, num_classes, device, pretrained)
        elif model_type == 'vit':
            return StudentModelLoader._load_vit(model_size, num_classes, device, pretrained)
        elif model_type == 'yolov8':
            return StudentModelLoader._load_yolov8(model_size, num_classes, device)
        elif model_type == 'unet':
            return StudentModelLoader._load_unet(model_size, num_classes, device)
        elif model_type == 'lstm':
            return StudentModelLoader._load_lstm(model_size, num_classes, device)
        else:
            raise ValueError(f"不支持的学生模型类型: {model_type}")

    @staticmethod
    def _load_resnet(size: str, num_classes: int, device, pretrained: bool):
        """加载ResNet模型"""
        resnet_variants = {
            'resnet18': models.resnet18,
            'resnet34': models.resnet34,
            'resnet50': models.resnet50,
            'resnet101': models.resnet101,
        }

        if size not in resnet_variants:
            raise ValueError(f"不支持的ResNet变体: {size}")

        model = resnet_variants[size](pretrained=pretrained)

        # 修改最后一层以适应类别数
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)

        model.to(device)
        print(f"✓ ResNet-{size}加载成功，参数量: {sum(p.numel() for p in model.parameters()):,}")
        return model

    @staticmethod
    def _load_vit(size: str, num_classes: int, device, pretrained: bool):
        """加载Vision Transformer模型"""
        vit_models = {
            'vit-base': 'google/vit-base-patch16-224',
            'vit-large': 'google/vit-large-patch16-224',
            'vit-tiny': 'WinKawaks/vit-tiny-patch16-224',
        }

        if size not in vit_models:
            raise ValueError(f"不支持的ViT变体: {size}")

        model_name = vit_models[size]
        model = ViTForImageClassification.from_pretrained(
            model_name,
            num_labels=num_classes,
            ignore_mismatched_sizes=True
        )

        model.to(device)
        print(f"✓ ViT-{size}加载成功，参数量: {sum(p.numel() for p in model.parameters()):,}")
        return model

    @staticmethod
    def _load_yolov8(size: str, num_classes: int, device):
        """加载YOLOv8模型"""
        if not YOLO_AVAILABLE:
            raise ImportError("YOLOv8未安装，请运行: pip install ultralytics")

        # YOLOv8模型大小
        yolo_sizes = {'n': 'yolov8n.pt', 's': 'yolov8s.pt', 'm': 'yolov8m.pt',
                     'l': 'yolov8l.pt', 'x': 'yolov8x.pt'}

        if size not in yolo_sizes:
            raise ValueError(f"不支持的YOLO大小: {size}")

        model = YOLO(yolo_sizes[size])
        print(f"✓ YOLOv8-{size}加载成功")
        return model

    @staticmethod
    def _load_unet(size: str, num_classes: int, device):
        """加载UNet模型（用于分割）"""
        # 简单的UNet实现
        class SimpleUNet(nn.Module):
            def __init__(self, in_channels=3, num_classes=10):
                super().__init__()
                # 编码器
                self.enc1 = self._conv_block(in_channels, 64)
                self.enc2 = self._conv_block(64, 128)
                self.enc3 = self._conv_block(128, 256)
                self.enc4 = self._conv_block(256, 512)

                # 解码器
                self.dec3 = self._conv_block(512 + 256, 256)
                self.dec2 = self._conv_block(256 + 128, 128)
                self.dec1 = self._conv_block(128 + 64, 64)

                self.final = nn.Conv2d(64, num_classes, 1)
                self.pool = nn.MaxPool2d(2)
                self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

            def _conv_block(self, in_ch, out_ch):
                return nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, 3, padding=1),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_ch, out_ch, 3, padding=1),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True)
                )

            def forward(self, x):
                # 编码
                e1 = self.enc1(x)
                e2 = self.enc2(self.pool(e1))
                e3 = self.enc3(self.pool(e2))
                e4 = self.enc4(self.pool(e3))

                # 解码
                d3 = self.dec3(torch.cat([self.upsample(e4), e3], dim=1))
                d2 = self.dec2(torch.cat([self.upsample(d3), e2], dim=1))
                d1 = self.dec1(torch.cat([self.upsample(d2), e1], dim=1))

                return self.final(d1)

        model = SimpleUNet(in_channels=3, num_classes=num_classes)
        model.to(device)
        print(f"✓ UNet加载成功，参数量: {sum(p.numel() for p in model.parameters()):,}")
        return model

    @staticmethod
    def _load_lstm(size: str, num_classes: int, device):
        """加载LSTM模型（用于序列特征提取+分类）"""
        class LSTMClassifier(nn.Module):
            def __init__(self, input_size=2048, hidden_size=512, num_layers=2, num_classes=10):
                super().__init__()
                # 将图像特征视为序列
                self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                                   batch_first=True, bidirectional=True)
                self.fc = nn.Linear(hidden_size * 2, num_classes)
                self.dropout = nn.Dropout(0.5)

                # 用于提取图像特征的CNN
                resnet = models.resnet50(pretrained=True)
                self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])

            def forward(self, x):
                # 提取特征
                batch_size = x.size(0)
                features = self.feature_extractor(x)
                features = features.view(batch_size, -1, 1)  # [B, C, 1]
                features = features.transpose(1, 2)  # [B, 1, C]

                # LSTM处理
                lstm_out, _ = self.lstm(features)
                lstm_out = lstm_out[:, -1, :]  # 取最后一个时间步
                lstm_out = self.dropout(lstm_out)

                # 分类
                output = self.fc(lstm_out)
                return output

        hidden_sizes = {'small': 256, 'medium': 512, 'large': 1024}
        hidden_size = hidden_sizes.get(size, 512)

        model = LSTMClassifier(hidden_size=hidden_size, num_classes=num_classes)
        model.to(device)
        print(f"✓ LSTM加载成功，参数量: {sum(p.numel() for p in model.parameters()):,}")
        return model


# ==================== 特征投影层 ====================

class FeatureAlignmentLayer(nn.Module):
    """特征对齐层：将教师模型特征投影到学生模型特征空间"""

    def __init__(self, teacher_dim: int, student_dim: int, use_attention: bool = False):
        super().__init__()
        self.use_attention = use_attention

        # 简单投影层
        self.projection = nn.Sequential(
            nn.Linear(teacher_dim, student_dim),
            nn.LayerNorm(student_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # 可选：注意力机制
        if use_attention:
            self.attention = nn.MultiheadAttention(student_dim, num_heads=8, batch_first=True)

    def forward(self, teacher_features, student_features=None):
        """
        Args:
            teacher_features: [B, N, D_teacher]
            student_features: [B, M, D_student] (可选，用于注意力)
        Returns:
            aligned_features: [B, N, D_student]
        """
        # 投影
        aligned = self.projection(teacher_features)

        # 可选：使用注意力进行进一步对齐
        if self.use_attention and student_features is not None:
            aligned, _ = self.attention(aligned, student_features, student_features)

        return aligned


# ==================== 知识蒸馏训练器 ====================

class MultiModelDistillationTrainer:
    """多模型协同训练器 - Qwen2.5-VL → 多种小模型"""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = self._setup_device()

        # 加载教师模型（Qwen2.5-VL）
        self.teacher_model, self.teacher_processor = TeacherModelLoader.load_qwen2vl(
            config.teacher_model_path,
            self.device
        )

        # 加载学生模型
        self.student_model = StudentModelLoader.load_model(
            config.student_model_type,
            config.student_model_size,
            config.num_classes,
            self.device
        )

        # 特征对齐层（如果需要）
        self.feature_aligner = None
        if config.align_feature and config.distillation_type in ['feature', 'hybrid']:
            # TODO: 根据实际模型确定特征维度
            teacher_dim = 1280  # Qwen2.5-VL视觉编码器输出维度（需根据实际调整）
            student_dim = self._get_student_feature_dim()
            self.feature_aligner = FeatureAlignmentLayer(
                teacher_dim, student_dim, use_attention=True
            ).to(self.device)

        # 优化器和调度器
        self._setup_optimizer()
        self._setup_scheduler()

        # 损失函数
        self.ce_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()
        self.cosine_loss = nn.CosineEmbeddingLoss()

        # 混合精度训练
        self.scaler = torch.cuda.amp.GradScaler() if config.use_amp else None

        # 训练状态
        self.current_epoch = 0
        self.global_step = 0
        self.best_acc = 0.0

    def _setup_device(self) -> torch.device:
        """设置训练设备"""
        if torch.cuda.is_available():
            device_id = self.config.gpu_devices[0]
            device = torch.device(f"cuda:{device_id}")
            print(f"✓ 使用GPU设备: cuda:{device_id}")
        else:
            device = torch.device("cpu")
            print("⚠️  使用CPU训练")
        return device

    def _get_student_feature_dim(self) -> int:
        """获取学生模型特征维度"""
        model_type = self.config.student_model_type
        size = self.config.student_model_size

        # 根据模型类型返回特征维度
        if model_type == 'resnet':
            dims = {'resnet18': 512, 'resnet34': 512, 'resnet50': 2048, 'resnet101': 2048}
            return dims.get(size, 2048)
        elif model_type == 'vit':
            dims = {'vit-tiny': 192, 'vit-base': 768, 'vit-large': 1024}
            return dims.get(size, 768)
        elif model_type == 'lstm':
            dims = {'small': 512, 'medium': 1024, 'large': 2048}
            return dims.get(size, 1024)
        else:
            return 512  # 默认值

    def _setup_optimizer(self):
        """设置优化器"""
        params = list(self.student_model.parameters())
        if self.feature_aligner is not None:
            params += list(self.feature_aligner.parameters())

        if self.config.optimizer_type == 'adamw':
            self.optimizer = torch.optim.AdamW(
                params,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer_type == 'adam':
            self.optimizer = torch.optim.Adam(
                params,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer_type == 'sgd':
            self.optimizer = torch.optim.SGD(
                params,
                lr=self.config.learning_rate,
                momentum=0.9,
                weight_decay=self.config.weight_decay
            )
        else:
            raise ValueError(f"不支持的优化器: {self.config.optimizer_type}")

    def _setup_scheduler(self):
        """设置学习率调度器"""
        # 计算总步数
        # TODO: 从数据集获取实际大小
        num_training_steps = (1000 // self.config.batch_size) * self.config.epochs

        if self.config.lr_scheduler == 'cosine':
            from torch.optim.lr_scheduler import CosineAnnealingLR
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=num_training_steps
            )
        elif self.config.lr_scheduler == 'linear':
            from torch.optim.lr_scheduler import LinearLR
            self.scheduler = LinearLR(
                self.optimizer,
                start_factor=1.0,
                end_factor=0.1,
                total_iters=num_training_steps
            )
        elif self.config.lr_scheduler == 'step':
            from torch.optim.lr_scheduler import StepLR
            self.scheduler = StepLR(
                self.optimizer,
                step_size=num_training_steps // 3,
                gamma=0.1
            )
        else:
            self.scheduler = None

    def extract_teacher_features(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        从Qwen2.5-VL提取视觉特征

        Returns:
            dict: {
                'vision_features': [B, N, D],  # 视觉编码器输出
                'hidden_states': List[[B, N, D]],  # 各层隐藏状态
            }
        """
        if self.teacher_model is None:
            # 模拟模式：返回随机特征
            batch_size = images.size(0)
            return {
                'vision_features': torch.randn(batch_size, 256, 1280).to(self.device),
                'hidden_states': [torch.randn(batch_size, 256, 1280).to(self.device)]
            }

        with torch.no_grad():
            # Qwen2.5-VL的视觉编码器提取特征
            # 注意：实际使用时需要根据Qwen2.5-VL的API调整
            outputs = self.teacher_model.visual(
                images,
                output_hidden_states=True
            )

            return {
                'vision_features': outputs.last_hidden_state,
                'hidden_states': outputs.hidden_states
            }

    def compute_distillation_loss(
        self,
        student_output: Any,
        teacher_features: Dict[str, torch.Tensor],
        labels: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        计算蒸馏损失

        Args:
            student_output: 学生模型输出
            teacher_features: 教师模型特征
            labels: 真实标签

        Returns:
            dict: {'total_loss': Tensor, 'hard_loss': Tensor, 'soft_loss': Tensor, ...}
        """
        losses = {}

        # 1. 硬标签损失（任务损失）
        if self.config.task_type == 'classification':
            if isinstance(student_output, dict):
                logits = student_output['logits']
            else:
                logits = student_output
            hard_loss = self.ce_loss(logits, labels)
            losses['hard_loss'] = hard_loss
        else:
            # TODO: 实现检测/分割的任务损失
            hard_loss = torch.tensor(0.0).to(self.device)
            losses['hard_loss'] = hard_loss

        # 2. 软标签损失（Logits蒸馏）
        if self.config.distillation_type in ['logit', 'hybrid']:
            # TODO: 从teacher_features提取软标签
            # 这需要Qwen2.5-VL输出分类logits
            soft_loss = torch.tensor(0.0).to(self.device)
            losses['soft_loss'] = soft_loss

        # 3. 特征蒸馏损失
        if self.config.distillation_type in ['feature', 'hybrid']:
            # 提取学生模型特征
            student_features = self._extract_student_features(student_output)

            # 对齐教师特征
            teacher_vis_features = teacher_features['vision_features']  # [B, N, D]

            if self.feature_aligner is not None:
                aligned_teacher_features = self.feature_aligner(
                    teacher_vis_features,
                    student_features
                )
            else:
                aligned_teacher_features = teacher_vis_features

            # 计算特征损失
            if self.config.feature_loss_type == 'mse':
                # 需要确保维度匹配
                if student_features.shape != aligned_teacher_features.shape:
                    # 使用池化或插值调整维度
                    student_features = F.adaptive_avg_pool1d(
                        student_features.transpose(1, 2),
                        aligned_teacher_features.size(1)
                    ).transpose(1, 2)

                feature_loss = self.mse_loss(student_features, aligned_teacher_features)

            elif self.config.feature_loss_type == 'cosine':
                # 余弦相似度损失
                student_norm = F.normalize(student_features.mean(dim=1), dim=-1)
                teacher_norm = F.normalize(aligned_teacher_features.mean(dim=1), dim=-1)
                target = torch.ones(student_norm.size(0)).to(self.device)
                feature_loss = self.cosine_loss(student_norm, teacher_norm, target)

            else:
                feature_loss = torch.tensor(0.0).to(self.device)

            losses['feature_loss'] = feature_loss

        # 4. 总损失
        total_loss = (
            self.config.alpha * losses.get('hard_loss', 0) +
            self.config.beta * losses.get('soft_loss', 0) +
            self.config.gamma * losses.get('feature_loss', 0)
        )

        losses['total_loss'] = total_loss
        return losses

    def _extract_student_features(self, student_output) -> torch.Tensor:
        """从学生模型输出中提取特征"""
        # 根据学生模型类型提取特征
        if isinstance(student_output, dict):
            if 'hidden_states' in student_output:
                # ViT等Transformer模型
                return student_output['hidden_states'][-1]  # 最后一层
            elif 'last_hidden_state' in student_output:
                return student_output['last_hidden_state']

        # 默认：假设student_output是特征张量
        if len(student_output.shape) == 3:
            return student_output
        elif len(student_output.shape) == 2:
            return student_output.unsqueeze(1)
        else:
            return student_output.flatten(start_dim=1).unsqueeze(1)

    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """训练一个epoch"""
        self.student_model.train()
        if self.feature_aligner is not None:
            self.feature_aligner.train()

        epoch_losses = {
            'total_loss': 0.0,
            'hard_loss': 0.0,
            'soft_loss': 0.0,
            'feature_loss': 0.0
        }

        num_batches = 0
        pbar = tqdm(train_loader, desc=f"Epoch {self.current_epoch + 1}/{self.config.epochs}")

        for batch_idx, batch in enumerate(pbar):
            images = batch['pixel_values'].to(self.device)
            labels = batch['labels'].to(self.device)

            # 1. 提取教师特征
            teacher_features = self.extract_teacher_features(images)

            # 2. 学生模型前向传播
            if self.config.use_amp:
                with torch.cuda.amp.autocast():
                    student_output = self.student_model(images)
                    losses = self.compute_distillation_loss(
                        student_output, teacher_features, labels
                    )
                    loss = losses['total_loss'] / self.config.grad_accum_steps
            else:
                student_output = self.student_model(images)
                losses = self.compute_distillation_loss(
                    student_output, teacher_features, labels
                )
                loss = losses['total_loss'] / self.config.grad_accum_steps

            # 3. 反向传播
            if self.config.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # 4. 梯度累积
            if (batch_idx + 1) % self.config.grad_accum_steps == 0:
                if self.config.use_amp:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.student_model.parameters(),
                        self.config.max_grad_norm
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(
                        self.student_model.parameters(),
                        self.config.max_grad_norm
                    )
                    self.optimizer.step()

                if self.scheduler is not None:
                    self.scheduler.step()

                self.optimizer.zero_grad()
                self.global_step += 1

            # 5. 记录损失
            for key in epoch_losses:
                if key in losses:
                    epoch_losses[key] += losses[key].item()
            num_batches += 1

            # 6. 更新进度条
            pbar.set_postfix({
                'loss': f"{losses['total_loss'].item():.4f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.6f}"
            })

            # 7. 定期日志
            if (batch_idx + 1) % self.config.log_interval == 0:
                self._log_training_step(losses)

        # 计算平均损失
        for key in epoch_losses:
            epoch_losses[key] /= num_batches

        return epoch_losses

    @torch.no_grad()
    def evaluate(self, val_loader: DataLoader) -> Dict[str, float]:
        """评估模型"""
        self.student_model.eval()

        total_loss = 0.0
        correct = 0
        total = 0

        for batch in tqdm(val_loader, desc="Evaluating"):
            images = batch['pixel_values'].to(self.device)
            labels = batch['labels'].to(self.device)

            # 前向传播
            outputs = self.student_model(images)

            # 计算损失
            if isinstance(outputs, dict):
                logits = outputs['logits']
            else:
                logits = outputs

            loss = self.ce_loss(logits, labels)
            total_loss += loss.item()

            # 计算准确率
            _, predicted = torch.max(logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        avg_loss = total_loss / len(val_loader)
        accuracy = 100.0 * correct / total

        return {
            'val_loss': avg_loss,
            'val_accuracy': accuracy
        }

    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        """完整训练流程"""
        print(f"\n{'='*60}")
        print("🚀 开始训练")
        print(f"{'='*60}")
        print(f"教师模型: Qwen2.5-VL 8B")
        print(f"学生模型: {self.config.student_model_type}-{self.config.student_model_size}")
        print(f"任务类型: {self.config.task_type}")
        print(f"蒸馏策略: {self.config.distillation_type}")
        print(f"训练轮数: {self.config.epochs}")
        print(f"批大小: {self.config.batch_size}")
        print(f"学习率: {self.config.learning_rate}")
        print(f"{'='*60}\n")

        for epoch in range(self.config.epochs):
            self.current_epoch = epoch

            # 训练
            train_metrics = self.train_epoch(train_loader)

            # 评估
            val_metrics = self.evaluate(val_loader)

            # 打印结果
            print(f"\nEpoch {epoch + 1}/{self.config.epochs} 完成:")
            print(f"  训练损失: {train_metrics['total_loss']:.4f}")
            print(f"  验证损失: {val_metrics['val_loss']:.4f}")
            print(f"  验证准确率: {val_metrics['val_accuracy']:.2f}%")

            # 回调后端API更新进度
            self._update_training_progress(epoch + 1, train_metrics, val_metrics)

            # 保存checkpoint
            if (epoch + 1) % self.config.checkpoint_interval == 0:
                self.save_checkpoint(epoch + 1, val_metrics['val_accuracy'])

            # 保存最佳模型
            if val_metrics['val_accuracy'] > self.best_acc:
                self.best_acc = val_metrics['val_accuracy']
                self.save_checkpoint(epoch + 1, val_metrics['val_accuracy'], is_best=True)

        print(f"\n{'='*60}")
        print("✓ 训练完成！")
        print(f"最佳验证准确率: {self.best_acc:.2f}%")
        print(f"{'='*60}\n")

    def save_checkpoint(self, epoch: int, accuracy: float, is_best: bool = False):
        """保存checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.student_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'accuracy': accuracy,
            'config': vars(self.config)
        }

        if self.feature_aligner is not None:
            checkpoint['aligner_state_dict'] = self.feature_aligner.state_dict()

        # 保存路径
        os.makedirs(self.config.output_dir, exist_ok=True)

        if is_best:
            path = os.path.join(self.config.output_dir, 'best_model.pt')
            print(f"💾 保存最佳模型: {path}")
        else:
            path = os.path.join(self.config.output_dir, f'checkpoint_epoch_{epoch}.pt')
            print(f"💾 保存checkpoint: {path}")

        torch.save(checkpoint, path)

    def _update_training_progress(self, epoch: int, train_metrics: Dict, val_metrics: Dict):
        """回调后端API更新训练进度"""
        try:
            url = f"{self.config.api_base_url}/training/progress/{self.config.task_id}"
            data = {
                'epoch': epoch,
                'total_epochs': self.config.epochs,
                'train_loss': train_metrics['total_loss'],
                'val_loss': val_metrics['val_loss'],
                'val_accuracy': val_metrics['val_accuracy'],
                'timestamp': datetime.now().isoformat()
            }

            response = requests.post(url, json=data, timeout=5)
            if response.status_code != 200:
                print(f"⚠️  进度更新失败: {response.text}")
        except Exception as e:
            print(f"⚠️  进度更新异常: {e}")

    def _log_training_step(self, losses: Dict[str, torch.Tensor]):
        """记录训练步骤"""
        # 这里可以集成TensorBoard或其他日志工具
        pass


# ==================== 命令行参数解析 ====================

def parse_args():
    parser = argparse.ArgumentParser(description='Qwen2.5-VL多模型协同训练')

    # 基础配置
    parser.add_argument('--task_id', type=str, required=True, help='任务ID')
    parser.add_argument('--api_base_url', type=str, required=True, help='后端API地址')

    # 教师模型配置
    parser.add_argument('--teacher_model_path', type=str, required=True,
                       help='Qwen2.5-VL模型路径')

    # 学生模型配置
    parser.add_argument('--student_model_type', type=str, required=True,
                       choices=['resnet', 'vit', 'yolov8', 'unet', 'lstm'],
                       help='学生模型类型')
    parser.add_argument('--student_model_size', type=str, required=True,
                       help='学生模型大小 (如resnet50, vit-base等)')
    parser.add_argument('--student_model_path', type=str, default=None,
                       help='学生模型预训练权重路径')

    # 任务配置
    parser.add_argument('--task_type', type=str, default='classification',
                       choices=['classification', 'detection', 'segmentation'],
                       help='任务类型')
    parser.add_argument('--num_classes', type=int, default=10, help='类别数')

    # 数据配置
    parser.add_argument('--dataset_path', type=str, required=True, help='训练数据集路径')
    parser.add_argument('--val_dataset_path', type=str, required=True, help='验证数据集路径')
    parser.add_argument('--image_size', type=int, default=224, help='图像大小')

    # 训练超参数
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=32, help='批大小')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='学习率')
    parser.add_argument('--optimizer_type', type=str, default='adamw',
                       choices=['adamw', 'adam', 'sgd'])
    parser.add_argument('--lr_scheduler', type=str, default='cosine',
                       choices=['cosine', 'linear', 'step'])
    parser.add_argument('--weight_decay', type=float, default=0.01, help='权重衰减')
    parser.add_argument('--grad_accum_steps', type=int, default=1, help='梯度累积步数')
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='梯度裁剪')

    # 蒸馏配置
    parser.add_argument('--distillation_type', type=str, default='hybrid',
                       choices=['feature', 'logit', 'layer', 'hybrid'],
                       help='蒸馏类型')
    parser.add_argument('--temperature', type=float, default=4.0, help='蒸馏温度')
    parser.add_argument('--alpha', type=float, default=0.5, help='硬标签权重')
    parser.add_argument('--beta', type=float, default=0.3, help='软标签权重')
    parser.add_argument('--gamma', type=float, default=0.2, help='特征蒸馏权重')
    parser.add_argument('--feature_loss_type', type=str, default='mse',
                       choices=['mse', 'cosine', 'attention'])
    parser.add_argument('--align_feature', action='store_true', help='使用特征对齐层')
    parser.add_argument('--feature_dim', type=int, default=768, help='特征对齐维度')

    # GPU配置
    parser.add_argument('--gpu_devices', type=str, default='0', help='GPU设备ID（逗号分隔）')
    parser.add_argument('--use_amp', action='store_true', help='使用混合精度训练')

    # 输出配置
    parser.add_argument('--output_dir', type=str, required=True, help='输出目录')
    parser.add_argument('--checkpoint_interval', type=int, default=10, help='保存checkpoint间隔')
    parser.add_argument('--log_interval', type=int, default=50, help='日志打印间隔')

    return parser.parse_args()


# ==================== 主函数 ====================

def main():
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║     Qwen2.5-VL 多模型协同训练系统                            ║
    ║     Multi-Model Collaborative Training with Qwen2.5-VL       ║
    ╚══════════════════════════════════════════════════════════════╝
    """)

    # 解析参数
    args = parse_args()
    config = TrainingConfig(args)

    # 创建数据集
    print("\n正在加载数据集...")
    train_dataset = MultiTaskDataset(
        config.dataset_path,
        config.task_type,
        config.image_size,
        config.num_classes,
        mode='train'
    )
    val_dataset = MultiTaskDataset(
        config.val_dataset_path,
        config.task_type,
        config.image_size,
        config.num_classes,
        mode='val'
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    print(f"✓ 训练集: {len(train_dataset)} 样本")
    print(f"✓ 验证集: {len(val_dataset)} 样本")

    # 创建训练器
    trainer = MultiModelDistillationTrainer(config)

    # 开始训练
    trainer.train(train_loader, val_loader)

    print("\n✅ 所有任务完成！")


if __name__ == '__main__':
    main()
