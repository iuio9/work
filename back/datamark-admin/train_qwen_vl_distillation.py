#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen2.5-VL到多架构小模型的知识蒸馏训练脚本

支持的教师模型：
- Qwen2.5-VL 3B（多模态视觉-语言模型）

支持的学生模型：
- LSTM：序列特征提取 + 图像分类
- UNet：图像分割
- YOLOv8：目标检测
- ResNet：图像分类
- Vision Transformer：图像分类

作者：Claude Assistant
日期：2026-01-11
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

# Qwen2.5-VL相关导入
try:
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    QWEN_AVAILABLE = True
except ImportError:
    QWEN_AVAILABLE = False
    warnings.warn("Qwen2_5_VL模型库未安装，将使用模拟模式")

# 小模型相关导入
import torchvision.models as models
from transformers import (
    AutoConfig,
    AutoModelForImageClassification,
    AutoImageProcessor,
    ViTForImageClassification,
    ViTImageProcessor
)
from peft import LoraConfig, get_peft_model, TaskType
from qwen_vl_utils import process_vision_info

# YOLO相关
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    warnings.warn("YOLOv8未安装，使用: pip install ultralytics")


# ==================== 配置类 ====================

class TrainingConfig:
    """训练配置类，与后端TrainingExecutionService保持一致"""

    def __init__(self, args):
        # 基础配置
        self.task_id = args.task_id
        self.api_base_url = args.api_base_url

        # 模型配置
        self.teacher_model = args.teacher_model  # "qwen2.5-vl-8b"
        self.student_model = args.student_model  # "resnet50", "vit-base", etc.
        self.teacher_path = args.teacher_path
        self.student_path = args.student_path

        # 学生模型类型和大小
        self.student_model_type = args.student_model_type  # resnet/vit/yolov8/unet/lstm
        self.student_model_size = args.student_model_size  # resnet50, vit-base, s, medium, etc.

        # 任务配置
        self.task_type = args.task_type  # classification/detection/segmentation
        self.num_classes = args.num_classes

        # 数据配置
        self.dataset_id = args.dataset_id
        self.val_dataset_id = args.val_dataset_id
        self.datasets_root = args.datasets_root
        self.image_size = args.image_size

        # 训练参数
        self.epochs = args.epochs
        self.batch_size = args.batch_size
        self.learning_rate = args.learning_rate

        # 优化器配置
        self.optimizer = args.optimizer
        self.lr_scheduler = args.lr_scheduler
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
        self.distill_loss_type = args.distill_loss_type

        # 蒸馏策略
        self.distillation_type = args.distillation_type  # feature/logit/hybrid
        self.feature_loss_type = args.feature_loss_type  # mse/cosine
        self.align_feature = args.align_feature

        # 输出配置
        self.output_dir = args.output_dir

    def _parse_gpu_devices(self, gpu_str: str) -> List[int]:
        if not gpu_str or gpu_str == "":
            return [0]
        return [int(x.strip()) for x in gpu_str.split(",")]

    def _parse_list(self, list_str: str) -> List[str]:
        if not list_str or list_str == "":
            return []
        return [x.strip() for x in list_str.split(",")]


# ==================== 数据集类 ====================

class MultiTaskDataset(Dataset):
    """
    多任务数据集，支持分类、检测、分割

    期望的目录结构（分类任务）：
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
        task_type: str = 'classification',
        image_size: int = 224,
        num_classes: int = 10,
        mode: str = 'train'
    ):
        self.dataset_path = dataset_path
        self.task_type = task_type
        self.image_size = image_size
        self.num_classes = num_classes
        self.mode = mode

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
            self._use_mock_data(1000 if mode == 'train' else 200)

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

    def _load_dataset(self):
        """从目录结构加载真实数据集（分类任务）"""
        print(f"📂 加载数据集: {self.dataset_path}")

        # 获取所有类别文件夹
        class_folders = sorted([d for d in os.listdir(self.dataset_path)
                               if os.path.isdir(os.path.join(self.dataset_path, d))])

        if not class_folders:
            print(f"⚠️ 警告: 在 {self.dataset_path} 中未找到类别文件夹")
            self._use_mock_data(1000 if self.mode == 'train' else 200)
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
        if self.num_classes is None or self.num_classes != len(self.class_names):
            self.num_classes = len(self.class_names)

    def _use_mock_data(self, num_samples):
        """使用模拟数据（当真实数据不可用时）"""
        print(f"🎭 使用模拟数据: {num_samples} 个样本")
        self.class_names = [f"class_{i}" for i in range(self.num_classes)]

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
                image_array = np.random.randint(0, 255, (self.image_size, self.image_size, 3), dtype=np.uint8)
                image = Image.fromarray(image_array)
        except Exception as e:
            print(f"⚠️ 加载图像失败 {img_path}: {e}")
            # 生成随机图像作为后备
            image_array = np.random.randint(0, 255, (self.image_size, self.image_size, 3), dtype=np.uint8)
            image = Image.fromarray(image_array)

        # 应用变换
        pixel_values = self.transform(image)

        if self.task_type == 'classification':
            return {'pixel_values': pixel_values, 'labels': label}
        elif self.task_type == 'detection':
            # 检测任务的模拟数据
            num_boxes = np.random.randint(1, 5)
            boxes = torch.rand(num_boxes, 4)
            box_labels = torch.randint(0, self.num_classes, (num_boxes,))
            return {'pixel_values': pixel_values, 'boxes': boxes, 'labels': box_labels}
        elif self.task_type == 'segmentation':
            # 分割任务的模拟数据
            mask = torch.randint(0, self.num_classes, (self.image_size, self.image_size))
            return {'pixel_values': pixel_values, 'mask': mask}


# ==================== 模型加载器 ====================

class TeacherModelLoader:
    """Qwen2.5-VL教师模型加载器"""

    @staticmethod
    def load_qwen2vl(model_path: str, device: torch.device):
        if not QWEN_AVAILABLE:
            print("⚠️  Qwen2VL未安装，使用模拟教师模型")
            return None, None

        print(f"正在加载Qwen2.5-VL教师模型: {model_path}")

        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map="cpu"
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
        resnet_variants = {
            'resnet18': models.resnet18,
            'resnet34': models.resnet34,
            'resnet50': models.resnet50,
            'resnet101': models.resnet101,
        }

        if size not in resnet_variants:
            raise ValueError(f"不支持的ResNet变体: {size}")

        model = resnet_variants[size](pretrained=pretrained)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        model.to(device)

        print(f"✓ ResNet-{size}加载成功，参数量: {sum(p.numel() for p in model.parameters()):,}")
        return model

    @staticmethod
    def _load_vit(size: str, num_classes: int, device, pretrained: bool):
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
        if not YOLO_AVAILABLE:
            raise ImportError("YOLOv8未安装，请运行: pip install ultralytics")

        yolo_sizes = {'n': 'yolov8n.pt', 's': 'yolov8s.pt', 'm': 'yolov8m.pt',
                     'l': 'yolov8l.pt', 'x': 'yolov8x.pt'}

        if size not in yolo_sizes:
            raise ValueError(f"不支持的YOLO大小: {size}")

        model = YOLO(yolo_sizes[size])
        print(f"✓ YOLOv8-{size}加载成功")
        return model

    @staticmethod
    def _load_unet(size: str, num_classes: int, device):
        class SimpleUNet(nn.Module):
            def __init__(self, in_channels=3, num_classes=10):
                super().__init__()
                self.enc1 = self._conv_block(in_channels, 64)
                self.enc2 = self._conv_block(64, 128)
                self.enc3 = self._conv_block(128, 256)
                self.enc4 = self._conv_block(256, 512)

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
                e1 = self.enc1(x)
                e2 = self.enc2(self.pool(e1))
                e3 = self.enc3(self.pool(e2))
                e4 = self.enc4(self.pool(e3))

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
        class LSTMClassifier(nn.Module):
            def __init__(self, input_size=2048, hidden_size=512, num_layers=2, num_classes=10):
                super().__init__()
                self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                                   batch_first=True, bidirectional=True)
                self.fc = nn.Linear(hidden_size * 2, num_classes)
                self.dropout = nn.Dropout(0.5)

                resnet = models.resnet50(pretrained=True)
                self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])

            def forward(self, x):
                batch_size = x.size(0)
                features = self.feature_extractor(x)
                features = features.view(batch_size, -1, 1)
                features = features.transpose(1, 2)

                lstm_out, _ = self.lstm(features)
                lstm_out = lstm_out[:, -1, :]
                lstm_out = self.dropout(lstm_out)

                output = self.fc(lstm_out)
                return output

        hidden_sizes = {'small': 256, 'medium': 512, 'large': 1024}
        hidden_size = hidden_sizes.get(size, 512)

        model = LSTMClassifier(hidden_size=hidden_size, num_classes=num_classes)
        model.to(device)
        print(f"✓ LSTM加载成功，参数量: {sum(p.numel() for p in model.parameters()):,}")
        return model


# ==================== 特征对齐层 ====================

class FeatureAlignmentLayer(nn.Module):
    """特征对齐层：将教师模型特征投影到学生模型特征空间"""

    def __init__(self, teacher_dim: int, student_dim: int, use_attention: bool = False):
        super().__init__()
        self.use_attention = use_attention

        self.projection = nn.Sequential(
            nn.Linear(teacher_dim, student_dim),
            nn.LayerNorm(student_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        if use_attention:
            self.attention = nn.MultiheadAttention(student_dim, num_heads=8, batch_first=True)

    def forward(self, teacher_features, student_features=None):
        aligned = self.projection(teacher_features)

        if self.use_attention and student_features is not None:
            aligned, _ = self.attention(aligned, student_features, student_features)

        return aligned


# ==================== 知识蒸馏训练器 ====================

class QwenMultiModelDistillationTrainer:
    """Qwen2.5-VL到多种小模型的知识蒸馏训练器"""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = self._setup_device()

        # 加载教师模型（Qwen2.5-VL）
        self.teacher_model, self.teacher_processor = TeacherModelLoader.load_qwen2vl(
            config.teacher_path,
            self.device
        )

        # 加载学生模型
        self.student_model = StudentModelLoader.load_model(
            config.student_model_type,
            config.student_model_size,
            config.num_classes,
            self.device
        )
        self.student_feature_map = None
        if config.student_model_type == "resnet":
            def hook_fn(module, input, output):
                self.student_feature_map = output

            self.student_model.avgpool.register_forward_hook(hook_fn)
        self.feature_aligner = None
        if config.align_feature and config.distillation_type in ['feature', 'hybrid']:

            student_dim = self._get_student_feature_dim()

            if self.teacher_model is not None:
                teacher_dim = self.teacher_model.config.vision_config.hidden_size
            else:
                teacher_dim = 1024  # fallback

            self.feature_aligner = FeatureAlignmentLayer(
                teacher_dim,
                student_dim,
                use_attention=False
            ).to(self.device)
        # 特征对齐层

#        '''' if config.align_feature and config.distillation_type in ['feature', 'hybrid']:
#             teacher_dim = 1280  # Qwen2.5-VL视觉编码器维度
#             student_dim = self._get_student_feature_dim()
#             self.feature_aligner = FeatureAlignmentLayer(
#                 teacher_dim, student_dim, use_attention=True
#             ).to(self.device)''''

        # 应用LoRA（如果需要）
        if config.lora_rank > 0 and config.student_model_type == "vit":
            self._apply_lora_to_student()

        # 优化器和调度器
        self._setup_optimizer()
        self._setup_scheduler()

        # 损失函数
        self.ce_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()
        self.cosine_loss = nn.CosineEmbeddingLoss()

        # 训练状态
        self.current_epoch = 0
        self.global_step = 0
        self.best_acc = 0.0

    def _setup_device(self) -> torch.device:
#         if torch.cuda.is_available():
#             device_id = self.config.gpu_devices[0]
#             device = torch.device(f"cuda:{device_id}")
#             print(f"✓ 使用GPU设备: cuda:{device_id}")
#         else:
        device = torch.device("cpu")
        print("⚠️  使用CPU训练")
        return device
    def _get_student_feature_dim(self) -> int:
        model_type = self.config.student_model_type
        size = self.config.student_model_size

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
            return 512

    def _apply_lora_to_student(self):
        """对学生模型应用LoRA"""
        if self.config.student_model_type == 'vit':
            task_type = TaskType.IMAGE_CLASSIFICATION
        else:
            task_type = TaskType.SEQ_CLS

        lora_config = LoraConfig(
            task_type=task_type,
            inference_mode=False,
            r=self.config.lora_rank,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=self.config.lora_target_modules if self.config.lora_target_modules else None,
            bias=self.config.lora_bias
        )

        self.student_model = get_peft_model(self.student_model, lora_config)
        print(f"✓ LoRA已应用到学生模型，可训练参数: {self.student_model.get_nb_trainable_parameters()}")

    def _setup_optimizer(self):
        params = list(self.student_model.parameters())
        if self.feature_aligner is not None:
            params += list(self.feature_aligner.parameters())

        if self.config.optimizer == 'adamw':
            self.optimizer = torch.optim.AdamW(
                params,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer == 'adam':
            self.optimizer = torch.optim.Adam(
                params,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer == 'sgd':
            self.optimizer = torch.optim.SGD(
                params,
                lr=self.config.learning_rate,
                momentum=0.9,
                weight_decay=self.config.weight_decay
            )

    def _setup_scheduler(self):
        num_training_steps = (1000 // self.config.batch_size) * self.config.epochs

        if self.config.lr_scheduler == 'cosine':
            from torch.optim.lr_scheduler import CosineAnnealingLR
            self.scheduler = CosineAnnealingLR(self.optimizer, T_max=num_training_steps)
        elif self.config.lr_scheduler == 'linear':
            from torch.optim.lr_scheduler import LinearLR
            self.scheduler = LinearLR(self.optimizer, start_factor=1.0,
                                     end_factor=0.1, total_iters=num_training_steps)
        else:
            self.scheduler = None
    def extract_teacher_features(self, images: torch.Tensor):
        """
        提取教师模型的视觉特征
        """
        batch_size = images.size(0)

        # 如果没有教师模型，返回随机特征
        if self.teacher_model is None:
            return {
                'vision_features': torch.randn(batch_size, 256, 1024).to(self.device)
            }

        # 转 PIL 图像
        pil_images = [transforms.ToPILImage()(img.cpu()) for img in images]

        # 获取 processor 输出
        inputs = self.teacher_processor(
            images=pil_images,
            text=["image"] * batch_size,
            return_tensors="pt",
        )

        # 单独处理 tensor 类型和 device
        for k, v in inputs.items():
            if k == "input_ids":  # embedding 输入必须是 LongTensor
                inputs[k] = v.long().to(self.device)
            else:
                inputs[k] = v.to(self.device)

        # 提取特征
        with torch.no_grad():
            outputs = self.teacher_model(**inputs, output_hidden_states=True)

        # 根据模型输出获取视觉特征
        if hasattr(outputs, "vision_hidden_states"):
            vision_features = outputs.vision_hidden_states[-1]
        else:
            vision_features = outputs.last_hidden_state  # 根据实际模型调整

        return {
            'vision_features': vision_features
        }

#     '''def extract_teacher_features(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
#         """从Qwen2.5-VL提取视觉特征"""
#         if self.teacher_model is None:
#             # 模拟模式
#             batch_size = images.size(0)
#             return {
#                 'vision_features': torch.randn(batch_size, 256, 1280).to(self.device),
#                 'hidden_states': [torch.randn(batch_size, 256, 1280).to(self.device)]
#             }
#
#         with torch.no_grad():
#             outputs = self.teacher_model.visual(images, output_hidden_states=True)
#             return {
#                 'vision_features': outputs.last_hidden_state,
#                 'hidden_states': outputs.hidden_states
#             }''''

    def compute_distillation_loss(
        self,
        student_output: Any,
        teacher_features: Dict[str, torch.Tensor],
        labels: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """计算蒸馏损失"""
        losses = {}

        # 硬标签损失
        if self.config.task_type == 'classification':
            if isinstance(student_output, dict):
                logits = student_output['logits']
            else:
                logits = student_output
            hard_loss = self.ce_loss(logits, labels)
            losses['hard_loss'] = hard_loss
        else:
            hard_loss = torch.tensor(0.0).to(self.device)
            losses['hard_loss'] = hard_loss

        # 软标签损失（暂时简化）
        soft_loss = torch.tensor(0.0).to(self.device)
        losses['soft_loss'] = soft_loss

        # 特征蒸馏损失
        if self.config.distillation_type in ['feature', 'hybrid']:
            student_features = self._extract_student_features(student_output)
            teacher_vis_features = teacher_features['vision_features']

            if self.feature_aligner is not None:
                aligned_teacher_features = self.feature_aligner(
                    teacher_vis_features, student_features
                )
            else:
                aligned_teacher_features = teacher_vis_features

            if self.config.feature_loss_type == 'mse':
                if student_features.shape != aligned_teacher_features.shape:
                    student_features = F.adaptive_avg_pool1d(
                        student_features.transpose(1, 2),
                        aligned_teacher_features.size(1)
                    ).transpose(1, 2)
                feature_loss = self.mse_loss(student_features, aligned_teacher_features)
            elif self.config.feature_loss_type == 'cosine':
                student_norm = F.normalize(student_features.mean(dim=1), dim=-1)
                teacher_norm = F.normalize(aligned_teacher_features.mean(dim=1), dim=-1)
                target = torch.ones(student_norm.size(0)).to(self.device)
                feature_loss = self.cosine_loss(student_norm, teacher_norm, target)
            else:
                feature_loss = torch.tensor(0.0).to(self.device)

            losses['feature_loss'] = feature_loss

        # 总损失
        total_loss = (
            self.config.hard_label_weight * losses.get('hard_loss', 0) +
            self.config.soft_label_weight * losses.get('soft_loss', 0)
        )

        if 'feature_loss' in losses:
            # 特征损失权重可以通过蒸馏配置调整
            feature_weight = 0.2
            total_loss += feature_weight * losses['feature_loss']

        losses['total_loss'] = total_loss
        return losses

    def _extract_student_features(self, student_output):

        if self.student_feature_map is not None:
            feat = self.student_feature_map  # [B, C, H, W]
            feat = feat.flatten(2).transpose(1, 2)
            return feat  # [B, HW, C]

        raise RuntimeError("❌ 未捕获学生模型 backbone 特征，请检查 hook")
#     '''def _extract_student_features(self, student_output) -> torch.Tensor:
#         if isinstance(student_output, dict):
#             if 'hidden_states' in student_output:
#                 return student_output['hidden_states'][-1]
#             elif 'last_hidden_state' in student_output:
#                 return student_output['last_hidden_state']''''

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

        epoch_losses = {'total_loss': 0.0, 'hard_loss': 0.0, 'soft_loss': 0.0, 'feature_loss': 0.0}
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {self.current_epoch + 1}/{self.config.epochs}")

        for batch_idx, batch in enumerate(pbar):
            images = batch['pixel_values'].to(self.device)
            labels = batch['labels'].to(self.device)

            # 提取教师特征
            teacher_features = self.extract_teacher_features(images)

            # 学生模型前向传播
            student_output = self.student_model(images)
            losses = self.compute_distillation_loss(student_output, teacher_features, labels)
            loss = losses['total_loss'] / self.config.grad_accum_steps

            # 反向传播
            loss.backward()

            # 梯度累积
            if (batch_idx + 1) % self.config.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.student_model.parameters(), self.config.max_grad_norm)
                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1

            # 记录损失
            for key in epoch_losses:
                if key in losses:
                    epoch_losses[key] += losses[key].item()
            num_batches += 1

            pbar.set_postfix({'loss': f"{losses['total_loss'].item():.4f}",
                            'lr': f"{self.optimizer.param_groups[0]['lr']:.6f}"})

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

            outputs = self.student_model(images)

            if isinstance(outputs, dict):
                logits = outputs['logits']
            else:
                logits = outputs

            loss = self.ce_loss(logits, labels)
            total_loss += loss.item()

            _, predicted = torch.max(logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        avg_loss = total_loss / len(val_loader)
        accuracy = 100.0 * correct / total

        return {'val_loss': avg_loss, 'val_accuracy': accuracy}

    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        """完整训练流程"""
        print(f"\n{'='*60}")
        print("🚀 开始训练 - Qwen2.5-VL多模型协同训练")
        print(f"{'='*60}")
        print(f"教师模型: Qwen2.5-VL 3B")
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

            # 回调后端API
            self._update_training_progress(epoch + 1, train_metrics, val_metrics)

            # 保存checkpoint
            if self.config.auto_save_checkpoint and (epoch + 1) % self.config.checkpoint_interval == 0:
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
            url = f"{self.config.api_base_url}/model-distillation/tasks/{self.config.task_id}/progress"
            data = {
                'currentEpoch': epoch,
                'totalEpochs': self.config.epochs,
                'trainLoss': train_metrics['total_loss'],
                'valLoss': val_metrics['val_loss'],
                'valAccuracy': val_metrics['val_accuracy'],
                'status': 'RUNNING'
            }

            response = requests.put(url, json=data, timeout=5)
            if response.status_code != 200:
                print(f"⚠️  进度更新失败: {response.text}")
        except Exception as e:
            print(f"⚠️  进度更新异常: {e}")


# ==================== 命令行参数解析 ====================

def parse_args():
    parser = argparse.ArgumentParser(description='Qwen2.5-VL多模型协同训练')

    # 基础配置
    parser.add_argument('--task_id', type=str, required=True)
    parser.add_argument('--api_base_url', type=str, required=True)

    # 模型配置
    parser.add_argument('--teacher_model', type=str, required=True)
    parser.add_argument('--student_model', type=str, required=True)
    parser.add_argument('--teacher_path', type=str, required=True)
    parser.add_argument('--student_path', type=str, default=None)

    # 学生模型类型和大小
    parser.add_argument('--student_model_type', type=str, required=True,
                       choices=['resnet', 'vit', 'yolov8', 'unet', 'lstm'])
    parser.add_argument('--student_model_size', type=str, required=True)

    # 任务配置
    parser.add_argument('--task_type', type=str, default='classification',
                       choices=['classification', 'detection', 'segmentation'])
    parser.add_argument('--num_classes', type=int, default=10)

    # 数据配置
    parser.add_argument('--dataset_id', type=str, required=True)
    parser.add_argument('--val_dataset_id', type=str, default=None)
    parser.add_argument('--image_size', type=int, default=224)

    # 训练参数
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=1e-4)

    # 优化器配置
    parser.add_argument('--optimizer', type=str, default='adamw',
                       choices=['adamw', 'adam', 'sgd'])
    parser.add_argument('--lr_scheduler', type=str, default='cosine',
                       choices=['cosine', 'linear', 'constant'])
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--grad_accum_steps', type=int, default=1)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)

    # GPU配置
    parser.add_argument('--gpu_devices', type=str, default='0')
    parser.add_argument('--auto_save_checkpoint', type=bool, default=True)
    #parser.add_argument('--auto_save_checkpoint', action='store_true')
    #parser.add_argument('--no_auto_save_checkpoint', action='store_false',
                        #dest='auto_save_checkpoint')
    parser.add_argument('--checkpoint_interval', type=int, default=10)

    # LoRA配置
    parser.add_argument('--lora_rank', type=int, default=0)
    parser.add_argument('--lora_alpha', type=int, default=16)
    parser.add_argument('--lora_dropout', type=float, default=0.1)
    parser.add_argument('--lora_target_modules', type=str, default='')
    parser.add_argument('--lora_bias', type=str, default='none')

    # 知识蒸馏配置
    parser.add_argument('--temperature', type=float, default=4.0)
    parser.add_argument('--hard_label_weight', type=float, default=0.5)
    parser.add_argument('--soft_label_weight', type=float, default=0.5)
    parser.add_argument('--distill_loss_type', type=str, default='kl_div')

    # 蒸馏策略
    parser.add_argument('--distillation_type', type=str, default='hybrid',
                       choices=['feature', 'logit', 'hybrid'])
    parser.add_argument('--feature_loss_type', type=str, default='mse',
                       choices=['mse', 'cosine'])
    parser.add_argument('--align_feature', type=bool, default=True)
#     parser.add_argument('--align_feature', action='store_true')
#     parser.add_argument('--no_align_feature', action='store_false', dest='align_feature')
    parser.set_defaults(align_feature=True)

    # 输出配置
    parser.add_argument('--output_dir', type=str, required=True)

    # 数据集根目录（新增，由后端配置文件传递）
    parser.add_argument('--datasets_root', type=str, required=True, help='数据集根目录')

    return parser.parse_args()


# ==================== 主函数 ====================

def main():
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║     Qwen2.5-VL 多模型协同训练系统                            ║
    ║     Multi-Model Collaborative Training with Qwen2.5-VL       ║
    ╚══════════════════════════════════════════════════════════════╝
    """)

    args = parse_args()
    config = TrainingConfig(args)

    print("\n正在加载数据集...")

    # 构建训练集路径
    train_dataset_path = os.path.join(config.datasets_root, config.dataset_id, "train")
    print(f"训练集路径: {train_dataset_path}")

    # 构建验证集路径
    val_dataset_id = config.val_dataset_id or config.dataset_id
    val_dataset_path = os.path.join(config.datasets_root, val_dataset_id, "val")
    print(f"验证集路径: {val_dataset_path}")

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

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    print(f"✓ 训练集: {len(train_dataset)} 样本")
    print(f"✓ 验证集: {len(val_dataset)} 样本")

    trainer = QwenMultiModelDistillationTrainer(config)
    trainer.train(train_loader, val_loader)

    print("\n✅ 所有任务完成！")


if __name__ == '__main__':
    main()