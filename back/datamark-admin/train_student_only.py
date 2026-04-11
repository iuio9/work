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

# OpenCV（目标检测数据加载需要）
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    warnings.warn("OpenCV未安装，目标检测训练将不可用，请运行: pip install opencv-python")

# pycocotools（目标检测评估需要）
try:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    PYCOCOTOOLS_AVAILABLE = True
except ImportError:
    PYCOCOTOOLS_AVAILABLE = False
    warnings.warn("pycocotools未安装，Faster R-CNN mAP评估将不可用: pip install pycocotools")


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
    def build_yolov8(model_size: str, num_classes: int):
        """构建YOLOv8模型"""
        if not YOLO_AVAILABLE:
            raise ImportError("YOLOv8未安装，请运行: pip install ultralytics")

        yolo_sizes = {'n': 'yolov8n.pt', 's': 'yolov8s.pt', 'm': 'yolov8m.pt',
                     'l': 'yolov8l.pt', 'x': 'yolov8x.pt'}

        if model_size not in yolo_sizes:
            raise ValueError(f"不支持的YOLO大小: {model_size}")

        model = YOLO(yolo_sizes[model_size])
        print(f"✓ YOLOv8-{model_size}加载成功")
        return model

    @staticmethod
    def build_unet(model_size: str, num_classes: int):
        """构建UNet模型"""
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
        print(f"✓ UNet-{model_size}加载成功，参数量: {sum(p.numel() for p in model.parameters()):,}")
        return model

    @staticmethod
    def build_lstm(model_size: str, num_classes: int):
        """构建LSTM模型"""
        hidden_sizes = {'small': 256, 'medium': 512, 'large': 1024}
        hidden_size = hidden_sizes.get(model_size, 512)

        class LSTMClassifier(nn.Module):
            def __init__(self, input_size=224*3, hidden_size=512, num_layers=2, num_classes=10):
                super().__init__()
                self.hidden_size = hidden_size
                self.num_layers = num_layers
                self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.3)
                self.fc = nn.Linear(hidden_size, num_classes)

            def forward(self, x):
                # x: (batch, C, H, W) -> (batch, H, W*C)
                b, c, h, w = x.shape
                x = x.permute(0, 2, 3, 1).reshape(b, h, w * c)
                out, _ = self.lstm(x)
                out = out[:, -1, :]  # 取最后一个时间步
                return self.fc(out)

        model = LSTMClassifier(input_size=224*3, hidden_size=hidden_size,
                              num_layers=2, num_classes=num_classes)
        print(f"✓ LSTM-{model_size}加载成功，参数量: {sum(p.numel() for p in model.parameters()):,}")
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
        elif model_type == 'yolov8':
            return StudentModelBuilder.build_yolov8(model_size, num_classes)
        elif model_type == 'unet':
            return StudentModelBuilder.build_unet(model_size, num_classes)
        elif model_type == 'lstm':
            return StudentModelBuilder.build_lstm(model_size, num_classes)
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")


# ==================== 目标检测数据集 ====================

class YoloDetectionDataset(Dataset):
    """YOLO格式目标检测数据集（用于Faster R-CNN等torchvision检测模型）

    期望目录结构：
        data_root/
          images/train/*.jpg
          images/val/*.jpg
          labels/train/*.txt  (YOLO格式: cls_id xc yc w h 归一化)
          labels/val/*.txt

    参数：
        label_offset: 类别ID偏移量。torchvision 检测模型约定 0=background，
                     因此需要 label_offset=1；HuggingFace YOLOS 等模型使用
                     0-indexed 类别，此时传 label_offset=0。
    """

    def __init__(self, data_root: str, split: str = 'train', label_offset: int = 1):
        if not CV2_AVAILABLE:
            raise ImportError("目标检测数据集需要 opencv-python，请运行: pip install opencv-python")

        self.data_root = Path(data_root)
        self.img_dir = self.data_root / "images" / split
        self.label_dir = self.data_root / "labels" / split
        self.label_offset = label_offset

        if not self.img_dir.exists():
            raise FileNotFoundError(f"图像目录不存在: {self.img_dir}")

        # 支持多种图像格式
        self.images = []
        for ext in ('*.jpg', '*.jpeg', '*.png', '*.bmp'):
            self.images.extend(list(self.img_dir.glob(ext)))
        self.images = sorted(self.images)

        print(f"✅ YOLO检测数据集: {self.img_dir} (split={split}, 图像数={len(self.images)})")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label_path = self.label_dir / f"{img_path.stem}.txt"

        img = cv2.imread(str(img_path))
        if img is None:
            raise RuntimeError(f"无法读取图像: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, _ = img.shape

        boxes = []
        labels = []

        if label_path.exists():
            with open(label_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    cid, xc, yc, bw, bh = map(float, parts[:5])
                    x1 = (xc - bw / 2) * w
                    y1 = (yc - bh / 2) * h
                    x2 = (xc + bw / 2) * w
                    y2 = (yc + bh / 2) * h
                    boxes.append([x1, y1, x2, y2])
                    # label_offset=1: torchvision (0=background)
                    # label_offset=0: HuggingFace YOLOS
                    labels.append(int(cid) + self.label_offset)

        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx]),
            "orig_size": torch.tensor([h, w]),
        }

        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        return img_tensor, target


def detection_collate_fn(batch):
    """目标检测专用collate函数（处理可变数量目标）"""
    return tuple(zip(*batch))


# ==================== COCO mAP 评估器 ====================

class CocoEvaluator:
    """将YoloDetectionDataset的GT转换为COCO格式并评估mAP"""

    def __init__(self, dataset: 'YoloDetectionDataset', num_classes: int):
        if not PYCOCOTOOLS_AVAILABLE:
            raise ImportError("mAP评估需要 pycocotools，请运行: pip install pycocotools")

        self.dataset = dataset

        images_meta = []
        anns = []
        ann_id = 1

        for i in range(len(dataset)):
            _, target = dataset[i]
            images_meta.append({"id": i})

            for box, label in zip(target["boxes"], target["labels"]):
                x1, y1, x2, y2 = box.tolist()
                anns.append({
                    "id": ann_id,
                    "image_id": i,
                    "category_id": int(label),
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "area": (x2 - x1) * (y2 - y1),
                    "iscrowd": 0,
                })
                ann_id += 1

        # 类别ID从1开始（0留给background）
        cats = [{"id": i} for i in range(1, num_classes + 1)]

        coco_dict = {
            "images": images_meta,
            "annotations": anns,
            "categories": cats,
        }

        self.coco_gt = COCO()
        self.coco_gt.dataset = coco_dict
        self.coco_gt.createIndex()

    def evaluate(self, model, dataloader, device):
        model.eval()
        coco_results = []

        with torch.no_grad():
            for imgs, targets in tqdm(dataloader, desc="验证中"):
                imgs = [i.to(device) for i in imgs]
                outputs = model(imgs)

                for out, tgt in zip(outputs, targets):
                    img_id = int(tgt["image_id"])
                    for box, score, label in zip(out["boxes"], out["scores"], out["labels"]):
                        coco_results.append({
                            "image_id": img_id,
                            "category_id": int(label),
                            "bbox": [
                                box[0].item(),
                                box[1].item(),
                                box[2].item() - box[0].item(),
                                box[3].item() - box[1].item(),
                            ],
                            "score": score.item(),
                        })

        if len(coco_results) == 0:
            print("⚠️ 验证集未产生任何检测结果，返回全零mAP")
            return [0.0] * 12

        coco_dt = self.coco_gt.loadRes(coco_results)
        coco_eval = COCOeval(self.coco_gt, coco_dt, "bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        return coco_eval.stats


# ==================== Faster R-CNN 检测训练器 ====================

class FasterRCNNDetectionTrainer:
    """Faster R-CNN (ResNet50-FPN) 目标检测训练器

    基于 resnet-ele.py 的训练逻辑改造而来。
    """

    def __init__(self, config: TrainingConfig):
        self.config = config

        print("\n" + "=" * 60)
        print("初始化 Faster R-CNN 目标检测训练器")
        print("=" * 60)

        from torchvision.models.detection import fasterrcnn_resnet50_fpn
        from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

        # num_classes + 1（+1 留给 background）
        num_classes_with_bg = config.num_classes + 1

        model = fasterrcnn_resnet50_fpn(weights="DEFAULT")
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes_with_bg)

        self.model = model.to(config.device)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        self.best_map = 0.0
        self.train_losses = []
        self.val_accuracies = []

    def train(self, train_loader: DataLoader, val_loader: DataLoader, val_dataset: YoloDetectionDataset):
        """完整训练流程"""
        print("\n" + "=" * 60)
        print("开始 Faster R-CNN 训练")
        print("=" * 60)

        evaluator = CocoEvaluator(val_dataset, self.config.num_classes)

        for epoch in range(1, self.config.epochs + 1):
            self.model.train()
            print(f"\n{'=' * 60}")
            print(f"Epoch {epoch}/{self.config.epochs}")
            print(f"{'=' * 60}")

            epoch_loss = 0.0
            n_batches = 0
            pbar = tqdm(train_loader, desc=f"Epoch {epoch}")

            for imgs, targets in pbar:
                # 过滤没有GT的样本（Faster R-CNN不支持空目标）
                valid_imgs, valid_targets = [], []
                for img, tgt in zip(imgs, targets):
                    if tgt["boxes"].shape[0] > 0:
                        valid_imgs.append(img)
                        valid_targets.append(tgt)

                if len(valid_imgs) == 0:
                    continue

                imgs_dev = [i.to(self.config.device) for i in valid_imgs]
                tgts_dev = [
                    {k: v.to(self.config.device) for k, v in t.items()}
                    for t in valid_targets
                ]

                loss_dict = self.model(imgs_dev, tgts_dev)
                loss = sum(loss_dict.values())

                self.optimizer.zero_grad()
                loss.backward()

                if self.config.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.max_grad_norm,
                    )

                self.optimizer.step()

                epoch_loss += float(loss)
                n_batches += 1
                pbar.set_postfix(loss=float(loss))

            avg_loss = epoch_loss / max(n_batches, 1)
            self.train_losses.append(avg_loss)

            # ========== 验证 ==========
            stats = evaluator.evaluate(self.model, val_loader, self.config.device)
            map5095 = float(stats[0]) * 100.0  # COCO mAP@[0.5:0.95]
            map50 = float(stats[1]) * 100.0    # COCO mAP@0.5

            self.val_accuracies.append(map50)

            print(f"\n📊 Epoch {epoch}: loss={avg_loss:.4f}, mAP50={map50:.2f}%, mAP50-95={map5095:.2f}%")

            # 保存定期checkpoint
            if self.config.auto_save_checkpoint and epoch % self.config.checkpoint_interval == 0:
                self._save_checkpoint(epoch)

            # 保存最佳模型（按mAP50-95）
            if map5095 > self.best_map:
                self.best_map = map5095
                self._save_best()
                print(f"🎉 新的最佳 mAP50-95: {map5095:.2f}%")

            # 上报进度（使用mAP50作为可视化指标）
            self.send_progress_update(epoch, avg_loss, map50)

        # 训练结束保存final
        self._save_final()
        print("\n" + "=" * 60)
        print("Faster R-CNN 训练完成!")
        print(f"最佳 mAP50-95: {self.best_map:.2f}%")
        print("=" * 60)

    def _save_checkpoint(self, epoch: int):
        ckpt_dir = os.path.join(self.config.output_dir, f'checkpoint-epoch-{epoch}')
        os.makedirs(ckpt_dir, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(ckpt_dir, 'model.pt'))
        info = {
            'epoch': epoch,
            'model_type': 'fasterrcnn_resnet50_fpn',
            'num_classes': self.config.num_classes,
            'best_map': self.best_map,
        }
        with open(os.path.join(ckpt_dir, 'training_info.json'), 'w') as f:
            json.dump(info, f, indent=2)
        print(f"✅ 检查点已保存: {ckpt_dir}")

    def _save_best(self):
        best_dir = os.path.join(self.config.output_dir, 'best')
        os.makedirs(best_dir, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(best_dir, 'best.pt'))

    def _save_final(self):
        final_dir = os.path.join(self.config.output_dir, 'checkpoint-epoch-final')
        os.makedirs(final_dir, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(final_dir, 'model.pt'))

    def send_progress_update(self, epoch: int, loss: float, accuracy: float):
        """向后端上报进度（与DirectTrainer保持一致）"""
        try:
            url = f"{self.config.api_base_url}/api/distillation/tasks/{self.config.task_id}/progress"
            data = {
                'currentEpoch': epoch,
                'totalEpochs': self.config.epochs,
                'loss': float(loss),
                'accuracy': float(accuracy),
                'status': 'RUNNING',
            }
            response = requests.post(url, json=data, timeout=5)
            if response.status_code == 200:
                print(f"✅ 进度已更新: Epoch {epoch}, Loss {loss:.4f}, mAP50 {accuracy:.2f}%")
        except Exception as e:
            print(f"⚠️ 发送进度失败: {e}")

    @property
    def best_accuracy(self) -> float:
        return self.best_map


# ==================== YOLOv8 检测训练器 ====================

class YoloV8DetectionTrainer:
    """YOLOv8 目标检测训练器（使用 ultralytics 原生 .train() API）

    基于 yolov8.py 的训练逻辑改造而来。
    数据集期望目录结构同 YoloDetectionDataset，外加可选的 data.yaml。
    如未提供 data.yaml，会自动根据 dataset_id 生成一个。
    """

    # 支持的模型大小 → 预训练权重
    YOLO_WEIGHTS = {
        'n': 'yolov8n.pt',
        's': 'yolov8s.pt',
        'm': 'yolov8m.pt',
        'l': 'yolov8l.pt',
        'x': 'yolov8x.pt',
        'yolov8n': 'yolov8n.pt',
        'yolov8s': 'yolov8s.pt',
        'yolov8m': 'yolov8m.pt',
        'yolov8l': 'yolov8l.pt',
        'yolov8x': 'yolov8x.pt',
    }

    def __init__(self, config: TrainingConfig):
        self.config = config

        if not YOLO_AVAILABLE:
            raise ImportError("YOLOv8未安装，请运行: pip install ultralytics")

        print("\n" + "=" * 60)
        print("初始化 YOLOv8 目标检测训练器")
        print("=" * 60)

        weights = self.YOLO_WEIGHTS.get(config.student_model_size, 'yolov8s.pt')
        # 允许 student_path 覆盖（用于加载用户的预训练权重）
        if getattr(config, 'student_path', None):
            student_path = config.student_path
            if student_path and os.path.exists(student_path):
                weights = student_path
                print(f"使用用户提供的权重: {weights}")

        print(f"加载 YOLOv8 预训练权重: {weights}")
        self.model = YOLO(weights)
        self.best_map = 0.0

    def _resolve_data_yaml(self) -> str:
        """查找或自动生成 data.yaml"""
        dataset_dir = Path(self.config.datasets_root) / self.config.dataset_id
        data_yaml_path = dataset_dir / "data.yaml"

        if data_yaml_path.exists():
            print(f"✅ 使用现有 data.yaml: {data_yaml_path}")
            return str(data_yaml_path)

        print(f"⚠️ 未找到 data.yaml，自动生成于: {data_yaml_path}")

        try:
            import yaml
        except ImportError:
            raise ImportError("生成 data.yaml 需要 PyYAML: pip install pyyaml")

        data_config = {
            'path': str(dataset_dir.resolve()),
            'train': 'images/train',
            'val': 'images/val',
            'nc': self.config.num_classes,
            'names': [f'class_{i}' for i in range(self.config.num_classes)],
        }

        dataset_dir.mkdir(parents=True, exist_ok=True)
        with open(data_yaml_path, 'w', encoding='utf-8') as f:
            yaml.safe_dump(data_config, f, default_flow_style=False, allow_unicode=True)

        print(f"✅ 已生成 data.yaml (nc={self.config.num_classes})")
        return str(data_yaml_path)

    def train(self):
        """执行 YOLOv8 原生训练流程"""
        data_yaml = self._resolve_data_yaml()

        # YOLOv8 device: 0 表示cuda:0，否则 'cpu'
        device = 0 if torch.cuda.is_available() else 'cpu'

        # ultralytics 会在 project/name 目录下生成 weights/best.pt
        run_name = f"yolov8_{self.config.task_id}"

        print(f"\n🚀 启动 YOLOv8 训练...")
        print(f"   - data: {data_yaml}")
        print(f"   - epochs: {self.config.epochs}")
        print(f"   - imgsz: {self.config.image_size}")
        print(f"   - batch: {self.config.batch_size}")
        print(f"   - device: {device}")
        print(f"   - output: {self.config.output_dir}/{run_name}")

        results = self.model.train(
            data=data_yaml,
            epochs=self.config.epochs,
            imgsz=self.config.image_size,
            batch=self.config.batch_size,
            device=device,
            project=self.config.output_dir,
            name=run_name,
            save=True,
            plots=True,
            exist_ok=True,
            lr0=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        # 从 results 中提取最佳 mAP50
        try:
            self.best_map = float(results.box.map50) * 100.0
            map5095 = float(results.box.map) * 100.0
            print(f"\n📊 最终指标: mAP50={self.best_map:.2f}%, mAP50-95={map5095:.2f}%")
        except Exception as e:
            print(f"⚠️ 无法从训练结果提取mAP: {e}")
            self.best_map = 0.0

        # 汇报训练完成的进度
        self.send_progress_update(self.config.epochs, 0.0, self.best_map)

        # 复制 best.pt 到标准的 best/ 目录（便于后续推理脚本定位）
        run_dir = Path(self.config.output_dir) / run_name / 'weights'
        best_pt = run_dir / 'best.pt'
        if best_pt.exists():
            standard_best_dir = Path(self.config.output_dir) / 'best'
            standard_best_dir.mkdir(parents=True, exist_ok=True)
            import shutil
            shutil.copy(str(best_pt), str(standard_best_dir / 'best.pt'))
            print(f"✅ 已将 best.pt 复制到: {standard_best_dir}/best.pt")

    def send_progress_update(self, epoch: int, loss: float, accuracy: float):
        """向后端上报进度（YOLOv8原生训练不支持逐epoch回调，只在结束时上报一次）"""
        try:
            url = f"{self.config.api_base_url}/api/distillation/tasks/{self.config.task_id}/progress"
            data = {
                'currentEpoch': epoch,
                'totalEpochs': self.config.epochs,
                'loss': float(loss),
                'accuracy': float(accuracy),
                'status': 'RUNNING',
            }
            response = requests.post(url, json=data, timeout=5)
            if response.status_code == 200:
                print(f"✅ 进度已更新: mAP50 {accuracy:.2f}%")
        except Exception as e:
            print(f"⚠️ 发送进度失败: {e}")

    @property
    def best_accuracy(self) -> float:
        return self.best_map


# ==================== ViT (YOLOS) 检测训练器 ====================

class _YolosDatasetWrapper(Dataset):
    """将 YoloDetectionDataset 转换为 HuggingFace YOLOS 可以消费的格式"""

    def __init__(self, yolo_dataset: 'YoloDetectionDataset', image_processor):
        self.yolo_dataset = yolo_dataset
        self.processor = image_processor

    def __len__(self):
        return len(self.yolo_dataset)

    def __getitem__(self, idx):
        # YoloDetectionDataset 返回 (C,H,W) float tensor 和 dict target
        img_tensor, target = self.yolo_dataset[idx]

        # Tensor (C,H,W, float [0,1]) → PIL Image
        img_np = (img_tensor.permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
        img_pil = Image.fromarray(img_np)
        w, h = img_pil.size

        # 把 xyxy 转成 COCO 的 xywh（像素坐标）
        annotations = []
        boxes_list = target["boxes"].tolist()
        labels_list = target["labels"].tolist()
        for box, label in zip(boxes_list, labels_list):
            x1, y1, x2, y2 = box
            bw = max(x2 - x1, 0.0)
            bh = max(y2 - y1, 0.0)
            annotations.append({
                "bbox": [x1, y1, bw, bh],
                "category_id": int(label),  # 已经是 0-indexed（label_offset=0）
                "area": float(bw * bh),
                "iscrowd": 0,
            })

        coco_target = {
            "image_id": int(idx),
            "annotations": annotations,
        }

        encoding = self.processor(
            images=img_pil,
            annotations=coco_target,
            return_tensors="pt",
        )

        pixel_values = encoding["pixel_values"].squeeze(0)
        labels = encoding["labels"][0]

        return pixel_values, labels, (h, w)


def yolos_collate_fn(batch):
    """HF YOLOS 专用 collate：pixel_values 做 padding，labels 保持 list"""
    pixel_values = [b[0] for b in batch]
    labels = [b[1] for b in batch]
    orig_sizes = [b[2] for b in batch]

    # 同一 batch 里图像可能尺寸不同，使用最大尺寸 padding
    max_h = max(p.shape[1] for p in pixel_values)
    max_w = max(p.shape[2] for p in pixel_values)
    padded = torch.zeros(len(pixel_values), 3, max_h, max_w, dtype=pixel_values[0].dtype)
    pixel_mask = torch.zeros(len(pixel_values), max_h, max_w, dtype=torch.long)
    for i, p in enumerate(pixel_values):
        c, ph, pw = p.shape
        padded[i, :, :ph, :pw] = p
        pixel_mask[i, :ph, :pw] = 1

    return {
        "pixel_values": padded,
        "pixel_mask": pixel_mask,
        "labels": labels,
        "orig_sizes": orig_sizes,
    }


class YolosViTDetectionTrainer:
    """基于 HuggingFace YOLOS 的 Vision Transformer 目标检测训练器

    YOLOS (You Only Look at One Sequence) 是纯 ViT 主干的检测模型，
    使用 DETR 风格的 set-prediction 训练。

    参考：https://huggingface.co/hustvl
    """

    VIT_VARIANTS = {
        'tiny': 'hustvl/yolos-tiny',
        'small': 'hustvl/yolos-small',
        'base': 'hustvl/yolos-base',
        'vit-tiny': 'hustvl/yolos-tiny',
        'vit-small': 'hustvl/yolos-small',
        'vit-base': 'hustvl/yolos-base',
    }

    def __init__(self, config: TrainingConfig):
        self.config = config

        print("\n" + "=" * 60)
        print("初始化 ViT (YOLOS) 目标检测训练器")
        print("=" * 60)

        try:
            from transformers import YolosImageProcessor, YolosForObjectDetection
        except ImportError as e:
            raise ImportError(
                "ViT 检测训练器需要 transformers 库，请运行: pip install transformers"
            ) from e

        model_id = self.VIT_VARIANTS.get(config.student_model_size, 'hustvl/yolos-small')
        print(f"加载 YOLOS 预训练权重: {model_id}")

        self.processor = YolosImageProcessor.from_pretrained(model_id)
        # num_labels 不包含 background；id2label/label2id 后续通过外部 API 管理
        id2label = {i: f"class_{i}" for i in range(config.num_classes)}
        label2id = {v: k for k, v in id2label.items()}

        self.model = YolosForObjectDetection.from_pretrained(
            model_id,
            num_labels=config.num_classes,
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True,
        )
        self.model.to(config.device)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        self.best_map = 0.0
        self.train_losses = []

    def train(self, train_loader: DataLoader, val_loader: DataLoader, val_yolo_dataset: 'YoloDetectionDataset'):
        """完整训练流程"""
        print("\n" + "=" * 60)
        print("开始 YOLOS (ViT) 训练")
        print("=" * 60)

        # 评估器基于未偏移 (label_offset=0) 的 val_yolo_dataset 构建
        # 注意：CocoEvaluator 默认类别从1开始，这里需要从0开始适配HF输出
        evaluator = _build_coco_evaluator_zero_indexed(val_yolo_dataset, self.config.num_classes)

        for epoch in range(1, self.config.epochs + 1):
            self.model.train()
            print(f"\n{'=' * 60}")
            print(f"Epoch {epoch}/{self.config.epochs}")
            print(f"{'=' * 60}")

            epoch_loss = 0.0
            n_batches = 0
            pbar = tqdm(train_loader, desc=f"Epoch {epoch}")

            for batch in pbar:
                pixel_values = batch["pixel_values"].to(self.config.device)
                pixel_mask = batch["pixel_mask"].to(self.config.device)
                labels = [
                    {k: v.to(self.config.device) for k, v in t.items()}
                    for t in batch["labels"]
                ]

                outputs = self.model(
                    pixel_values=pixel_values,
                    pixel_mask=pixel_mask,
                    labels=labels,
                )
                loss = outputs.loss

                self.optimizer.zero_grad()
                loss.backward()

                if self.config.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.max_grad_norm,
                    )

                self.optimizer.step()

                epoch_loss += float(loss)
                n_batches += 1
                pbar.set_postfix(loss=float(loss))

            avg_loss = epoch_loss / max(n_batches, 1)
            self.train_losses.append(avg_loss)

            # ========== 验证 ==========
            map50, map5095 = self._evaluate(val_loader, evaluator)

            print(f"\n📊 Epoch {epoch}: loss={avg_loss:.4f}, mAP50={map50:.2f}%, mAP50-95={map5095:.2f}%")

            if self.config.auto_save_checkpoint and epoch % self.config.checkpoint_interval == 0:
                self._save_checkpoint(epoch)

            if map5095 > self.best_map:
                self.best_map = map5095
                self._save_best()
                print(f"🎉 新的最佳 mAP50-95: {map5095:.2f}%")

            self.send_progress_update(epoch, avg_loss, map50)

        self._save_final()
        print("\n" + "=" * 60)
        print("YOLOS (ViT) 训练完成!")
        print(f"最佳 mAP50-95: {self.best_map:.2f}%")
        print("=" * 60)

    def _evaluate(self, val_loader: DataLoader, evaluator) -> Tuple[float, float]:
        """运行验证，返回 (mAP50, mAP50-95)"""
        self.model.eval()

        # 收集 COCO 格式的预测结果
        coco_results = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="验证中"):
                pixel_values = batch["pixel_values"].to(self.config.device)
                pixel_mask = batch["pixel_mask"].to(self.config.device)
                orig_sizes = batch["orig_sizes"]
                labels = batch["labels"]

                outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask)

                # target_sizes: HF 后处理要求的原始图像尺寸 (h, w)
                target_sizes = torch.tensor([s for s in orig_sizes], device=self.config.device)
                processed = self.processor.post_process_object_detection(
                    outputs,
                    threshold=0.05,
                    target_sizes=target_sizes,
                )

                for i, result in enumerate(processed):
                    img_id = int(labels[i]["image_id"].item()) if "image_id" in labels[i] else i
                    for box, score, label in zip(result["boxes"], result["scores"], result["labels"]):
                        x1, y1, x2, y2 = box.tolist()
                        coco_results.append({
                            "image_id": img_id,
                            "category_id": int(label),
                            "bbox": [x1, y1, x2 - x1, y2 - y1],
                            "score": float(score),
                        })

        if len(coco_results) == 0:
            print("⚠️ 验证集未产生任何检测结果，返回全零 mAP")
            return 0.0, 0.0

        coco_dt = evaluator.coco_gt.loadRes(coco_results)
        coco_eval = COCOeval(evaluator.coco_gt, coco_dt, "bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        stats = coco_eval.stats
        return float(stats[1]) * 100.0, float(stats[0]) * 100.0

    def _save_checkpoint(self, epoch: int):
        ckpt_dir = os.path.join(self.config.output_dir, f'checkpoint-epoch-{epoch}')
        os.makedirs(ckpt_dir, exist_ok=True)
        self.model.save_pretrained(ckpt_dir)
        self.processor.save_pretrained(ckpt_dir)
        print(f"✅ 检查点已保存: {ckpt_dir}")

    def _save_best(self):
        best_dir = os.path.join(self.config.output_dir, 'best')
        os.makedirs(best_dir, exist_ok=True)
        self.model.save_pretrained(best_dir)
        self.processor.save_pretrained(best_dir)

    def _save_final(self):
        final_dir = os.path.join(self.config.output_dir, 'checkpoint-epoch-final')
        os.makedirs(final_dir, exist_ok=True)
        self.model.save_pretrained(final_dir)
        self.processor.save_pretrained(final_dir)

    def send_progress_update(self, epoch: int, loss: float, accuracy: float):
        try:
            url = f"{self.config.api_base_url}/api/distillation/tasks/{self.config.task_id}/progress"
            data = {
                'currentEpoch': epoch,
                'totalEpochs': self.config.epochs,
                'loss': float(loss),
                'accuracy': float(accuracy),
                'status': 'RUNNING',
            }
            response = requests.post(url, json=data, timeout=5)
            if response.status_code == 200:
                print(f"✅ 进度已更新: Epoch {epoch}, Loss {loss:.4f}, mAP50 {accuracy:.2f}%")
        except Exception as e:
            print(f"⚠️ 发送进度失败: {e}")

    @property
    def best_accuracy(self) -> float:
        return self.best_map


def _build_coco_evaluator_zero_indexed(yolo_dataset: 'YoloDetectionDataset', num_classes: int) -> 'CocoEvaluator':
    """为 0-indexed 类别构建 CocoEvaluator（适用于 HF 模型）"""
    if not PYCOCOTOOLS_AVAILABLE:
        raise ImportError("mAP 评估需要 pycocotools，请运行: pip install pycocotools")

    # 创建一个符合 HF 约定的临时 evaluator，类别 0..num_classes-1
    images_meta = []
    anns = []
    ann_id = 1

    for i in range(len(yolo_dataset)):
        _, target = yolo_dataset[i]
        images_meta.append({"id": i})
        for box, label in zip(target["boxes"], target["labels"]):
            x1, y1, x2, y2 = box.tolist()
            anns.append({
                "id": ann_id,
                "image_id": i,
                "category_id": int(label),
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "area": (x2 - x1) * (y2 - y1),
                "iscrowd": 0,
            })
            ann_id += 1

    cats = [{"id": i} for i in range(num_classes)]
    coco_dict = {
        "images": images_meta,
        "annotations": anns,
        "categories": cats,
    }

    class _EvalHolder:
        pass

    holder = _EvalHolder()
    holder.coco_gt = COCO()
    holder.coco_gt.dataset = coco_dict
    holder.coco_gt.createIndex()
    return holder


# ==================== UNet 检测训练器（语义分割→边界框） ====================

class SegmentationFromBoxesDataset(Dataset):
    """把 YOLO 框标注在线转换成矩形语义分割掩码

    输出:
        image:  (3, H, W) float tensor
        mask:   (H, W)    long tensor，像素值 0..num_classes（0=background）

    多框重叠时后写入者覆盖先写入者（不影响最终 mAP 数量级）。
    """

    def __init__(self, data_root: str, split: str, num_classes: int, image_size: int):
        if not CV2_AVAILABLE:
            raise ImportError("UNet 数据集需要 opencv-python")

        self.data_root = Path(data_root)
        self.img_dir = self.data_root / "images" / split
        self.label_dir = self.data_root / "labels" / split
        self.num_classes = num_classes
        self.image_size = image_size

        self.images = []
        for ext in ('*.jpg', '*.jpeg', '*.png', '*.bmp'):
            self.images.extend(list(self.img_dir.glob(ext)))
        self.images = sorted(self.images)

        print(f"✅ 分割数据集: {self.img_dir} (split={split}, 图像数={len(self.images)})")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label_path = self.label_dir / f"{img_path.stem}.txt"

        img = cv2.imread(str(img_path))
        if img is None:
            raise RuntimeError(f"无法读取图像: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, _ = img.shape

        # 在原图尺寸构建 mask，然后统一 resize
        mask = np.zeros((h, w), dtype=np.int64)

        if label_path.exists():
            with open(label_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    cid, xc, yc, bw, bh = map(float, parts[:5])
                    x1 = int(max(0, (xc - bw / 2) * w))
                    y1 = int(max(0, (yc - bh / 2) * h))
                    x2 = int(min(w, (xc + bw / 2) * w))
                    y2 = int(min(h, (yc + bh / 2) * h))
                    # class_id + 1 是因为 0 留给背景
                    mask[y1:y2, x1:x2] = int(cid) + 1

        # 统一 resize
        img_resized = cv2.resize(img, (self.image_size, self.image_size))
        mask_resized = cv2.resize(mask.astype(np.int32), (self.image_size, self.image_size),
                                  interpolation=cv2.INTER_NEAREST)

        # ImageNet 标准化
        img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_tensor = (img_tensor - mean) / std

        mask_tensor = torch.from_numpy(mask_resized).long()

        return img_tensor, mask_tensor, (h, w), idx


def seg_collate_fn(batch):
    """SegmentationFromBoxesDataset 的 collate 函数

    输出:
        imgs:       (B, 3, H, W) float tensor
        masks:      (B, H, W) long tensor
        orig_sizes: List[Tuple[int, int]] — 每张图原始 (h, w)
        ids:        List[int]
    """
    imgs = torch.stack([b[0] for b in batch], dim=0)
    masks = torch.stack([b[1] for b in batch], dim=0)
    orig_sizes = [b[2] for b in batch]
    ids = [b[3] for b in batch]
    return imgs, masks, orig_sizes, ids


class UNetDetectionTrainer:
    """基于 segmentation-models-pytorch UNet 的目标检测训练器

    训练阶段：把框转成矩形 mask，当作语义分割训练
    推理阶段：预测 mask → 按类别取连通分量 → 每个分量取外接矩形作为 bbox
    评估指标：与 Faster R-CNN / YOLOv8 相同，使用 pycocotools 的 COCO mAP
    """

    # 前端 size → smp encoder 名
    UNET_ENCODERS = {
        'small': 'resnet18',
        'medium': 'resnet34',
        'large': 'resnet50',
        'unet-small': 'resnet18',
        'unet-medium': 'resnet34',
        'unet-large': 'resnet50',
    }

    def __init__(self, config: TrainingConfig):
        self.config = config

        print("\n" + "=" * 60)
        print("初始化 UNet 检测训练器（语义分割→边界框）")
        print("=" * 60)

        try:
            import segmentation_models_pytorch as smp
        except ImportError as e:
            raise ImportError(
                "UNet 检测训练器需要 segmentation-models-pytorch，请运行: "
                "pip install segmentation-models-pytorch"
            ) from e

        encoder_name = self.UNET_ENCODERS.get(config.student_model_size, 'resnet34')
        print(f"UNet encoder: {encoder_name}")

        # +1 是为 background 类留位置
        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights='imagenet',
            in_channels=3,
            classes=config.num_classes + 1,
        ).to(config.device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        self.best_map = 0.0
        self.train_losses = []

    def train(self, train_loader: DataLoader, val_loader: DataLoader, val_yolo_dataset: 'YoloDetectionDataset'):
        """训练入口"""
        print("\n" + "=" * 60)
        print("开始 UNet 检测训练")
        print("=" * 60)

        # 使用 0-indexed evaluator 匹配 mask→box 后的类别
        evaluator = _build_coco_evaluator_zero_indexed(val_yolo_dataset, self.config.num_classes)

        for epoch in range(1, self.config.epochs + 1):
            self.model.train()
            print(f"\n{'=' * 60}")
            print(f"Epoch {epoch}/{self.config.epochs}")
            print(f"{'=' * 60}")

            epoch_loss = 0.0
            n_batches = 0
            pbar = tqdm(train_loader, desc=f"Epoch {epoch}")

            for imgs, masks, _orig_sizes, _ids in pbar:
                imgs = imgs.to(self.config.device)
                masks = masks.to(self.config.device)

                logits = self.model(imgs)  # (B, C+1, H, W)
                loss = self.criterion(logits, masks)

                self.optimizer.zero_grad()
                loss.backward()
                if self.config.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.max_grad_norm
                    )
                self.optimizer.step()

                epoch_loss += float(loss)
                n_batches += 1
                pbar.set_postfix(loss=float(loss))

            avg_loss = epoch_loss / max(n_batches, 1)
            self.train_losses.append(avg_loss)

            # ========== 验证：mask → 连通分量 → bbox → mAP ==========
            map50, map5095 = self._evaluate(val_loader, evaluator)

            print(f"\n📊 Epoch {epoch}: loss={avg_loss:.4f}, mAP50={map50:.2f}%, mAP50-95={map5095:.2f}%")

            if self.config.auto_save_checkpoint and epoch % self.config.checkpoint_interval == 0:
                self._save_checkpoint(epoch)

            if map5095 > self.best_map:
                self.best_map = map5095
                self._save_best()
                print(f"🎉 新的最佳 mAP50-95: {map5095:.2f}%")

            self.send_progress_update(epoch, avg_loss, map50)

        self._save_final()
        print("\n" + "=" * 60)
        print("UNet 检测训练完成!")
        print(f"最佳 mAP50-95: {self.best_map:.2f}%")
        print("=" * 60)

    def _evaluate(self, val_loader: DataLoader, evaluator) -> Tuple[float, float]:
        """预测 mask → 连通分量 → bbox → COCO mAP"""
        self.model.eval()
        coco_results = []

        with torch.no_grad():
            for imgs, masks, orig_sizes, ids in tqdm(val_loader, desc="验证中"):
                imgs = imgs.to(self.config.device)
                logits = self.model(imgs)  # (B, C+1, H, W)
                probs = F.softmax(logits, dim=1)
                # 每个像素取概率最大的类
                pred_classes = probs.argmax(dim=1).cpu().numpy()  # (B, H, W)
                pred_probs = probs.cpu().numpy()  # (B, C+1, H, W)

                # seg_collate_fn 输出:
                #   orig_sizes: List[Tuple[int, int]] = [(h1, w1), (h2, w2), ...]
                #   ids:        List[int]
                for b in range(pred_classes.shape[0]):
                    img_id = int(ids[b])
                    orig_h = int(orig_sizes[b][0])
                    orig_w = int(orig_sizes[b][1])

                    mask_pred = pred_classes[b]  # (Hs, Ws)
                    prob_pred = pred_probs[b]    # (C+1, Hs, Ws)

                    # 对每个前景类提取连通分量
                    detections = _mask_to_detections(
                        mask_pred, prob_pred, self.config.num_classes,
                        target_h=orig_h, target_w=orig_w
                    )

                    for det in detections:
                        coco_results.append({
                            "image_id": img_id,
                            "category_id": det["category_id"],
                            "bbox": det["bbox"],
                            "score": det["score"],
                        })

        if len(coco_results) == 0:
            print("⚠️ 验证集未产生任何检测结果，返回全零 mAP")
            return 0.0, 0.0

        coco_dt = evaluator.coco_gt.loadRes(coco_results)
        coco_eval = COCOeval(evaluator.coco_gt, coco_dt, "bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        stats = coco_eval.stats
        return float(stats[1]) * 100.0, float(stats[0]) * 100.0

    def _save_checkpoint(self, epoch: int):
        ckpt_dir = os.path.join(self.config.output_dir, f'checkpoint-epoch-{epoch}')
        os.makedirs(ckpt_dir, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(ckpt_dir, 'model.pt'))
        info = {
            'epoch': epoch,
            'model_type': 'unet_smp',
            'encoder': self.UNET_ENCODERS.get(self.config.student_model_size, 'resnet34'),
            'num_classes': self.config.num_classes,
            'best_map': self.best_map,
        }
        with open(os.path.join(ckpt_dir, 'training_info.json'), 'w') as f:
            json.dump(info, f, indent=2)

    def _save_best(self):
        best_dir = os.path.join(self.config.output_dir, 'best')
        os.makedirs(best_dir, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(best_dir, 'best.pt'))

    def _save_final(self):
        final_dir = os.path.join(self.config.output_dir, 'checkpoint-epoch-final')
        os.makedirs(final_dir, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(final_dir, 'model.pt'))

    def send_progress_update(self, epoch: int, loss: float, accuracy: float):
        try:
            url = f"{self.config.api_base_url}/api/distillation/tasks/{self.config.task_id}/progress"
            data = {
                'currentEpoch': epoch,
                'totalEpochs': self.config.epochs,
                'loss': float(loss),
                'accuracy': float(accuracy),
                'status': 'RUNNING',
            }
            response = requests.post(url, json=data, timeout=5)
            if response.status_code == 200:
                print(f"✅ 进度已更新: Epoch {epoch}, Loss {loss:.4f}, mAP50 {accuracy:.2f}%")
        except Exception as e:
            print(f"⚠️ 发送进度失败: {e}")

    @property
    def best_accuracy(self) -> float:
        return self.best_map


def _mask_to_detections(mask_pred: np.ndarray, prob_pred: np.ndarray,
                       num_classes: int, target_h: int, target_w: int) -> List[Dict]:
    """把 (H, W) 的类别预测 mask 转换成 detection list

    对每个前景类做连通分量分析，每个分量的外接矩形即为一个 bbox，
    score 取该分量内的平均类别概率。

    返回的 bbox 坐标会 resize 回原图尺寸 (target_h, target_w)。
    """
    detections = []
    pred_h, pred_w = mask_pred.shape

    scale_x = target_w / float(pred_w)
    scale_y = target_h / float(pred_h)

    # category_id 是 0-indexed（和 _build_coco_evaluator_zero_indexed 对齐）
    # mask 中的 0 是 background，1..num_classes 是前景（对应 category 0..num_classes-1）
    for cls_mask_val in range(1, num_classes + 1):
        binary = (mask_pred == cls_mask_val).astype(np.uint8)
        if binary.sum() == 0:
            continue

        # 连通分量
        num_labels, labels_img, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

        # stats: [x, y, w, h, area]
        for comp_id in range(1, num_labels):  # 0 是背景
            x, y, bw, bh, area = stats[comp_id]
            if area < 10:  # 过滤太小的噪点
                continue

            # 计算该分量内的平均类别概率作为 score
            comp_mask = (labels_img == comp_id)
            score = float(prob_pred[cls_mask_val][comp_mask].mean())

            # resize bbox 到原图
            x1 = float(x) * scale_x
            y1 = float(y) * scale_y
            w = float(bw) * scale_x
            h = float(bh) * scale_y

            detections.append({
                "category_id": int(cls_mask_val - 1),  # 转 0-indexed
                "bbox": [x1, y1, w, h],
                "score": score,
            })

    return detections


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


def _report_completion(config: TrainingConfig, accuracy: float, model_path: str):
    """通知后端任务完成"""
    try:
        url = f"{config.api_base_url}/api/distillation/tasks/{config.task_id}/complete"
        data = {
            'status': 'COMPLETED',
            'accuracy': float(accuracy),
            'modelPath': model_path,
        }
        requests.post(url, json=data, timeout=5)
    except Exception as e:
        print(f"⚠️ 更新任务状态失败: {e}")


def _report_failure(config: TrainingConfig, error: Exception):
    """通知后端任务失败"""
    try:
        url = f"{config.api_base_url}/api/distillation/tasks/{config.task_id}/fail"
        data = {'errorMessage': str(error)}
        requests.post(url, json=data, timeout=5)
    except Exception:
        pass


def _run_classification_training(config: TrainingConfig):
    """分类任务训练流程（原有 DirectTrainer 逻辑）"""
    train_dataset_path = os.path.join(config.datasets_root, config.dataset_id, "train")
    val_dataset_path = os.path.join(config.datasets_root, config.val_dataset_id, "val")

    print(f"\n📂 数据集路径:")
    print(f"   - 训练集: {train_dataset_path}")
    print(f"   - 验证集: {val_dataset_path}")

    train_dataset = MultiTaskDataset(
        train_dataset_path,
        config.task_type,
        config.image_size,
        config.num_classes,
        mode='train',
    )

    val_dataset = MultiTaskDataset(
        val_dataset_path,
        config.task_type,
        config.image_size,
        config.num_classes,
        mode='val',
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4,
    )

    trainer = DirectTrainer(config)
    trainer.train(train_loader, val_loader)

    final_dir = os.path.join(config.output_dir, 'checkpoint-epoch-final')
    _report_completion(config, trainer.best_accuracy, final_dir)


def _run_detection_training(config: TrainingConfig):
    """目标检测训练流程"""
    print("\n" + "=" * 60)
    print(f"目标检测训练: {config.student_model_type} - {config.student_model_size}")
    print("=" * 60)

    model_type = (config.student_model_type or '').lower()

    if model_type == 'yolov8':
        # YOLOv8 使用 ultralytics 原生训练（自己管理数据加载器）
        trainer = YoloV8DetectionTrainer(config)
        trainer.train()
        best_path = os.path.join(config.output_dir, 'best', 'best.pt')
        _report_completion(config, trainer.best_accuracy, best_path)

    elif model_type == 'resnet':
        # Faster R-CNN 使用我们自己的 YoloDetectionDataset
        dataset_dir = os.path.join(config.datasets_root, config.dataset_id)
        val_dataset_dir = os.path.join(config.datasets_root, config.val_dataset_id)

        print(f"\n📂 检测数据根目录:")
        print(f"   - 训练集: {dataset_dir}/images/train")
        print(f"   - 验证集: {val_dataset_dir}/images/val")

        train_set = YoloDetectionDataset(dataset_dir, split='train', label_offset=1)
        val_set = YoloDetectionDataset(val_dataset_dir, split='val', label_offset=1)

        train_loader = DataLoader(
            train_set,
            batch_size=config.batch_size,
            shuffle=True,
            collate_fn=detection_collate_fn,
            num_workers=4,
        )
        val_loader = DataLoader(
            val_set,
            batch_size=1,
            shuffle=False,
            collate_fn=detection_collate_fn,
            num_workers=4,
        )

        trainer = FasterRCNNDetectionTrainer(config)
        trainer.train(train_loader, val_loader, val_set)

        best_path = os.path.join(config.output_dir, 'best', 'best.pt')
        _report_completion(config, trainer.best_accuracy, best_path)

    elif model_type in ('vit', 'yolos'):
        # HuggingFace YOLOS (ViT) 目标检测
        dataset_dir = os.path.join(config.datasets_root, config.dataset_id)
        val_dataset_dir = os.path.join(config.datasets_root, config.val_dataset_id)

        print(f"\n📂 检测数据根目录:")
        print(f"   - 训练集: {dataset_dir}/images/train")
        print(f"   - 验证集: {val_dataset_dir}/images/val")

        # HF 模型使用 0-indexed 类别（无 background class）
        train_yolo = YoloDetectionDataset(dataset_dir, split='train', label_offset=0)
        val_yolo = YoloDetectionDataset(val_dataset_dir, split='val', label_offset=0)

        # 先构建 trainer，因为需要拿 processor 来包装 dataset
        trainer = YolosViTDetectionTrainer(config)

        train_wrapped = _YolosDatasetWrapper(train_yolo, trainer.processor)
        val_wrapped = _YolosDatasetWrapper(val_yolo, trainer.processor)

        train_loader = DataLoader(
            train_wrapped,
            batch_size=config.batch_size,
            shuffle=True,
            collate_fn=yolos_collate_fn,
            num_workers=4,
        )
        val_loader = DataLoader(
            val_wrapped,
            batch_size=1,
            shuffle=False,
            collate_fn=yolos_collate_fn,
            num_workers=4,
        )

        trainer.train(train_loader, val_loader, val_yolo)

        # YOLOS 模型用 save_pretrained，best 是一个目录
        best_path = os.path.join(config.output_dir, 'best')
        _report_completion(config, trainer.best_accuracy, best_path)

    elif model_type == 'unet':
        # UNet 语义分割 → 边界框
        dataset_dir = os.path.join(config.datasets_root, config.dataset_id)
        val_dataset_dir = os.path.join(config.datasets_root, config.val_dataset_id)

        print(f"\n📂 检测数据根目录:")
        print(f"   - 训练集: {dataset_dir}/images/train")
        print(f"   - 验证集: {val_dataset_dir}/images/val")

        train_set = SegmentationFromBoxesDataset(
            dataset_dir, 'train', config.num_classes, config.image_size
        )
        val_set = SegmentationFromBoxesDataset(
            val_dataset_dir, 'val', config.num_classes, config.image_size
        )
        # 评估用的 GT 数据集（0-indexed 匹配 _mask_to_detections）
        val_yolo_for_eval = YoloDetectionDataset(
            val_dataset_dir, split='val', label_offset=0
        )

        train_loader = DataLoader(
            train_set,
            batch_size=config.batch_size,
            shuffle=True,
            collate_fn=seg_collate_fn,
            num_workers=4,
        )
        val_loader = DataLoader(
            val_set,
            batch_size=config.batch_size,
            shuffle=False,
            collate_fn=seg_collate_fn,
            num_workers=4,
        )

        trainer = UNetDetectionTrainer(config)
        trainer.train(train_loader, val_loader, val_yolo_for_eval)

        best_path = os.path.join(config.output_dir, 'best', 'best.pt')
        _report_completion(config, trainer.best_accuracy, best_path)

    else:
        raise ValueError(
            f"目标检测任务暂不支持的模型类型: {config.student_model_type}. "
            f"当前支持: yolov8, resnet, vit, unet"
        )


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
    print(f"任务类型: {config.task_type}")
    print(f"数据集: {config.dataset_id}")
    print(f"训练轮数: {config.epochs}")
    print(f"批次大小: {config.batch_size}")
    print(f"学习率: {config.learning_rate}")
    print(f"设备: {config.device}")
    print("=" * 60)

    # 创建输出目录
    os.makedirs(config.output_dir, exist_ok=True)

    # 根据任务类型分发到对应的训练流程
    task_type = (config.task_type or 'classification').lower()

    try:
        if task_type == 'detection':
            _run_detection_training(config)
        elif task_type in ('classification', 'segmentation'):
            # 分割暂时复用分类流程（UNet classifier head）
            _run_classification_training(config)
        else:
            raise ValueError(f"不支持的任务类型: {config.task_type}")

    except Exception as e:
        print(f"\n❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()
        _report_failure(config, e)
        sys.exit(1)


if __name__ == '__main__':
    main()
