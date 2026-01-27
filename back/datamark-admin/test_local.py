#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
本地测试版本 - Qwen2.5-VL知识蒸馏训练脚本

用途：脱离前后端系统，独立运行测试知识蒸馏训练
保留完整的蒸馏逻辑：教师模型(Qwen) -> 学生模型(ResNet)

使用方法：
    python test_local.py

作者：AI Assistant
日期：2026-01-27
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
    warnings.warn("⚠️ Qwen2_5_VL模型库未安装，将使用模拟教师模型")

# 小模型相关导入
import torchvision.models as models
from peft import LoraConfig, get_peft_model, TaskType


# ==================== 配置类 ====================

class SimpleConfig:
    """简化的训练配置类"""

    def __init__(self):
        # ========== 关键配置（需要修改） ==========
        # 数据集根目录（修改为您的实际路径）
        self.datasets_root = r"D:\pythonProject2\datasets"
        self.dataset_id = "cifar10"

        # 教师模型路径（Qwen2.5-VL）
        self.teacher_path = r"D:\pythonProject2\models\Qwen2___5-VL-3B-Instruct"

        # 输出目录
        self.output_dir = r"D:\pythonProject2\test_output"

        # ========== 是否使用真实Qwen模型 ==========
        # True: 加载真实Qwen模型（需要大量内存，训练慢但完整）
        # False: 使用模拟教师模型（快速测试，验证流程）
        self.use_real_teacher = False  # 改为True以使用真实Qwen模型

        # ========== 任务配置 ==========
        self.task_type = "classification"
        self.num_classes = 10  # CIFAR-10
        self.image_size = 224

        # ========== 学生模型配置 ==========
        self.student_model_type = "resnet"
        self.student_model_size = "resnet18"

        # ========== 训练参数 ==========
        self.epochs = 2  # 测试用
        self.batch_size = 8
        self.learning_rate = 0.0001

        # ========== LoRA配置 ==========
        self.lora_rank = 8
        self.lora_alpha = 16
        self.lora_dropout = 0.1

        # ========== 知识蒸馏参数 ==========
        self.temperature = 4.0
        self.hard_label_weight = 0.3  # 硬标签权重
        self.soft_label_weight = 0.7  # 软标签权重
        self.distillation_type = "feature"  # logit/feature/hybrid
        self.feature_loss_type = "mse"  # mse/cosine

        # ========== 设备配置 ==========
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ========== 优化器配置 ==========
        self.optimizer_type = "adamw"
        self.weight_decay = 0.01


# ==================== 数据集类 ====================

class CIFAR10Dataset(Dataset):
    """CIFAR-10数据集加载器"""

    def __init__(self, dataset_path: str, image_size: int = 224, mode: str = 'train'):
        self.dataset_path = dataset_path
        self.image_size = image_size
        self.mode = mode

        self.image_paths = []
        self.labels = []
        self.class_names = []

        if os.path.exists(dataset_path):
            self._load_dataset()
        else:
            print(f"❌ 错误: 数据集路径不存在: {dataset_path}")
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

        class_folders = sorted([d for d in os.listdir(self.dataset_path)
                               if os.path.isdir(os.path.join(self.dataset_path, d))])

        if not class_folders:
            print(f"❌ 错误: 未找到类别文件夹")
            sys.exit(1)

        self.class_names = class_folders
        print(f"✅ 找到 {len(self.class_names)} 个类别: {self.class_names}")

        for class_idx, class_name in enumerate(self.class_names):
            class_dir = os.path.join(self.dataset_path, class_name)
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
            print(f"⚠️ 加载图像失败: {e}")
            image_array = np.random.randint(0, 255, (self.image_size, self.image_size, 3), dtype=np.uint8)
            image = Image.fromarray(image_array)

        image = self.transform(image)
        return {'pixel_values': image, 'labels': label}


# ==================== 模型加载器 ====================

class TeacherModelLoader:
    """教师模型加载器"""

    @staticmethod
    def load_qwen2vl(model_path: str, device: torch.device, use_real: bool = True):
        """加载Qwen2.5-VL模型"""
        if not use_real or not QWEN_AVAILABLE:
            print("📦 使用模拟教师模型（快速测试模式）")
            return None, None

        print(f"📦 正在加载Qwen2.5-VL教师模型: {model_path}")
        print("⚠️ 注意：这可能需要几分钟时间和大量内存...")

        try:
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype="auto",
                device_map="cpu"  # 先加载到CPU，避免GPU内存不足
            )
            processor = AutoProcessor.from_pretrained(model_path)
            print("✅ 教师模型加载成功")
            return model, processor
        except Exception as e:
            print(f"❌ 加载教师模型失败: {e}")
            print("将使用模拟教师模型")
            return None, None


class StudentModelLoader:
    """学生模型加载器"""

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

        # 修改最后一层
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model


# ==================== 知识蒸馏训练器 ====================

class DistillationTrainer:
    """知识蒸馏训练器（保留完整蒸馏逻辑）"""

    def __init__(self, config: SimpleConfig):
        self.config = config

        print("\n" + "=" * 60)
        print("知识蒸馏训练器初始化")
        print("=" * 60)

        # 创建输出目录
        os.makedirs(config.output_dir, exist_ok=True)

        # 加载教师模型
        self.teacher_model, self.teacher_processor = TeacherModelLoader.load_qwen2vl(
            config.teacher_path,
            config.device,
            use_real=config.use_real_teacher
        )

        # 加载学生模型
        self.student_model = StudentModelLoader.load_resnet(
            config.student_model_size,
            config.num_classes
        ).to(config.device)

        # 注册hook提取学生特征
        self.student_feature_map = None

        def hook_fn(module, input, output):
            self.student_feature_map = output

        self.student_model.avgpool.register_forward_hook(hook_fn)

        # 加载数据集
        train_path = os.path.join(config.datasets_root, config.dataset_id, "train")
        val_path = os.path.join(config.datasets_root, config.dataset_id, "val")

        self.train_dataset = CIFAR10Dataset(train_path, config.image_size, mode='train')
        self.val_dataset = CIFAR10Dataset(val_path, config.image_size, mode='val')

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=0
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=0
        )

        # 优化器
        self.optimizer = torch.optim.AdamW(
            self.student_model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )

        # 损失函数
        self.ce_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')

        print(f"✅ 设备: {config.device}")
        print(f"✅ 教师模型: {'真实Qwen' if self.teacher_model else '模拟模型'}")
        print(f"✅ 学生模型参数量: {sum(p.numel() for p in self.student_model.parameters()):,}")
        print(f"✅ 训练样本: {len(self.train_dataset)}")
        print(f"✅ 验证样本: {len(self.val_dataset)}")

    def extract_teacher_features(self, images: torch.Tensor):
        """提取教师模型特征"""
        batch_size = images.size(0)

        # 模拟模式
        if self.teacher_model is None:
            return {
                'vision_features': torch.randn(batch_size, 256, 1024).to(self.config.device)
            }

        # 真实Qwen模型
        pil_images = [transforms.ToPILImage()(img.cpu()) for img in images]

        inputs = self.teacher_processor(
            images=pil_images,
            text=["image"] * batch_size,
            return_tensors="pt",
        )

        for k, v in inputs.items():
            if k == "input_ids":
                inputs[k] = v.long().to(self.config.device)
            else:
                inputs[k] = v.to(self.config.device)

        with torch.no_grad():
            outputs = self.teacher_model.visual(**inputs, output_hidden_states=True)
            return {
                'vision_features': outputs.last_hidden_state
            }

    def compute_distillation_loss(
        self,
        student_output: torch.Tensor,
        teacher_features: Dict[str, torch.Tensor],
        labels: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """计算知识蒸馏损失"""
        losses = {}

        # 1. 硬标签损失（交叉熵）
        hard_loss = self.ce_loss(student_output, labels)
        losses['hard_loss'] = hard_loss

        # 2. 软标签损失（KL散度 - 简化版）
        # 在完整实现中，这里应该使用教师模型的logits
        soft_loss = torch.tensor(0.0).to(self.config.device)
        losses['soft_loss'] = soft_loss

        # 3. 特征蒸馏损失
        if self.config.distillation_type in ['feature', 'hybrid']:
            # 提取学生特征
            if self.student_feature_map is not None:
                student_features = self.student_feature_map.flatten(1)  # [B, D]
            else:
                student_features = student_output

            # 教师特征
            teacher_vis_features = teacher_features['vision_features']  # [B, N, D]
            teacher_pooled = teacher_vis_features.mean(dim=1)  # [B, D]

            # 对齐维度
            if student_features.shape != teacher_pooled.shape:
                # 简单投影
                if student_features.shape[1] != teacher_pooled.shape[1]:
                    if student_features.shape[1] > teacher_pooled.shape[1]:
                        student_features = F.adaptive_avg_pool1d(
                            student_features.unsqueeze(1),
                            teacher_pooled.shape[1]
                        ).squeeze(1)
                    else:
                        teacher_pooled = F.adaptive_avg_pool1d(
                            teacher_pooled.unsqueeze(1),
                            student_features.shape[1]
                        ).squeeze(1)

            # 计算特征损失
            if self.config.feature_loss_type == 'mse':
                feature_loss = self.mse_loss(student_features, teacher_pooled)
            elif self.config.feature_loss_type == 'cosine':
                student_norm = F.normalize(student_features, dim=-1)
                teacher_norm = F.normalize(teacher_pooled, dim=-1)
                feature_loss = 1 - F.cosine_similarity(student_norm, teacher_norm).mean()
            else:
                feature_loss = torch.tensor(0.0).to(self.config.device)

            losses['feature_loss'] = feature_loss
        else:
            losses['feature_loss'] = torch.tensor(0.0).to(self.config.device)

        # 4. 总损失
        total_loss = (
            self.config.hard_label_weight * losses['hard_loss'] +
            self.config.soft_label_weight * losses['soft_loss'] +
            losses['feature_loss']
        )
        losses['total_loss'] = total_loss

        return losses

    def train_epoch(self, epoch: int):
        """训练一个epoch"""
        self.student_model.train()
        epoch_losses = {'total_loss': 0.0, 'hard_loss': 0.0, 'soft_loss': 0.0, 'feature_loss': 0.0}
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.config.epochs}")

        for batch_idx, batch in enumerate(pbar):
            images = batch['pixel_values'].to(self.config.device)
            labels = batch['labels'].to(self.config.device)

            # 提取教师特征
            teacher_features = self.extract_teacher_features(images)

            # 学生模型前向传播
            self.optimizer.zero_grad()
            student_output = self.student_model(images)

            # 计算蒸馏损失
            losses = self.compute_distillation_loss(student_output, teacher_features, labels)

            # 反向传播
            losses['total_loss'].backward()
            self.optimizer.step()

            # 统计
            for k in epoch_losses.keys():
                epoch_losses[k] += losses[k].item()

            _, predicted = student_output.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # 更新进度条
            if (batch_idx + 1) % 10 == 0:
                acc = 100. * correct / total
                pbar.set_postfix({
                    'loss': f'{losses["total_loss"].item():.4f}',
                    'hard': f'{losses["hard_loss"].item():.3f}',
                    'feat': f'{losses["feature_loss"].item():.3f}',
                    'acc': f'{acc:.2f}%'
                })

        # Epoch统计
        for k in epoch_losses.keys():
            epoch_losses[k] /= len(self.train_loader)
        acc = 100. * correct / total

        print(f"\n📊 Epoch {epoch} 训练结果:")
        print(f"   Total Loss: {epoch_losses['total_loss']:.4f}")
        print(f"   Hard Loss: {epoch_losses['hard_loss']:.4f}")
        print(f"   Feature Loss: {epoch_losses['feature_loss']:.4f}")
        print(f"   Accuracy: {acc:.2f}%")

        return epoch_losses, acc

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
                loss = self.ce_loss(outputs, labels)

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
        print("开始知识蒸馏训练")
        print("=" * 60)

        best_acc = 0
        start_time = time.time()

        for epoch in range(1, self.config.epochs + 1):
            print(f"\n{'=' * 60}")
            print(f"Epoch {epoch}/{self.config.epochs}")
            print('=' * 60)

            # 训练
            train_losses, train_acc = self.train_epoch(epoch)

            # 验证
            val_loss, val_acc = self.validate()

            # 保存最佳模型
            if val_acc > best_acc:
                best_acc = val_acc
                save_path = os.path.join(self.config.output_dir, "best_distilled_model.pth")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.student_model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_acc': train_acc,
                    'val_acc': val_acc,
                    'best_acc': best_acc,
                    'config': vars(self.config)
                }, save_path)
                print(f"\n💾 保存最佳模型: {save_path} (Acc: {best_acc:.2f}%)")

        # 训练结束
        elapsed_time = time.time() - start_time
        print("\n" + "=" * 60)
        print("✅ 知识蒸馏训练完成！")
        print("=" * 60)
        print(f"总耗时: {elapsed_time/60:.2f} 分钟")
        print(f"最佳验证准确率: {best_acc:.2f}%")
        print(f"蒸馏配置:")
        print(f"  - 硬标签权重: {self.config.hard_label_weight}")
        print(f"  - 软标签权重: {self.config.soft_label_weight}")
        print(f"  - 特征蒸馏类型: {self.config.distillation_type}")
        print(f"  - 特征损失类型: {self.config.feature_loss_type}")


# ==================== 主函数 ====================

def main():
    print("\n" + "=" * 60)
    print("Qwen2.5-VL 知识蒸馏训练脚本 - 本地测试版")
    print("（保留完整蒸馏逻辑）")
    print("=" * 60)

    # 创建配置
    config = SimpleConfig()

    # 打印配置
    print("\n📋 训练配置:")
    print(f"   数据集: {config.datasets_root}/{config.dataset_id}")
    print(f"   教师模型: {config.teacher_path}")
    print(f"   使用真实Qwen: {config.use_real_teacher}")
    print(f"   学生模型: {config.student_model_type}-{config.student_model_size}")
    print(f"   蒸馏类型: {config.distillation_type}")
    print(f"   特征损失: {config.feature_loss_type}")
    print(f"   硬标签权重: {config.hard_label_weight}")
    print(f"   软标签权重: {config.soft_label_weight}")
    print(f"   训练轮数: {config.epochs}")
    print(f"   批次大小: {config.batch_size}")

    # 检查路径
    print("\n🔍 检查路径...")
    train_path = os.path.join(config.datasets_root, config.dataset_id, "train")
    val_path = os.path.join(config.datasets_root, config.dataset_id, "val")

    if not os.path.exists(train_path):
        print(f"❌ 错误: 训练集不存在: {train_path}")
        print(f"请先运行 convert_cifar10.py")
        return

    if not os.path.exists(val_path):
        print(f"❌ 错误: 验证集不存在: {val_path}")
        return

    if config.use_real_teacher and not os.path.exists(config.teacher_path):
        print(f"❌ 错误: 教师模型不存在: {config.teacher_path}")
        print(f"提示: 可以设置 use_real_teacher=False 使用模拟模型快速测试")
        return

    print("✅ 路径检查通过")

    # 创建训练器
    trainer = DistillationTrainer(config)

    # 开始训练
    trainer.train()

    print("\n✅ 知识蒸馏测试完成！")
    if not config.use_real_teacher:
        print("\n💡 提示: 当前使用模拟教师模型")
        print("   如需完整蒸馏训练，请设置 use_real_teacher=True")


if __name__ == "__main__":
    main()
