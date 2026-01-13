#!/usr/bin/env python3
"""
Qwen2.5-VL蒸馏模型推理脚本 - 图像自动标注

支持模型类型：
- ResNet (图像分类)
- Vision Transformer (图像分类)
- YOLOv8 (目标检测)
- UNet (图像分割)
- LSTM (序列特征提取+分类)

输出格式：Labelme JSON格式，可直接用于数据标注系统
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import base64
from io import BytesIO

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from torchvision import transforms
from tqdm import tqdm

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ========== 模型定义 ==========

class ResNetStudent(nn.Module):
    """ResNet学生模型（分类）"""
    def __init__(self, num_classes: int = 1000, model_size: str = "resnet50"):
        super().__init__()
        if model_size == "resnet18":
            from torchvision.models import resnet18
            self.backbone = resnet18(pretrained=False)
        elif model_size == "resnet34":
            from torchvision.models import resnet34
            self.backbone = resnet34(pretrained=False)
        elif model_size == "resnet50":
            from torchvision.models import resnet50
            self.backbone = resnet50(pretrained=False)
        else:
            from torchvision.models import resnet101
            self.backbone = resnet101(pretrained=False)

        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)


class ViTStudent(nn.Module):
    """Vision Transformer学生模型（分类）"""
    def __init__(self, num_classes: int = 1000, model_size: str = "vit-base",
                 image_size: int = 224, patch_size: int = 16):
        super().__init__()
        if model_size == "vit-tiny":
            embed_dim, num_heads, depth = 192, 3, 12
        elif model_size == "vit-small":
            embed_dim, num_heads, depth = 384, 6, 12
        elif model_size == "vit-base":
            embed_dim, num_heads, depth = 768, 12, 12
        else:  # vit-large
            embed_dim, num_heads, depth = 1024, 16, 24

        from torchvision.models import VisionTransformer
        self.vit = VisionTransformer(
            image_size=image_size,
            patch_size=patch_size,
            num_layers=depth,
            num_heads=num_heads,
            hidden_dim=embed_dim,
            mlp_dim=embed_dim * 4,
            num_classes=num_classes
        )

    def forward(self, x):
        return self.vit(x)


class YOLOv8Student(nn.Module):
    """YOLOv8学生模型（目标检测）"""
    def __init__(self, num_classes: int = 80, model_size: str = "n"):
        super().__init__()
        try:
            from ultralytics import YOLO
            self.model = YOLO(f'yolov8{model_size}.pt')
            # 修改类别数
            self.model.model.model[-1].nc = num_classes
        except ImportError:
            logger.warning("ultralytics未安装，使用简化版本")
            self.model = None

    def forward(self, x):
        if self.model is None:
            return torch.zeros(x.size(0), 100, 6)  # [batch, max_det, 6(x1,y1,x2,y2,conf,cls)]
        return self.model(x)


class UNetStudent(nn.Module):
    """UNet学生模型（图像分割）"""
    def __init__(self, num_classes: int = 21, in_channels: int = 3):
        super().__init__()

        def double_conv(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, 3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True)
            )

        # Encoder
        self.enc1 = double_conv(in_channels, 64)
        self.enc2 = double_conv(64, 128)
        self.enc3 = double_conv(128, 256)
        self.enc4 = double_conv(256, 512)

        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = double_conv(512, 1024)

        # Decoder
        self.upconv4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = double_conv(1024, 512)

        self.upconv3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = double_conv(512, 256)

        self.upconv2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = double_conv(256, 128)

        self.upconv1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = double_conv(128, 64)

        self.out = nn.Conv2d(64, num_classes, 1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        # Bottleneck
        b = self.bottleneck(self.pool(e4))

        # Decoder
        d4 = self.upconv4(b)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)

        d3 = self.upconv3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.upconv2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.upconv1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        return self.out(d1)


class LSTMStudent(nn.Module):
    """LSTM学生模型（序列特征提取+分类）"""
    def __init__(self, num_classes: int = 1000, input_size: int = 2048,
                 hidden_size: int = 512, num_layers: int = 2):
        super().__init__()
        # CNN特征提取
        from torchvision.models import resnet18
        backbone = resnet18(pretrained=False)
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-2])

        # LSTM
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )

        # 分类头
        self.classifier = nn.Linear(hidden_size * 2, num_classes)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        batch_size = x.size(0)

        # 提取CNN特征
        features = self.feature_extractor(x)  # [B, C, H, W]
        features = self.adaptive_pool(features)  # [B, C, 1, 1]
        features = features.view(batch_size, -1, 1)  # [B, C, 1]
        features = features.transpose(1, 2)  # [B, 1, C]

        # LSTM处理
        lstm_out, _ = self.lstm(features)  # [B, 1, hidden*2]

        # 分类
        output = self.classifier(lstm_out[:, -1, :])  # [B, num_classes]
        return output


# ========== 数据集 ==========

class InferenceDataset(Dataset):
    """推理数据集"""
    def __init__(self, image_paths: List[str], transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, img_path


# ========== 推理引擎 ==========

class ModelInference:
    """模型推理引擎"""

    def __init__(self, model_path: str, model_type: str, config: Dict):
        self.model_path = model_path
        self.model_type = model_type
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        logger.info(f"使用设备: {self.device}")
        logger.info(f"加载模型类型: {model_type}")

        # 加载模型
        self.model = self._load_model()
        self.model.eval()

        # 数据预处理
        self.transform = self._get_transform()

        # 加载类别名称
        self.class_names = config.get('class_names', [f"class_{i}" for i in range(config['num_classes'])])

    def _load_model(self) -> nn.Module:
        """加载模型"""
        num_classes = self.config['num_classes']

        if self.model_type == 'resnet':
            model = ResNetStudent(
                num_classes=num_classes,
                model_size=self.config.get('model_size', 'resnet50')
            )
        elif self.model_type == 'vit':
            model = ViTStudent(
                num_classes=num_classes,
                model_size=self.config.get('model_size', 'vit-base'),
                image_size=self.config.get('image_size', 224)
            )
        elif self.model_type == 'yolov8':
            model = YOLOv8Student(
                num_classes=num_classes,
                model_size=self.config.get('model_size', 'n')
            )
        elif self.model_type == 'unet':
            model = UNetStudent(num_classes=num_classes)
        elif self.model_type == 'lstm':
            model = LSTMStudent(num_classes=num_classes)
        else:
            raise ValueError(f"不支持的模型类型: {self.model_type}")

        # 加载检查点
        checkpoint = torch.load(self.model_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)

        model.to(self.device)
        logger.info(f"模型加载成功: {self.model_path}")

        return model

    def _get_transform(self):
        """获取数据转换"""
        image_size = self.config.get('image_size', 224)

        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def predict_classification(self, images: torch.Tensor, image_paths: List[str]) -> List[Dict]:
        """分类任务推理"""
        results = []

        with torch.no_grad():
            outputs = self.model(images)
            probs = F.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

        for i, img_path in enumerate(image_paths):
            class_id = preds[i].item()
            confidence = probs[i, class_id].item()

            # 转换为Labelme格式
            result = self._create_labelme_classification(
                img_path,
                self.class_names[class_id],
                confidence
            )
            results.append(result)

        return results

    def predict_detection(self, images: torch.Tensor, image_paths: List[str]) -> List[Dict]:
        """检测任务推理"""
        results = []

        with torch.no_grad():
            predictions = self.model(images)

        for i, img_path in enumerate(image_paths):
            # YOLOv8返回格式: [x1, y1, x2, y2, conf, cls]
            if isinstance(predictions, list):
                pred = predictions[i]
            else:
                pred = predictions[i]

            # 转换为Labelme格式
            result = self._create_labelme_detection(img_path, pred)
            results.append(result)

        return results

    def predict_segmentation(self, images: torch.Tensor, image_paths: List[str]) -> List[Dict]:
        """分割任务推理"""
        results = []

        with torch.no_grad():
            outputs = self.model(images)
            masks = torch.argmax(outputs, dim=1)

        for i, img_path in enumerate(image_paths):
            mask = masks[i].cpu().numpy()

            # 转换为Labelme格式
            result = self._create_labelme_segmentation(img_path, mask)
            results.append(result)

        return results

    def predict(self, image_paths: List[str], batch_size: int = 8) -> List[Dict]:
        """批量推理"""
        dataset = InferenceDataset(image_paths, transform=self.transform)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        all_results = []

        for images, paths in tqdm(dataloader, desc="推理中"):
            images = images.to(self.device)

            if self.model_type in ['resnet', 'vit', 'lstm']:
                results = self.predict_classification(images, paths)
            elif self.model_type == 'yolov8':
                results = self.predict_detection(images, paths)
            elif self.model_type == 'unet':
                results = self.predict_segmentation(images, paths)
            else:
                raise ValueError(f"不支持的任务类型")

            all_results.extend(results)

        return all_results

    def _create_labelme_classification(self, image_path: str, label: str, confidence: float) -> Dict:
        """创建分类标注的Labelme JSON"""
        image = Image.open(image_path)
        width, height = image.size

        return {
            "version": "5.0.1",
            "flags": {},
            "shapes": [
                {
                    "label": f"{label} ({confidence:.2%})",
                    "points": [[0, 0], [width, height]],
                    "group_id": None,
                    "shape_type": "rectangle",
                    "flags": {}
                }
            ],
            "imagePath": os.path.basename(image_path),
            "imageData": None,
            "imageHeight": height,
            "imageWidth": width
        }

    def _create_labelme_detection(self, image_path: str, detections: torch.Tensor) -> Dict:
        """创建检测标注的Labelme JSON"""
        image = Image.open(image_path)
        width, height = image.size

        shapes = []
        for det in detections:
            if len(det) >= 6:
                x1, y1, x2, y2, conf, cls = det[:6]
                if conf > 0.25:  # 置信度阈值
                    shapes.append({
                        "label": f"{self.class_names[int(cls)]} ({conf:.2%})",
                        "points": [[float(x1), float(y1)], [float(x2), float(y2)]],
                        "group_id": None,
                        "shape_type": "rectangle",
                        "flags": {}
                    })

        return {
            "version": "5.0.1",
            "flags": {},
            "shapes": shapes,
            "imagePath": os.path.basename(image_path),
            "imageData": None,
            "imageHeight": height,
            "imageWidth": width
        }

    def _create_labelme_segmentation(self, image_path: str, mask: np.ndarray) -> Dict:
        """创建分割标注的Labelme JSON"""
        image = Image.open(image_path)
        width, height = image.size

        shapes = []
        unique_labels = np.unique(mask)

        for label_id in unique_labels:
            if label_id == 0:  # 跳过背景
                continue

            # 提取当前类别的mask
            binary_mask = (mask == label_id).astype(np.uint8)

            # 查找轮廓
            import cv2
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                if len(contour) > 2:
                    points = contour.reshape(-1, 2).tolist()
                    shapes.append({
                        "label": self.class_names[label_id] if label_id < len(self.class_names) else f"class_{label_id}",
                        "points": points,
                        "group_id": None,
                        "shape_type": "polygon",
                        "flags": {}
                    })

        return {
            "version": "5.0.1",
            "flags": {},
            "shapes": shapes,
            "imagePath": os.path.basename(image_path),
            "imageData": None,
            "imageHeight": height,
            "imageWidth": width
        }


# ========== 主函数 ==========

def main():
    parser = argparse.ArgumentParser(description='Qwen2.5-VL蒸馏模型推理')

    # 必需参数
    parser.add_argument('--model_path', type=str, required=True, help='模型检查点路径')
    parser.add_argument('--model_type', type=str, required=True,
                       choices=['resnet', 'vit', 'yolov8', 'unet', 'lstm'],
                       help='模型类型')
    parser.add_argument('--input_dir', type=str, required=True, help='输入图像目录')
    parser.add_argument('--output_dir', type=str, required=True, help='输出JSON目录')

    # 配置参数
    parser.add_argument('--num_classes', type=int, required=True, help='类别数')
    parser.add_argument('--class_names_file', type=str, help='类别名称文件（每行一个类别）')
    parser.add_argument('--model_size', type=str, help='模型大小（如resnet50, vit-base）')
    parser.add_argument('--image_size', type=int, default=224, help='输入图像尺寸')
    parser.add_argument('--batch_size', type=int, default=8, help='批次大小')

    args = parser.parse_args()

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 加载类别名称
    class_names = None
    if args.class_names_file and os.path.exists(args.class_names_file):
        with open(args.class_names_file, 'r', encoding='utf-8') as f:
            class_names = [line.strip() for line in f]

    # 配置
    config = {
        'num_classes': args.num_classes,
        'image_size': args.image_size,
        'model_size': args.model_size,
        'class_names': class_names
    }

    # 初始化推理引擎
    inference = ModelInference(args.model_path, args.model_type, config)

    # 获取所有图像
    image_exts = ['.jpg', '.jpeg', '.png', '.bmp']
    image_paths = []
    for ext in image_exts:
        image_paths.extend(Path(args.input_dir).glob(f'**/*{ext}'))
        image_paths.extend(Path(args.input_dir).glob(f'**/*{ext.upper()}'))

    image_paths = [str(p) for p in image_paths]
    logger.info(f"找到 {len(image_paths)} 张图像")

    if len(image_paths) == 0:
        logger.warning("未找到图像文件")
        return

    # 执行推理
    results = inference.predict(image_paths, batch_size=args.batch_size)

    # 保存结果
    for result in results:
        output_path = os.path.join(
            args.output_dir,
            os.path.splitext(result['imagePath'])[0] + '.json'
        )
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

    logger.info(f"推理完成！结果已保存到: {args.output_dir}")

    # 输出统计信息
    print("\n" + "="*50)
    print("推理统计:")
    print(f"  总图像数: {len(results)}")
    print(f"  模型类型: {args.model_type}")
    print(f"  输出目录: {args.output_dir}")
    print("="*50)


if __name__ == '__main__':
    main()
