#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Python环境测试脚本
用于验证大小模型协同训练系统的Python环境是否正确配置

测试项目：
1. Python版本
2. PyTorch和CUDA
3. 必需依赖包
4. 可选依赖包
5. 模型加载能力
6. 数据集访问

作者：Claude Assistant
日期：2026-01-14
"""

import sys
import os
from pathlib import Path

print("=" * 70)
print("🔍 大小模型协同训练系统 - Python环境测试")
print("=" * 70)
print()

# ==================== 测试1: Python版本 ====================
print("📌 测试1: Python版本")
print("-" * 70)
print(f"Python版本: {sys.version}")
print(f"Python路径: {sys.executable}")

py_version = sys.version_info
if py_version >= (3, 8):
    print("✅ Python版本符合要求 (>= 3.8)")
else:
    print(f"❌ Python版本过低，需要 >= 3.8，当前: {py_version.major}.{py_version.minor}")
    sys.exit(1)
print()

# ==================== 测试2: PyTorch ====================
print("📌 测试2: PyTorch")
print("-" * 70)
try:
    import torch
    print(f"✅ PyTorch版本: {torch.__version__}")
    print(f"   安装路径: {torch.__file__}")

    # 测试CUDA
    if torch.cuda.is_available():
        print(f"✅ CUDA可用")
        print(f"   CUDA版本: {torch.version.cuda}")
        print(f"   GPU数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
            # 显存信息
            total_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"        显存: {total_memory:.2f} GB")
    else:
        print("⚠️  CUDA不可用，将使用CPU训练（速度会很慢）")
        print("   如果有NVIDIA GPU，请安装CUDA版本的PyTorch")
        print("   参考: https://pytorch.org/get-started/locally/")
except ImportError as e:
    print(f"❌ PyTorch未安装: {e}")
    print("   安装命令: pip install torch torchvision")
    sys.exit(1)
print()

# ==================== 测试3: TorchVision ====================
print("📌 测试3: TorchVision")
print("-" * 70)
try:
    import torchvision
    print(f"✅ TorchVision版本: {torchvision.__version__}")
except ImportError:
    print("❌ TorchVision未安装")
    print("   安装命令: pip install torchvision")
print()

# ==================== 测试4: Transformers ====================
print("📌 测试4: Transformers (Hugging Face)")
print("-" * 70)
try:
    import transformers
    print(f"✅ Transformers版本: {transformers.__version__}")

    # 测试AutoProcessor和AutoModel
    try:
        from transformers import AutoProcessor, AutoModelForImageClassification
        print("✅ AutoProcessor和AutoModel可用")
    except ImportError as e:
        print(f"⚠️  部分功能不可用: {e}")
except ImportError:
    print("❌ Transformers未安装")
    print("   安装命令: pip install transformers")
    print("   用途: 用于ViT等Transformer模型")
print()

# ==================== 测试5: PEFT (LoRA) ====================
print("📌 测试5: PEFT (LoRA支持)")
print("-" * 70)
try:
    import peft
    print(f"✅ PEFT版本: {peft.__version__}")
    from peft import LoraConfig, get_peft_model
    print("✅ LoRA配置可用")
except ImportError:
    print("❌ PEFT未安装")
    print("   安装命令: pip install peft")
    print("   用途: 用于LoRA高效微调")
print()

# ==================== 测试6: YOLOv8 ====================
print("📌 测试6: YOLOv8 (Ultralytics)")
print("-" * 70)
try:
    from ultralytics import YOLO
    print(f"✅ Ultralytics可用")
    # 尝试列出YOLO版本
    import ultralytics
    print(f"   版本: {ultralytics.__version__}")
except ImportError:
    print("⚠️  YOLOv8未安装（如果需要目标检测任务才需要）")
    print("   安装命令: pip install ultralytics")
print()

# ==================== 测试7: 其他依赖包 ====================
print("📌 测试7: 其他必需依赖")
print("-" * 70)

required_packages = {
    'numpy': '数值计算',
    'pandas': '数据处理',
    'Pillow': '图像处理',
    'tqdm': '进度条',
    'requests': 'HTTP请求',
    'scikit-learn': '机器学习工具'
}

for package, description in required_packages.items():
    try:
        module = __import__(package.replace('-', '_').lower())
        version = getattr(module, '__version__', 'unknown')
        print(f"✅ {package:20s} {version:15s} - {description}")
    except ImportError:
        print(f"⚠️  {package:20s} {'未安装':15s} - {description}")
print()

# ==================== 测试8: 可选依赖包 ====================
print("📌 测试8: 可选依赖包")
print("-" * 70)

optional_packages = {
    'matplotlib': '数据可视化',
    'seaborn': '高级可视化',
    'timm': '更多预训练模型',
    'segmentation_models_pytorch': 'UNet等分割模型',
    'accelerate': '模型加速',
}

for package, description in optional_packages.items():
    try:
        module = __import__(package.replace('-', '_'))
        version = getattr(module, '__version__', 'unknown')
        print(f"✅ {package:30s} {version:15s} - {description}")
    except ImportError:
        print(f"⚠️  {package:30s} {'未安装':15s} - {description}")
print()

# ==================== 测试9: 简单的张量操作 ====================
print("📌 测试9: PyTorch张量操作")
print("-" * 70)
try:
    # CPU测试
    x = torch.randn(3, 3)
    y = torch.randn(3, 3)
    z = torch.matmul(x, y)
    print(f"✅ CPU张量运算正常")

    # GPU测试
    if torch.cuda.is_available():
        x_gpu = x.cuda()
        y_gpu = y.cuda()
        z_gpu = torch.matmul(x_gpu, y_gpu)
        print(f"✅ GPU张量运算正常")
        print(f"   结果形状: {z_gpu.shape}")
except Exception as e:
    print(f"❌ 张量运算失败: {e}")
print()

# ==================== 测试10: 预训练模型加载测试 ====================
print("📌 测试10: 预训练模型加载测试")
print("-" * 70)
try:
    # 测试ResNet（轻量级）
    print("测试ResNet18...")
    import torchvision.models as models
    model = models.resnet18(pretrained=False)  # 不下载权重，只测试结构
    print(f"✅ ResNet18结构加载成功")
    print(f"   参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    # 如果有GPU，测试移动到GPU
    if torch.cuda.is_available():
        model = model.cuda()
        print(f"✅ 模型成功移动到GPU")

        # 测试前向传播
        dummy_input = torch.randn(1, 3, 224, 224).cuda()
        with torch.no_grad():
            output = model(dummy_input)
        print(f"✅ GPU前向传播成功，输出形状: {output.shape}")
except Exception as e:
    print(f"⚠️  模型加载测试失败: {e}")
print()

# ==================== 测试11: 数据集访问测试 ====================
print("📌 测试11: 数据集路径检查")
print("-" * 70)

dataset_root = Path("/home/user/datasets")
if dataset_root.exists():
    print(f"✅ 数据集根目录存在: {dataset_root}")
    # 列出子目录
    subdirs = [d for d in dataset_root.iterdir() if d.is_dir()]
    if subdirs:
        print(f"   找到 {len(subdirs)} 个数据集目录:")
        for d in subdirs[:5]:  # 最多显示5个
            print(f"   - {d.name}")
        if len(subdirs) > 5:
            print(f"   ... 还有 {len(subdirs) - 5} 个")
    else:
        print(f"   ⚠️  目录为空，请添加数据集")
else:
    print(f"⚠️  数据集根目录不存在: {dataset_root}")
    print(f"   请创建目录: mkdir -p {dataset_root}")
print()

# ==================== 测试12: 模型路径检查 ====================
print("📌 测试12: 模型路径检查")
print("-" * 70)

models_root = Path("/home/user/models")
if models_root.exists():
    print(f"✅ 模型根目录存在: {models_root}")
    # 检查Qwen2.5-VL
    qwen_path = models_root / "qwen2.5-vl-8b"
    if qwen_path.exists():
        print(f"✅ Qwen2.5-VL模型路径存在: {qwen_path}")
        # 检查关键文件
        required_files = ['config.json', 'pytorch_model.bin', 'tokenizer_config.json']
        for f in required_files:
            if (qwen_path / f).exists():
                print(f"   ✅ {f}")
            else:
                print(f"   ⚠️  缺少 {f}")
    else:
        print(f"⚠️  Qwen2.5-VL模型未找到: {qwen_path}")
        print(f"   请下载模型或修改配置")
else:
    print(f"⚠️  模型根目录不存在: {models_root}")
    print(f"   请创建目录: mkdir -p {models_root}")
print()

# ==================== 总结 ====================
print("=" * 70)
print("📊 测试总结")
print("=" * 70)
print()

print("✅ 必需组件:")
print("   - Python >= 3.8")
print("   - PyTorch")
print("   - TorchVision")
print()

print("⚠️  可选组件（根据需要安装）:")
print("   - Transformers (用于ViT、Qwen2.5-VL)")
print("   - PEFT (用于LoRA)")
print("   - Ultralytics (用于YOLOv8)")
print("   - CUDA (用于GPU加速)")
print()

print("📝 下一步建议:")
if not torch.cuda.is_available():
    print("   1. 如果有NVIDIA GPU，安装CUDA版本的PyTorch")
print("   2. 准备数据集到 /home/user/datasets/")
print("   3. （可选）下载Qwen2.5-VL模型到 /home/user/models/")
print("   4. 运行简单训练测试: python test_simple_training.py")
print()

print("=" * 70)
print("测试完成！")
print("=" * 70)
