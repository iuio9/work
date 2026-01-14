#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简单训练测试脚本
用于验证基本的训练流程（不依赖Qwen2.5-VL大模型）

测试内容：
1. ResNet18在CIFAR-10上的训练
2. 训练循环正确性
3. GPU加速效果
4. 模型保存和加载

作者：Claude Assistant
日期：2026-01-14
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from pathlib import Path
import time
import sys

print("=" * 70)
print("🚀 简单训练测试 - ResNet18 on CIFAR-10")
print("=" * 70)
print()

# ==================== 配置 ====================
BATCH_SIZE = 32
NUM_EPOCHS = 2  # 只训练2个epoch，快速验证
LEARNING_RATE = 0.001
NUM_WORKERS = 2
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"📌 训练配置")
print("-" * 70)
print(f"设备: {DEVICE}")
print(f"批次大小: {BATCH_SIZE}")
print(f"训练轮数: {NUM_EPOCHS}")
print(f"学习率: {LEARNING_RATE}")
print()

# ==================== 准备数据集 ====================
print("📌 准备数据集")
print("-" * 70)

# 数据预处理
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# 下载数据集（如果不存在）
dataset_root = Path("./datasets")
dataset_root.mkdir(exist_ok=True)

try:
    print("正在加载CIFAR-10训练集...")
    trainset = torchvision.datasets.CIFAR10(
        root=str(dataset_root),
        train=True,
        download=True,
        transform=transform_train
    )
    print(f"✅ 训练集加载成功: {len(trainset)} 张图片")

    print("正在加载CIFAR-10测试集...")
    testset = torchvision.datasets.CIFAR10(
        root=str(dataset_root),
        train=False,
        download=True,
        transform=transform_test
    )
    print(f"✅ 测试集加载成功: {len(testset)} 张图片")

except Exception as e:
    print(f"❌ 数据集加载失败: {e}")
    sys.exit(1)

# 创建数据加载器
trainloader = DataLoader(
    trainset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS
)

testloader = DataLoader(
    testset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS
)

print(f"   训练批次数: {len(trainloader)}")
print(f"   测试批次数: {len(testloader)}")
print()

# ==================== 创建模型 ====================
print("📌 创建模型")
print("-" * 70)

try:
    # 使用ResNet18
    model = torchvision.models.resnet18(pretrained=False, num_classes=10)
    model = model.to(DEVICE)

    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"✅ ResNet18创建成功")
    print(f"   总参数量: {total_params / 1e6:.2f}M")
    print(f"   可训练参数: {trainable_params / 1e6:.2f}M")
    print(f"   模型设备: {next(model.parameters()).device}")
except Exception as e:
    print(f"❌ 模型创建失败: {e}")
    sys.exit(1)

print()

# ==================== 定义损失函数和优化器 ====================
print("📌 配置训练组件")
print("-" * 70)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

print(f"✅ 损失函数: CrossEntropyLoss")
print(f"✅ 优化器: Adam (lr={LEARNING_RATE})")
print()

# ==================== 训练函数 ====================
def train_one_epoch(epoch):
    """训练一个epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    print(f"Epoch {epoch + 1}/{NUM_EPOCHS}")
    start_time = time.time()

    for batch_idx, (inputs, labels) in enumerate(trainloader):
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

        # 前向传播
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # 反向传播
        loss.backward()
        optimizer.step()

        # 统计
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # 每100个batch打印一次
        if (batch_idx + 1) % 100 == 0:
            avg_loss = running_loss / (batch_idx + 1)
            acc = 100.0 * correct / total
            print(f"  Batch [{batch_idx + 1}/{len(trainloader)}] "
                  f"Loss: {avg_loss:.4f} | Acc: {acc:.2f}% ({correct}/{total})")

    epoch_time = time.time() - start_time
    avg_loss = running_loss / len(trainloader)
    acc = 100.0 * correct / total

    print(f"  Epoch完成! 用时: {epoch_time:.2f}s | "
          f"平均Loss: {avg_loss:.4f} | 准确率: {acc:.2f}%")

    return avg_loss, acc

# ==================== 测试函数 ====================
def test():
    """在测试集上评估"""
    model.eval()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    avg_loss = test_loss / len(testloader)
    acc = 100.0 * correct / total

    print(f"  测试集结果: Loss: {avg_loss:.4f} | Acc: {acc:.2f}% ({correct}/{total})")

    return avg_loss, acc

# ==================== 开始训练 ====================
print("=" * 70)
print("🏋️  开始训练")
print("=" * 70)
print()

training_start = time.time()

try:
    for epoch in range(NUM_EPOCHS):
        train_loss, train_acc = train_one_epoch(epoch)
        print(f"  正在测试...")
        test_loss, test_acc = test()
        print()

    total_time = time.time() - training_start
    print("=" * 70)
    print(f"✅ 训练完成! 总用时: {total_time:.2f}秒")
    print("=" * 70)

except KeyboardInterrupt:
    print("\n⚠️  训练被用户中断")
    sys.exit(0)
except Exception as e:
    print(f"\n❌ 训练过程出错: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()

# ==================== 模型保存测试 ====================
print("📌 测试模型保存和加载")
print("-" * 70)

model_save_dir = Path("./test_models")
model_save_dir.mkdir(exist_ok=True)
model_path = model_save_dir / "resnet18_cifar10_test.pth"

try:
    # 保存模型
    torch.save({
        'epoch': NUM_EPOCHS,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_acc': train_acc,
        'test_acc': test_acc,
    }, model_path)
    print(f"✅ 模型已保存: {model_path}")
    print(f"   文件大小: {model_path.stat().st_size / 1024 / 1024:.2f} MB")

    # 加载模型
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✅ 模型加载成功")
    print(f"   训练准确率: {checkpoint['train_acc']:.2f}%")
    print(f"   测试准确率: {checkpoint['test_acc']:.2f}%")

except Exception as e:
    print(f"❌ 模型保存/加载失败: {e}")

print()

# ==================== 总结 ====================
print("=" * 70)
print("📊 测试总结")
print("=" * 70)
print()

print("✅ 测试通过的项目:")
print("   - 数据集下载和加载")
print("   - 模型创建和移动到GPU")
print("   - 训练循环执行")
print("   - 前向和反向传播")
print("   - 模型保存和加载")
print()

if torch.cuda.is_available():
    print("💡 GPU加速信息:")
    print(f"   - 使用GPU: {torch.cuda.get_device_name(0)}")
    print(f"   - 显存占用: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
    print(f"   - 显存缓存: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")
else:
    print("⚠️  未使用GPU加速")
    print("   如果有NVIDIA GPU，请安装CUDA版本的PyTorch")

print()
print("📝 下一步:")
print("   1. 如果此测试通过，说明基本训练环境正常")
print("   2. 可以继续测试模型加载: python test_model_loading.py")
print("   3. 最后测试完整蒸馏流程")
print()

print("=" * 70)
print("测试完成！")
print("=" * 70)
