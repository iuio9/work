#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
知识蒸馏测试脚本
测试大模型（教师）→ 小模型（学生）的知识蒸馏训练流程

核心测试：
1. 教师模型（大模型）：ResNet50 或 Qwen2.5-VL
2. 学生模型（小模型）：ResNet18
3. 蒸馏损失：KL散度 + 交叉熵
4. 温度参数调节
5. 完整蒸馏训练流程

作者：Claude Assistant
日期：2026-01-14
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from pathlib import Path
import time
import sys

print("=" * 80)
print("🎓 知识蒸馏测试 - 大小模型协同训练")
print("=" * 80)
print()

# ==================== 配置 ====================
BATCH_SIZE = 64
NUM_EPOCHS = 3
LEARNING_RATE = 0.001
NUM_WORKERS = 2
TEMPERATURE = 4.0  # 蒸馏温度
ALPHA = 0.7  # 蒸馏损失权重（1-alpha为硬标签权重）
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"📌 训练配置")
print("-" * 80)
print(f"设备: {DEVICE}")
print(f"批次大小: {BATCH_SIZE}")
print(f"训练轮数: {NUM_EPOCHS}")
print(f"学习率: {LEARNING_RATE}")
print(f"蒸馏温度: {TEMPERATURE}")
print(f"蒸馏权重 α: {ALPHA}")
print()

# ==================== 准备数据集 ====================
print("📌 准备数据集")
print("-" * 80)

dataset_root = Path("./datasets")
dataset_root.mkdir(exist_ok=True)

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.Resize(224),  # ResNet需要224x224
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

try:
    print("正在加载CIFAR-10...")
    trainset = torchvision.datasets.CIFAR10(
        root=str(dataset_root),
        train=True,
        download=True,
        transform=transform_train
    )

    testset = torchvision.datasets.CIFAR10(
        root=str(dataset_root),
        train=False,
        download=True,
        transform=transform_test
    )

    print(f"✅ 数据集加载成功")
    print(f"   训练集: {len(trainset)} 张")
    print(f"   测试集: {len(testset)} 张")

except Exception as e:
    print(f"❌ 数据集加载失败: {e}")
    sys.exit(1)

trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

print()

# ==================== 创建教师模型（大模型）====================
print("📌 创建教师模型（大模型）")
print("-" * 80)

try:
    # 使用ResNet101作为教师模型（也可以用Qwen2.5-VL，但需要很大显存）
    teacher_model = torchvision.models.resnet101(pretrained=False, num_classes=10)
    teacher_model = teacher_model.to(DEVICE)
    teacher_model.eval()  # 教师模型设为评估模式

    # 冻结教师模型参数（不训练）
    for param in teacher_model.parameters():
        param.requires_grad = False

    teacher_params = sum(p.numel() for p in teacher_model.parameters()) / 1e6

    print(f"✅ 教师模型创建成功: ResNet101")
    print(f"   参数量: {teacher_params:.2f}M")
    print(f"   模式: 评估模式（冻结参数）")

    # 选项：如果有预训练的教师模型，可以加载
    teacher_checkpoint_path = Path("./test_models/teacher_resnet101.pth")
    if teacher_checkpoint_path.exists():
        print(f"   正在加载预训练教师模型...")
        checkpoint = torch.load(teacher_checkpoint_path)
        teacher_model.load_state_dict(checkpoint['model_state_dict'])
        print(f"   ✅ 预训练模型已加载")
    else:
        print(f"   ⚠️  未找到预训练教师模型，将使用随机初始化")
        print(f"   提示: 真实场景中教师模型应该是预训练好的")

except Exception as e:
    print(f"❌ 教师模型创建失败: {e}")
    sys.exit(1)

print()

# ==================== 创建学生模型（小模型）====================
print("📌 创建学生模型（小模型）")
print("-" * 80)

try:
    # 使用ResNet18作为学生模型
    student_model = torchvision.models.resnet18(pretrained=False, num_classes=10)
    student_model = student_model.to(DEVICE)

    student_params = sum(p.numel() for p in student_model.parameters()) / 1e6

    print(f"✅ 学生模型创建成功: ResNet18")
    print(f"   参数量: {student_params:.2f}M")
    print(f"   压缩比: {teacher_params / student_params:.2f}x")

except Exception as e:
    print(f"❌ 学生模型创建失败: {e}")
    sys.exit(1)

print()

# ==================== 定义蒸馏损失 ====================
print("📌 定义蒸馏损失函数")
print("-" * 80)

class DistillationLoss(nn.Module):
    """知识蒸馏损失函数

    结合两部分：
    1. 蒸馏损失：学生输出与教师输出的KL散度（软标签）
    2. 学生损失：学生输出与真实标签的交叉熵（硬标签）
    """

    def __init__(self, temperature=4.0, alpha=0.7):
        super(DistillationLoss, self).__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.kl_div = nn.KLDivLoss(reduction='batchmean')
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, student_logits, teacher_logits, labels):
        """
        Args:
            student_logits: 学生模型输出 (batch_size, num_classes)
            teacher_logits: 教师模型输出 (batch_size, num_classes)
            labels: 真实标签 (batch_size,)

        Returns:
            total_loss: 总损失
            distill_loss: 蒸馏损失
            student_loss: 学生损失
        """
        # 蒸馏损失：KL散度（使用温度软化）
        student_soft = F.log_softmax(student_logits / self.temperature, dim=1)
        teacher_soft = F.softmax(teacher_logits / self.temperature, dim=1)
        distill_loss = self.kl_div(student_soft, teacher_soft) * (self.temperature ** 2)

        # 学生损失：交叉熵（硬标签）
        student_loss = self.ce_loss(student_logits, labels)

        # 总损失：加权组合
        total_loss = self.alpha * distill_loss + (1 - self.alpha) * student_loss

        return total_loss, distill_loss, student_loss

criterion = DistillationLoss(temperature=TEMPERATURE, alpha=ALPHA)
optimizer = optim.Adam(student_model.parameters(), lr=LEARNING_RATE)

print(f"✅ 蒸馏损失函数配置完成")
print(f"   温度参数 T: {TEMPERATURE}")
print(f"   蒸馏权重 α: {ALPHA}")
print(f"   学生权重 (1-α): {1-ALPHA}")
print(f"   公式: Loss = α * KL(S||T) + (1-α) * CE(S, y)")
print()

# ==================== 训练函数 ====================
def train_one_epoch(epoch):
    """训练一个epoch（知识蒸馏）"""
    student_model.train()
    teacher_model.eval()

    running_total_loss = 0.0
    running_distill_loss = 0.0
    running_student_loss = 0.0
    correct = 0
    total = 0

    print(f"Epoch {epoch + 1}/{NUM_EPOCHS}")
    start_time = time.time()

    for batch_idx, (inputs, labels) in enumerate(trainloader):
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

        # 教师模型前向传播（不计算梯度）
        with torch.no_grad():
            teacher_logits = teacher_model(inputs)

        # 学生模型前向传播
        optimizer.zero_grad()
        student_logits = student_model(inputs)

        # 计算蒸馏损失
        total_loss, distill_loss, student_loss = criterion(
            student_logits, teacher_logits, labels
        )

        # 反向传播
        total_loss.backward()
        optimizer.step()

        # 统计
        running_total_loss += total_loss.item()
        running_distill_loss += distill_loss.item()
        running_student_loss += student_loss.item()

        _, predicted = student_logits.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # 每50个batch打印一次
        if (batch_idx + 1) % 50 == 0:
            avg_total = running_total_loss / (batch_idx + 1)
            avg_distill = running_distill_loss / (batch_idx + 1)
            avg_student = running_student_loss / (batch_idx + 1)
            acc = 100.0 * correct / total

            print(f"  [{batch_idx + 1}/{len(trainloader)}] "
                  f"Total: {avg_total:.4f} | "
                  f"Distill: {avg_distill:.4f} | "
                  f"Student: {avg_student:.4f} | "
                  f"Acc: {acc:.2f}%")

    epoch_time = time.time() - start_time
    avg_total = running_total_loss / len(trainloader)
    avg_distill = running_distill_loss / len(trainloader)
    avg_student = running_student_loss / len(trainloader)
    acc = 100.0 * correct / total

    print(f"  Epoch完成! 用时: {epoch_time:.2f}s")
    print(f"  总损失: {avg_total:.4f} | 蒸馏损失: {avg_distill:.4f} | "
          f"学生损失: {avg_student:.4f} | 准确率: {acc:.2f}%")

    return avg_total, avg_distill, avg_student, acc

# ==================== 测试函数 ====================
def test():
    """在测试集上评估学生模型"""
    student_model.eval()
    test_loss = 0
    correct = 0
    total = 0

    ce_loss = nn.CrossEntropyLoss()

    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            outputs = student_model(inputs)
            loss = ce_loss(outputs, labels)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    avg_loss = test_loss / len(testloader)
    acc = 100.0 * correct / total

    print(f"  测试集结果: Loss: {avg_loss:.4f} | Acc: {acc:.2f}% ({correct}/{total})")

    return avg_loss, acc

# ==================== 对比：不使用蒸馏 ====================
def train_without_distillation_one_epoch(model, optimizer, epoch):
    """不使用蒸馏的普通训练（对比）"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    ce_loss = nn.CrossEntropyLoss()

    for batch_idx, (inputs, labels) in enumerate(trainloader):
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = ce_loss(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    avg_loss = running_loss / len(trainloader)
    acc = 100.0 * correct / total

    return avg_loss, acc

# ==================== 开始训练 ====================
print("=" * 80)
print("🏋️  开始知识蒸馏训练")
print("=" * 80)
print()

training_start = time.time()
best_acc = 0.0

try:
    for epoch in range(NUM_EPOCHS):
        # 蒸馏训练
        train_total_loss, train_distill_loss, train_student_loss, train_acc = train_one_epoch(epoch)

        # 测试
        print(f"  正在测试...")
        test_loss, test_acc = test()

        if test_acc > best_acc:
            best_acc = test_acc
            print(f"  🎉 新的最佳准确率: {best_acc:.2f}%")

        print()

    total_time = time.time() - training_start
    print("=" * 80)
    print(f"✅ 蒸馏训练完成! 总用时: {total_time:.2f}秒")
    print(f"   最佳测试准确率: {best_acc:.2f}%")
    print("=" * 80)

except KeyboardInterrupt:
    print("\n⚠️  训练被用户中断")
except Exception as e:
    print(f"\n❌ 训练过程出错: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()

# ==================== 对比实验（可选）====================
print("=" * 80)
print("📊 对比实验：不使用蒸馏的训练")
print("=" * 80)
print()

print("正在创建新的ResNet18（不使用蒸馏）进行对比...")
baseline_model = torchvision.models.resnet18(pretrained=False, num_classes=10)
baseline_model = baseline_model.to(DEVICE)
baseline_optimizer = optim.Adam(baseline_model.parameters(), lr=LEARNING_RATE)

print("训练1个epoch进行对比...")
baseline_train_loss, baseline_train_acc = train_without_distillation_one_epoch(
    baseline_model, baseline_optimizer, 0
)
baseline_test_loss, baseline_test_acc = test()

print()
print("-" * 80)
print("🔍 对比结果")
print("-" * 80)
print(f"{'方法':<20s} {'训练准确率':<15s} {'测试准确率':<15s}")
print("-" * 80)
print(f"{'知识蒸馏':<20s} {train_acc:<15.2f} {test_acc:<15.2f}")
print(f"{'普通训练（基线）':<20s} {baseline_train_acc:<15.2f} {baseline_test_acc:<15.2f}")
print("-" * 80)

if test_acc > baseline_test_acc:
    improvement = test_acc - baseline_test_acc
    print(f"✅ 知识蒸馏提升: +{improvement:.2f}%")
else:
    print(f"⚠️  基线更好（可能需要更多训练轮数或调整超参数）")

print()

# ==================== 保存模型 ====================
print("📌 保存蒸馏后的学生模型")
print("-" * 80)

model_save_dir = Path("./test_models")
model_save_dir.mkdir(exist_ok=True)

try:
    student_path = model_save_dir / "student_resnet18_distilled.pth"
    torch.save({
        'epoch': NUM_EPOCHS,
        'model_state_dict': student_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'test_acc': test_acc,
        'best_acc': best_acc,
        'temperature': TEMPERATURE,
        'alpha': ALPHA,
    }, student_path)
    print(f"✅ 学生模型已保存: {student_path}")
    print(f"   文件大小: {student_path.stat().st_size / 1024 / 1024:.2f} MB")
    print(f"   测试准确率: {test_acc:.2f}%")

except Exception as e:
    print(f"❌ 模型保存失败: {e}")

print()

# ==================== 总结 ====================
print("=" * 80)
print("📊 知识蒸馏测试总结")
print("=" * 80)
print()

print("✅ 测试完成的项目:")
print("   - 教师模型（ResNet101）创建和推理")
print("   - 学生模型（ResNet18）创建")
print("   - 蒸馏损失计算（KL散度 + 交叉熵）")
print("   - 完整蒸馏训练流程")
print("   - 模型保存")
print("   - 与基线对比")
print()

print("🎓 知识蒸馏核心概念验证:")
print(f"   ✅ 大模型（教师）参数量: {teacher_params:.2f}M")
print(f"   ✅ 小模型（学生）参数量: {student_params:.2f}M")
print(f"   ✅ 模型压缩比: {teacher_params / student_params:.2f}x")
print(f"   ✅ 软标签温度调节: T={TEMPERATURE}")
print(f"   ✅ 损失加权平衡: α={ALPHA}")
print()

if torch.cuda.is_available():
    print("💡 GPU使用情况:")
    print(f"   - GPU: {torch.cuda.get_device_name(0)}")
    print(f"   - 显存占用: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
    print(f"   - 显存峰值: {torch.cuda.max_memory_allocated(0) / 1024**3:.2f} GB")
else:
    print("⚠️  未使用GPU加速")

print()

print("📝 真实应用中的改进方向:")
print("   1. 使用更强的教师模型（ResNet152, ViT-Large, Qwen2.5-VL）")
print("   2. 教师模型应该是预训练好的（accuracy > 95%）")
print("   3. 增加训练轮数（50-200 epochs）")
print("   4. 使用更大的数据集（ImageNet）")
print("   5. 调整温度和alpha参数获得最佳效果")
print("   6. 可以加入中间层特征蒸馏（Feature Distillation）")
print()

print("🚀 下一步:")
print("   1. 如果此测试通过，说明知识蒸馏流程正确")
print("   2. 可以训练一个强教师模型（高准确率）")
print("   3. 然后用强教师模型蒸馏到多个小模型")
print("   4. 部署完整系统并在前端界面配置蒸馏任务")
print()

print("=" * 80)
print("测试完成！")
print("=" * 80)
