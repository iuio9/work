# Qwen2.5-VL 多模型协同训练指南

## 📖 概述

本系统实现了从 **Qwen2.5-VL 8B** 多模态大模型到多种小模型架构的知识蒸馏训练框架，支持：

### 支持的学生模型

| 模型类型 | 模型变体 | 适用任务 | 参数量 |
|---------|---------|---------|--------|
| **ResNet** | resnet18/34/50/101 | 图像分类 | 11M-44M |
| **Vision Transformer** | vit-tiny/base/large | 图像分类 | 5M-300M |
| **YOLOv8** | n/s/m/l/x | 目标检测 | 3M-68M |
| **UNet** | small/medium/large | 图像分割 | 7M-30M |
| **LSTM** | small/medium/large | 序列特征提取+分类 | 10M-50M |

### 支持的蒸馏策略

1. **特征蒸馏 (Feature-based)**：从Qwen2.5-VL的视觉编码器提取特征
2. **Logits蒸馏 (Logit-based)**：用于分类任务的软标签蒸馏
3. **中间层蒸馏 (Layer-wise)**：适用于Transformer架构（如ViT）
4. **混合蒸馏 (Hybrid)**：结合上述多种策略

---

## 🏗️ 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                  前端配置页面 (Vue3)                          │
│  ┌────────────────────────────────────────────────────┐    │
│  │ 教师模型选择: Qwen2.5-VL 8B                        │    │
│  │ 学生模型选择: [ResNet/ViT/YOLO/UNet/LSTM]          │    │
│  │ 蒸馏策略: [特征/Logits/中间层/混合]                 │    │
│  │ 训练参数: epochs, batch_size, lr, optimizer...    │    │
│  └────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
                         ↓ JSON配置
┌─────────────────────────────────────────────────────────────┐
│              后端 Spring Boot (Java)                         │
│  ┌────────────────────────────────────────────────────┐    │
│  │ ModelDistillationController                        │    │
│  │  - POST /api/training/create (创建训练任务)        │    │
│  │  - POST /api/training/{id}/start (启动训练)       │    │
│  │  - GET /api/training/{id}/progress (查询进度)     │    │
│  │  - POST /api/training/{id}/stop (停止训练)        │    │
│  └────────────────────────────────────────────────────┘    │
│                         ↓                                   │
│  ┌────────────────────────────────────────────────────┐    │
│  │ TrainingExecutionService                           │    │
│  │  - 解析JSON配置                                     │    │
│  │  - 构建Python命令                                   │    │
│  │  - 异步执行训练进程                                 │    │
│  │  - 管理训练生命周期                                 │    │
│  └────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│      Python训练脚本 (train_qwen_vl_distillation.py)         │
│  ┌────────────────────────────────────────────────────┐    │
│  │ 1. 教师模型加载                                     │    │
│  │    Qwen2.5-VL (冻结权重)                           │    │
│  │         ↓                                          │    │
│  │    视觉编码器 → 提取特征 [B, N, 1280]              │    │
│  │                                                    │    │
│  │ 2. 学生模型加载                                     │    │
│  │    [ResNet | ViT | YOLO | UNet | LSTM]            │    │
│  │         ↓                                          │    │
│  │    任务头 → 输出 [分类/检测/分割]                   │    │
│  │                                                    │    │
│  │ 3. 特征对齐层 (可选)                                │    │
│  │    Projector: D_teacher → D_student               │    │
│  │    Attention: 跨模态特征对齐                        │    │
│  │                                                    │    │
│  │ 4. 损失计算                                         │    │
│  │    ├─ 硬标签损失 (Task Loss)                       │    │
│  │    ├─ 软标签损失 (KL Divergence)                   │    │
│  │    ├─ 特征蒸馏损失 (MSE/Cosine)                    │    │
│  │    └─ 总损失 = α*L_hard + β*L_soft + γ*L_feat     │    │
│  │                                                    │    │
│  │ 5. 训练循环                                         │    │
│  │    - 梯度累积                                       │    │
│  │    - 混合精度训练 (AMP)                             │    │
│  │    - 学习率调度                                     │    │
│  │    - 定期验证                                       │    │
│  │    - 保存checkpoints                               │    │
│  │    - HTTP回调更新进度                               │    │
│  └────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

---

## 🚀 快速开始

### 1. 环境准备

#### Python环境 (推荐Python 3.9+)

```bash
# 创建虚拟环境
conda create -n qwen-distill python=3.9
conda activate qwen-distill

# 安装核心依赖
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 安装Qwen2.5-VL
pip install transformers>=4.37.0
pip install qwen-vl-utils  # Qwen VL工具库

# 安装其他依赖
pip install pillow numpy requests tqdm

# 可选：安装YOLOv8
pip install ultralytics

# 可选：安装分布式训练支持
pip install accelerate deepspeed
```

#### 模型下载

```bash
# 方式1：使用Hugging Face Hub
# Qwen2.5-VL 8B模型会自动下载

# 方式2：手动下载
mkdir -p /data/models/qwen2.5-vl-8b
cd /data/models/qwen2.5-vl-8b
# 从ModelScope或Hugging Face下载模型文件
```

### 2. 数据准备

#### 数据集结构

```
/data/datasets/
├── train/
│   ├── class_0/
│   │   ├── img_001.jpg
│   │   ├── img_002.jpg
│   │   └── ...
│   ├── class_1/
│   │   └── ...
│   └── ...
└── val/
    ├── class_0/
    └── ...
```

#### 数据库集成 (TODO)

修改 `train_qwen_vl_distillation.py` 中的 `MultiTaskDataset` 类：

```python
def __getitem__(self, idx):
    # 从数据库加载图像路径和标注
    image_record = self.db.query(
        "SELECT image_path, label FROM dataset WHERE id=?",
        (idx,)
    )

    # 加载图像
    image = Image.open(image_record['image_path'])
    pixel_values = self.transform(image)

    return {
        'pixel_values': pixel_values,
        'labels': image_record['label']
    }
```

### 3. 配置训练任务

#### 示例1：ResNet50 分类任务

```bash
python train_qwen_vl_distillation.py \
    --task_id "task_001" \
    --api_base_url "http://localhost:8080/api" \
    --teacher_model_path "/data/models/qwen2.5-vl-8b" \
    --student_model_type "resnet" \
    --student_model_size "resnet50" \
    --task_type "classification" \
    --num_classes 10 \
    --dataset_path "/data/datasets/train" \
    --val_dataset_path "/data/datasets/val" \
    --image_size 224 \
    --epochs 100 \
    --batch_size 32 \
    --learning_rate 1e-4 \
    --optimizer_type "adamw" \
    --lr_scheduler "cosine" \
    --distillation_type "hybrid" \
    --alpha 0.5 --beta 0.3 --gamma 0.2 \
    --align_feature \
    --use_amp \
    --gpu_devices "0" \
    --output_dir "/data/outputs/task_001"
```

#### 示例2：Vision Transformer 分类任务

```bash
python train_qwen_vl_distillation.py \
    --task_id "task_002" \
    --api_base_url "http://localhost:8080/api" \
    --teacher_model_path "/data/models/qwen2.5-vl-8b" \
    --student_model_type "vit" \
    --student_model_size "vit-base" \
    --task_type "classification" \
    --num_classes 100 \
    --dataset_path "/data/datasets/imagenet_subset/train" \
    --val_dataset_path "/data/datasets/imagenet_subset/val" \
    --image_size 224 \
    --epochs 50 \
    --batch_size 64 \
    --learning_rate 5e-5 \
    --optimizer_type "adamw" \
    --lr_scheduler "cosine" \
    --weight_decay 0.05 \
    --distillation_type "layer" \
    --temperature 4.0 \
    --align_feature \
    --feature_loss_type "cosine" \
    --use_amp \
    --gpu_devices "0,1" \
    --output_dir "/data/outputs/task_002"
```

#### 示例3：YOLOv8 目标检测任务

```bash
python train_qwen_vl_distillation.py \
    --task_id "task_003" \
    --api_base_url "http://localhost:8080/api" \
    --teacher_model_path "/data/models/qwen2.5-vl-8b" \
    --student_model_type "yolov8" \
    --student_model_size "s" \
    --task_type "detection" \
    --num_classes 80 \
    --dataset_path "/data/datasets/coco/train" \
    --val_dataset_path "/data/datasets/coco/val" \
    --image_size 640 \
    --epochs 300 \
    --batch_size 16 \
    --learning_rate 1e-3 \
    --optimizer_type "sgd" \
    --lr_scheduler "cosine" \
    --distillation_type "feature" \
    --gamma 1.0 \
    --feature_loss_type "mse" \
    --gpu_devices "0" \
    --output_dir "/data/outputs/task_003"
```

#### 示例4：UNet 图像分割任务

```bash
python train_qwen_vl_distillation.py \
    --task_id "task_004" \
    --api_base_url "http://localhost:8080/api" \
    --teacher_model_path "/data/models/qwen2.5-vl-8b" \
    --student_model_type "unet" \
    --student_model_size "medium" \
    --task_type "segmentation" \
    --num_classes 21 \
    --dataset_path "/data/datasets/voc2012/train" \
    --val_dataset_path "/data/datasets/voc2012/val" \
    --image_size 512 \
    --epochs 150 \
    --batch_size 8 \
    --learning_rate 1e-4 \
    --optimizer_type "adam" \
    --distillation_type "feature" \
    --gamma 1.0 \
    --use_amp \
    --gpu_devices "0" \
    --output_dir "/data/outputs/task_004"
```

#### 示例5：LSTM 序列分类任务

```bash
python train_qwen_vl_distillation.py \
    --task_id "task_005" \
    --api_base_url "http://localhost:8080/api" \
    --teacher_model_path "/data/models/qwen2.5-vl-8b" \
    --student_model_type "lstm" \
    --student_model_size "medium" \
    --task_type "classification" \
    --num_classes 10 \
    --dataset_path "/data/datasets/train" \
    --val_dataset_path "/data/datasets/val" \
    --image_size 224 \
    --epochs 100 \
    --batch_size 32 \
    --learning_rate 1e-4 \
    --optimizer_type "adam" \
    --distillation_type "feature" \
    --gamma 0.5 \
    --gpu_devices "0" \
    --output_dir "/data/outputs/task_005"
```

---

## ⚙️ 配置参数详解

### 基础配置

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `--task_id` | str | ✅ | 任务唯一标识 |
| `--api_base_url` | str | ✅ | 后端API地址 |
| `--teacher_model_path` | str | ✅ | Qwen2.5-VL模型路径 |
| `--student_model_type` | str | ✅ | 学生模型类型：resnet/vit/yolov8/unet/lstm |
| `--student_model_size` | str | ✅ | 模型大小：resnet50, vit-base等 |

### 任务配置

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--task_type` | str | classification | 任务类型：classification/detection/segmentation |
| `--num_classes` | int | 10 | 分类类别数 |
| `--dataset_path` | str | 必填 | 训练数据集路径 |
| `--val_dataset_path` | str | 必填 | 验证数据集路径 |
| `--image_size` | int | 224 | 输入图像大小 |

### 训练超参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--epochs` | int | 100 | 训练轮数 |
| `--batch_size` | int | 32 | 批大小 |
| `--learning_rate` | float | 1e-4 | 学习率 |
| `--optimizer_type` | str | adamw | 优化器：adamw/adam/sgd |
| `--lr_scheduler` | str | cosine | 学习率调度：cosine/linear/step |
| `--weight_decay` | float | 0.01 | 权重衰减 |
| `--grad_accum_steps` | int | 1 | 梯度累积步数 |
| `--max_grad_norm` | float | 1.0 | 梯度裁剪阈值 |

### 蒸馏配置

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--distillation_type` | str | hybrid | 蒸馏类型：feature/logit/layer/hybrid |
| `--temperature` | float | 4.0 | 蒸馏温度 (用于softmax) |
| `--alpha` | float | 0.5 | 硬标签权重 (任务损失) |
| `--beta` | float | 0.3 | 软标签权重 (KL散度) |
| `--gamma` | float | 0.2 | 特征蒸馏权重 |
| `--feature_loss_type` | str | mse | 特征损失类型：mse/cosine/attention |
| `--align_feature` | flag | False | 是否使用特征对齐层 |
| `--feature_dim` | int | 768 | 特征对齐维度 |

### GPU配置

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--gpu_devices` | str | 0 | GPU设备ID，逗号分隔 (如"0,1,2") |
| `--use_amp` | flag | False | 使用混合精度训练 (推荐) |

### 输出配置

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `--output_dir` | str | ✅ | 输出目录（checkpoints等） |
| `--checkpoint_interval` | int | 10 | 保存checkpoint间隔（epoch） |
| `--log_interval` | int | 50 | 日志打印间隔（steps） |

---

## 📊 蒸馏策略详解

### 1. 特征蒸馏 (Feature-based)

**原理**：从Qwen2.5-VL的视觉编码器提取特征，让学生模型学习教师模型的中间表示。

**适用场景**：
- ✅ 所有模型类型（ResNet、ViT、YOLO、UNet、LSTM）
- ✅ 跨架构蒸馏（Transformer → CNN）
- ✅ 任务不完全匹配的情况

**损失函数**：
```python
# MSE损失
L_feature = MSE(student_features, aligned_teacher_features)

# 余弦相似度损失
L_feature = 1 - cosine_similarity(student_features, teacher_features)
```

**配置示例**：
```bash
--distillation_type "feature" \
--gamma 1.0 \
--feature_loss_type "mse" \
--align_feature
```

### 2. Logits蒸馏 (Logit-based)

**原理**：使用Qwen2.5-VL的零样本分类能力生成软标签，通过KL散度蒸馏到学生模型。

**适用场景**：
- ✅ 分类任务（ResNet、ViT、LSTM）
- ❌ 不适用于检测/分割任务

**损失函数**：
```python
# 软标签
soft_teacher = softmax(teacher_logits / T)
soft_student = log_softmax(student_logits / T)

# KL散度
L_soft = T^2 * KL_divergence(soft_student, soft_teacher)

# 总损失
L_total = α * L_hard + β * L_soft
```

**配置示例**：
```bash
--distillation_type "logit" \
--temperature 4.0 \
--alpha 0.5 \
--beta 0.5
```

### 3. 中间层蒸馏 (Layer-wise)

**原理**：对齐教师模型和学生模型的中间层表示，特别适用于Transformer架构。

**适用场景**：
- ✅ Vision Transformer → ViT (架构相似)
- ⚠️  需要学生模型也是Transformer架构

**损失函数**：
```python
# 逐层对齐
L_layer = Σ MSE(student_layer_i, teacher_layer_j)
```

**配置示例**：
```bash
--distillation_type "layer" \
--student_model_type "vit"
```

### 4. 混合蒸馏 (Hybrid)

**原理**：结合特征蒸馏、Logits蒸馏和任务损失。

**适用场景**：
- ✅ 分类任务，追求最佳性能
- ✅ 学生模型需要同时学习特征和决策边界

**损失函数**：
```python
L_total = α * L_hard + β * L_soft + γ * L_feature
```

**配置示例**：
```bash
--distillation_type "hybrid" \
--alpha 0.5 --beta 0.3 --gamma 0.2 \
--temperature 4.0 \
--feature_loss_type "cosine" \
--align_feature
```

---

## 🔧 进阶配置

### 多GPU训练

```bash
# 数据并行
python train_qwen_vl_distillation.py \
    --gpu_devices "0,1,2,3" \
    --batch_size 128 \
    # 其他参数...

# 注意：当前实现使用单GPU，多GPU需要集成torch.nn.DataParallel或DDP
```

### 梯度累积（模拟大批量）

```bash
# 实际batch_size = 32，模拟batch_size = 32 * 4 = 128
python train_qwen_vl_distillation.py \
    --batch_size 32 \
    --grad_accum_steps 4 \
    # 其他参数...
```

### 混合精度训练（节省显存）

```bash
python train_qwen_vl_distillation.py \
    --use_amp \
    --batch_size 64 \  # 可以增大batch_size
    # 其他参数...
```

### 继续训练（从checkpoint恢复）

```python
# 修改train_qwen_vl_distillation.py
# 在main()函数中添加：

if args.resume_checkpoint:
    checkpoint = torch.load(args.resume_checkpoint)
    trainer.student_model.load_state_dict(checkpoint['model_state_dict'])
    trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    trainer.current_epoch = checkpoint['epoch']
```

---

## 📈 监控和调试

### 训练进度监控

训练脚本会通过HTTP回调自动上报进度到后端API：

```python
# 每个epoch结束后
POST http://localhost:8080/api/training/progress/{task_id}
{
    "epoch": 10,
    "total_epochs": 100,
    "train_loss": 0.234,
    "val_loss": 0.456,
    "val_accuracy": 92.3,
    "timestamp": "2026-01-11T10:30:00"
}
```

### TensorBoard集成（可选）

```python
# 在train_qwen_vl_distillation.py中添加
from torch.utils.tensorboard import SummaryWriter

class MultiModelDistillationTrainer:
    def __init__(self, config):
        # ...
        self.writer = SummaryWriter(log_dir=config.output_dir)

    def _log_training_step(self, losses):
        self.writer.add_scalar('Loss/total', losses['total_loss'], self.global_step)
        self.writer.add_scalar('Loss/hard', losses['hard_loss'], self.global_step)
        self.writer.add_scalar('Loss/feature', losses['feature_loss'], self.global_step)

# 启动TensorBoard
tensorboard --logdir=/data/outputs/task_001
```

### 常见问题排查

#### Q1: CUDA Out of Memory

**解决方案**：
```bash
# 1. 减小batch_size
--batch_size 16  # 从32减到16

# 2. 使用梯度累积
--batch_size 16 --grad_accum_steps 2

# 3. 使用混合精度
--use_amp

# 4. 减小图像尺寸
--image_size 192  # 从224减到192
```

#### Q2: Qwen2.5-VL加载失败

**检查**：
```bash
# 验证模型文件完整性
ls -lh /data/models/qwen2.5-vl-8b/
# 应包含：config.json, model.safetensors, tokenizer.json等

# 测试模型加载
python -c "from transformers import Qwen2VLForConditionalGeneration; \
           model = Qwen2VLForConditionalGeneration.from_pretrained('/data/models/qwen2.5-vl-8b')"
```

#### Q3: 特征维度不匹配

**症状**：`RuntimeError: The size of tensor a (1280) must match the size of tensor b (768)`

**解决方案**：
```bash
# 启用特征对齐层
--align_feature \
--feature_dim 768  # 设置为学生模型的特征维度
```

#### Q4: 训练损失不下降

**排查步骤**：
1. 检查学习率是否过小或过大
   ```bash
   --learning_rate 1e-3  # 尝试调整
   ```

2. 检查蒸馏权重配置
   ```bash
   # 初期增大硬标签权重
   --alpha 0.7 --beta 0.2 --gamma 0.1
   ```

3. 检查数据增强是否过强
   ```python
   # 减少数据增强强度
   transforms.ColorJitter(brightness=0.1, contrast=0.1)  # 从0.2减到0.1
   ```

---

## 🎯 最佳实践

### 1. 学生模型选择建议

| 任务类型 | 推荐模型 | 理由 |
|---------|---------|------|
| **图像分类** | ResNet50, ViT-Base | 平衡准确率和速度 |
| **目标检测** | YOLOv8-s/m | 实时检测性能好 |
| **图像分割** | UNet | 专门设计用于分割 |
| **视频/序列任务** | LSTM | 处理时序信息 |
| **边缘部署** | ResNet18, YOLOv8-n | 参数量小，推理快 |

### 2. 蒸馏策略选择建议

| 场景 | 推荐策略 | 配置 |
|------|---------|------|
| **分类任务，追求高准确率** | Hybrid | `--distillation_type hybrid --alpha 0.5 --beta 0.3 --gamma 0.2` |
| **分类任务，快速收敛** | Feature | `--distillation_type feature --gamma 1.0` |
| **ViT学生模型** | Layer-wise | `--distillation_type layer` |
| **检测/分割任务** | Feature | `--distillation_type feature --gamma 1.0` |
| **跨架构蒸馏** | Feature + Align | `--distillation_type feature --align_feature` |

### 3. 超参数调优建议

**学习率**：
```
ResNet: 1e-4
ViT: 5e-5 (Transformer对学习率敏感)
YOLO: 1e-3 (检测任务通常需要更大学习率)
UNet: 1e-4
LSTM: 1e-4
```

**Batch Size**：
```
分类 (ResNet/ViT): 32-64
检测 (YOLO): 16-32 (受图像尺寸影响)
分割 (UNet): 8-16 (显存占用大)
```

**蒸馏温度**：
```
简单任务 (10类): T=2-3
中等任务 (100类): T=4-5
复杂任务 (1000类): T=6-8
```

### 4. 训练流程建议

**阶段1：热身训练（10% epochs）**
```bash
# 仅使用硬标签损失，让学生模型先学会基本任务
--alpha 1.0 --beta 0.0 --gamma 0.0 \
--lr_scheduler "linear"
```

**阶段2：蒸馏训练（80% epochs）**
```bash
# 完整蒸馏
--alpha 0.5 --beta 0.3 --gamma 0.2 \
--lr_scheduler "cosine"
```

**阶段3：微调（10% epochs）**
```bash
# 降低学习率，仅使用硬标签
--learning_rate 1e-5 \
--alpha 1.0 --beta 0.0 --gamma 0.0
```

---

## 📝 注意事项

### 1. Qwen2.5-VL特性

- **输入格式**：Qwen2.5-VL接受图像+文本作为输入
- **视觉编码器**：提取的特征维度通常为1280或更高
- **零样本能力**：可以用于生成软标签（分类任务）
- **显存占用**：8B模型加载需要约16GB显存（FP16）

### 2. 数据集要求

- **图像格式**：支持JPEG、PNG等常见格式
- **图像大小**：建议统一resize到224x224或更大
- **标注格式**：
  - 分类：类别ID (0-N)
  - 检测：COCO格式或YOLO格式
  - 分割：像素级标签图

### 3. 计算资源需求

| 配置 | GPU显存 | 训练时间 (100 epochs, 1000样本) |
|------|---------|--------------------------------|
| **最低配置** | 16GB | ~6小时 |
| **推荐配置** | 24GB | ~3小时 |
| **高性能配置** | 32GB+ | ~1.5小时 |

### 4. 许可证和使用限制

- **Qwen2.5-VL**：遵循阿里云通义千问模型许可协议
- **学生模型**：遵循各自的开源许可证（MIT、Apache等）
- **商业使用**：需查阅Qwen2.5-VL的商业使用条款

---

## 🔗 参考资源

### 官方文档

- [Qwen2.5-VL官方文档](https://github.com/QwenLM/Qwen-VL)
- [Hugging Face Transformers文档](https://huggingface.co/docs/transformers)
- [PyTorch官方文档](https://pytorch.org/docs/stable/index.html)

### 相关论文

1. **知识蒸馏**：
   - [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531) (Hinton et al., 2015)

2. **特征蒸馏**：
   - [FitNets: Hints for Thin Deep Nets](https://arxiv.org/abs/1412.6550) (Romero et al., 2014)

3. **多模态模型**：
   - [Qwen-VL: A Versatile Vision-Language Model](https://arxiv.org/abs/2308.12966)

### 社区支持

- **GitHub Issues**：报告bug和功能请求
- **Discussions**：技术讨论和经验分享
- **微信群**：加入开发者社区

---

## 📄 许可证

本项目代码遵循 MIT 许可证。使用的模型需遵循各自的许可协议。

---

**最后更新**: 2026-01-11
**维护者**: Claude Assistant
**版本**: 1.0.0
