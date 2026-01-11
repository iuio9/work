# Qwen2.5-VL 多模型协同训练完整指南

## 📖 概述

本系统实现了从 **Qwen2.5-VL 8B** 多模态大模型到5种不同架构小模型的知识蒸馏训练框架。

### 支持的学生模型

| 模型类型 | 模型变体 | 适用任务 | 参数量 | 特点 |
|---------|---------|---------|--------|------|
| **LSTM** | small/medium/large | 序列特征提取+分类 | 10M-50M | 处理时序信息 |
| **UNet** | small/medium/large | 图像分割 | 7M-30M | 像素级预测 |
| **YOLOv8** | n/s/m/l/x | 目标检测 | 3M-68M | 实时检测 |
| **ResNet** | resnet18/34/50/101 | 图像分类 | 11M-44M | 经典CNN架构 |
| **Vision Transformer** | vit-tiny/base/large | 图像分类 | 5M-300M | Transformer架构 |

---

## 🏗️ 系统架构

```
┌─────────────────────────────────────────────────────────┐
│             前端 Vue3 (已有)                              │
│  - 创建训练任务                                          │
│  - 配置教师模型：Qwen2.5-VL 8B                          │
│  - 配置学生模型类型和大小                                │
│  - JSON配置编辑                                          │
└────────────────────────┬────────────────────────────────┘
                         ↓ HTTP POST
┌─────────────────────────────────────────────────────────┐
│        Spring Boot后端 (TrainingExecutionService)        │
│  1. 解析training_config JSON                           │
│  2. 构建Python命令                                      │
│  3. 启动训练进程                                         │
└────────────────────────┬────────────────────────────────┘
                         ↓ ProcessBuilder
┌─────────────────────────────────────────────────────────┐
│   Python训练脚本: train_qwen_vl_distillation.py         │
│                                                         │
│   Qwen2.5-VL 8B (教师模型) - 冻结权重                   │
│         ↓                                               │
│   视觉编码器提取特征 [B, N, 1280]                        │
│         ↓                                               │
│   特征对齐层 (可选)                                      │
│         ↓                                               │
│   ┌──────┬──────┬──────┬──────┬──────┐                │
│   │ LSTM │ UNet │ YOLO │ResNet│  ViT │ (学生模型)      │
│   └──────┴──────┴──────┴──────┴──────┘                │
│         ↓                                               │
│   蒸馏损失 = α*L_hard + β*L_feature                     │
│         ↓                                               │
│   训练&保存模型                                          │
└─────────────────────────────────────────────────────────┘
```

---

## 🚀 快速开始

### 1. 环境准备

#### Python环境（Python 3.9+）

```bash
# 创建虚拟环境
conda create -n qwen-distill python=3.9
conda activate qwen-distill

# 安装PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 安装核心依赖
pip install transformers>=4.37.0
pip install pillow numpy requests tqdm
pip install peft  # LoRA支持

# 可选：安装Qwen2.5-VL（如果模型库可用）
pip install qwen-vl-utils

# 可选：安装YOLOv8
pip install ultralytics
```

#### 模型下载

```bash
# Qwen2.5-VL 8B模型
# 方式1：自动下载（首次运行时）
# 方式2：手动下载
mkdir -p /data/models/qwen2.5-vl-8b
# 从ModelScope或Hugging Face下载模型文件
```

### 2. 后端配置

#### application-distillation.yml

在 `back/datamark-admin/src/main/resources/` 目录下添加或更新：

```yaml
distillation:
  python:
    path: python3  # Python解释器路径，改为你的虚拟环境路径
  script:
    path: /home/user/work/back/datamark-admin/train_qwen_vl_distillation.py
  api:
    base-url: http://localhost:8080
  models:
    root: /data/models  # 模型存储根目录
  datasets:
    root: /data/datasets  # 数据集根目录
  output:
    root: /data/training_output  # 训练输出目录
```

### 3. 使用方式

#### 方式1：通过前端界面（推荐）

1. **选择教师模型**
   - 教师模型名称：`qwen2.5-vl-8b`
   - 教师模型路径：`/data/models/qwen2.5-vl-8b`

2. **选择学生模型**
   - 学生模型类型：从下拉框选择 `LSTM` / `UNet` / `YOLOv8` / `ResNet` / `ViT`
   - 学生模型大小：根据类型选择对应的变体
     - ResNet: `resnet18`, `resnet34`, `resnet50`, `resnet101`
     - ViT: `vit-tiny`, `vit-base`, `vit-large`
     - YOLO: `n`, `s`, `m`, `l`, `x`
     - UNet: `small`, `medium`, `large`
     - LSTM: `small`, `medium`, `large`

3. **配置训练参数**
   - Epochs: 100
   - Batch Size: 32
   - Learning Rate: 1e-4
   - 优化器：AdamW
   - 学习率调度器：Cosine

4. **配置蒸馏策略**
   - 蒸馏类型：`hybrid`（混合蒸馏）或 `feature`（特征蒸馏）
   - 硬标签权重：0.5
   - 软标签权重：0.5
   - 特征对齐：启用

5. **点击创建任务并启动训练**

#### 方式2：直接调用Python脚本（测试用）

```bash
# 示例1：ResNet50分类任务
python back/datamark-admin/train_qwen_vl_distillation.py \
    --task_id "test_resnet50" \
    --api_base_url "http://localhost:8080" \
    --teacher_model "qwen2.5-vl-8b" \
    --student_model "resnet50" \
    --teacher_path "/data/models/qwen2.5-vl-8b" \
    --student_model_type "resnet" \
    --student_model_size "resnet50" \
    --task_type "classification" \
    --num_classes 10 \
    --dataset_id "dataset_001" \
    --image_size 224 \
    --epochs 10 \
    --batch_size 16 \
    --learning_rate 0.0001 \
    --optimizer "adamw" \
    --lr_scheduler "cosine" \
    --distillation_type "hybrid" \
    --hard_label_weight 0.5 \
    --soft_label_weight 0.5 \
    --feature_loss_type "cosine" \
    --align_feature True \
    --gpu_devices "0" \
    --output_dir "/data/training_output/test_resnet50"
```

---

## 📋 配置参数详解

### 模型配置参数

| 参数 | 类型 | 必填 | 说明 | 示例值 |
|------|------|------|------|--------|
| `--teacher_model` | str | ✅ | 教师模型名称 | `qwen2.5-vl-8b` |
| `--teacher_path` | str | ✅ | 教师模型路径 | `/data/models/qwen2.5-vl-8b` |
| `--student_model` | str | ✅ | 学生模型名称 | `resnet50` |
| `--student_model_type` | str | ✅ | 学生模型类型 | `resnet`, `vit`, `yolov8`, `unet`, `lstm` |
| `--student_model_size` | str | ✅ | 学生模型大小 | `resnet50`, `vit-base`, `s` |
| `--student_path` | str | ❌ | 学生预训练权重路径 | `/data/models/resnet50_pretrained.pth` |

### 任务配置参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--task_type` | str | classification | 任务类型：classification/detection/segmentation |
| `--num_classes` | int | 10 | 分类类别数 |
| `--image_size` | int | 224 | 输入图像大小 |

### 训练参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--epochs` | int | 100 | 训练轮数 |
| `--batch_size` | int | 32 | 批大小 |
| `--learning_rate` | float | 1e-4 | 学习率 |
| `--optimizer` | str | adamw | 优化器：adamw/adam/sgd |
| `--lr_scheduler` | str | cosine | 学习率调度器：cosine/linear/constant |
| `--weight_decay` | float | 0.01 | 权重衰减 |
| `--grad_accum_steps` | int | 1 | 梯度累积步数 |
| `--max_grad_norm` | float | 1.0 | 梯度裁剪 |

### 蒸馏配置参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--distillation_type` | str | hybrid | 蒸馏类型：feature/logit/hybrid |
| `--temperature` | float | 4.0 | 蒸馏温度 |
| `--hard_label_weight` | float | 0.5 | 硬标签权重（任务损失） |
| `--soft_label_weight` | float | 0.5 | 软标签权重（蒸馏损失） |
| `--feature_loss_type` | str | mse | 特征损失类型：mse/cosine |
| `--align_feature` | bool | True | 是否使用特征对齐层 |

### LoRA配置参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--lora_rank` | int | 0 | LoRA秩（0表示不使用LoRA） |
| `--lora_alpha` | int | 16 | LoRA缩放因子 |
| `--lora_dropout` | float | 0.1 | LoRA Dropout率 |
| `--lora_target_modules` | str | '' | 目标模块，逗号分隔 |
| `--lora_bias` | str | none | Bias训练策略：none/all/lora_only |

---

## 🎯 5种学生模型使用指南

### 1. ResNet 图像分类

**适用场景**：标准图像分类任务，平衡准确率和速度

**推荐配置**：
```bash
--student_model_type "resnet" \
--student_model_size "resnet50" \
--task_type "classification" \
--num_classes 10 \
--image_size 224 \
--batch_size 32 \
--learning_rate 1e-4 \
--distillation_type "hybrid" \
--feature_loss_type "cosine"
```

**模型变体**：
- `resnet18` - 11M参数，最快
- `resnet34` - 21M参数
- `resnet50` - 25M参数，推荐
- `resnet101` - 44M参数，最准确

---

### 2. Vision Transformer (ViT) 图像分类

**适用场景**：需要高准确率的分类任务，Transformer架构

**推荐配置**：
```bash
--student_model_type "vit" \
--student_model_size "vit-base" \
--task_type "classification" \
--num_classes 100 \
--image_size 224 \
--batch_size 64 \
--learning_rate 5e-5 \
--distillation_type "hybrid" \
--align_feature True
```

**模型变体**：
- `vit-tiny` - 5M参数，轻量级
- `vit-base` - 86M参数，推荐
- `vit-large` - 307M参数，最准确

**注意事项**：
- ViT对学习率敏感，建议使用较小的学习率（5e-5）
- 需要较大的显存，建议使用24GB+显卡

---

### 3. YOLOv8 目标检测

**适用场景**：实时目标检测，边缘设备部署

**推荐配置**：
```bash
--student_model_type "yolov8" \
--student_model_size "s" \
--task_type "detection" \
--num_classes 80 \
--image_size 640 \
--batch_size 16 \
--learning_rate 1e-3 \
--optimizer "sgd" \
--distillation_type "feature" \
--feature_loss_type "mse"
```

**模型变体**：
- `n` (nano) - 3M参数，极速
- `s` (small) - 11M参数，推荐
- `m` (medium) - 26M参数
- `l` (large) - 44M参数
- `x` (xlarge) - 68M参数，最准确

**注意事项**：
- 检测任务通常需要较大的学习率（1e-3）
- 建议使用SGD优化器
- 图像尺寸推荐640x640

---

### 4. UNet 图像分割

**适用场景**：像素级图像分割任务

**推荐配置**：
```bash
--student_model_type "unet" \
--student_model_size "medium" \
--task_type "segmentation" \
--num_classes 21 \
--image_size 512 \
--batch_size 8 \
--learning_rate 1e-4 \
--distillation_type "feature" \
--feature_loss_type "mse"
```

**模型变体**：
- `small` - 7M参数
- `medium` - 17M参数，推荐
- `large` - 31M参数

**注意事项**：
- 分割任务显存占用大，建议较小的batch_size（8-16）
- 图像尺寸推荐512x512或更大
- 目前实现的是简化版UNet，可根据需求扩展

---

### 5. LSTM 序列特征提取+分类

**适用场景**：处理时序信息，视频分类，行为识别

**推荐配置**：
```bash
--student_model_type "lstm" \
--student_model_size "medium" \
--task_type "classification" \
--num_classes 10 \
--image_size 224 \
--batch_size 32 \
--learning_rate 1e-4 \
--distillation_type "feature" \
--feature_loss_type "cosine"
```

**模型变体**：
- `small` - hidden_size=256, 10M参数
- `medium` - hidden_size=512, 25M参数，推荐
- `large` - hidden_size=1024, 50M参数

**注意事项**：
- LSTM使用ResNet50作为特征提取器
- 特别适合处理视频帧序列
- 可以结合注意力机制进一步优化

---

## 📊 蒸馏策略选择

### 特征蒸馏 (feature)

**原理**：从Qwen2.5-VL的视觉编码器提取特征，让学生模型学习教师的中间表示

**适用场景**：
- ✅ 所有模型类型
- ✅ 跨架构蒸馏（Transformer → CNN）
- ✅ 检测和分割任务

**配置**：
```bash
--distillation_type "feature" \
--feature_loss_type "mse" \
--align_feature True
```

### 混合蒸馏 (hybrid)

**原理**：结合任务损失和特征蒸馏损失

**适用场景**：
- ✅ 分类任务
- ✅ 追求最佳性能

**配置**：
```bash
--distillation_type "hybrid" \
--hard_label_weight 0.5 \
--soft_label_weight 0.5 \
--feature_loss_type "cosine" \
--align_feature True
```

---

## 🔧 后端集成

### 扩展TrainingExecutionService

现有的 `TrainingExecutionService` 已经支持大部分参数。如需支持新增的参数，可以在 `buildPythonCommand` 方法中添加：

```java
// 在TrainingExecutionService.java的buildPythonCommand方法中添加

// 学生模型类型和大小
if (task.getStudentModel() != null) {
    // 解析学生模型类型和大小
    // 例如：student_model = "resnet/resnet50"
    String[] parts = task.getStudentModel().split("/");
    if (parts.length == 2) {
        command.add("--student_model_type");
        command.add(parts[0]);  // "resnet"

        command.add("--student_model_size");
        command.add(parts[1]);  // "resnet50"
    }
}

// 任务类型
command.add("--task_type");
command.add("classification");  // 从配置读取

// 类别数
command.add("--num_classes");
command.add("10");  // 从配置读取

// 蒸馏策略
command.add("--distillation_type");
command.add("hybrid");  // 从JSON配置读取

command.add("--feature_loss_type");
command.add("cosine");  // 从JSON配置读取

command.add("--align_feature");
command.add("True");
```

### 数据库表扩展（可选）

如需在数据库中存储学生模型类型等新字段，可执行以下SQL：

```sql
ALTER TABLE md_training_task
ADD COLUMN student_model_type VARCHAR(50) COMMENT '学生模型类型：resnet/vit/yolov8/unet/lstm';

ALTER TABLE md_training_task
ADD COLUMN student_model_size VARCHAR(50) COMMENT '学生模型大小：resnet50/vit-base等';

ALTER TABLE md_training_task
ADD COLUMN task_type VARCHAR(50) DEFAULT 'classification' COMMENT '任务类型：classification/detection/segmentation';

ALTER TABLE md_training_task
ADD COLUMN num_classes INT DEFAULT 10 COMMENT '分类类别数';
```

---

## 🐛 故障排查

### Q1: Qwen2.5-VL模型加载失败

**症状**：
```
ImportError: No module named 'transformers.models.qwen2_vl'
```

**解决方案**：
```bash
# 升级transformers到最新版本
pip install --upgrade transformers>=4.37.0

# 如果仍然失败，脚本会使用模拟模式继续运行
```

### Q2: CUDA Out of Memory

**解决方案**：
```bash
# 1. 减小batch_size
--batch_size 16  # 从32减到16

# 2. 使用梯度累积
--batch_size 16 --grad_accum_steps 2

# 3. 减小图像尺寸
--image_size 192  # 从224减到192

# 4. 选择更小的学生模型
--student_model_size "resnet18"  # 而非resnet50
```

### Q3: 训练损失不下降

**排查步骤**：

1. **检查学习率**
   ```bash
   # ResNet/UNet/LSTM: 1e-4
   # ViT: 5e-5 (更敏感)
   # YOLO: 1e-3 (检测任务需要更大)
   ```

2. **检查蒸馏权重**
   ```bash
   # 初期可以增大硬标签权重
   --hard_label_weight 0.7 \
   --soft_label_weight 0.3
   ```

3. **检查数据加载**
   - 确保数据集路径正确
   - 确认数据预处理正常

### Q4: 特征维度不匹配

**症状**：
```
RuntimeError: The size of tensor a (1280) must match the size of tensor b (768)
```

**解决方案**：
```bash
# 启用特征对齐层
--align_feature True
```

---

## 📈 性能优化建议

### 1. 学生模型选择

| 任务 | 推荐模型 | 理由 |
|------|---------|------|
| **通用图像分类** | ResNet50 | 平衡准确率和速度 |
| **高准确率分类** | ViT-Base | Transformer优势 |
| **实时检测** | YOLOv8-s | 速度快 |
| **图像分割** | UNet-medium | 专门设计 |
| **视频/序列** | LSTM-medium | 处理时序 |

### 2. 超参数建议

```bash
# ResNet
--learning_rate 1e-4 --optimizer adamw --batch_size 32

# ViT
--learning_rate 5e-5 --optimizer adamw --batch_size 64 --weight_decay 0.05

# YOLO
--learning_rate 1e-3 --optimizer sgd --batch_size 16

# UNet
--learning_rate 1e-4 --optimizer adam --batch_size 8

# LSTM
--learning_rate 1e-4 --optimizer adam --batch_size 32
```

### 3. 蒸馏策略建议

| 学生模型 | 推荐策略 | 配置 |
|---------|---------|------|
| ResNet | Hybrid | `--distillation_type hybrid --hard_label_weight 0.5` |
| ViT | Hybrid | `--distillation_type hybrid --align_feature True` |
| YOLO | Feature | `--distillation_type feature --feature_loss_type mse` |
| UNet | Feature | `--distillation_type feature` |
| LSTM | Feature | `--distillation_type feature --feature_loss_type cosine` |

---

## 📝 TODO 后续工作

- [ ] **数据库集成**：修改`MultiTaskDataset`类，从数据库加载真实图像和标注
- [ ] **完善检测任务损失**：实现YOLOv8的完整检测损失函数
- [ ] **完善分割任务损失**：实现UNet的像素级分割损失
- [ ] **Qwen2.5-VL特征提取优化**：根据实际API调整视觉编码器调用
- [ ] **分布式训练支持**：集成PyTorch DDP实现多GPU训练
- [ ] **TensorBoard可视化**：添加训练过程可视化
- [ ] **模型量化**：支持训练后量化和量化感知训练

---

## 📞 技术支持

如遇到问题，请检查：

1. **Python环境**：确认所有依赖已安装
   ```bash
   python -c "import torch; import transformers; import peft; print('OK')"
   ```

2. **模型路径**：确认Qwen2.5-VL模型路径正确
   ```bash
   ls /data/models/qwen2.5-vl-8b/
   ```

3. **GPU可用性**：
   ```bash
   python -c "import torch; print(torch.cuda.is_available())"
   ```

4. **后端日志**：查看Spring Boot日志中的训练命令和输出

---

**版本**: 1.0.0
**最后更新**: 2026-01-11
**维护者**: Claude Assistant
