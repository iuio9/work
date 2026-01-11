# Qwen2.5-VL 蒸馏训练配置示例

本目录包含了各种学生模型的配置示例，展示如何使用Qwen2.5-VL作为教师模型进行知识蒸馏训练。

## 📁 配置文件列表

### 1. ResNet50 分类任务
**文件**: `qwen_vl_resnet50_example.json`

**适用场景**: 图像分类，平衡准确率和速度

**关键配置**:
- 学生模型: ResNet50
- 蒸馏策略: Hybrid (特征+Logits+任务损失)
- 批大小: 32
- 学习率: 1e-4
- 推荐数据集: CIFAR-10, CIFAR-100, ImageNet子集

**命令行运行**:
```bash
python train_qwen_vl_distillation.py \
    --task_id "resnet50_task" \
    --api_base_url "http://localhost:8080/api" \
    --teacher_model_path "/data/models/qwen2.5-vl-8b" \
    --student_model_type "resnet" \
    --student_model_size "resnet50" \
    --task_type "classification" \
    --num_classes 10 \
    --dataset_path "/data/datasets/cifar10/train" \
    --val_dataset_path "/data/datasets/cifar10/val" \
    --image_size 224 \
    --epochs 100 \
    --batch_size 32 \
    --learning_rate 1e-4 \
    --optimizer_type "adamw" \
    --lr_scheduler "cosine" \
    --distillation_type "hybrid" \
    --alpha 0.5 --beta 0.3 --gamma 0.2 \
    --feature_loss_type "cosine" \
    --align_feature \
    --use_amp \
    --gpu_devices "0" \
    --output_dir "/data/outputs/qwen_resnet50"
```

---

### 2. Vision Transformer 分类任务
**文件**: `qwen_vl_vit_example.json`

**适用场景**: 需要高准确率的图像分类任务，架构相似的Transformer蒸馏

**关键配置**:
- 学生模型: ViT-Base
- 蒸馏策略: Layer-wise (逐层对齐)
- 批大小: 64
- 学习率: 5e-5 (Transformer对学习率敏感)
- 推荐数据集: ImageNet子集, 细粒度分类数据集

**命令行运行**:
```bash
python train_qwen_vl_distillation.py \
    --task_id "vit_base_task" \
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
    --grad_accum_steps 2 \
    --distillation_type "layer" \
    --alpha 0.4 --beta 0.4 --gamma 0.2 \
    --feature_loss_type "mse" \
    --align_feature \
    --feature_dim 768 \
    --use_amp \
    --gpu_devices "0,1" \
    --output_dir "/data/outputs/qwen_vit_base"
```

---

### 3. YOLOv8 目标检测任务
**文件**: `qwen_vl_yolov8_example.json`

**适用场景**: 实时目标检测，边缘设备部署

**关键配置**:
- 学生模型: YOLOv8-s
- 蒸馏策略: Feature-only (特征蒸馏)
- 批大小: 16
- 学习率: 1e-3 (检测任务需要较大学习率)
- 图像大小: 640x640
- 推荐数据集: COCO, VOC, 自定义检测数据集

**命令行运行**:
```bash
python train_qwen_vl_distillation.py \
    --task_id "yolov8_task" \
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
    --weight_decay 0.0005 \
    --max_grad_norm 10.0 \
    --distillation_type "feature" \
    --gamma 1.0 \
    --feature_loss_type "mse" \
    --align_feature \
    --feature_dim 512 \
    --gpu_devices "0" \
    --output_dir "/data/outputs/qwen_yolov8s"
```

---

## 🎯 如何选择配置

### 根据任务类型选择

| 任务类型 | 推荐配置 | 理由 |
|---------|---------|------|
| **图像分类 (小规模, <50类)** | ResNet50 | 快速收敛，性能好 |
| **图像分类 (大规模, >100类)** | ViT-Base | 更强的表达能力 |
| **实时目标检测** | YOLOv8-s/n | 速度快，适合边缘设备 |
| **高精度目标检测** | YOLOv8-m/l | 准确率更高 |
| **图像分割** | UNet | 专门为分割设计 |
| **视频分类/行为识别** | LSTM | 处理时序信息 |

### 根据计算资源选择

| GPU显存 | 推荐配置 | 调整建议 |
|---------|---------|---------|
| **12GB-16GB** | ResNet50, batch=16 | 减小batch_size，使用混合精度 |
| **24GB** | ViT-Base, batch=32 | 标准配置 |
| **32GB+** | ViT-Large, batch=64 | 可以训练更大模型 |

### 根据数据集大小选择

| 数据集大小 | 训练策略 | epochs |
|-----------|---------|--------|
| **< 1000样本** | 使用预训练 + 轻量蒸馏 | 50-100 |
| **1000-10000样本** | 标准蒸馏配置 | 100-200 |
| **> 10000样本** | 完整蒸馏训练 | 200-500 |

---

## 🔧 配置文件使用方法

### 方式1: 直接使用JSON配置（需要实现配置解析器）

```python
# config_parser.py
import json
import argparse

def load_config_from_json(json_path):
    with open(json_path, 'r') as f:
        config = json.load(f)

    # 转换为命令行参数格式
    args = argparse.Namespace(
        task_id=config.get('task_name', 'default_task'),
        api_base_url=config.get('api_base_url', 'http://localhost:8080/api'),
        teacher_model_path=config['teacher_model']['model_path'],
        student_model_type=config['student_model']['type'],
        student_model_size=config['student_model']['size'],
        # ... 其他参数映射
    )
    return args

# 使用
args = load_config_from_json('config_examples/qwen_vl_resnet50_example.json')
```

### 方式2: 参考JSON，使用命令行（推荐）

直接复制上面提供的命令行示例，根据实际路径调整参数。

### 方式3: 在前端配置页面导入JSON

前端Vue页面支持JSON导入功能，可以直接加载这些配置文件：

```javascript
// 前端代码示例
const loadConfig = async (jsonFile) => {
  const response = await fetch(jsonFile);
  const config = await response.json();

  // 填充表单
  taskModel.value = {
    teacherModel: config.teacher_model.model_path,
    studentModel: `${config.student_model.type}-${config.student_model.size}`,
    epochs: config.training.epochs,
    batchSize: config.training.batch_size,
    // ... 其他字段
  };
};
```

---

## 📝 配置参数说明

### 教师模型 (teacher_model)

```json
{
  "type": "qwen2.5-vl",           // 固定值
  "model_path": "/path/to/model", // Qwen2.5-VL模型路径
  "freeze_weights": true          // 是否冻结权重（推荐true）
}
```

### 学生模型 (student_model)

```json
{
  "type": "resnet|vit|yolov8|unet|lstm",  // 模型类型
  "size": "resnet50|vit-base|s|medium",   // 模型大小
  "pretrained": true,                     // 是否使用预训练权重
  "num_classes": 10                       // 分类类别数
}
```

### 训练配置 (training)

```json
{
  "task_type": "classification|detection|segmentation",
  "epochs": 100,                   // 训练轮数
  "batch_size": 32,                // 批大小
  "image_size": 224,               // 图像尺寸
  "learning_rate": 0.0001,         // 学习率
  "optimizer": "adamw|adam|sgd",   // 优化器
  "lr_scheduler": "cosine|linear|step",  // 学习率调度器
  "weight_decay": 0.01,            // 权重衰减
  "grad_accum_steps": 1,           // 梯度累积步数
  "max_grad_norm": 1.0,            // 梯度裁剪
  "use_amp": true                  // 混合精度训练
}
```

### 蒸馏配置 (distillation)

```json
{
  "type": "feature|logit|layer|hybrid",  // 蒸馏类型
  "temperature": 4.0,                    // 蒸馏温度
  "alpha": 0.5,                          // 硬标签权重
  "beta": 0.3,                           // 软标签权重
  "gamma": 0.2,                          // 特征蒸馏权重
  "feature_loss_type": "mse|cosine",     // 特征损失类型
  "align_feature": true,                 // 使用特征对齐层
  "feature_dim": 768                     // 特征对齐维度
}
```

**注意**: alpha + beta + gamma 不一定等于1，可以根据实际效果调整。

---

## 🚀 快速测试

### 最小化配置（用于快速验证）

```bash
python train_qwen_vl_distillation.py \
    --task_id "quick_test" \
    --api_base_url "http://localhost:8080/api" \
    --teacher_model_path "/data/models/qwen2.5-vl-8b" \
    --student_model_type "resnet" \
    --student_model_size "resnet18" \
    --task_type "classification" \
    --num_classes 10 \
    --dataset_path "/data/datasets/test/train" \
    --val_dataset_path "/data/datasets/test/val" \
    --image_size 224 \
    --epochs 5 \
    --batch_size 16 \
    --learning_rate 1e-4 \
    --distillation_type "feature" \
    --gamma 1.0 \
    --gpu_devices "0" \
    --output_dir "/tmp/quick_test"
```

**用途**: 快速验证环境配置是否正确，训练流程是否可以运行。

---

## 💡 优化建议

### 提升训练速度

1. **使用混合精度训练**
   ```json
   "use_amp": true
   ```

2. **梯度累积模拟大批量**
   ```json
   "batch_size": 16,
   "grad_accum_steps": 4  // 等效batch_size=64
   ```

3. **减小图像尺寸**
   ```json
   "image_size": 192  // 从224减小到192
   ```

### 提升模型性能

1. **增加蒸馏权重**
   ```json
   "alpha": 0.3,  // 减小任务损失
   "beta": 0.4,   // 增大软标签
   "gamma": 0.3   // 增大特征蒸馏
   ```

2. **使用混合蒸馏策略**
   ```json
   "type": "hybrid"
   ```

3. **启用特征对齐**
   ```json
   "align_feature": true,
   "feature_dim": 768
   ```

### 处理过拟合

1. **增加数据增强**（修改代码）
   ```python
   transforms.RandomAffine(degrees=20, translate=(0.1, 0.1))
   transforms.RandomErasing(p=0.2)
   ```

2. **增加权重衰减**
   ```json
   "weight_decay": 0.05  // 从0.01增加到0.05
   ```

3. **降低学习率**
   ```json
   "learning_rate": 0.00005  // 从1e-4降到5e-5
   ```

---

## 🔗 相关文档

- [主文档: QWEN_VL_DISTILLATION_GUIDE.md](../QWEN_VL_DISTILLATION_GUIDE.md)
- [训练脚本: train_qwen_vl_distillation.py](../train_qwen_vl_distillation.py)

---

**最后更新**: 2026-01-11
**维护者**: Claude Assistant
