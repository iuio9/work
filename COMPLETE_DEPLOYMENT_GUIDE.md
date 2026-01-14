# 大小模型协同训练系统 - 完整部署和运行指南

## 📋 目录
- [系统架构](#系统架构)
- [必需组件清单](#必需组件清单)
- [环境准备](#环境准备)
- [安装步骤](#安装步骤)
- [配置说明](#配置说明)
- [运行系统](#运行系统)
- [测试流程](#测试流程)
- [常见问题](#常见问题)

---

## 系统架构

```
┌─────────────┐      ┌─────────────┐      ┌──────────────┐
│  前端 Vue3  │ ───> │ 后端 Spring │ ───> │ Python 训练  │
│             │      │    Boot     │      │    脚本      │
└─────────────┘      └─────────────┘      └──────────────┘
                            │                     │
                            ▼                     ▼
                      ┌──────────┐        ┌─────────────┐
                      │  MySQL   │        │ Qwen2.5-VL  │
                      │ 数据库   │        │ + 学生模型  │
                      └──────────┘        └─────────────┘
```

---

## 必需组件清单

### ✅ 1. 数据库
- [x] MySQL 5.7+ 或 8.0+
- [x] 数据库名：`mark`
- [x] 执行：`model_distillation_schema.sql`（创建表）
- [x] 执行：`update_dataset_id_to_string.sql`（字段更新）

### ✅ 2. 后端服务
- [x] Java 8+
- [x] Maven 3.6+
- [x] Spring Boot 应用（已有）
- [x] 配置文件：`application-test.yml`

### ✅ 3. 前端服务
- [x] Node.js 16+
- [x] npm 或 pnpm
- [x] Vue 3 + Vite（已有）

### ⚠️ 4. Python 环境
- [ ] **Python 3.8+**
- [ ] **PyTorch 2.0+**
- [ ] **依赖包**（见 `requirements.txt`）

### ⚠️ 5. 模型文件
- [ ] **教师模型：Qwen2.5-VL-8B**
  - 大小：~16GB
  - 来源：Hugging Face
  - 路径：需配置
- [ ] **学生模型（预训练权重）**
  - ResNet50：~100MB
  - ViT-Base：~300MB
  - YOLOv8s：~20MB
  - 等等

### ⚠️ 6. 数据集
- [ ] **训练数据集**
  - 格式：图像 + 标注
  - 位置：需配置
  - 示例：COCO、ImageNet 子集
- [ ] **验证数据集**
  - 格式：同上
  - 用于验证模型效果

### ⚠️ 7. 硬件
- [ ] **GPU**（强烈推荐）
  - NVIDIA GPU（支持 CUDA）
  - 显存：至少 16GB（用于 Qwen2.5-VL）
  - 建议：RTX 3090 / 4090 或 A100
- [ ] **内存**：至少 32GB RAM
- [ ] **存储**：至少 100GB 可用空间

---

## 环境准备

### 1. 安装 Python 环境

```bash
# 检查 Python 版本
python3 --version  # 应该 >= 3.8

# 创建虚拟环境（推荐）
cd /home/user/work/back/datamark-admin
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows

# 升级 pip
pip install --upgrade pip
```

### 2. 安装 Python 依赖

```bash
# 安装基础依赖
pip install -r requirements.txt

# 安装 Qwen2.5-VL 相关
pip install transformers>=4.35.0
pip install accelerate  # 用于模型加速
pip install peft        # 用于 LoRA

# 如果有 GPU，安装 CUDA 版本的 PyTorch
# 访问 https://pytorch.org/ 选择适合你的版本
# 示例（CUDA 11.8）：
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 3. 下载 Qwen2.5-VL 模型

#### 方式 A：使用 Hugging Face CLI（推荐）

```bash
# 安装 HuggingFace CLI
pip install huggingface-hub

# 登录（如果模型需要授权）
huggingface-cli login

# 下载模型（示例）
# 注意：Qwen2.5-VL 可能需要从魔搭社区下载
huggingface-cli download Qwen/Qwen2.5-VL-8B-Instruct \
  --local-dir /home/user/models/qwen2.5-vl-8b
```

#### 方式 B：从魔搭社区下载

```bash
# 安装 ModelScope
pip install modelscope

# 使用 Python 下载
python3 << EOF
from modelscope import snapshot_download
model_dir = snapshot_download('qwen/Qwen2.5-VL-8B-Instruct',
                               cache_dir='/home/user/models')
print(f'Model downloaded to: {model_dir}')
EOF
```

#### 方式 C：手动下载（备用）

1. 访问魔搭社区：https://modelscope.cn/models/qwen/Qwen2.5-VL-8B-Instruct
2. 点击"模型文件"下载所有文件
3. 放到 `/home/user/models/qwen2.5-vl-8b/` 目录

### 4. 准备数据集

#### 选项 1：使用示例数据集

```bash
# 创建数据集目录
mkdir -p /home/user/datasets/demo-classification
mkdir -p /home/user/datasets/demo-classification/train
mkdir -p /home/user/datasets/demo-classification/val

# 下载 CIFAR-10 示例（小型数据集）
python3 << EOF
from torchvision import datasets
import os

# 下载训练集
train_dataset = datasets.CIFAR10(
    root='/home/user/datasets/cifar10',
    train=True,
    download=True
)

# 下载验证集
val_dataset = datasets.CIFAR10(
    root='/home/user/datasets/cifar10',
    train=False,
    download=True
)

print("数据集下载完成！")
print(f"训练样本：{len(train_dataset)}")
print(f"验证样本：{len(val_dataset)}")
EOF
```

#### 选项 2：使用自己的数据集

数据集目录结构：
```
/home/user/datasets/my-dataset/
├── train/
│   ├── class1/
│   │   ├── img1.jpg
│   │   ├── img2.jpg
│   │   └── ...
│   ├── class2/
│   │   └── ...
│   └── ...
└── val/
    ├── class1/
    └── class2/
```

---

## 安装步骤

### 第一步：数据库初始化

```bash
cd /home/user/work/back/datamark-admin

# 1. 创建数据库（如果还没有）
mysql -u root -p << EOF
CREATE DATABASE IF NOT EXISTS mark CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
EOF

# 2. 创建表结构
mysql -u root -p mark < model_distillation_schema.sql

# 3. 执行字段更新
mysql -u root -p mark < update_dataset_id_to_string.sql

# 4. 验证
mysql -u root -p mark -e "SHOW TABLES; DESCRIBE md_training_task;"
```

### 第二步：配置后端

编辑 `src/main/resources/application-test.yml`：

```yaml
# 数据库配置
spring:
  datasource:
    url: jdbc:mysql://localhost:3306/mark?useUnicode=true&characterEncoding=utf8
    username: root
    password: 你的密码

# 蒸馏训练配置
distillation:
  # Python 解释器路径
  python:
    path: /home/user/work/back/datamark-admin/venv/bin/python3  # 虚拟环境的 Python

  # 训练脚本路径
  script:
    path: /home/user/work/back/datamark-admin/train_qwen_vl_distillation.py

  # 模型路径
  models:
    teacher:
      qwen-2.5-vl-8b: /home/user/models/qwen2.5-vl-8b  # Qwen2.5-VL 模型路径
    student:
      resnet50: torchvision  # 使用 torchvision 预训练模型
      vit-base: huggingface/vit-base-patch16-224
      yolov8s: ultralytics/yolov8s.pt

  # 数据集路径
  datasets:
    root: /home/user/datasets  # 数据集根目录

  # 输出路径
  output:
    models: /home/user/outputs/models      # 训练后的模型保存路径
    logs: /home/user/outputs/logs          # 训练日志
    checkpoints: /home/user/outputs/checkpoints  # 检查点
```

### 第三步：启动后端服务

```bash
cd /home/user/work/back/datamark-admin

# 编译
mvn clean package -DskipTests

# 启动（开发模式）
mvn spring-boot:run

# 或者直接运行 JAR
java -jar target/datamark-admin.jar

# 检查启动状态
curl http://localhost:8081/actuator/health
```

### 第四步：启动前端服务

```bash
cd /home/user/work/front/data-mark-v3

# 安装依赖（首次）
npm install
# 或 pnpm install

# 启动开发服务器
npm run dev

# 访问：http://localhost:5173
```

### 第五步：验证 Python 环境

```bash
# 激活虚拟环境
source /home/user/work/back/datamark-admin/venv/bin/activate

# 测试 PyTorch
python3 -c "import torch; print(f'PyTorch: {torch.__version__}')"

# 测试 CUDA（如果有 GPU）
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

# 测试模型加载（需要先下载 Qwen2.5-VL）
python3 << EOF
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
import torch

model_path = "/home/user/models/qwen2.5-vl-8b"
print(f"加载模型：{model_path}")

processor = AutoProcessor.from_pretrained(model_path)
model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto"
)

print("✅ 模型加载成功！")
print(f"模型参数量：{sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")
EOF
```

---

## 配置说明

### 关键配置文件位置

1. **后端配置**
   - 主配置：`back/datamark-admin/src/main/resources/application.yml`
   - 测试环境：`back/datamark-admin/src/main/resources/application-test.yml`

2. **Python 配置**
   - 依赖：`back/datamark-admin/requirements.txt`
   - 训练脚本：`back/datamark-admin/train_qwen_vl_distillation.py`

3. **前端配置**
   - Vite 配置：`front/data-mark-v3/vite.config.ts`
   - API 配置：`front/data-mark-v3/src/service/request/index.ts`

### 需要配置的路径

在 `application-test.yml` 中**必须**配置：

```yaml
distillation:
  python:
    path: /path/to/python3  # ✅ Python 解释器

  script:
    path: /path/to/train_qwen_vl_distillation.py  # ✅ 训练脚本

  models:
    teacher:
      qwen-2.5-vl-8b: /path/to/qwen2.5-vl-8b  # ✅ 教师模型路径

  datasets:
    root: /path/to/datasets  # ✅ 数据集根目录

  output:
    models: /path/to/outputs/models  # ✅ 输出模型路径
```

---

## 运行系统

### 完整启动流程

#### 1. 启动所有服务

```bash
# 终端 1：启动后端
cd /home/user/work/back/datamark-admin
mvn spring-boot:run

# 终端 2：启动前端
cd /home/user/work/front/data-mark-v3
npm run dev
```

#### 2. 访问系统

打开浏览器访问：http://localhost:5173

---

## 测试流程

### 快速测试（使用演示数据）

#### 步骤 1：创建训练任务

1. 打开浏览器：http://localhost:5173
2. 进入"大小模型协同训练"页面

3. **Tab 1：模型配置**
   - 教师模型：Qwen2.5-VL-8B
   - 学生模型：ResNet-50
   - LoRA 配置：
     - Rank: 16
     - Alpha: 32
     - Dropout: 0.05
   - 蒸馏参数：
     - Temperature: 3.0
     - Alpha: 0.7

4. **Tab 2：创建训练任务**
   - 任务名称：`测试-图像分类-ResNet50`
   - 数据集 ID：`cifar10-train`（对应你的数据集）
   - 验证数据集 ID：`cifar10-val`
   - 训练轮数：10
   - 批次大小：32
   - 学习率：0.001

5. 点击"创建训练任务"

#### 步骤 2：启动训练

1. 在"训练任务"列表中找到刚创建的任务
2. 点击"启动"按钮
3. 后端会调用 Python 脚本开始训练

#### 步骤 3：监控训练

1. 点击"监控"按钮
2. 查看训练进度、损失曲线、准确率等

#### 步骤 4：使用模型推理

1. 训练完成后，进入"已训练模型"tab
2. 找到训练好的模型
3. 点击"使用模型进行自动标注"
4. 配置推理参数
5. 提交推理任务

---

## 常见问题

### Q1: Python 找不到模块

**错误**：`ModuleNotFoundError: No module named 'transformers'`

**解决**：
```bash
# 激活虚拟环境
source /home/user/work/back/datamark-admin/venv/bin/activate

# 重新安装依赖
pip install -r requirements.txt
```

### Q2: CUDA out of memory

**错误**：`CUDA out of memory`

**解决**：
1. 减小 batch_size（例如从 32 改为 16 或 8）
2. 使用模型量化：
   ```python
   model = Qwen2VLForConditionalGeneration.from_pretrained(
       model_path,
       torch_dtype=torch.float16,  # 使用 FP16
       load_in_8bit=True,          # 或使用 8-bit 量化
       device_map="auto"
   )
   ```
3. 使用梯度累积

### Q3: 模型下载太慢

**解决**：
1. 使用国内镜像：
   ```bash
   export HF_ENDPOINT=https://hf-mirror.com
   ```
2. 或使用魔搭社区（ModelScope）
3. 或手动下载后放到指定目录

### Q4: 后端启动失败

**检查**：
1. 数据库是否运行：`mysql -u root -p -e "SELECT 1"`
2. 端口是否被占用：`lsof -i:8081`
3. Java 版本：`java -version`（应该 >= 8）
4. 查看日志：`tail -f logs/spring.log`

### Q5: 前端连接后端失败

**检查**：
1. 后端是否启动：`curl http://localhost:8081/actuator/health`
2. 跨域配置是否正确
3. API 基础路径配置：检查 `front/data-mark-v3/src/service/request/index.ts`

### Q6: 训练一直卡在 PENDING 状态

**原因**：
1. Python 脚本路径配置错误
2. Python 环境依赖缺失
3. 模型文件路径错误
4. 数据集路径不存在

**调试**：
```bash
# 查看后端日志
tail -f /home/user/work/back/datamark-admin/logs/spring.log

# 查看训练进程
ps aux | grep python | grep train

# 手动测试 Python 脚本
source venv/bin/activate
python3 train_qwen_vl_distillation.py --help
```

---

## 最小可运行配置（快速开始）

如果你想**先测试系统是否能跑通**，不需要实际训练：

### 1. 使用模拟模式

修改 `TrainingExecutionService.java`，添加模拟模式：

```java
// 在 startTraining 方法中
if (simulationMode) {
    // 模拟训练进度
    simulateTraining(task);
    return;
}
```

### 2. 使用小型数据集

- 下载 MNIST 或 CIFAR-10（很小，几百MB）
- 使用小模型：ResNet-18 而不是 Qwen2.5-VL

### 3. 测试功能流程

```bash
1. ✅ 创建任务
2. ✅ 启动任务（模拟）
3. ✅ 查看监控（模拟数据）
4. ✅ 停止/删除任务
5. ✅ Tab 切换正常
```

---

## 总结

### 已经具备的组件 ✅

- [x] 前端 Vue3 应用
- [x] 后端 Spring Boot 应用
- [x] Python 训练脚本
- [x] 数据库表结构
- [x] API 接口
- [x] 依赖配置文件

### 还需要准备的组件 ⚠️

- [ ] **Qwen2.5-VL 模型文件**（~16GB）
- [ ] **训练数据集**
- [ ] **Python 环境和依赖**
- [ ] **GPU 环境**（可选但强烈推荐）
- [ ] **配置文件更新**（路径）

### 建议的启动顺序

1. **先测试系统功能**（无需实际模型）
   - 启动数据库
   - 启动后端（模拟模式）
   - 启动前端
   - 测试 UI 和 API

2. **再准备实际训练环境**
   - 下载小模型（先用 ResNet）
   - 准备小数据集（CIFAR-10）
   - 测试训练流程

3. **最后部署完整系统**
   - 下载 Qwen2.5-VL
   - 准备大型数据集
   - 正式训练

---

**文档版本**：v1.0
**最后更新**：2026-01-14
**相关分支**：claude/training-execution-impl-01UtXwNRFMxFL54Nhfv5AN1y
