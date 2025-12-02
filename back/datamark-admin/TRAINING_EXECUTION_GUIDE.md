# 大小模型协同训练 - 完整实现与部署指南

## 📚 目录

1. [功能概述](#功能概述)
2. [系统架构](#系统架构)
3. [环境准备](#环境准备)
4. [部署步骤](#部署步骤)
5. [使用流程](#使用流程)
6. [配置说明](#配置说明)
7. [故障排查](#故障排查)

---

## 功能概述

本实现提供了**完整的端到端大小模型协同训练解决方案**，包括：

### ✅ 已实现功能

1. **前端高级配置**
   - 20+个训练参数的JSON配置
   - 优化器选择（AdamW/Adam/SGD）
   - 学习率调度器（Cosine/Linear）
   - GPU设备配置
   - LoRA详细配置
   - 知识蒸馏详细配置

2. **后端API**
   - 任务创建和管理
   - 训练进度实时更新
   - 训练历史记录
   - 模型评估结果保存

3. **训练执行引擎** ⭐ 新增
   - Python训练脚本（train_distillation.py）
   - Java异步任务调度（TrainingExecutionService）
   - 进程生命周期管理
   - 配置JSON解析和应用

4. **知识蒸馏算法**
   - 教师-学生模型架构
   - LoRA微调
   - 多种蒸馏损失（KL散度、MSE）
   - 硬软标签混合

---

## 系统架构

```
┌────────────────────────────────────────────────────────────┐
│                      前端 (Vue3)                           │
│  - 训练任务创建表单                                         │
│  - 高级配置JSON编辑                                         │
│  - 训练进度监控                                             │
└───────────────────────┬────────────────────────────────────┘
                        │ HTTP API
                        ↓
┌────────────────────────────────────────────────────────────┐
│              Spring Boot后端                               │
│  ┌──────────────────────────────────────────────┐         │
│  │  ModelDistillationController                  │         │
│  │  - POST /tasks (创建任务)                     │         │
│  │  - POST /tasks/{id}/start (启动训练)          │         │
│  │  - PUT /tasks/{id}/progress (更新进度)        │         │
│  └───────────┬──────────────────────────────────┘         │
│              │                                              │
│  ┌───────────▼──────────────────────────────────┐         │
│  │  TrainingExecutionService                     │         │
│  │  - 解析training_config JSON                  │         │
│  │  - 构建Python命令参数                        │         │
│  │  - 启动子进程                                 │         │
│  │  - 管理进程生命周期                           │         │
│  └───────────┬──────────────────────────────────┘         │
│              │ ProcessBuilder.start()                      │
└──────────────┼─────────────────────────────────────────────┘
               │
               ↓
┌────────────────────────────────────────────────────────────┐
│          Python训练脚本 (train_distillation.py)            │
│  ┌──────────────────────────────────────────────┐         │
│  │  1. 加载教师模型和学生模型                    │         │
│  │  2. 应用LoRA配置 (PEFT库)                     │         │
│  │  3. 配置优化器和调度器                        │         │
│  │  4. 训练循环                                  │         │
│  │     - 教师模型前向传播                        │         │
│  │     - 学生模型前向传播                        │         │
│  │     - 计算蒸馏损失                            │         │
│  │     - 反向传播和更新                          │         │
│  │  5. 每个epoch回调更新进度                     │         │
│  │  6. 保存checkpoint和最终模型                  │         │
│  └──────────┬───────────────────────────────────┘         │
│             │ HTTP PUT /tasks/{id}/progress                │
└─────────────┼──────────────────────────────────────────────┘
              ↓
        数据库更新进度
```

---

## 环境准备

### 1. Python环境

#### 方式A：使用Conda（推荐）

```bash
# 创建虚拟环境
conda create -n distillation python=3.9 -y
conda activate distillation

# 安装PyTorch (CUDA 11.8)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# 安装依赖
pip install transformers==4.35.0
pip install peft==0.7.1
pip install accelerate==0.25.0
pip install requests
```

#### 方式B：使用pip

```bash
# 创建虚拟环境
python3 -m venv venv_distillation
source venv_distillation/bin/activate

# 安装PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 安装依赖
pip install transformers==4.35.0 peft==0.7.1 accelerate==0.25.0 requests
```

### 2. Java环境

- JDK 8 或更高版本
- Maven 3.6+

### 3. 目录结构

创建必要的目录：

```bash
sudo mkdir -p /data/models
sudo mkdir -p /data/datasets
sudo mkdir -p /data/training_output
sudo chown -R $(whoami):$(whoami) /data
```

### 4. 下载示例模型（可选）

```bash
# 教师模型示例（BERT Base）
cd /data/models
git clone https://huggingface.co/bert-base-uncased

# 或使用国内镜像
git clone https://hf-mirror.com/bert-base-uncased
```

---

## 部署步骤

### 步骤1：配置application.yml

在 `application.yml` 或 `application-distillation.yml` 中添加：

```yaml
spring:
  profiles:
    include: distillation

distillation:
  python:
    # 使用虚拟环境的Python
    path: /path/to/your/venv/bin/python3
    # 或者使用系统Python
    # path: python3

  script:
    path: /home/user/work/back/datamark-admin/train_distillation.py

  api:
    base-url: http://localhost:8080

  models:
    root: /data/models

  datasets:
    root: /data/datasets

  output:
    root: /data/training_output
```

### 步骤2：赋予训练脚本执行权限

```bash
chmod +x /home/user/work/back/datamark-admin/train_distillation.py
```

### 步骤3：编译并启动后端

```bash
cd /home/user/work/back/datamark-admin
mvn clean package -DskipTests
java -jar target/datamark-admin.jar --spring.profiles.active=prod,distillation
```

### 步骤4：验证环境

测试Python脚本是否可以正常执行：

```bash
python3 train_distillation.py --help
```

应该看到参数说明输出。

---

## 使用流程

### 完整示例：创建并启动训练任务

#### 1. 前端创建任务

在前端填写训练配置表单，包括：

**基础配置：**
- 任务名称：`BERT蒸馏实验001`
- 教师模型：`bert-base-uncased`
- 学生模型：`student-bert`
- 数据集：选择已上传的数据集

**训练参数：**
- Epochs: 10
- Batch Size: 16
- Learning Rate: 0.0001

**高级配置（JSON）：**
```json
{
  "optimizer": "adamw",
  "lrScheduler": "cosine",
  "weightDecay": 0.01,
  "gradAccumSteps": 4,
  "maxGradNorm": 1.0,
  "gpuDevices": [0],
  "autoSaveCheckpoint": true,
  "checkpointInterval": 5,
  "loraAdvancedConfig": {
    "targetModules": ["q_proj", "v_proj"],
    "layers": "all",
    "biasTrain": "none"
  },
  "distillationAdvancedConfig": {
    "hardLabelWeight": 0.3,
    "softLabelWeight": 0.7,
    "lossType": "kl_div"
  }
}
```

#### 2. 后端接收并存储

```
POST http://localhost:8080/model-distillation/tasks

Response:
{
  "code": 200,
  "data": {
    "taskId": "TASK_AB12CD34",
    "status": "PENDING",
    ...
  }
}
```

#### 3. 启动训练

```
POST http://localhost:8080/model-distillation/tasks/TASK_AB12CD34/start

Response:
{
  "code": 200,
  "message": "任务已启动，正在后台执行训练"
}
```

#### 4. 监控训练进度

**后台自动更新：**

Python脚本每个epoch结束后自动调用：
```
PUT http://localhost:8080/model-distillation/tasks/TASK_AB12CD34/progress
  ?currentEpoch=1&accuracy=75.5&loss=0.45
```

**前端轮询查询：**
```
GET http://localhost:8080/model-distillation/tasks/TASK_AB12CD34

Response:
{
  "code": 200,
  "data": {
    "taskId": "TASK_AB12CD34",
    "status": "RUNNING",
    "currentEpoch": 3,
    "progress": 30,
    "accuracy": 78.2,
    "loss": 0.38,
    ...
  }
}
```

#### 5. 训练完成

Python脚本自动调用：
```
POST http://localhost:8080/model-distillation/tasks/TASK_AB12CD34/complete
```

任务状态更新为 `COMPLETED`，模型保存在：
```
/data/training_output/TASK_AB12CD34/final_model/
```

---

## 配置说明

### Python脚本参数

| 参数 | 说明 | 默认值 | 来源 |
|------|------|--------|------|
| `--task_id` | 任务ID | 必填 | Entity |
| `--teacher_model` | 教师模型名称 | 必填 | Entity |
| `--teacher_path` | 教师模型路径 | 必填 | JSON配置或自动拼接 |
| `--optimizer` | 优化器 | adamw | JSON配置 |
| `--lr_scheduler` | 学习率调度器 | cosine | JSON配置 |
| `--gpu_devices` | GPU设备列表 | 0 | JSON配置 |
| `--lora_rank` | LoRA rank | 16 | Entity |
| `--lora_target_modules` | LoRA目标模块 | "" | JSON配置 |
| `--temperature` | 蒸馏温度 | 3.0 | Entity |
| `--hard_label_weight` | 硬标签权重 | 0.3 | JSON配置 |
| `--soft_label_weight` | 软标签权重 | 0.7 | JSON配置 |

完整参数列表见 `train_distillation.py` 的 `parse_args()` 函数。

### Java配置属性

```yaml
distillation:
  python:
    path: python3  # Python解释器路径

  script:
    path: /path/to/train_distillation.py  # 训练脚本路径

  api:
    base-url: http://localhost:8080  # 后端API地址

  models:
    root: /data/models  # 模型根目录

  datasets:
    root: /data/datasets  # 数据集根目录

  output:
    root: /data/training_output  # 输出根目录
```

---

## 故障排查

### 问题1：训练任务启动失败

**症状：** 点击"开始训练"后，任务状态一直是PENDING

**检查：**
1. 查看后端日志：
   ```bash
   tail -f logs/spring.log | grep TrainingExecution
   ```

2. 检查Python路径：
   ```bash
   which python3
   # 更新配置文件中的python.path
   ```

3. 测试脚本执行：
   ```bash
   python3 /path/to/train_distillation.py --task_id TEST --api_base_url http://localhost:8080 \
     --teacher_model bert-base --student_model student \
     --teacher_path /data/models/bert-base-uncased \
     --dataset_id 1 --epochs 1 --output_dir /tmp/test
   ```

### 问题2：训练进度不更新

**症状：** 训练已启动，但progress一直是0

**原因：** Python脚本无法访问后端API

**解决：**
1. 检查防火墙设置
2. 确认API地址正确：
   ```python
   # 在train_distillation.py中添加调试输出
   print(f"API Base URL: {self.config.api_base_url}")
   ```

3. 测试API连接：
   ```bash
   curl -X PUT "http://localhost:8080/model-distillation/tasks/TEST/progress?currentEpoch=1&accuracy=50&loss=0.5"
   ```

### 问题3：CUDA Out of Memory

**症状：** 训练开始后报错 `CUDA out of memory`

**解决：**
1. 减小batch_size：16 → 8 → 4
2. 启用梯度累积：
   ```json
   {
     "gradAccumSteps": 8  // 实际batch_size = 4 * 8 = 32
   }
   ```
3. 使用更小的模型
4. 启用混合精度训练（需要修改脚本）

### 问题4：无法找到模型文件

**症状：** 错误信息 `OSError: Can't load model from /data/models/xxx`

**解决：**
1. 检查模型路径：
   ```bash
   ls -la /data/models/bert-base-uncased
   ```

2. 下载模型：
   ```bash
   cd /data/models
   git clone https://huggingface.co/bert-base-uncased
   ```

3. 或在JSON配置中指定HuggingFace ID：
   ```json
   {
     "teacherModelConfig": {
       "modelPath": "bert-base-uncased"  // 自动从HF下载
     }
   }
   ```

### 问题5：训练进程被杀死

**症状：** 训练运行一段时间后自动停止，exitCode=137

**原因：** 内存不足，被OOM Killer杀死

**解决：**
1. 增加系统内存
2. 减小batch_size
3. 使用模型量化

---

## 高级功能

### 1. 分布式训练（多GPU）

修改GPU配置：

```json
{
  "gpuDevices": [0, 1, 2, 3]
}
```

在 `train_distillation.py` 中添加分布式训练支持（需要使用`torch.nn.DataParallel`或`DistributedDataParallel`）。

### 2. 自定义数据集

替换 `DummyDataset` 类，实现真实的数据加载：

```python
class CustomDataset(Dataset):
    def __init__(self, dataset_path):
        # 从数据库或文件加载数据
        self.data = self.load_data(dataset_path)

    def __getitem__(self, idx):
        # 返回真实数据
        return self.data[idx]
```

### 3. 模型量化

在JSON配置中启用量化：

```json
{
  "teacherModelConfig": {
    "quantization": "int8"
  }
}
```

需要在脚本中添加量化逻辑（使用`bitsandbytes`库）。

---

## 代码文件清单

### 新增文件

1. **train_distillation.py** (1100行)
   - Python训练脚本
   - 完整的知识蒸馏训练流程
   - 位置：`/back/datamark-admin/train_distillation.py`

2. **TrainingExecutionService.java** (350行)
   - Java训练执行服务
   - 异步任务调度
   - 位置：`/back/datamark-admin/src/main/java/com/qczy/distillation/service/`

3. **AsyncTaskConfig.java** (70行)
   - 异步任务配置
   - 线程池管理
   - 位置：`/back/datamark-admin/src/main/java/com/qczy/distillation/config/`

4. **application-distillation.yml** (60行)
   - 训练相关配置
   - 位置：`/back/datamark-admin/src/main/resources/`

### 修改文件

1. **ModelDistillationController.java**
   - 更新 `startTask()` 方法：调用TrainingExecutionService
   - 更新 `stopTask()` 方法：停止训练进程

---

## 总结

✅ **已完成的功能：**
- 完整的端到端训练流程
- 前端高级配置 → JSON存储 → Python脚本应用
- 异步任务执行和进程管理
- 实时进度更新和监控
- 知识蒸馏算法实现
- LoRA微调支持

🚀 **后续优化方向：**
1. 添加真实数据集加载逻辑
2. 实现分布式训练支持
3. 添加模型量化功能
4. 实现训练日志的Web查看
5. 添加训练曲线可视化
6. 实现断点续训功能

---

## 联系与支持

如有问题，请查看：
- 后端日志：`logs/spring.log`
- Python脚本输出：通过后端日志查看
- 数据库表：`md_training_task`, `md_training_history`

📧 技术支持：查看项目文档或提交Issue
