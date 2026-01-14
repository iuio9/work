# Python训练环境集成指南

## 📋 目录

1. [系统架构](#系统架构)
2. [Python环境配置](#python环境配置)
3. [训练流程](#训练流程)
4. [配置说明](#配置说明)
5. [常见问题](#常见问题)

---

## 🏗️ 系统架构

### 整体架构图

```
┌─────────────┐         ┌──────────────────┐         ┌─────────────────┐
│             │         │                  │         │                 │
│  Vue 3前端  │ ──HTTP─▶│  Spring Boot后端 │ ──启动─▶│  Python训练进程 │
│             │         │                  │         │                 │
└─────────────┘         └──────────────────┘         └─────────────────┘
      │                         │                             │
      │                         │                             │
      │                    操作数据库                      读写模型/数据
      │                         │                             │
      ▼                         ▼                             ▼
 WebSocket实时          ┌─────────────┐              ┌──────────────┐
   推送进度             │   MySQL     │              │  文件系统    │
                        └─────────────┘              │ /home/user/  │
                                                     └──────────────┘
```

### 核心组件

1. **前端（Vue 3）**
   - 用户界面：创建训练任务、配置参数
   - 实时监控：显示训练进度、Loss、Accuracy
   - 文件路径：`/home/user/work/front/data-mark-v3/`

2. **后端（Spring Boot）**
   - API接口：接收前端请求
   - 训练管理：`DistillationTrainingManager.java`
   - 进程控制：启动/停止/暂停Python进程
   - 文件路径：`/home/user/work/back/datamark-admin/`

3. **Python训练脚本**
   - 实际训练逻辑：知识蒸馏算法
   - 模型加载：教师模型、学生模型
   - 数据处理：数据集加载、增强
   - 文件路径：`/home/user/work/back/datamark-admin/python_scripts/`

---

## 🐍 Python环境配置

### 方式1: 使用系统Python（简单但不推荐）

```bash
# 检查Python版本
python3 --version  # 需要 >= 3.8

# 全局安装依赖（需要root权限）
sudo pip3 install torch torchvision transformers peft ultralytics

# 配置文件：application-distillation.yml
training:
  python:
    executable: python3
```

**优点**: 配置简单
**缺点**:
- 需要root权限
- 可能与系统其他Python项目冲突
- 难以管理依赖版本

---

### 方式2: 使用Python虚拟环境（推荐⭐）

#### 步骤1: 创建虚拟环境

```bash
cd /home/user/work/back/datamark-admin

# 创建虚拟环境
python3 -m venv venv

# 查看创建结果
ls -la venv/
# 应该看到: bin/  include/  lib/  pyvenv.cfg
```

#### 步骤2: 激活虚拟环境并安装依赖

```bash
# 激活虚拟环境
source venv/bin/activate

# 你会看到命令行前缀变为: (venv)

# 升级pip
pip install --upgrade pip

# 安装PyTorch（CUDA版本，根据你的CUDA版本选择）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 安装其他依赖
pip install transformers>=4.37.0
pip install peft>=0.7.0
pip install ultralytics>=8.0.0
pip install pillow
pip install numpy
pip install matplotlib

# 保存依赖列表（方便其他人安装）
pip freeze > requirements.txt

# 退出虚拟环境
deactivate
```

#### 步骤3: 配置Spring Boot使用虚拟环境

编辑 `application-dev.yml` 或 `application.yml`：

```yaml
training:
  python:
    # 方式A: 直接指定虚拟环境中的python
    executable: /home/user/work/back/datamark-admin/venv/bin/python

    # 方式B: 配置虚拟环境路径，系统自动使用venv/bin/python
    venv-path: /home/user/work/back/datamark-admin/venv
```

**优点**:
- ✅ 环境隔离，不影响系统Python
- ✅ 依赖管理清晰
- ✅ 可以精确控制版本
- ✅ 不需要root权限

**这是生产环境推荐的方式！**

---

### 方式3: 使用Conda环境

```bash
# 创建conda环境
conda create -n distillation python=3.10

# 激活环境
conda activate distillation

# 安装依赖
conda install pytorch torchvision -c pytorch
conda install transformers -c huggingface
pip install peft ultralytics

# 查看python路径
which python
# 例如: /home/user/anaconda3/envs/distillation/bin/python

# 配置
training:
  python:
    executable: /home/user/anaconda3/envs/distillation/bin/python
```

---

## 🚀 训练流程

### 完整流程图

```
1. 前端：用户点击"开始训练"
         ↓
2. 前端：发送POST请求到后端
   POST /api/distillation/training/start
   {
     "teacherModel": "qwen2.5-vl-3b",
     "studentModel": "resnet18",
     "dataset": "cifar10",
     "batchSize": 32,
     ...
   }
         ↓
3. 后端：Controller接收请求
   DistillationTrainingController.startTraining()
         ↓
4. 后端：Service处理业务逻辑
   - 验证参数
   - 保存任务到数据库（状态：PENDING）
   - 调用TrainingManager
         ↓
5. 后端：DistillationTrainingManager启动Python进程
   - 构建命令: python3 distillation_train.py --task-id xxx ...
   - ProcessBuilder.start()
   - 异步读取输出
   - 更新任务状态：RUNNING
         ↓
6. Python：训练脚本执行
   - 加载教师模型（Qwen2.5-VL）
   - 加载学生模型（ResNet18）
   - 加载数据集（CIFAR-10）
   - 开始训练循环
   - 每个epoch输出：Epoch 1/50, Loss: 2.345, Acc: 65.2%
         ↓
7. 后端：实时读取Python输出
   - 解析训练指标
   - 更新数据库
   - 通过WebSocket推送给前端
         ↓
8. 前端：实时显示
   - 更新进度条
   - 绘制Loss/Accuracy曲线
   - 显示当前epoch信息
         ↓
9. Python：训练完成
   - 保存学生模型
   - 输出最终指标
   - 退出进程（exit code 0）
         ↓
10. 后端：检测到进程结束
    - 更新任务状态：COMPLETED
    - 保存最终结果
    - 通知前端
```

### 关键代码位置

#### 1. Java后端调用Python

**文件**: `DistillationTrainingManager.java`

```java
public boolean startTraining(
    String taskId,
    String teacherModel,
    String studentModel,
    ...
) {
    // 构建命令
    ProcessBuilder pb = new ProcessBuilder(
        pythonExecutable,           // "/path/to/venv/bin/python"
        scriptPath,                 // "python_scripts/distillation_train.py"
        "--task-id", taskId,
        "--teacher-model", teacherModel,
        "--student-model", studentModel,
        ...
    );

    // 启动进程
    Process process = pb.start();

    // 异步读取输出
    new Thread(() -> {
        BufferedReader reader = new BufferedReader(
            new InputStreamReader(process.getInputStream())
        );
        String line;
        while ((line = reader.readLine()) != null) {
            log.info("[训练 {}]: {}", taskId, line);
            // TODO: 解析并推送到前端
        }
    }).start();

    return true;
}
```

#### 2. Python训练脚本

**文件**: `python_scripts/distillation_train.py`

```python
import argparse
import torch

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--task-id', required=True)
    parser.add_argument('--teacher-model', required=True)
    parser.add_argument('--student-model', required=True)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--batch-size', type=int, default=32)
    # ... 更多参数
    args = parser.parse_args()

    # 加载模型和数据
    teacher = load_teacher_model(args.teacher_model)
    student = load_student_model(args.student_model)
    train_loader, test_loader = load_dataset(args.dataset)

    # 训练循环
    for epoch in range(args.epochs):
        loss, acc = train_one_epoch(teacher, student, train_loader)

        # 输出到stdout（Java会读取）
        print(f"Epoch {epoch+1}/{args.epochs}, Loss: {loss:.4f}, Acc: {acc:.2f}%")

    # 保存模型
    torch.save(student.state_dict(), f"models/{args.task_id}_student.pth")
    print("Training completed!")

if __name__ == '__main__':
    main()
```

---

## ⚙️ 配置说明

### 完整配置文件示例

**文件**: `application-distillation.yml`

```yaml
training:
  python:
    # Python可执行文件
    executable: python3
    # 或者使用虚拟环境
    venv-path: /home/user/work/back/datamark-admin/venv

    # 脚本目录
    script-dir: /home/user/work/back/datamark-admin/python_scripts
    distillation-script: distillation_train.py

  models:
    # 教师模型路径
    teacher-model-path: /home/user/models/qwen2.5-vl-3b-instruct
    # 学生模型保存路径
    student-model-path: /home/user/work/models/students

  datasets:
    # 数据集根目录
    root-path: /home/user/datasets

  gpu:
    enabled: true
    default-device: 0  # 使用第一个GPU

  hyperparameters:
    default-batch-size: 32
    default-epochs: 50
    default-learning-rate: 0.001
    default-temperature: 4.0
    default-alpha: 0.7
```

### 配置优先级

1. **Python路径**:
   ```
   venv-path (如果配置) > executable
   ```

2. **数据集路径**:
   ```
   具体数据集路径 (cifar10-path) > root-path/dataset-name
   ```

3. **超参数**:
   ```
   前端传入的参数 > 配置文件默认值
   ```

---

## ❓ 常见问题

### Q1: 如何验证Python环境配置正确？

```bash
# 方式1: 直接运行Python脚本
cd /home/user/work/back/datamark-admin
source venv/bin/activate  # 如果使用虚拟环境
python3 python_scripts/distillation_train.py --help

# 方式2: 检查Java能否调用Python
java -cp target/classes com.qczy.distillation.manager.DistillationTrainingManager
```

### Q2: Java无法找到Python脚本

**错误**: `训练脚本不存在: /path/to/script.py`

**解决**:
```bash
# 检查脚本是否存在
ls -l /home/user/work/back/datamark-admin/python_scripts/

# 检查权限
chmod +x python_scripts/distillation_train.py

# 确认配置路径正确
grep "script-dir" src/main/resources/application-distillation.yml
```

### Q3: Python进程启动失败

**检查步骤**:
```bash
# 1. 手动运行Python命令
/home/user/work/venv/bin/python \
  python_scripts/distillation_train.py \
  --task-id test \
  --teacher-model qwen2.5-vl-3b \
  --student-model resnet18 \
  --dataset cifar10

# 2. 检查Python依赖
source venv/bin/activate
pip list | grep torch
pip list | grep transformers

# 3. 检查Java日志
tail -f logs/spring.log | grep "训练"
```

### Q4: GPU不可用

**检查**:
```bash
# 1. 检查CUDA
nvidia-smi

# 2. 检查PyTorch能否识别GPU
python3 -c "import torch; print(torch.cuda.is_available())"

# 3. 如果返回False，重新安装CUDA版PyTorch
source venv/bin/activate
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Q5: 如何查看训练日志？

**方式1**: Spring Boot日志
```bash
tail -f logs/spring.log | grep "训练任务"
```

**方式2**: 训练专用日志文件
```bash
tail -f /home/user/work/logs/training/task-xxx.log
```

**方式3**: 前端实时查看
- 打开浏览器，进入训练任务详情页
- 点击"查看日志"标签

---

## 🎯 快速开始

### 5分钟完成配置

```bash
# 1. 创建虚拟环境
cd /home/user/work/back/datamark-admin
python3 -m venv venv
source venv/bin/activate

# 2. 安装依赖
pip install torch torchvision transformers peft ultralytics
pip freeze > requirements.txt

# 3. 配置Spring Boot
cat >> src/main/resources/application-dev.yml << 'EOF'

training:
  python:
    venv-path: /home/user/work/back/datamark-admin/venv
    script-dir: /home/user/work/back/datamark-admin/python_scripts
  models:
    teacher-model-path: /home/user/models/qwen2.5-vl-3b-instruct
    student-model-path: /home/user/work/models/students
  datasets:
    root-path: /home/user/datasets
EOF

# 4. 创建必要目录
mkdir -p python_scripts
mkdir -p /home/user/work/models/students
mkdir -p /home/user/datasets

# 5. 复制训练脚本（从测试脚本修改）
cp test_distillation_qwen.py python_scripts/distillation_train.py

# 6. 启动后端
mvn spring-boot:run

# 7. 打开前端，创建训练任务
```

---

## 📚 相关文档

- [测试脚本使用指南](README_TESTING.md)
- [Qwen2.5-VL蒸馏指南](QWEN_DISTILLATION_GUIDE.md)
- [完整部署指南](../../COMPLETE_DEPLOYMENT_GUIDE.md)

---

**现在你已经了解了整个Python集成架构！** 🎉
