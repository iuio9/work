# Qwen2.5-VL多模型训练系统 - 实现细节与集成指南

## 📋 目录
1. [整体架构设计](#整体架构设计)
2. [完整数据流程](#完整数据流程)
3. [设计思路详解](#设计思路详解)
4. [后端修改方案](#后端修改方案)
5. [前端修改方案](#前端修改方案)
6. [集成步骤](#集成步骤)
7. [测试验证](#测试验证)

---

## 🏗️ 整体架构设计

### 现有系统架构（已实现）

```
┌─────────────────────────────────────────────────────────────┐
│                    前端 Vue3                                 │
│  /views/model-distillation/index.vue                        │
│  - 创建训练任务表单                                          │
│  - 基础配置：教师模型、学生模型、数据集                      │
│  - 高级配置：优化器、学习率、LoRA、蒸馏参数                 │
│  - JSON配置编辑器                                           │
└────────────────────────┬────────────────────────────────────┘
                         │ POST /model-distillation/tasks
                         ↓
┌─────────────────────────────────────────────────────────────┐
│              Spring Boot 后端                                │
│  ┌────────────────────────────────────────────────┐         │
│  │ ModelDistillationController                     │         │
│  │ - createTask(@RequestBody CreateTaskRequestDTO)│         │
│  │ - startTask(@PathVariable taskId)              │         │
│  │ - stopTask(@PathVariable taskId)               │         │
│  │ - updateProgress(@PathVariable taskId)         │         │
│  └───────────┬────────────────────────────────────┘         │
│              │                                               │
│  ┌───────────▼────────────────────────────────────┐         │
│  │ MdTrainingTaskService                           │         │
│  │ - createTask(entity)                            │         │
│  │ - startTask(taskId)                             │         │
│  └───────────┬────────────────────────────────────┘         │
│              │                                               │
│  ┌───────────▼────────────────────────────────────┐         │
│  │ TrainingExecutionService                        │         │
│  │ - @Async startTrainingAsync(taskId)             │         │
│  │ - stopTraining(taskId)                          │         │
│  │ - buildPythonCommand(task, config)              │         │
│  └───────────┬────────────────────────────────────┘         │
│              │ ProcessBuilder.start()                        │
└──────────────┼───────────────────────────────────────────────┘
               │
               ↓
┌─────────────────────────────────────────────────────────────┐
│        Python训练脚本                                        │
│  train_distillation.py (现有)                               │
│  - 支持图像分类任务                                         │
│  - AutoModelForImageClassification                          │
│  - LoRA微调                                                 │
│  - 知识蒸馏                                                 │
└─────────────────────────────────────────────────────────────┘
```

### 新增功能架构（本次实现）

```
┌─────────────────────────────────────────────────────────────┐
│              前端 Vue3 (需要扩展)                            │
│  /views/model-distillation/index.vue                        │
│  【新增】学生模型选择区域：                                  │
│    ┌──────────────────────────────────────┐                │
│    │ 学生模型类型: [下拉框]               │                │
│    │  - ResNet (图像分类)                 │                │
│    │  - Vision Transformer (图像分类)     │                │
│    │  - YOLOv8 (目标检测)                 │                │
│    │  - UNet (图像分割)                   │                │
│    │  - LSTM (序列分类)                   │                │
│    ├──────────────────────────────────────┤                │
│    │ 模型大小: [下拉框，根据类型动态变化]  │                │
│    │  ResNet: resnet18/34/50/101          │                │
│    │  ViT: vit-tiny/base/large            │                │
│    │  YOLO: n/s/m/l/x                     │                │
│    │  UNet: small/medium/large            │                │
│    │  LSTM: small/medium/large            │                │
│    └──────────────────────────────────────┘                │
│  【新增】任务类型选择：                                      │
│    - 图像分类 (classification)                             │
│    - 目标检测 (detection)                                  │
│    - 图像分割 (segmentation)                               │
│  【新增】蒸馏策略选择：                                      │
│    - 特征蒸馏 (feature)                                    │
│    - 混合蒸馏 (hybrid)                                     │
└────────────────────────┬────────────────────────────────────┘
                         │ POST /model-distillation/tasks
                         ↓
┌─────────────────────────────────────────────────────────────┐
│       Spring Boot 后端 (需要扩展)                            │
│  ┌────────────────────────────────────────────────┐         │
│  │ ModelDistillationController                     │         │
│  │ 【扩展】createTask 方法：                       │         │
│  │   - 接收新字段：studentModelType, studentModelSize │     │
│  │   - 接收新字段：taskType, numClasses           │         │
│  │   - 接收新字段：distillationType                │         │
│  └───────────┬────────────────────────────────────┘         │
│              │                                               │
│  ┌───────────▼────────────────────────────────────┐         │
│  │ MdTrainingTaskEntity (需要扩展)                 │         │
│  │ 【新增字段】                                     │         │
│  │ - studentModelType (resnet/vit/yolov8/...)     │         │
│  │ - studentModelSize (resnet50/vit-base/...)     │         │
│  │ - taskType (classification/detection/seg)      │         │
│  │ - numClasses (类别数)                           │         │
│  │ 【或者】存在training_config JSON中               │         │
│  └───────────┬────────────────────────────────────┘         │
│              │                                               │
│  ┌───────────▼────────────────────────────────────┐         │
│  │ TrainingExecutionService                        │         │
│  │ 【扩展】buildPythonCommand 方法：               │         │
│  │   1. 检测教师模型类型                           │         │
│  │   2. 如果是Qwen2.5-VL，使用新脚本               │         │
│  │   3. 添加新参数：student_model_type/size等      │         │
│  └───────────┬────────────────────────────────────┘         │
│              │ 根据模型类型选择脚本                          │
│              ├─ train_distillation.py (现有)                 │
│              └─ train_qwen_vl_distillation.py (新增)         │
└──────────────┼───────────────────────────────────────────────┘
               │
               ↓
┌─────────────────────────────────────────────────────────────┐
│   Python训练脚本: train_qwen_vl_distillation.py (新增)      │
│   - 支持Qwen2.5-VL作为教师模型                              │
│   - 支持5种学生模型架构                                     │
│   - 灵活的蒸馏策略                                          │
└─────────────────────────────────────────────────────────────┘
```

---

## 🔄 完整数据流程

### 流程1：用户创建训练任务

```
用户操作                   前端                    后端                    数据库
   │                        │                       │                       │
   │ 1. 填写训练表单         │                       │                       │
   │ ─────────────────────>│                       │                       │
   │   - 教师模型: qwen2.5-vl-8b                    │                       │
   │   - 学生模型类型: resnet                       │                       │
   │   - 学生模型大小: resnet50                     │                       │
   │   - 任务类型: classification                   │                       │
   │   - 类别数: 10                                 │                       │
   │   - Epochs: 100                                │                       │
   │   - 蒸馏策略: hybrid                           │                       │
   │                        │                       │                       │
   │ 2. 点击"创建任务"       │                       │                       │
   │ ─────────────────────>│                       │                       │
   │                        │                       │                       │
   │                        │ 3. 构建请求数据       │                       │
   │                        │    {                  │                       │
   │                        │      taskName: "...", │                       │
   │                        │      teacherModel: "qwen2.5-vl-8b",           │
   │                        │      studentModel: "resnet/resnet50", ←─ 拼接 │
   │                        │      studentModelType: "resnet",              │
   │                        │      studentModelSize: "resnet50",            │
   │                        │      taskType: "classification",              │
   │                        │      numClasses: 10,                          │
   │                        │      epochs: 100,                             │
   │                        │      // ... 其他参数                          │
   │                        │      distillationType: "hybrid",              │
   │                        │      featureLossType: "cosine",               │
   │                        │      alignFeature: true                       │
   │                        │    }                  │                       │
   │                        │                       │                       │
   │                        │ 4. POST /model-distillation/tasks              │
   │                        │ ──────────────────────>│                       │
   │                        │                       │                       │
   │                        │                       │ 5. 接收请求            │
   │                        │                       │    (CreateTaskRequestDTO)
   │                        │                       │                       │
   │                        │                       │ 6. 拆分数据：          │
   │                        │                       │    基础字段 → Entity   │
   │                        │                       │    高级配置 → JSON     │
   │                        │                       │    {                  │
   │                        │                       │      "studentModelType": "resnet",
   │                        │                       │      "studentModelSize": "resnet50",
   │                        │                       │      "taskType": "classification",
   │                        │                       │      "numClasses": 10,           │
   │                        │                       │      "distillationType": "hybrid",
   │                        │                       │      ...               │
   │                        │                       │    }                  │
   │                        │                       │                       │
   │                        │                       │ 7. 保存到数据库        │
   │                        │                       │ ──────────────────────>│
   │                        │                       │                       │
   │                        │                       │                       │ INSERT
   │                        │                       │                       │ md_training_task
   │                        │                       │                       │ - task_id
   │                        │                       │                       │ - task_name
   │                        │                       │                       │ - teacher_model
   │                        │                       │                       │ - student_model
   │                        │                       │                       │ - training_config (JSON)
   │                        │                       │                       │ - status: 'PENDING'
   │                        │                       │                       │
   │                        │                       │<───────────────────────│
   │                        │                       │                       │
   │                        │ 8. 返回任务ID          │                       │
   │                        │<───────────────────────│                       │
   │<───────────────────────│                       │                       │
   │ 显示"任务创建成功"      │                       │                       │
```

### 流程2：启动训练任务

```
用户操作                   前端                    后端                    Python脚本
   │                        │                       │                       │
   │ 1. 点击"启动训练"       │                       │                       │
   │ ─────────────────────>│                       │                       │
   │                        │                       │                       │
   │                        │ 2. POST /tasks/{taskId}/start                 │
   │                        │ ──────────────────────>│                       │
   │                        │                       │                       │
   │                        │                       │ 3. 调用startTask       │
   │                        │                       │    - 更新status='RUNNING'
   │                        │                       │    - 调用TrainingExecutionService
   │                        │                       │                       │
   │                        │                       │ 4. @Async startTrainingAsync
   │                        │                       │    (异步执行)          │
   │                        │                       │                       │
   │                        │                       │ 5. 从DB读取任务配置    │
   │                        │                       │    - 基础字段          │
   │                        │                       │    - training_config JSON
   │                        │                       │                       │
   │                        │                       │ 6. 解析JSON配置        │
   │                        │                       │    TrainingConfigDTO config =
   │                        │                       │      JSON.parseObject(...);
   │                        │                       │                       │
   │                        │                       │ 7. 判断教师模型类型    │
   │                        │                       │    if (teacher.contains("qwen")) {
   │                        │                       │      scriptPath = train_qwen_vl_distillation.py
   │                        │                       │    } else {            │
   │                        │                       │      scriptPath = train_distillation.py
   │                        │                       │    }                   │
   │                        │                       │                       │
   │                        │                       │ 8. 构建Python命令      │
   │                        │                       │    List<String> cmd = [
   │                        │                       │      "python3",        │
   │                        │                       │      "/path/to/train_qwen_vl_distillation.py",
   │                        │                       │      "--task_id", "task_001",
   │                        │                       │      "--teacher_model", "qwen2.5-vl-8b",
   │                        │                       │      "--student_model", "resnet50",
   │                        │                       │      "--student_model_type", "resnet",
   │                        │                       │      "--student_model_size", "resnet50",
   │                        │                       │      "--task_type", "classification",
   │                        │                       │      "--num_classes", "10",
   │                        │                       │      "--epochs", "100",
   │                        │                       │      "--batch_size", "32",
   │                        │                       │      "--distillation_type", "hybrid",
   │                        │                       │      "--feature_loss_type", "cosine",
   │                        │                       │      "--align_feature", "True",
   │                        │                       │      ...               │
   │                        │                       │    ]                   │
   │                        │                       │                       │
   │                        │                       │ 9. 启动Python进程      │
   │                        │                       │    ProcessBuilder pb = new ProcessBuilder(cmd);
   │                        │                       │    Process process = pb.start();
   │                        │                       │    runningProcesses.put(taskId, process);
   │                        │                       │                       │
   │                        │                       │ ──────────────────────>│
   │                        │                       │                       │
   │                        │                       │                       │ 10. Python脚本启动
   │                        │                       │                       │     - 解析命令行参数
   │                        │                       │                       │     - 加载Qwen2.5-VL
   │                        │                       │                       │     - 加载学生模型
   │                        │                       │                       │     - 创建训练器
   │                        │                       │                       │
   │                        │ 11. 返回"训练已启动"   │                       │
   │                        │<───────────────────────│                       │
   │<───────────────────────│                       │                       │
   │ 显示"训练中..."         │                       │                       │
```

### 流程3：训练进度更新

```
Python脚本                后端API                 数据库                  前端
   │                        │                       │                       │
   │ 每个Epoch结束后：       │                       │                       │
   │                        │                       │                       │
   │ 1. 计算训练指标         │                       │                       │
   │    - train_loss        │                       │                       │
   │    - val_loss          │                       │                       │
   │    - val_accuracy      │                       │                       │
   │                        │                       │                       │
   │ 2. HTTP回调            │                       │                       │
   │    PUT /model-distillation/tasks/{taskId}/progress                     │
   │ ──────────────────────>│                       │                       │
   │    {                   │                       │                       │
   │      currentEpoch: 10, │                       │                       │
   │      totalEpochs: 100, │                       │                       │
   │      trainLoss: 0.234, │                       │                       │
   │      valLoss: 0.456,   │                       │                       │
   │      valAccuracy: 92.3,│                       │                       │
   │      status: 'RUNNING' │                       │                       │
   │    }                   │                       │                       │
   │                        │                       │                       │
   │                        │ 3. 更新数据库          │                       │
   │                        │ ──────────────────────>│                       │
   │                        │                       │                       │
   │                        │                       │ UPDATE md_training_task
   │                        │                       │ SET current_epoch = 10,
   │                        │                       │     train_loss = 0.234,
   │                        │                       │     val_accuracy = 92.3
   │                        │                       │                       │
   │                        │ 4. 返回成功            │                       │
   │<───────────────────────│                       │                       │
   │                        │                       │                       │
   │ 5. 继续训练下一个Epoch  │                       │                       │
   │                        │                       │                       │
   │                        │                       │                       │
   │                        │                       │                       │ 前端定时轮询
   │                        │                       │                       │ (每5秒一次)
   │                        │                       │                       │
   │                        │                       │                       │ GET /tasks/{taskId}
   │                        │                       │<───────────────────────│
   │                        │<───────────────────────│                       │
   │                        │                       │                       │
   │                        │ 返回任务详情（包括进度）                        │
   │                        │ ──────────────────────────────────────────────>│
   │                        │                       │                       │
   │                        │                       │                       │ 更新进度条
   │                        │                       │                       │ Epoch: 10/100
   │                        │                       │                       │ 准确率: 92.3%
```

---

## 💡 设计思路详解

### 核心设计原则

#### 1. 向后兼容性 ✅
**现有功能不受影响**：
- 现有的 `train_distillation.py` 继续支持原有的图像分类任务
- 新增的 `train_qwen_vl_distillation.py` 作为补充
- 通过教师模型类型自动选择使用哪个脚本

**实现方式**：
```java
// TrainingExecutionService.java
private String getTrainingScript(String teacherModel) {
    if (teacherModel != null &&
        (teacherModel.contains("qwen") || teacherModel.contains("Qwen"))) {
        // 使用新脚本
        return qwenScriptPath;
    } else {
        // 使用原有脚本
        return scriptPath;
    }
}
```

#### 2. 最小侵入性 ✅
**数据库表不强制修改**：
- 优先使用现有的 `training_config` JSON字段存储新参数
- 如果需要频繁查询，可以选择性添加新字段
- 新字段可以为NULL，不影响旧数据

**实现方式**：
```json
// training_config JSON中添加新字段
{
  "optimizer": "adamw",
  "lrScheduler": "cosine",
  // ... 原有字段 ...

  // 新增字段
  "studentModelType": "resnet",
  "studentModelSize": "resnet50",
  "taskType": "classification",
  "numClasses": 10,
  "distillationType": "hybrid",
  "featureLossType": "cosine",
  "alignFeature": true
}
```

#### 3. 灵活扩展性 ✅
**易于添加新模型**：
- 在 `StudentModelLoader` 中添加新的 `_load_xxx` 方法
- 前端下拉框添加新选项
- 后端无需修改，自动传递参数

**示例**：要添加新的模型架构（如EfficientNet）：
```python
# 1. 在StudentModelLoader中添加
@staticmethod
def _load_efficientnet(size: str, num_classes: int, device):
    model = models.efficientnet_b0(pretrained=True)
    # ... 配置
    return model

# 2. 在load_model中添加分支
elif model_type == 'efficientnet':
    return StudentModelLoader._load_efficientnet(...)

# 3. 前端添加选项（就这么简单！）
```

#### 4. 配置驱动 ✅
**所有参数通过配置传递**：
- 不硬编码任何路径或参数
- 通过 `application-distillation.yml` 配置
- 支持不同环境的配置

---

## 🔧 后端修改方案

### 方案A：最小修改方案（推荐）⭐

**优点**：
- ✅ 修改最少
- ✅ 立即可用
- ✅ 不需要改数据库

**需要修改的文件**：

#### 1. TrainingExecutionService.java

```java
package com.qczy.distillation.service;

@Service
public class TrainingExecutionService {

    // 新增配置项
    @Value("${distillation.qwen-script.path:/home/user/work/back/datamark-admin/train_qwen_vl_distillation.py}")
    private String qwenScriptPath;

    /**
     * 根据教师模型类型选择训练脚本
     */
    private String getTrainingScript(String teacherModel) {
        if (teacherModel != null &&
            (teacherModel.toLowerCase().contains("qwen") ||
             teacherModel.toLowerCase().contains("qwen2"))) {
            return qwenScriptPath;
        }
        return scriptPath; // 原有脚本
    }

    /**
     * 构建Python训练命令（扩展版）
     */
    private List<String> buildPythonCommand(MdTrainingTaskEntity task, TrainingConfigDTO config) {
        List<String> command = new ArrayList<>();

        // Python解释器
        command.add(pythonPath);

        // 【修改】根据教师模型选择脚本
        command.add(getTrainingScript(task.getTeacherModel()));

        // ========== 基础配置 ==========
        command.add("--task_id");
        command.add(task.getTaskId());

        command.add("--api_base_url");
        command.add(apiBaseUrl);

        // ========== 模型配置 ==========
        command.add("--teacher_model");
        command.add(task.getTeacherModel());

        command.add("--student_model");
        command.add(task.getStudentModel());

        // 【新增】解析学生模型类型和大小
        // 假设studentModel格式为 "resnet/resnet50" 或 "resnet50"
        String studentModelType = null;
        String studentModelSize = null;

        // 优先从JSON配置读取
        if (config != null) {
            studentModelType = (String) getConfigValue(config, "studentModelType");
            studentModelSize = (String) getConfigValue(config, "studentModelSize");
        }

        // 如果JSON中没有，尝试从studentModel字段解析
        if (studentModelType == null && task.getStudentModel() != null) {
            String[] parts = task.getStudentModel().split("/");
            if (parts.length == 2) {
                studentModelType = parts[0];
                studentModelSize = parts[1];
            } else if (parts.length == 1) {
                // 尝试推断类型
                String model = parts[0].toLowerCase();
                if (model.startsWith("resnet")) {
                    studentModelType = "resnet";
                    studentModelSize = parts[0];
                } else if (model.startsWith("vit")) {
                    studentModelType = "vit";
                    studentModelSize = parts[0];
                } else if (model.startsWith("yolo")) {
                    studentModelType = "yolov8";
                    studentModelSize = model.replace("yolov8", "").replace("yolo", "");
                }
            }
        }

        if (studentModelType != null) {
            command.add("--student_model_type");
            command.add(studentModelType);
        }

        if (studentModelSize != null) {
            command.add("--student_model_size");
            command.add(studentModelSize);
        }

        // 【新增】任务类型和类别数
        String taskType = (String) getConfigValue(config, "taskType");
        if (taskType != null) {
            command.add("--task_type");
            command.add(taskType);
        } else {
            command.add("--task_type");
            command.add("classification"); // 默认
        }

        Integer numClasses = (Integer) getConfigValue(config, "numClasses");
        if (numClasses != null) {
            command.add("--num_classes");
            command.add(String.valueOf(numClasses));
        }

        Integer imageSize = (Integer) getConfigValue(config, "imageSize");
        if (imageSize != null) {
            command.add("--image_size");
            command.add(String.valueOf(imageSize));
        }

        // 教师模型路径
        String teacherPath = getModelPath(task.getTeacherModel(), config);
        command.add("--teacher_path");
        command.add(teacherPath);

        // ... 其他参数（保持原有逻辑）...

        // 【新增】蒸馏策略配置
        String distillationType = (String) getConfigValue(config, "distillationType");
        if (distillationType != null) {
            command.add("--distillation_type");
            command.add(distillationType);
        }

        String featureLossType = (String) getConfigValue(config, "featureLossType");
        if (featureLossType != null) {
            command.add("--feature_loss_type");
            command.add(featureLossType);
        }

        Boolean alignFeature = (Boolean) getConfigValue(config, "alignFeature");
        if (alignFeature != null) {
            command.add("--align_feature");
            command.add(String.valueOf(alignFeature));
        }

        // ... 输出配置 ...

        return command;
    }

    /**
     * 从JSON配置中安全获取值的辅助方法
     */
    private Object getConfigValue(TrainingConfigDTO config, String fieldName) {
        if (config == null) return null;

        try {
            // 使用反射或者手动映射
            // 这里简化处理，实际可以使用BeanUtils或反射
            switch (fieldName) {
                case "studentModelType":
                    // 假设在TrainingConfigDTO中添加了这些字段
                    // 或者从JSON字符串中解析
                    break;
                // ... 其他字段
            }
        } catch (Exception e) {
            logger.warn("无法获取配置值: {}", fieldName, e);
        }

        return null;
    }
}
```

#### 2. application-distillation.yml

```yaml
distillation:
  python:
    path: /path/to/conda/envs/qwen-distill/bin/python3

  script:
    path: /home/user/work/back/datamark-admin/train_distillation.py

  # 【新增】Qwen2.5-VL训练脚本路径
  qwen-script:
    path: /home/user/work/back/datamark-admin/train_qwen_vl_distillation.py

  api:
    base-url: http://localhost:8080

  models:
    root: /data/models

  datasets:
    root: /data/datasets

  output:
    root: /data/training_output
```

#### 3. TrainingConfigDTO.java（可选扩展）

如果想要类型安全，可以在DTO中添加新字段：

```java
@Data
public class TrainingConfigDTO implements Serializable {

    // ========== 原有字段 ==========
    private String optimizer;
    private String lrScheduler;
    // ...

    // ========== 【新增】Qwen2.5-VL相关配置 ==========

    /**
     * 学生模型类型：resnet, vit, yolov8, unet, lstm
     */
    private String studentModelType;

    /**
     * 学生模型大小：resnet50, vit-base, s, medium等
     */
    private String studentModelSize;

    /**
     * 任务类型：classification, detection, segmentation
     */
    private String taskType;

    /**
     * 分类类别数
     */
    private Integer numClasses;

    /**
     * 图像尺寸
     */
    private Integer imageSize;

    /**
     * 蒸馏类型：feature, logit, hybrid
     */
    private String distillationType;

    /**
     * 特征损失类型：mse, cosine
     */
    private String featureLossType;

    /**
     * 是否启用特征对齐
     */
    private Boolean alignFeature;
}
```

---

### 方案B：完整修改方案（可选）

如果要完全优化，可以添加数据库字段：

#### 1. 数据库迁移SQL

```sql
-- 添加新字段到md_training_task表
ALTER TABLE md_training_task
ADD COLUMN student_model_type VARCHAR(50) COMMENT '学生模型类型：resnet/vit/yolov8/unet/lstm';

ALTER TABLE md_training_task
ADD COLUMN student_model_size VARCHAR(50) COMMENT '学生模型大小';

ALTER TABLE md_training_task
ADD COLUMN task_type VARCHAR(50) DEFAULT 'classification'
    COMMENT '任务类型：classification/detection/segmentation';

ALTER TABLE md_training_task
ADD COLUMN num_classes INT DEFAULT 10 COMMENT '分类类别数';

ALTER TABLE md_training_task
ADD COLUMN image_size INT DEFAULT 224 COMMENT '图像尺寸';

-- 添加索引（如果需要频繁查询）
CREATE INDEX idx_student_model_type ON md_training_task(student_model_type);
CREATE INDEX idx_task_type ON md_training_task(task_type);
```

#### 2. MdTrainingTaskEntity.java

```java
@Data
@TableName("md_training_task")
public class MdTrainingTaskEntity implements Serializable {

    // ========== 原有字段 ==========
    private String taskId;
    private String taskName;
    private String teacherModel;
    private String studentModel;
    // ...

    // ========== 【新增】字段 ==========

    /**
     * 学生模型类型
     */
    private String studentModelType;

    /**
     * 学生模型大小
     */
    private String studentModelSize;

    /**
     * 任务类型
     */
    private String taskType;

    /**
     * 类别数
     */
    private Integer numClasses;

    /**
     * 图像尺寸
     */
    private Integer imageSize;
}
```

---

## 🎨 前端修改方案

### 需要修改的文件：

`/front/data-mark-v3/src/views/model-distillation/index.vue`

### 修改内容：

#### 1. 添加学生模型选择区域

```vue
<template>
  <!-- 现有的教师模型配置 -->
  <n-form-item label="教师模型" path="teacherModel">
    <n-select v-model:value="taskForm.teacherModel" :options="teacherModelOptions" />
  </n-form-item>

  <!-- 【新增】学生模型配置区域 -->
  <n-divider title-placement="left">学生模型配置</n-divider>

  <!-- 学生模型类型选择 -->
  <n-form-item label="学生模型类型" path="studentModelType">
    <n-select
      v-model:value="taskForm.studentModelType"
      :options="studentModelTypeOptions"
      @update:value="handleStudentModelTypeChange"
      placeholder="选择模型架构"
    >
      <template #prefix>
        <n-icon :component="CubeOutline" />
      </template>
    </n-select>
  </n-form-item>

  <!-- 学生模型大小选择（根据类型动态变化） -->
  <n-form-item label="模型大小" path="studentModelSize">
    <n-select
      v-model:value="taskForm.studentModelSize"
      :options="studentModelSizeOptions"
      placeholder="选择模型大小"
    />
  </n-form-item>

  <!-- 【新增】任务类型选择 -->
  <n-form-item label="任务类型" path="taskType">
    <n-select
      v-model:value="taskForm.taskType"
      :options="taskTypeOptions"
      @update:value="handleTaskTypeChange"
    />
  </n-form-item>

  <!-- 类别数（仅分类任务显示） -->
  <n-form-item
    v-if="taskForm.taskType === 'classification'"
    label="类别数"
    path="numClasses"
  >
    <n-input-number
      v-model:value="taskForm.numClasses"
      :min="2"
      :max="1000"
      placeholder="输入分类类别数"
    />
  </n-form-item>

  <!-- 图像尺寸 -->
  <n-form-item label="图像尺寸" path="imageSize">
    <n-select
      v-model:value="taskForm.imageSize"
      :options="imageSizeOptions"
    />
  </n-form-item>

  <!-- 【新增】蒸馏策略配置 -->
  <n-divider title-placement="left">蒸馏策略</n-divider>

  <n-form-item label="蒸馏类型" path="distillationType">
    <n-select
      v-model:value="taskForm.distillationType"
      :options="distillationTypeOptions"
    />
  </n-form-item>

  <n-form-item label="特征损失类型" path="featureLossType">
    <n-select
      v-model:value="taskForm.featureLossType"
      :options="featureLossTypeOptions"
    />
  </n-form-item>

  <n-form-item label="启用特征对齐">
    <n-switch v-model:value="taskForm.alignFeature" />
  </n-form-item>
</template>

<script setup lang="ts">
import { ref, reactive, computed, watch } from 'vue';
import { CubeOutline } from '@vicons/ionicons5';

// ========== 数据定义 ==========

const taskForm = ref({
  // ... 原有字段 ...

  // 【新增】学生模型配置
  studentModelType: 'resnet',
  studentModelSize: 'resnet50',
  taskType: 'classification',
  numClasses: 10,
  imageSize: 224,

  // 【新增】蒸馏配置
  distillationType: 'hybrid',
  featureLossType: 'cosine',
  alignFeature: true
});

// 学生模型类型选项
const studentModelTypeOptions = [
  {
    label: 'ResNet (图像分类)',
    value: 'resnet',
    description: '经典CNN架构，平衡准确率和速度'
  },
  {
    label: 'Vision Transformer (图像分类)',
    value: 'vit',
    description: 'Transformer架构，高准确率'
  },
  {
    label: 'YOLOv8 (目标检测)',
    value: 'yolov8',
    description: '实时目标检测'
  },
  {
    label: 'UNet (图像分割)',
    value: 'unet',
    description: '像素级图像分割'
  },
  {
    label: 'LSTM (序列分类)',
    value: 'lstm',
    description: '处理时序信息'
  }
];

// 学生模型大小选项（动态变化）
const studentModelSizeOptions = computed(() => {
  const sizeOptionsMap = {
    resnet: [
      { label: 'ResNet-18 (11M参数)', value: 'resnet18' },
      { label: 'ResNet-34 (21M参数)', value: 'resnet34' },
      { label: 'ResNet-50 (25M参数，推荐)', value: 'resnet50' },
      { label: 'ResNet-101 (44M参数)', value: 'resnet101' }
    ],
    vit: [
      { label: 'ViT-Tiny (5M参数)', value: 'vit-tiny' },
      { label: 'ViT-Base (86M参数，推荐)', value: 'vit-base' },
      { label: 'ViT-Large (307M参数)', value: 'vit-large' }
    ],
    yolov8: [
      { label: 'YOLOv8-nano (3M参数，极速)', value: 'n' },
      { label: 'YOLOv8-small (11M参数，推荐)', value: 's' },
      { label: 'YOLOv8-medium (26M参数)', value: 'm' },
      { label: 'YOLOv8-large (44M参数)', value: 'l' },
      { label: 'YOLOv8-xlarge (68M参数)', value: 'x' }
    ],
    unet: [
      { label: 'UNet-Small (7M参数)', value: 'small' },
      { label: 'UNet-Medium (17M参数，推荐)', value: 'medium' },
      { label: 'UNet-Large (31M参数)', value: 'large' }
    ],
    lstm: [
      { label: 'LSTM-Small (10M参数)', value: 'small' },
      { label: 'LSTM-Medium (25M参数，推荐)', value: 'medium' },
      { label: 'LSTM-Large (50M参数)', value: 'large' }
    ]
  };

  return sizeOptionsMap[taskForm.value.studentModelType] || [];
});

// 任务类型选项
const taskTypeOptions = [
  {
    label: '图像分类 (Classification)',
    value: 'classification',
    description: '将图像分为不同类别'
  },
  {
    label: '目标检测 (Detection)',
    value: 'detection',
    description: '检测图像中的物体位置'
  },
  {
    label: '图像分割 (Segmentation)',
    value: 'segmentation',
    description: '像素级图像分割'
  }
];

// 图像尺寸选项
const imageSizeOptions = [
  { label: '192×192', value: 192 },
  { label: '224×224 (推荐)', value: 224 },
  { label: '256×256', value: 256 },
  { label: '320×320', value: 320 },
  { label: '512×512', value: 512 },
  { label: '640×640 (YOLO推荐)', value: 640 }
];

// 蒸馏类型选项
const distillationTypeOptions = [
  {
    label: '特征蒸馏 (Feature)',
    value: 'feature',
    description: '学习教师模型的特征表示'
  },
  {
    label: '混合蒸馏 (Hybrid，推荐)',
    value: 'hybrid',
    description: '结合任务损失和特征蒸馏'
  }
];

// 特征损失类型选项
const featureLossTypeOptions = [
  { label: 'MSE损失', value: 'mse' },
  { label: 'Cosine相似度 (推荐)', value: 'cosine' }
];

// ========== 事件处理 ==========

// 学生模型类型变化时，重置模型大小
function handleStudentModelTypeChange(value: string) {
  const firstOption = studentModelSizeOptions.value[0];
  if (firstOption) {
    taskForm.value.studentModelSize = firstOption.value;
  }

  // 根据模型类型调整推荐配置
  if (value === 'yolov8') {
    taskForm.value.taskType = 'detection';
    taskForm.value.imageSize = 640;
    taskForm.value.distillationType = 'feature';
  } else if (value === 'unet') {
    taskForm.value.taskType = 'segmentation';
    taskForm.value.imageSize = 512;
    taskForm.value.distillationType = 'feature';
  } else {
    taskForm.value.taskType = 'classification';
    taskForm.value.imageSize = 224;
    taskForm.value.distillationType = 'hybrid';
  }
}

// 任务类型变化时，调整相关配置
function handleTaskTypeChange(value: string) {
  if (value === 'classification') {
    taskForm.value.numClasses = 10;
  } else if (value === 'detection') {
    taskForm.value.numClasses = 80; // COCO数据集
    taskForm.value.imageSize = 640;
  } else if (value === 'segmentation') {
    taskForm.value.numClasses = 21; // VOC数据集
    taskForm.value.imageSize = 512;
  }
}

// ========== API调用 ==========

async function handleCreateTask() {
  try {
    await taskFormRef.value?.validate();

    creatingTask.value = true;

    // 构建提交数据
    const submitData = {
      ...taskForm.value,

      // 拼接学生模型字段
      studentModel: `${taskForm.value.studentModelType}/${taskForm.value.studentModelSize}`,

      // 所有新增字段都包含在请求中
      studentModelType: taskForm.value.studentModelType,
      studentModelSize: taskForm.value.studentModelSize,
      taskType: taskForm.value.taskType,
      numClasses: taskForm.value.numClasses,
      imageSize: taskForm.value.imageSize,
      distillationType: taskForm.value.distillationType,
      featureLossType: taskForm.value.featureLossType,
      alignFeature: taskForm.value.alignFeature,

      // GPU设备转为逗号分隔字符串
      gpuDevices: taskForm.value.gpuDevices?.join(',')
    };

    // 调用API
    const res = await createDistillationTask(submitData);

    if (res.code === 200) {
      message.success('训练任务创建成功');
      showCreateTaskModal.value = false;
      refreshTasks();
    } else {
      message.error(res.message || '创建任务失败');
    }
  } catch (error) {
    console.error('创建任务失败:', error);
    message.error('创建任务失败');
  } finally {
    creatingTask.value = false;
  }
}
</script>
```

---

## 📝 集成步骤

### Step 1: 后端修改（30分钟）

1. **修改TrainingExecutionService.java**
   - 添加 `getTrainingScript()` 方法
   - 扩展 `buildPythonCommand()` 方法
   - 添加新参数处理逻辑

2. **更新application-distillation.yml**
   - 添加 `qwen-script.path` 配置

3. **（可选）扩展TrainingConfigDTO.java**
   - 添加新字段定义

### Step 2: 前端修改（1小时）

1. **修改index.vue**
   - 添加学生模型选择组件
   - 添加任务类型选择
   - 添加蒸馏策略配置
   - 更新表单提交逻辑

2. **测试前端页面**
   - 确认下拉框正常显示
   - 确认动态联动正常

### Step 3: 环境准备（30分钟）

1. **安装Python依赖**
   ```bash
   conda create -n qwen-distill python=3.9
   conda activate qwen-distill
   pip install torch transformers peft pillow numpy requests tqdm
   ```

2. **配置路径**
   - 更新 `application-distillation.yml` 中的Python路径
   - 确认脚本路径正确

### Step 4: 测试验证（30分钟）

1. **单元测试**
   - 测试命令构建逻辑
   - 测试脚本选择逻辑

2. **集成测试**
   - 创建一个测试任务
   - 确认Python脚本正常启动
   - 确认参数传递正确

3. **端到端测试**
   - 通过前端创建完整任务
   - 监控训练日志
   - 验证进度回调

---

## ✅ 测试验证

### 测试清单

```
□ 1. 后端代码编译通过
□ 2. 前端页面正常显示
□ 3. 创建ResNet任务成功
□ 4. 创建ViT任务成功
□ 5. 创建YOLO任务成功
□ 6. 创建UNet任务成功
□ 7. 创建LSTM任务成功
□ 8. Python脚本正常启动
□ 9. 训练进度正常更新
□ 10. Checkpoint正常保存
```

### 测试命令

```bash
# 1. 测试Python脚本是否可执行
python /path/to/train_qwen_vl_distillation.py --help

# 2. 测试最小配置运行
python train_qwen_vl_distillation.py \
  --task_id "test" \
  --api_base_url "http://localhost:8080" \
  --teacher_model "qwen2.5-vl-8b" \
  --student_model "resnet50" \
  --teacher_path "/data/models/qwen2.5-vl-8b" \
  --student_model_type "resnet" \
  --student_model_size "resnet50" \
  --task_type "classification" \
  --num_classes 10 \
  --dataset_id "test_dataset" \
  --epochs 2 \
  --batch_size 8 \
  --output_dir "/tmp/test_output"

# 3. 查看后端日志
tail -f logs/application.log | grep "Training"

# 4. 测试进度回调
curl -X PUT http://localhost:8080/model-distillation/tasks/test_task/progress \
  -H "Content-Type: application/json" \
  -d '{"currentEpoch": 1, "totalEpochs": 10, "valAccuracy": 85.5}'
```

---

## 🎓 总结

### 设计亮点

1. **最小侵入性** - 无需大规模重构
2. **向后兼容** - 现有功能不受影响
3. **灵活扩展** - 易于添加新模型
4. **配置驱动** - 所有参数可配置
5. **完整文档** - 详细的集成指南

### 工作量评估

| 任务 | 工作量 | 优先级 |
|------|--------|--------|
| 后端修改 | 2小时 | 🔴 高 |
| 前端修改 | 2小时 | 🔴 高 |
| 环境配置 | 1小时 | 🟡 中 |
| 测试验证 | 2小时 | 🔴 高 |
| **总计** | **7小时** | - |

### 下一步行动

1. ✅ 阅读本文档，理解整体架构
2. ⬜ 按照后端修改方案修改代码
3. ⬜ 按照前端修改方案更新界面
4. ⬜ 配置Python环境
5. ⬜ 运行测试验证
6. ⬜ 部署到生产环境

---

**文档版本**: 1.0
**最后更新**: 2026-01-11
**作者**: Claude Assistant
