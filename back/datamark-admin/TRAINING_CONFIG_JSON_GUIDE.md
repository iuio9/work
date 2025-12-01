# 训练配置JSON完整实现指南

## 📝 概述

本文档说明如何通过JSON字段存储前端提交的所有高级训练配置参数。

## 🔄 完整数据流

```
前端表单 (taskForm)
    ↓
前端API调用 (POST /model-distillation/tasks)
    ↓
后端Controller接收 (CreateTaskRequestDTO)
    ↓
【数据拆分】
├─ 基础字段 → Entity直接属性 (taskName, batchSize, learningRate等)
└─ 高级配置 → TrainingConfigDTO → JSON序列化 → training_config字段
    ↓
Service层保存 (MdTrainingTaskService.createTask)
    ↓
数据库存储 (md_training_task表)
```

## 📊 数据库表结构变更

### 1. 执行SQL脚本

```bash
mysql -u your_username -p your_database < add_training_config_fields.sql
```

**新增字段：**
- `training_config` TEXT - 存储所有高级配置的JSON
- `description` VARCHAR(500) - 任务描述
- `val_dataset_id` BIGINT - 验证数据集ID
- `val_dataset_name` VARCHAR(255) - 验证数据集名称

### 2. training_config JSON结构示例

```json
{
  "optimizer": "adamw",
  "lrScheduler": "cosine",
  "weightDecay": 0.01,
  "gradAccumSteps": 4,
  "maxGradNorm": 1.0,
  "gpuDevices": [0, 1],
  "autoSaveCheckpoint": true,
  "checkpointInterval": 5,
  "teacherModelConfig": {
    "paramSize": "7B",
    "modelPath": "meta-llama/Llama-2-7b-hf",
    "quantization": "int8"
  },
  "studentModelConfig": {
    "paramSize": "350M",
    "initMethod": "random",
    "pretrainPath": ""
  },
  "loraAdvancedConfig": {
    "targetModules": ["q_proj", "v_proj"],
    "layers": "all",
    "biasTrain": "none"
  },
  "distillationAdvancedConfig": {
    "hardLabelWeight": 0.3,
    "softLabelWeight": 0.7,
    "lossType": "kl_div",
    "intermediateLayers": false,
    "attentionDistill": false
  }
}
```

## 💾 后端实现

### 1. 新增类文件

#### TrainingConfigDTO.java
**位置**: `com.qczy.distillation.model.dto.TrainingConfigDTO`

**作用**: 定义高级配置的数据结构，用于JSON序列化和反序列化

**包含的配置类**:
- 优化器和调度器配置
- 硬件配置（GPU设备、检查点）
- TeacherModelConfig - 教师模型详细配置
- StudentModelConfig - 学生模型详细配置
- LoraAdvancedConfig - LoRA高级配置
- DistillationAdvancedConfig - 知识蒸馏高级配置

#### CreateTaskRequestDTO.java
**位置**: `com.qczy.distillation.model.dto.CreateTaskRequestDTO`

**作用**: 接收前端提交的完整表单数据

**包含字段**: 所有基础字段 + 所有高级配置字段（扁平结构）

### 2. Entity类更新

**文件**: `MdTrainingTaskEntity.java`

**新增字段**:
```java
// 任务描述
private String description;

// 验证数据集
private Long valDatasetId;
private String valDatasetName;

// 训练高级配置JSON
private String trainingConfig;
```

### 3. Controller更新

**文件**: `ModelDistillationController.java`

**关键方法**: `createTask(@RequestBody CreateTaskRequestDTO requestDTO)`

**处理逻辑**:
```java
// 1. 基础字段直接赋值
task.setTaskName(requestDTO.getTaskName());
task.setBatchSize(requestDTO.getBatchSize());
// ...

// 2. 构建TrainingConfigDTO对象
TrainingConfigDTO config = new TrainingConfigDTO();
config.setOptimizer(requestDTO.getOptimizer());
config.setLrScheduler(requestDTO.getLrScheduler());
// ...

// 3. 序列化为JSON
String trainingConfigJson = JSON.toJSONString(config);
task.setTrainingConfig(trainingConfigJson);

// 4. 保存到数据库
trainingTaskService.createTask(task);
```

## 🎯 前端调用示例

### 当前前端代码（需更新）

**文件**: `/front/data-mark-v3/src/views/model-distillation/index.vue`

**taskForm对象** (行875-891):
```javascript
const taskForm = ref({
  taskName: '',
  description: '',
  datasetId: '',
  valDatasetId: '',
  epochs: 10,
  batchSize: 16,
  learningRate: 0.0001,
  weightDecay: 0.01,
  gradAccumSteps: 4,
  maxGradNorm: 1.0,
  lrScheduler: 'cosine',
  optimizer: 'adamw',
  gpuDevices: [],
  autoSaveCheckpoint: true,
  checkpointInterval: 5
});
```

### 需要实现的API调用

**文件**: `/front/data-mark-v3/src/service/api/model-distillation.ts`

添加创建任务方法：
```typescript
/** 创建训练任务 */
export function createDistillationTask(taskData: any) {
  return request<any>({
    url: '/model-distillation/tasks',
    method: 'post',
    data: taskData
  });
}
```

**文件**: `/front/data-mark-v3/src/views/model-distillation/index.vue`

更新handleCreateTask函数 (行1532):
```typescript
import { createDistillationTask } from '@/service/api/model-distillation';

async function handleCreateTask() {
  try {
    // 验证表单
    await taskFormRef.value?.validate();

    creatingTask.value = true;

    // 准备提交数据
    const submitData = {
      ...taskForm.value,
      // 从Tab1的模型配置中获取
      teacherModel: teacherModel.value.modelId,
      studentModel: studentModel.value.modelId,
      teacherParamSize: teacherModel.value.paramSize,
      teacherModelPath: teacherModel.value.modelPath,
      teacherQuantization: teacherModel.value.quantization,
      studentParamSize: studentModel.value.paramSize,
      studentInitMethod: studentModel.value.initMethod,
      studentPretrainPath: studentModel.value.pretrainPath,
      // LoRA配置
      loraRank: loraConfig.value.rank,
      loraAlpha: loraConfig.value.alpha,
      loraDropout: loraConfig.value.dropout,
      loraTargetModules: loraConfig.value.targetModules?.join(','),
      loraLayers: loraConfig.value.layers,
      loraBiasTrain: loraConfig.value.biasTrain,
      // 知识蒸馏配置
      temperature: distillConfig.value.temperature,
      alpha: distillConfig.value.softLabelWeight,
      hardLabelWeight: distillConfig.value.hardLabelWeight,
      softLabelWeight: distillConfig.value.softLabelWeight,
      distillLossType: distillConfig.value.lossType,
      intermediateLayers: distillConfig.value.intermediateLayers,
      attentionDistill: distillConfig.value.attentionDistill,
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
```

## 🔍 如何读取和使用JSON配置

### 在Service或其他业务代码中反序列化

```java
import com.alibaba.fastjson.JSON;

// 读取任务
MdTrainingTaskEntity task = trainingTaskMapper.selectByTaskId(taskId);

// 反序列化JSON配置
if (task.getTrainingConfig() != null) {
    TrainingConfigDTO config = JSON.parseObject(
        task.getTrainingConfig(),
        TrainingConfigDTO.class
    );

    // 使用配置
    String optimizer = config.getOptimizer();  // "adamw"
    List<Integer> gpuDevices = config.getGpuDevices();  // [0, 1]

    // 访问嵌套配置
    if (config.getTeacherModelConfig() != null) {
        String paramSize = config.getTeacherModelConfig().getParamSize();
        String modelPath = config.getTeacherModelConfig().getModelPath();
    }
}
```

## ✅ 实现优势

1. **灵活性高**: 新增配置项只需修改DTO，不需要修改数据库表结构
2. **完整保存**: 前端的所有配置都能完整保存
3. **易于扩展**: 可以随时添加新的配置项
4. **类型安全**: 使用DTO保证类型安全
5. **向后兼容**: 旧数据training_config为NULL也不影响使用

## 📋 部署清单

- [x] 数据库SQL脚本: `add_training_config_fields.sql`
- [x] DTO类: `TrainingConfigDTO.java`
- [x] 请求DTO: `CreateTaskRequestDTO.java`
- [x] Entity更新: `MdTrainingTaskEntity.java`
- [x] Controller更新: `ModelDistillationController.java`
- [ ] 前端API方法: `model-distillation.ts`
- [ ] 前端调用实现: `index.vue handleCreateTask()`

## 🚀 下一步

1. **执行数据库脚本** - 添加新字段
2. **重启后端服务** - 加载新代码
3. **更新前端代码** - 实现handleCreateTask方法
4. **测试完整流程** - 创建任务并验证数据保存

## 📞 FAQ

**Q: 为什么用JSON而不是直接加字段？**
A: 前端有20+个高级配置项，都加字段会导致表结构非常复杂，且不灵活。JSON方案更适合快速迭代。

**Q: JSON性能如何？**
A: TEXT类型存储JSON，查询时反序列化。对于训练任务这种低频操作，性能完全够用。

**Q: 如何查询JSON里的字段？**
A: MySQL 5.7+支持JSON字段查询，或者在应用层反序列化后过滤。

**Q: 旧数据怎么办？**
A: training_config为NULL时，代码中做NULL判断即可，不影响旧任务。
