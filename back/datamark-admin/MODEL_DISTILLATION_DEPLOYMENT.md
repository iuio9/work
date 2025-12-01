# 大小模型协同训练模块 - 部署说明

## 概述

本文档说明如何部署和使用大小模型协同训练（Model Distillation）功能模块。

## 功能特性

- ✅ 训练任务管理（创建、启动、停止、删除）
- ✅ 实时训练进度跟踪
- ✅ 训练历史记录（每个epoch的详细指标）
- ✅ LoRA配置预设管理
- ✅ 模型评估结果保存
- ✅ 与自动标注功能集成
- ✅ RESTful API接口
- ✅ Swagger API文档

## 数据库部署

### 1. 创建数据库表

执行SQL脚本创建4个核心表：

```bash
mysql -u your_username -p your_database < model_distillation_schema.sql
```

或者在数据库管理工具中直接执行 `model_distillation_schema.sql` 文件。

**创建的表：**
- `md_training_task` - 训练任务主表
- `md_training_history` - 训练历史记录表
- `md_lora_preset` - LoRA配置预设表（包含4个默认预设）
- `md_model_evaluation` - 模型评估结果表

### 2. 插入示例数据（可选，用于测试和演示）

```bash
mysql -u your_username -p your_database < model_distillation_sample_data.sql
```

**示例数据包括：**
- 4个已完成的训练任务（用于自动标注集成测试）
- 1个正在运行的训练任务
- TASK_001的训练历史记录（5个epoch）
- TASK_001的模型评估结果

## 后端代码结构

```
com.qczy.distillation/
├── model/
│   └── entity/
│       ├── MdTrainingTaskEntity.java       # 训练任务实体
│       ├── MdTrainingHistoryEntity.java    # 训练历史实体
│       ├── MdLoraPresetEntity.java         # LoRA预设实体
│       └── MdModelEvaluationEntity.java    # 模型评估实体
├── mapper/
│   ├── MdTrainingTaskMapper.java           # 训练任务Mapper
│   ├── MdTrainingHistoryMapper.java        # 训练历史Mapper
│   ├── MdLoraPresetMapper.java             # LoRA预设Mapper
│   └── MdModelEvaluationMapper.java        # 模型评估Mapper
├── service/
│   └── MdTrainingTaskService.java          # 训练任务服务（核心业务逻辑）
└── controller/
    └── ModelDistillationController.java     # REST API控制器
```

## API接口说明

### 基础URL
```
http://your-server:port/model-distillation
```

### 主要接口

#### 1. 训练任务管理

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/tasks` | 获取所有训练任务 |
| GET | `/completed-models` | 获取已完成的训练任务（用于自动标注） |
| GET | `/tasks/{taskId}` | 获取任务详情 |
| POST | `/tasks` | 创建训练任务 |
| POST | `/tasks/{taskId}/start` | 启动训练任务 |
| POST | `/tasks/{taskId}/stop` | 停止训练任务 |
| POST | `/tasks/{taskId}/complete` | 完成训练任务 |
| DELETE | `/tasks/{taskId}` | 删除训练任务 |
| PUT | `/tasks/{taskId}/progress` | 更新训练进度 |

#### 2. 训练历史记录

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/tasks/{taskId}/history` | 获取任务训练历史 |
| GET | `/tasks/{taskId}/history/latest` | 获取最新训练记录 |
| POST | `/tasks/{taskId}/history` | 记录训练历史 |

#### 3. LoRA预设管理

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/lora-presets` | 获取所有LoRA预设 |
| GET | `/lora-presets/{presetName}` | 根据名称获取预设 |
| POST | `/lora-presets` | 创建LoRA预设 |

#### 4. 模型评估

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/tasks/{taskId}/evaluations` | 获取任务评估结果 |
| GET | `/tasks/{taskId}/evaluations/latest` | 获取最新评估结果 |
| GET | `/tasks/{taskId}/evaluations/best` | 获取最佳评估结果 |
| POST | `/tasks/{taskId}/evaluations` | 保存模型评估结果 |

## 前端集成

### 自动标注功能集成

前端已完成集成，位于：
- `/front/data-mark-v3/src/views/data-expansion/add/index.vue`
- `/front/data-mark-v3/src/views/data-ano/autoAno/modules/user-search.vue`

**调用示例：**
```typescript
import { fetchCompletedDistillationModels } from "@/service/api/model-distillation";

// 获取已完成的训练模型（准确率>=70%）
const res = await fetchCompletedDistillationModels({ minAccuracy: 70 });
const models = res.data.map((task: any) => ({
  label: `[协同训练] ${task.taskName} (准确率: ${task.accuracy?.toFixed(2)}%)`,
  value: `distillation_${task.taskId}`,
  taskId: task.taskId,
  type: 'distillation'
}));
```

## 测试步骤

### 1. 验证数据库表创建成功

```sql
SHOW TABLES LIKE 'md_%';
```

应显示4个表。

### 2. 验证示例数据插入成功

```sql
SELECT task_id, task_name, status, accuracy FROM md_training_task;
```

应显示5条记录（4个COMPLETED，1个RUNNING）。

### 3. 测试API接口

使用Swagger UI或Postman测试：

```bash
# 获取所有任务
GET http://localhost:8080/model-distillation/tasks

# 获取已完成的模型（用于自动标注）
GET http://localhost:8080/model-distillation/completed-models?minAccuracy=70

# 获取任务详情
GET http://localhost:8080/model-distillation/tasks/TASK_001
```

### 4. 测试前端集成

1. 启动前端项目
2. 进入"数据标注 > 自动标注"页面
3. 点击"创建任务"按钮
4. 勾选"目标检测"或"元器件检测"
5. 在"请选择模型"下拉框中，应该能看到带有"[协同训练]"标签的模型选项

## 常见问题

### Q1: 数据库表创建失败

**A:** 检查MySQL版本是否支持`DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP`语法（需要MySQL 5.6+）

### Q2: API返回空数据

**A:**
1. 确认数据库表已创建
2. 确认示例数据已插入
3. 检查数据库连接配置
4. 查看后端日志是否有错误信息

### Q3: 前端下拉框看不到协同训练模型

**A:**
1. 打开浏览器开发者工具，查看Network标签
2. 检查`/model-distillation/completed-models`接口是否调用成功
3. 确认返回的data不为空
4. 检查前端console是否有JavaScript错误

## 后续扩展

本实现为核心功能，后续可扩展：

1. **模型文件存储**
   - 实现模型文件上传接口
   - 集成MinIO或OSS对象存储
   - 添加模型版本管理

2. **实际训练执行**
   - 集成PyTorch训练脚本
   - 实现异步任务队列（Celery/RabbitMQ）
   - GPU资源调度和管理

3. **监控和可视化**
   - 实时训练曲线图表
   - 资源使用监控（GPU、内存）
   - 训练日志查看

4. **权限和安全**
   - 用户权限控制
   - 任务所有者验证
   - API访问限流

## 技术栈

- **后端框架**: Spring Boot 2.2.1
- **ORM**: MyBatis Plus 3.x
- **数据库**: MySQL 5.7+
- **API文档**: Swagger 2.x
- **日志**: SLF4J + Logback

## 联系方式

如有问题，请联系开发团队或查看项目文档。
