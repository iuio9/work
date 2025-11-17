# 联邦学习模块代码修改说明

## 概述

本次修改实现了基于 Flower 框架的联邦学习模块，支持多节点（≥10个）联邦训练，包含精度保护机制（5%阈值），并支持 YOLOv8、LSTM、UNet、ResNet、Vision Transformer 等模型。

## 新增文件

### Python 文件

1. **`flower_server.py`** - Flower 服务器端实现
   - 位置：`datamark-admin/flower_server.py`
   - 功能：协调多个客户端进行联邦学习，使用 FedAvg 策略聚合参数，监控精度下降
   - 关键类：`AccuracyMonitorStrategy`（扩展 FedAvg，增加精度监控）

2. **`flower_client.py`** - Flower 客户端实现
   - 位置：`datamark-admin/flower_client.py`
   - 功能：客户端本地训练、参数更新、模型评估
   - 关键类：`ModelAdapter`（模型适配器基类）及各种模型适配器

3. **`FLOWER_README.md`** - 使用说明文档
   - 位置：`datamark-admin/FLOWER_README.md`
   - 内容：安装、配置、使用流程说明

### Java 文件

1. **`FlowerServerManager.java`** - Flower Server 进程管理器
   - 位置：`datamark-admin/src/main/java/com/qczy/federated/flower/FlowerServerManager.java`
   - 功能：启动/停止 Flower Server 进程，监控进程状态
   - 关键方法：
     - `startServer()` - 启动 Flower Server
     - `stopServer()` - 停止 Flower Server
     - `isServerRunning()` - 检查运行状态

2. **`FederatedNode.java`** - 联邦节点实体
   - 位置：`datamark-admin/src/main/java/com/qczy/federated/model/FederatedNode.java`
   - 功能：存储节点信息（ID、地址、状态、心跳时间等）

3. **`ModelType.java`** - 模型类型枚举
   - 位置：`datamark-admin/src/main/java/com/qczy/federated/model/ModelType.java`
   - 功能：定义支持的模型类型（YOLO_V8, LSTM, UNET, RESNET, VISION_TRANSFORMER）

4. **`TrainingJob.java`** - 训练任务实体
   - 位置：`datamark-admin/src/main/java/com/qczy/federated/model/TrainingJob.java`
   - 功能：存储训练任务信息（任务ID、模型类型、超参数、参与节点、状态等）

5. **`FederatedCoordinatorService.java`** - 联邦协调服务
   - 位置：`datamark-admin/src/main/java/com/qczy/federated/service/FederatedCoordinatorService.java`
   - 功能：节点注册、任务管理、与 Flower Server 集成

6. **`FederatedController.java`** - 联邦学习 REST API 控制器
   - 位置：`datamark-admin/src/main/java/com/qczy/federated/controller/FederatedController.java`
   - 功能：提供节点注册、任务创建、启动/停止等 REST 接口

7. **`FedAvgOptimizer.java`** - FedAvg 优化器
   - 位置：`datamark-admin/src/main/java/com/qczy/federated/optimizer/FedAvgOptimizer.java`
   - 功能：实现联邦平均算法（FedAvg）的参数聚合

8. **`ModelAdapter.java`** - 模型适配器接口
   - 位置：`datamark-admin/src/main/java/com/qczy/federated/spi/ModelAdapter.java`
   - 功能：定义模型适配器接口（本地训练、应用全局参数、评估）

9. **`DefaultAdapterFactory.java`** - 适配器工厂
   - 位置：`datamark-admin/src/main/java/com/qczy/federated/adapters/DefaultAdapterFactory.java`
   - 功能：根据模型类型创建对应的适配器

10. **`PlaceholderAdapter.java`** - 占位适配器
    - 位置：`datamark-admin/src/main/java/com/qczy/federated/adapters/PlaceholderAdapter.java`
    - 功能：占位实现，可替换为真实模型训练逻辑

## 修改的现有文件

### 1. `MyApplication.java`
- **位置**：`datamark-admin/src/main/java/com/qczy/MyApplication.java`
- **修改内容**：已包含 `@EnableScheduling` 注解（无需修改）
- **说明**：用于启用定时任务，支持节点监控和参数同步

### 2. `AlgorithmTaskEntity.java`
- **位置**：`datamark-admin/src/main/java/com/qczy/model/entity/AlgorithmTaskEntity.java`
- **修改内容**：为 `testResult` 字段添加 `@TableField(exist = false)` 注解
- **原因**：该字段不存在于数据库表中，避免 MyBatis-Plus 查询错误
- **修改行**：第 145 行

## 代码结构

```
datamark-admin/
├── flower_server.py                    # Flower 服务器（Python）
├── flower_client.py                    # Flower 客户端（Python）
├── FLOWER_README.md                    # 使用说明
├── CODE_CHANGES.md                     # 本文档
└── src/main/java/com/qczy/federated/
    ├── controller/
    │   └── FederatedController.java    # REST API 控制器
    ├── service/
    │   └── FederatedCoordinatorService.java  # 协调服务
    ├── flower/
    │   └── FlowerServerManager.java    # Flower Server 管理器
    ├── model/
    │   ├── FederatedNode.java          # 节点实体
    │   ├── ModelType.java              # 模型类型枚举
    │   └── TrainingJob.java           # 训练任务实体
    ├── optimizer/
    │   └── FedAvgOptimizer.java       # FedAvg 优化器
    ├── spi/
    │   └── ModelAdapter.java          # 模型适配器接口
    └── adapters/
        ├── DefaultAdapterFactory.java  # 适配器工厂
        └── PlaceholderAdapter.java     # 占位适配器
```

## 关键功能实现

### 1. 节点管理
- **注册**：`POST /federated/register` - 注册新节点
- **心跳**：`POST /federated/heartbeat/{nodeId}` - 节点心跳上报
- **监控**：定时任务每 5 秒检查节点心跳，30 秒无心跳标记为不活跃

### 2. 任务管理
- **创建**：`POST /federated/jobs` - 创建训练任务
- **启动**：`POST /federated/jobs/{jobId}/start` - 启动任务（自动启动 Flower Server）
- **停止**：`POST /federated/jobs/{jobId}/stop` - 停止任务（自动停止 Flower Server）
- **查询**：`GET /federated/jobs` - 查询所有任务

### 3. 精度保护
- **基线记录**：首次评估时记录基线精度
- **下降监控**：每轮训练后计算精度下降百分比
- **阈值保护**：如果下降超过阈值（默认 5%），标记任务为 DEGRADED

### 4. Flower Server 集成
- **自动启动**：启动任务时自动启动 Flower Server 进程
- **进程管理**：使用 `ProcessBuilder` 管理 Python 进程
- **日志捕获**：异步读取 Flower Server 输出并记录到日志

## 配置说明

在 `application.yml` 中添加以下配置：

```yaml
flower:
  server:
    script:
      path: flower_server.py  # Flower Server 脚本路径（相对于项目根目录）
    port: 8080                 # 默认端口（多个任务会自动分配不同端口）
  python:
    executable: python3        # Python 可执行文件路径
```

## 使用流程

1. **安装依赖**：`pip install flwr`
2. **启动 Java 后端**：`mvn spring-boot:run`
3. **创建任务**：调用 `POST /federated/jobs` 创建训练任务
4. **启动任务**：调用 `POST /federated/jobs/{jobId}/start`（自动启动 Flower Server）
5. **启动客户端**：在各节点运行 `python3 flower_client.py --model-type YOLO_V8 --node-id node-1 --server-address localhost:8080`
6. **监控状态**：通过 `GET /federated/jobs` 查询任务状态

## 注意事项

1. **Python 环境**：确保已安装 `flwr` 包
2. **端口冲突**：多个任务会自动分配不同端口（8080 + 任务数）
3. **网络访问**：Flower Server 和 Client 需要在同一网络或可访问的地址
4. **模型适配**：`flower_client.py` 中的模型适配器是占位实现，需要替换为真实模型训练逻辑

## 后续扩展

1. **真实模型实现**：在 `flower_client.py` 中实现 YOLOv8/LSTM/UNet/ResNet/ViT 的真实训练逻辑
2. **数据加载**：实现各节点的本地数据加载逻辑
3. **持久化**：将训练任务和节点信息持久化到数据库
4. **WebSocket**：实时推送训练进度和状态更新
5. **更多优化策略**：实现 FedProx、FedNova 等其他联邦学习算法




