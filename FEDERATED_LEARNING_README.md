# 联邦学习框架使用文档

## 📋 目录

- [系统概述](#系统概述)
- [核心功能](#核心功能)
- [技术架构](#技术架构)
- [环境要求](#环境要求)
- [安装部署](#安装部署)
- [使用指南](#使用指南)
- [API文档](#api文档)
- [常见问题](#常见问题)

---

## 🎯 系统概述

本系统是一个基于 **Flower 框架**的分布式联邦学习平台，支持多节点协同训练深度学习模型，保护数据隐私的同时实现模型的高效训练。

### 主要特性

✅ **多节点支持**：支持至少10个节点同时参与联邦学习训练
✅ **多模型支持**：支持 YOLOv8、LSTM、UNet、ResNet、Vision Transformer
✅ **精度监控**：实时监控模型精度，精度损失控制在5%以内
✅ **容错机制**：节点故障时自动检测并保证训练连续性
✅ **可视化管理**：Web界面管理节点和训练任务
✅ **数据持久化**：训练过程和结果数据库持久化存储

---

## 🚀 核心功能

### 1. 节点管理

- **节点注册**：新节点加入联邦学习网络
- **心跳监控**：实时监控节点在线状态（30秒超时）
- **节点信息**：显示节点硬件配置（CPU、GPU、内存等）
- **自动恢复**：节点重新上线后自动恢复参与训练

### 2. 训练任务管理

- **任务创建**：选择模型类型和超参数创建训练任务
- **任务启动**：自动启动 Flower Server 并协调客户端训练
- **任务停止**：随时停止训练任务
- **进度跟踪**：实时查看训练轮次和精度

### 3. 模型训练

#### 支持的模型类型

| 模型 | 用途 | 特点 |
|------|------|------|
| **YOLOv8** | 目标检测 | 实时物体检测，速度快 |
| **LSTM** | 序列预测 | 时间序列数据处理 |
| **UNet** | 图像分割 | 医学影像分割 |
| **ResNet** | 图像分类 | 残差网络，深层网络训练 |
| **Vision Transformer** | 图像分类 | 基于注意力机制 |

#### 训练策略

- **FedAvg**（联邦平均）：默认策略，简单高效
- **FedProx**（联邦近端）：处理数据异构性
- **FedAdam**（联邦自适应）：自适应学习率调整

### 4. 精度保护机制

- 首次评估建立基线精度
- 实时监控每轮训练后的模型精度
- 精度下降超过阈值（默认5%）时触发告警
- 任务状态变为 `DEGRADED`，管理员可决定是否继续

### 5. 数据可视化

- 节点统计看板（总数、在线、离线、GPU总数）
- 训练精度曲线图
- 任务进度实时显示
- 节点状态实时更新

---

## 🏗️ 技术架构

### 系统架构图

```
┌─────────────────────────────────────────────────────────────┐
│                        前端界面 (Vue 3)                      │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │节点管理  │  │任务管理  │  │训练监控  │  │数据可视化│   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘   │
└───────────────────────┬─────────────────────────────────────┘
                        │ REST API
┌───────────────────────┴─────────────────────────────────────┐
│              后端服务 (Spring Boot)                          │
│  ┌─────────────────────────────────────────────────────┐   │
│  │     FederatedCoordinatorService (协调器)            │   │
│  │  - 节点注册与心跳管理                                 │   │
│  │  - 训练任务调度                                       │   │
│  │  - 精度监控与容错                                     │   │
│  └─────────────────────────────────────────────────────┘   │
│  ┌─────────────────┐     ┌──────────────────────────┐     │
│  │ FlowerServer    │────▶│  数据库 (MySQL)           │     │
│  │ Manager         │     │  - 节点信息表             │     │
│  │ (进程管理器)     │     │  - 训练任务表             │     │
│  └─────────────────┘     │  - 训练轮次记录表         │     │
└───────────┬───────────────│  - 节点训练历史表         │     │
            │               │  - 模型版本管理表         │     │
            │               └──────────────────────────┘     │
┌───────────┴───────────────────────────────────────────────┐
│              Flower Framework (Python)                     │
│  ┌─────────────────┐                                      │
│  │  Flower Server  │  (flower_server.py)                  │
│  │  - 参数聚合      │                                      │
│  │  - 策略管理      │                                      │
│  │  - 精度监控      │                                      │
│  └────────┬────────┘                                      │
│           │                                                │
│  ┌────────┴────────┬────────────┬────────────┐           │
│  ▼                 ▼            ▼            ▼           │
│ Client 1        Client 2     Client 3    Client N        │
│ (node-1)        (node-2)     (node-3)    (node-n)        │
│ - 数据加载       - 数据加载    - 数据加载   - 数据加载    │
│ - 本地训练       - 本地训练    - 本地训练   - 本地训练    │
│ - 模型更新       - 模型更新    - 模型更新   - 模型更新    │
└────────────────────────────────────────────────────────────┘
```

### 技术栈

#### 后端
- **框架**：Spring Boot 2.2.1
- **ORM**：MyBatis Plus 3.4.2
- **数据库**：MySQL 8.0+
- **API文档**：Swagger 2.7.0
- **安全**：Spring Security + JWT

#### 前端
- **框架**：Vue 3.4 + TypeScript 5.5
- **构建工具**：Vite 5.4
- **UI库**：Naive UI 2.39 + Element Plus 2.8
- **状态管理**：Pinia 2.2
- **图表**：ECharts 5.5

#### 联邦学习
- **框架**：Flower 1.6+
- **深度学习**：PyTorch 2.0+
- **模型库**：Ultralytics (YOLOv8), Timm, Transformers

---

## 💻 环境要求

### 服务器端（后端 + Flower Server）

**硬件要求：**
- CPU：4核心及以上
- 内存：8GB及以上
- 磁盘：50GB及以上
- 网络：千兆网卡

**软件要求：**
- **操作系统**：Linux (Ubuntu 20.04+) / Windows 10+ / macOS
- **Java**：JDK 1.8+
- **Python**：Python 3.8+
- **数据库**：MySQL 8.0+
- **Maven**：3.6+

### 客户端节点（Flower Client）

**硬件要求：**
- CPU：2核心及以上
- 内存：4GB及以上
- GPU：推荐 NVIDIA GPU（可选，用于加速训练）
- 磁盘：10GB及以上

**软件要求：**
- **Python**：Python 3.8+
- **CUDA**：11.0+（如果使用 GPU）

---

## 📦 安装部署

### 1. 数据库初始化

```bash
# 1. 创建数据库
mysql -u root -p
CREATE DATABASE datamark CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

# 2. 导入初始化脚本
mysql -u root -p datamark < 初始化数据库文件/datamark.sql

# 3. 导入联邦学习表结构
mysql -u root -p datamark < back/datamark-admin/federated_learning_schema.sql
```

### 2. 后端部署

```bash
cd back/datamark-admin

# 1. 修改配置文件
vim src/main/resources/application-dev.yml
# 修改数据库连接信息：
#   spring.datasource.url
#   spring.datasource.username
#   spring.datasource.password

# 2. 安装 Python 依赖
pip3 install -r requirements.txt

# 3. 编译打包
mvn clean package -DskipTests

# 4. 运行
java -jar target/datamark-admin-1.0-SNAPSHOT.jar
# 或者使用 IDE 直接运行 MyApplication.java
```

### 3. 前端部署

```bash
cd front/data-mark-v3

# 1. 安装依赖
pnpm install

# 2. 修改 API 地址（如需要）
vim src/service/request/index.ts
# 修改 baseURL

# 3. 运行开发服务器
pnpm dev

# 4. 打包生产版本
pnpm build
```

### 4. 客户端节点部署

在每个参与训练的节点上执行：

```bash
cd back/datamark-admin

# 1. 安装 Python 依赖
pip3 install -r requirements.txt

# 2. 启动客户端（方式1：使用脚本）
./start_client.sh node-1 localhost:8080 RESNET

# 3. 启动客户端（方式2：直接运行）
python3 flower_client.py \
  --node-id node-1 \
  --server-address localhost:8080 \
  --model-type RESNET \
  --data-path /path/to/data
```

---

## 📖 使用指南

### 步骤1：启动系统

1. **启动后端服务**
   ```bash
   cd back/datamark-admin
   mvn spring-boot:run
   ```
   后端服务运行在：`http://localhost:9091`

2. **启动前端服务**
   ```bash
   cd front/data-mark-v3
   pnpm dev
   ```
   前端服务运行在：`http://localhost:3000`

3. **访问系统**
   - 打开浏览器访问：`http://localhost:3000`
   - 登录系统（使用现有账号）

### 步骤2：启动客户端节点

在至少2个节点上启动客户端：

```bash
# 节点1
./start_client.sh node-1 192.168.1.100:8080 RESNET

# 节点2
./start_client.sh node-2 192.168.1.100:8080 RESNET

# 节点3
./start_client.sh node-3 192.168.1.100:8080 RESNET
# ... 更多节点
```

**注意**：
- `node-id`：节点唯一标识
- `server-address`：后端服务器地址:Flower端口
- `model-type`：模型类型（需与训练任务一致）

### 步骤3：创建训练任务

1. **进入联邦学习管理页面**
   - 点击左侧菜单"联邦学习"

2. **查看节点状态**
   - 点击"节点管理"标签
   - 确认所有客户端节点都显示为"在线"状态

3. **创建训练任务**
   - 点击"训练任务"标签
   - 点击"创建训练任务"按钮
   - 填写训练参数：
     - **模型类型**：选择模型（如 RESNET）
     - **训练轮数**：10（默认）
     - **学习率**：0.001
     - **批次大小**：32
     - **本地训练轮数**：5
     - **精度下降阈值**：5.0%
     - **参与节点**：选择在线的节点
   - 点击"创建"

4. **启动训练**
   - 在任务列表中找到刚创建的任务
   - 点击"启动"按钮
   - 等待 Flower Server 启动（约5秒）
   - 客户端自动连接并开始训练

### 步骤4：监控训练过程

1. **查看任务状态**
   - 任务状态会实时更新：
     - `CREATED`：已创建
     - `RUNNING`：训练中
     - `COMPLETED`：已完成
     - `DEGRADED`：精度下降
     - `FAILED`：失败
     - `STOPPED`：已停止

2. **查看训练进度**
   - 当前轮次/总轮数
   - 当前精度
   - 参与节点数

3. **查看详细监控**
   - 点击任务的"查看"按钮
   - 进入"训练监控"页面
   - 查看精度变化曲线
   - 查看任务详细信息

### 步骤5：处理异常情况

#### 节点离线
- 系统自动检测（30秒超时）
- 节点状态变为"离线"
- 训练任务继续进行（其他节点）
- 节点重新上线后自动恢复

#### 精度下降
- 系统实时监控每轮精度
- 精度下降超过阈值时：
  - 任务状态变为 `DEGRADED`
  - 发出告警
  - 管理员可选择：
    - 继续训练
    - 停止任务
    - 调整超参数

#### 训练失败
- 查看错误信息
- 检查日志：`logs/spring.log`
- 常见原因：
  - 客户端未启动
  - 数据路径错误
  - 模型加载失败
  - 端口被占用

---

## 📡 API文档

### 节点管理 API

#### 1. 注册节点
```http
POST /federated/register
Content-Type: application/json

{
  "nodeId": "node-1",
  "host": "192.168.1.101",
  "port": 5000,
  "metadata": {
    "cpuCores": 8,
    "memoryGb": 16,
    "gpuCount": 1,
    "gpuModel": "NVIDIA RTX 3090",
    "datasetSize": 10000
  }
}
```

**响应：**
```json
{
  "code": 200,
  "msg": "success",
  "data": {
    "nodeId": "node-1",
    "host": "192.168.1.101",
    "port": 5000,
    "active": true,
    "lastHeartbeatAt": "2025-01-20T10:30:00"
  }
}
```

#### 2. 发送心跳
```http
POST /federated/heartbeat/{nodeId}
Content-Type: application/json

{
  "status": "training",
  "currentEpoch": 5,
  "gpuUtil": 85.5
}
```

#### 3. 获取所有节点
```http
GET /federated/nodes
```

**响应：**
```json
{
  "code": 200,
  "msg": "success",
  "data": [
    {
      "nodeId": "node-1",
      "host": "192.168.1.101",
      "port": 5000,
      "active": true,
      "gpuCount": 1,
      "datasetSize": 10000,
      "lastHeartbeatAt": "2025-01-20T10:30:00"
    }
  ]
}
```

### 训练任务 API

#### 1. 创建训练任务
```http
POST /federated/jobs?modelType=RESNET
Content-Type: application/json

{
  "hyperParameters": {
    "numRounds": 10,
    "learningRate": 0.001,
    "batchSize": 32,
    "localEpochs": 5
  },
  "participantNodeIds": ["node-1", "node-2", "node-3"],
  "allowedDropPercent": 5.0
}
```

**响应：**
```json
{
  "code": 200,
  "msg": "success",
  "data": {
    "jobId": "job-12345",
    "modelType": "RESNET",
    "status": "CREATED",
    "participantCount": 3,
    "createdAt": "2025-01-20T10:35:00"
  }
}
```

#### 2. 启动训练任务
```http
POST /federated/jobs/{jobId}/start
```

#### 3. 停止训练任务
```http
POST /federated/jobs/{jobId}/stop
```

#### 4. 获取所有任务
```http
GET /federated/jobs
```

---

## ❓ 常见问题

### Q1: 客户端无法连接到 Flower Server？

**解决方案：**
1. 检查防火墙设置，开放端口（默认8080-8090）
2. 检查服务器地址是否正确
3. 确认 Flower Server 是否已启动：
   ```bash
   ps aux | grep flower_server
   ```
4. 查看后端日志：
   ```bash
   tail -f logs/spring.log
   ```

### Q2: 训练任务启动失败？

**解决方案：**
1. 检查 Python 环境：
   ```bash
   python3 --version  # 需要 3.8+
   python3 -c "import flwr; import torch"
   ```
2. 检查依赖安装：
   ```bash
   pip3 install -r requirements.txt
   ```
3. 检查端口占用：
   ```bash
   netstat -tuln | grep 8080
   ```
4. 查看详细错误信息（后端日志）

### Q3: 精度监控显示为0？

**可能原因：**
- 模型尚未评估（首轮训练后才有精度）
- 评估失败（数据集问题）
- 客户端未正确实现评估方法

**解决方案：**
等待第一轮训练完成，或检查客户端评估代码

### Q4: 节点一直显示离线？

**解决方案：**
1. 检查节点是否启动：
   ```bash
   ps aux | grep flower_client
   ```
2. 检查心跳配置（默认30秒超时）
3. 检查网络连接
4. 手动发送心跳测试：
   ```bash
   curl -X POST http://localhost:9091/federated/heartbeat/node-1 \
     -H "Content-Type: application/json" \
     -d '{}'
   ```

### Q5: 如何添加新的模型类型？

**步骤：**
1. 在 `ModelType.java` 中添加新的枚举值
2. 在 `flower_client.py` 中实现对应的模型适配器
3. 在 `flower_server.py` 中添加模型类型支持
4. 在前端 `federated-learning/index.vue` 中添加选项

### Q6: 训练速度太慢？

**优化建议：**
1. **使用 GPU**：
   - 安装 CUDA 版本的 PyTorch
   - 确保客户端有可用 GPU
2. **调整批次大小**：
   - 增大 `batchSize`（如 64, 128）
3. **减少本地轮数**：
   - 减小 `localEpochs`（如 3）
4. **减少训练轮数**：
   - 减小 `numRounds`（如 5）
5. **使用更少的节点**：
   - 减少参与节点数量（通信开销）

### Q7: 如何备份训练数据？

**备份内容：**
1. **数据库备份**：
   ```bash
   mysqldump -u root -p datamark > backup_$(date +%Y%m%d).sql
   ```
2. **模型文件备份**：
   ```bash
   cp -r /usr/local/qczyDataMark/fl_models /backup/
   ```
3. **配置文件备份**：
   ```bash
   cp src/main/resources/application*.yml /backup/
   ```

---

## 📞 技术支持

如有问题或建议，请联系：

- **项目文档**：本 README 文件
- **API 文档**：`http://localhost:9091/swagger-ui.html`
- **GitHub Issues**：提交问题和建议

---

## 📄 许可证

本项目采用 [MIT License](LICENSE)

---

## 🎉 致谢

感谢以下开源项目：

- [Flower](https://flower.dev/) - 联邦学习框架
- [PyTorch](https://pytorch.org/) - 深度学习框架
- [Spring Boot](https://spring.io/projects/spring-boot) - 后端框架
- [Vue.js](https://vuejs.org/) - 前端框架
- [Naive UI](https://www.naiveui.com/) - UI 组件库

---

**最后更新：2025-01-20**
