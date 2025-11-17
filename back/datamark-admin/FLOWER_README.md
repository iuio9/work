# Flower 联邦学习框架集成说明

## 架构说明

- **Java 后端**：作为编排层，管理 Flower Server 进程，提供 REST API
- **Flower Server (Python)**：实际的联邦学习协调服务器，通过 gRPC 与客户端通信
- **Flower Client (Python)**：各节点的客户端，执行本地训练

## 安装依赖

### Python 环境

```bash
pip install flwr
# 如果需要使用具体模型，还需安装：
# pip install ultralytics  # YOLOv8
# pip install torch torchvision  # PyTorch 模型
# pip install tensorflow  # TensorFlow 模型
```

## 配置

在 `application.yml` 中添加：

```yaml
flower:
  server:
    script:
      path: flower_server.py  # Flower Server 脚本路径
    port: 8080  # 默认端口
  python:
    executable: python3  # Python 可执行文件路径
```

## 使用流程

### 1. 启动 Java 后端

```bash
mvn spring-boot:run
```

### 2. 注册节点（可选，用于监控）

```bash
POST /federated/register
{
  "nodeId": "node-1",
  "host": "10.0.0.1",
  "port": 9001
}
```

### 3. 创建训练任务

```bash
POST /federated/jobs?modelType=YOLO_V8
{
  "hyperParameters": {
    "numRounds": 10,
    "learningRate": 0.001,
    "batchSize": 32
  },
  "participantNodeIds": ["node-1", "node-2", "node-3"],
  "baselineAccuracy": 0.9,
  "allowedDropPercent": 5.0
}
```

### 4. 启动任务（会自动启动 Flower Server）

```bash
POST /federated/jobs/{jobId}/start
```

### 5. 在各节点启动 Flower Client

```bash
python3 flower_client.py \
  --model-type YOLO_V8 \
  --node-id node-1 \
  --server-address localhost:8080
```

### 6. 查询状态

```bash
GET /federated/jobs  # 查看所有任务
GET /federated/nodes  # 查看所有节点
```

### 7. 停止任务

```bash
POST /federated/jobs/{jobId}/stop
```

## 支持的模型类型

- `YOLO_V8`: YOLOv8 目标检测
- `LSTM`: LSTM 序列模型
- `UNET`: UNet 分割模型
- `RESNET`: ResNet 分类模型
- `VISION_TRANSFORMER`: Vision Transformer 模型

## 自定义模型适配器

在 `flower_client.py` 中，可以扩展 `ModelAdapter` 类来实现具体的模型训练逻辑：

```python
class MyYOLOv8Adapter(ModelAdapter):
    def __init__(self, node_id: str):
        super().__init__("YOLO_V8", node_id)
        from ultralytics import YOLO
        self.model = YOLO("yolov8n.pt")
    
    def fit(self, parameters, config):
        # 加载本地数据集
        # 训练模型
        # 返回更新后的参数
        pass
```

## 精度监控

Flower Server 会自动监控精度下降：
- 首次评估时建立基线精度
- 每轮训练后检查精度是否下降超过阈值（默认 5%）
- 如果超过阈值，任务状态会变为 `DEGRADED`

## 注意事项

1. 确保 Python 环境已安装 `flwr` 包
2. Flower Server 和 Client 需要在同一网络或可访问的地址
3. 端口不要冲突，多个任务会自动分配不同端口
4. 客户端需要能够访问到服务器地址




