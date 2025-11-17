#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Flower Client for Federated Learning
=====================================
联邦学习客户端实现，基于 Flower 框架

功能：
1. 连接到 Flower Server，参与联邦学习训练
2. 在本地数据集上训练模型
3. 将训练后的参数更新发送给服务器
4. 接收服务器聚合后的全局参数并应用到本地模型
5. 支持多种模型类型：YOLOv8, LSTM, UNet, ResNet, Vision Transformer

工作流程：
1. 连接到指定的 Flower Server
2. 等待服务器发送全局模型参数
3. 在本地数据集上训练一个 epoch
4. 将参数更新发送给服务器
5. 接收服务器聚合后的新全局参数
6. 重复步骤 2-5 直到训练完成

作者：AI Assistant
日期：2025-01-20
"""
import flwr as fl
import numpy as np
from typing import Dict, Tuple, Optional
import argparse
import logging
from flwr.common import NDArrays, Scalar

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# 模型适配器基类：定义统一的模型接口
# ============================================================================

class ModelAdapter:
    """
    模型适配器基类
    
    功能：
    1. 管理模型参数（获取、设置）
    2. 本地训练（fit）
    3. 模型评估（evaluate）
    
    说明：
    这是一个占位实现，实际使用时需要替换为真实模型（如 PyTorch/TensorFlow）
    子类应实现具体的模型初始化和训练逻辑
    """
    
    def __init__(self, model_type: str, node_id: str):
        """
        初始化模型适配器
        
        参数:
            model_type: 模型类型（YOLO_V8, LSTM, UNET, RESNET, VISION_TRANSFORMER）
            node_id: 节点ID（用于标识不同的客户端）
        """
        self.model_type = model_type
        self.node_id = node_id
        # 简化的模型参数（实际应为真实模型）
        # 注意：这是占位实现，实际应初始化真实模型（如 PyTorch/TensorFlow）
        self.parameters = self._init_parameters()
    
    def _init_parameters(self) -> NDArrays:
        """
        初始化模型参数
        
        返回:
            NDArrays: 模型参数列表（NumPy 数组列表）
        
        说明:
            这是占位实现，返回一个简单的随机参数
            实际应根据模型类型初始化真实模型：
            - PyTorch: model.state_dict() 转换为 NumPy
            - TensorFlow: model.get_weights()
            - YOLOv8: model.model.state_dict() 转换为 NumPy
        """
        # 占位：返回一个简单的参数列表
        # 实际应根据模型类型初始化真实模型（如 PyTorch/TensorFlow）
        return [np.random.randn(10, 10).astype(np.float32)]
    
    def get_parameters(self) -> NDArrays:
        """
        获取当前模型参数
        
        返回:
            NDArrays: 模型参数列表
        
        说明:
            Flower 需要将模型参数转换为 NumPy 数组列表
            实际实现中需要将模型权重（如 PyTorch 的 state_dict）转换为 NumPy
        """
        return self.parameters
    
    def set_parameters(self, parameters: NDArrays) -> None:
        """
        设置模型参数（从服务器接收的全局参数）
        
        参数:
            parameters: 模型参数列表（NumPy 数组列表）
        
        说明:
            将服务器聚合后的全局参数应用到本地模型
            实际实现中需要将 NumPy 数组转换回模型权重格式
        """
        self.parameters = parameters
    
    def fit(self, parameters: NDArrays, config: Dict[str, Scalar]) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        """
        本地训练一轮（FedAvg 的核心：客户端本地训练）
        
        参数:
            parameters: 服务器发送的全局模型参数
            config: 训练配置（包含学习率、批次大小等）
        
        返回:
            (更新后的参数, 训练样本数, 训练指标字典)
            - 更新后的参数：本地训练后的模型参数
            - 训练样本数：用于加权聚合（数据量大的客户端权重更大）
            - 训练指标：包含 loss、accuracy 等
        
        流程：
        1. 应用全局参数到本地模型
        2. 加载本地数据集
        3. 训练一个 epoch（或指定步数）
        4. 计算参数更新（新参数 - 旧参数）
        5. 返回更新后的参数和训练指标
        
        说明：
        这是占位实现，实际应：
        1. 加载本地数据集（如：train_loader = DataLoader(local_dataset, ...)）
        2. 训练模型（如：for batch in train_loader: loss.backward()）
        3. 提取模型参数（如：model.state_dict() 转换为 NumPy）
        4. 计算训练指标（loss、accuracy 等）
        """
        # 应用服务器发送的全局参数到本地模型
        self.set_parameters(parameters)
        
        # ====================================================================
        # 占位：模拟本地训练
        # 实际应：
        # 1. 加载本地数据集
        #    train_loader = DataLoader(local_dataset, batch_size=config["batch_size"])
        # 2. 训练一个 epoch
        #    for batch in train_loader:
        #        optimizer.zero_grad()
        #        loss = criterion(model(batch.x), batch.y)
        #        loss.backward()
        #        optimizer.step()
        # 3. 提取更新后的参数
        #    updated_params = model_to_numpy(model)
        # ====================================================================
        
        # 简单模拟：参数加一个小的随机更新（占位实现）
        updated = [p + np.random.randn(*p.shape).astype(np.float32) * 0.01 for p in self.parameters]
        self.set_parameters(updated)
        
        num_examples = 100  # 占位：实际应为本地数据集大小（len(local_dataset)）
        
        # 返回更新后的参数、样本数和训练指标
        return self.get_parameters(), num_examples, {"loss": 0.5, "accuracy": 0.85}
    
    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]) -> Tuple[float, int, Dict[str, Scalar]]:
        """
        评估模型性能
        
        参数:
            parameters: 要评估的模型参数（通常是聚合后的全局参数）
            config: 评估配置
        
        返回:
            (损失值, 评估样本数, 评估指标字典)
            - 损失值：模型在测试集上的平均损失
            - 评估样本数：用于加权聚合
            - 评估指标：包含 accuracy、precision、recall 等
        
        流程：
        1. 应用参数到本地模型
        2. 加载测试集
        3. 在测试集上评估模型
        4. 计算损失和指标
        
        说明：
        这是占位实现，实际应：
        1. 加载测试集（如：test_loader = DataLoader(test_dataset, ...)）
        2. 评估模型（如：with torch.no_grad(): predictions = model(test_x)）
        3. 计算指标（如：accuracy = (predictions == test_y).float().mean()）
        """
        # 应用参数到本地模型
        self.set_parameters(parameters)
        
        # ====================================================================
        # 占位：模拟评估
        # 实际应：
        # 1. 加载测试集
        #    test_loader = DataLoader(test_dataset, batch_size=32)
        # 2. 评估模型
        #    model.eval()
        #    with torch.no_grad():
        #        for batch in test_loader:
        #            outputs = model(batch.x)
        #            loss += criterion(outputs, batch.y).item()
        #            accuracy += calculate_accuracy(outputs, batch.y)
        # 3. 计算平均指标
        #    avg_loss = loss / len(test_loader)
        #    avg_accuracy = accuracy / len(test_loader)
        # ====================================================================
        
        num_examples = 50  # 占位：实际应为测试集大小（len(test_dataset)）
        loss = 0.3          # 占位：实际应从评估中计算
        accuracy = 0.88     # 占位：实际应从评估中计算
        
        return loss, num_examples, {"accuracy": accuracy}

# ============================================================================
# 具体模型适配器：为不同模型类型提供专门实现
# ============================================================================

class YOLOv8Adapter(ModelAdapter):
    """
    YOLOv8 目标检测模型适配器
    
    说明：
    实际实现时应：
    1. 初始化 YOLOv8 模型：from ultralytics import YOLO; self.model = YOLO("yolov8n.pt")
    2. 实现 fit() 方法：调用 model.train() 进行训练
    3. 实现 evaluate() 方法：调用 model.val() 进行评估
    4. 实现参数转换：YOLOv8 模型参数与 NumPy 数组的相互转换
    """
    def __init__(self, node_id: str):
        super().__init__("YOLO_V8", node_id)
        # 实际应初始化 YOLOv8 模型
        # from ultralytics import YOLO
        # self.model = YOLO("yolov8n.pt")
        # self.model_path = f"/models/yolov8_{node_id}.pt"  # 每个节点保存自己的模型
    
    def fit(self, parameters: NDArrays, config: Dict[str, Scalar]) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        """
        实际应调用 YOLOv8 训练
        
        示例代码（实际实现）：
        # 1. 将 NumPy 参数转换为 YOLOv8 模型权重
        # 2. 加载本地数据集
        #    results = self.model.train(data="local_dataset.yaml", epochs=1, ...)
        # 3. 提取训练后的参数
        #    updated_params = model_to_numpy(self.model)
        # 4. 返回结果
        """
        # 占位实现：调用父类方法
        return super().fit(parameters, config)

class LSTMAdapter(ModelAdapter):
    """
    LSTM 序列模型适配器
    
    说明：
    适用于时间序列预测、文本分类等任务
    实际实现时应使用 PyTorch 的 nn.LSTM 或 TensorFlow 的 LSTM 层
    """
    def __init__(self, node_id: str):
        super().__init__("LSTM", node_id)
        # 实际应初始化 LSTM 模型
        # import torch.nn as nn
        # self.model = nn.LSTM(input_size=..., hidden_size=..., ...)

class UNetAdapter(ModelAdapter):
    """
    UNet 分割模型适配器
    
    说明：
    适用于图像分割任务（如医学图像分割）
    实际实现时应使用 PyTorch 的 segmentation_models 或自定义 UNet
    """
    def __init__(self, node_id: str):
        super().__init__("UNET", node_id)
        # 实际应初始化 UNet 模型
        # from segmentation_models_pytorch import Unet
        # self.model = Unet(encoder_name="resnet34", ...)

class ResNetAdapter(ModelAdapter):
    """
    ResNet 分类模型适配器
    
    说明：
    适用于图像分类任务
    实际实现时应使用 torchvision.models.resnet 或 TensorFlow 的 ResNet
    """
    def __init__(self, node_id: str):
        super().__init__("RESNET", node_id)
        # 实际应初始化 ResNet 模型
        # import torchvision.models as models
        # self.model = models.resnet50(pretrained=False, ...)

class ViTAdapter(ModelAdapter):
    """
    Vision Transformer (ViT) 适配器
    
    说明：
    适用于图像分类、目标检测等任务
    实际实现时应使用 timm 库或 transformers 库的 ViT 模型
    """
    def __init__(self, node_id: str):
        super().__init__("VISION_TRANSFORMER", node_id)
        # 实际应初始化 ViT 模型
        # import timm
        # self.model = timm.create_model("vit_base_patch16_224", pretrained=False, ...)

# ============================================================================
# Flower 客户端创建函数
# ============================================================================

def create_client(model_type: str, node_id: str) -> fl.client.Client:
    """
    创建 Flower 客户端
    
    参数:
        model_type: 模型类型（YOLO_V8, LSTM, UNET, RESNET, VISION_TRANSFORMER）
        node_id: 节点ID（用于标识不同的客户端）
    
    返回:
        Flower 客户端实例（NumPyClient）
    
    流程：
    1. 根据模型类型选择对应的适配器类
    2. 实例化适配器
    3. 创建 FlowerClient 类，将适配器方法映射到 Flower 接口
    4. 返回客户端实例
    
    说明：
    FlowerClient 是 Flower 框架要求的接口，需要实现：
    - get_parameters(): 获取模型参数
    - fit(): 本地训练
    - evaluate(): 模型评估
    """
    # 根据模型类型选择适配器
    adapter_map = {
        "YOLO_V8": YOLOv8Adapter,
        "LSTM": LSTMAdapter,
        "UNET": UNetAdapter,
        "RESNET": ResNetAdapter,
        "VISION_TRANSFORMER": ViTAdapter,
    }
    
    # 获取适配器类（如果不存在则使用基类）
    adapter_class = adapter_map.get(model_type, ModelAdapter)
    # 实例化适配器
    adapter = adapter_class(node_id)
    
    # 创建 Flower 客户端类（实现 Flower 框架要求的接口）
    class FlowerClient(fl.client.NumPyClient):
        """
        Flower 客户端实现
        
        将适配器的方法映射到 Flower 框架的接口
        """
        def get_parameters(self, config):
            """获取模型参数（服务器调用）"""
            return adapter.get_parameters()
        
        def fit(self, parameters, config):
            """本地训练（服务器调用）"""
            return adapter.fit(parameters, config)
        
        def evaluate(self, parameters, config):
            """模型评估（服务器调用）"""
            return adapter.evaluate(parameters, config)
    
    return FlowerClient()

# ============================================================================
# 客户端启动函数
# ============================================================================

def start_client(
    model_type: str,
    node_id: str,
    server_address: str = "localhost:8080"
):
    """
    启动 Flower 客户端并连接到服务器
    
    参数:
        model_type: 模型类型（YOLO_V8, LSTM, UNET, RESNET, VISION_TRANSFORMER）
        node_id: 节点ID（用于标识不同的客户端）
        server_address: 服务器地址（格式：host:port，默认 localhost:8080）
    
    流程：
    1. 创建客户端实例
    2. 连接到 Flower Server
    3. 参与联邦学习训练（阻塞运行，直到训练完成）
    
    说明：
    - 客户端会一直运行直到服务器完成所有训练轮次
    - 如果服务器关闭，客户端会自动断开连接
    - 可以在多个节点上运行多个客户端实例（使用不同的 node_id）
    """
    logger.info(f"Starting Flower Client: {node_id} for {model_type}")
    logger.info(f"Connecting to server: {server_address}")
    
    # 创建客户端实例
    client = create_client(model_type, node_id)
    
    # 启动客户端并连接到服务器（阻塞运行）
    fl.client.start_numpy_client(
        server_address=server_address,  # 服务器地址
        client=client                   # 客户端实例
    )

# ============================================================================
# 主程序入口
# ============================================================================

if __name__ == "__main__":
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description="Flower Federated Learning Client")
    
    # 必需参数：模型类型
    parser.add_argument("--model-type", type=str, required=True,
                       choices=["YOLO_V8", "LSTM", "UNET", "RESNET", "VISION_TRANSFORMER"],
                       help="模型类型：YOLO_V8, LSTM, UNET, RESNET, VISION_TRANSFORMER")
    
    # 必需参数：节点ID
    parser.add_argument("--node-id", type=str, required=True, 
                       help="节点ID（用于标识不同的客户端，如：node-1, node-2）")
    
    # 可选参数：服务器地址（默认 localhost:8080）
    parser.add_argument("--server-address", type=str, default="localhost:8080",
                       help="Flower 服务器地址（格式：host:port，默认：localhost:8080）")
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 启动客户端
    start_client(
        model_type=args.model_type,
        node_id=args.node_id,
        server_address=args.server_address
    )

