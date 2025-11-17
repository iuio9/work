#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Flower Server for Federated Learning
=====================================
联邦学习服务器端实现，基于 Flower 框架

功能：
1. 协调多个客户端进行联邦学习训练
2. 使用 FedAvg 策略聚合客户端模型参数
3. 监控训练精度，防止精度下降超过阈值（默认 5%）
4. 支持多种模型类型：YOLOv8, LSTM, UNet, ResNet, Vision Transformer

工作流程：
1. 等待客户端连接（至少 min_clients 个）
2. 每轮训练：
   - 向所有客户端发送全局模型参数
   - 客户端本地训练后返回参数更新
   - 使用 FedAvg 聚合所有更新
   - 评估聚合后的模型精度
   - 检查精度是否下降超过阈值
3. 重复步骤 2 直到达到指定轮数

作者：AI Assistant
日期：2025-01-20
"""
import flwr as fl
from flwr.server import ServerConfig
from flwr.server.strategy import FedAvg
from typing import List, Tuple, Dict, Any
import argparse
import json
import logging
from flwr.common import Metrics

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# 全局状态字典：用于存储训练状态，可与 Java 后端通信
# ============================================================================
TRAINING_STATE = {
    "round": 0,                    # 当前训练轮次
    "accuracy": 0.0,               # 当前平均精度
    "loss": 0.0,                   # 当前平均损失
    "baseline_accuracy": None,     # 基线精度（首次评估时记录）
    "allowed_drop_percent": 5.0,   # 允许的精度下降百分比（默认 5%）
    "status": "CREATED"            # 训练状态：CREATED/RUNNING/DEGRADED/STOPPED
}

# ============================================================================
# 指标聚合函数
# ============================================================================

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """
    聚合客户端指标（加权平均）
    
    参数:
        metrics: 客户端指标列表，每个元素为 (样本数, 指标字典)
                例如: [(100, {"accuracy": 0.9, "loss": 0.1}), ...]
    
    返回:
        聚合后的指标字典，包含加权平均的 accuracy 和 loss
    
    说明:
        使用样本数作为权重，确保数据量大的客户端对最终指标影响更大
        公式：加权平均 = Σ(样本数 × 指标值) / Σ(样本数)
    """
    # 计算加权精度：每个客户端的精度乘以样本数
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    # 计算加权损失：每个客户端的损失乘以样本数
    losses = [num_examples * m["loss"] for num_examples, m in metrics]
    # 总样本数
    examples = [num_examples for num_examples, _ in metrics]
    
    # 返回加权平均值
    return {
        "accuracy": sum(accuracies) / sum(examples),  # 加权平均精度
        "loss": sum(losses) / sum(examples),          # 加权平均损失
    }

def fit_config(server_round: int) -> Dict[str, Any]:
    """
    返回每轮训练的配置参数
    
    参数:
        server_round: 当前服务器轮次（从 1 开始）
    
    返回:
        配置字典，包含：
        - server_round: 当前轮次
        - learning_rate: 学习率（可根据轮次动态调整）
        - batch_size: 批次大小
    
    说明:
        此函数在每轮训练开始时被调用，返回的配置会发送给所有客户端
        可以根据轮次动态调整超参数（如学习率衰减）
    """
    return {
        "server_round": server_round,
        "learning_rate": 0.001,  # 可根据轮次调整，例如：0.001 * (0.9 ** server_round)
        "batch_size": 32,
    }

def evaluate_config(server_round: int) -> Dict[str, Any]:
    """
    返回每轮评估的配置参数
    
    参数:
        server_round: 当前服务器轮次
    
    返回:
        评估配置字典
    
    说明:
        此函数在每轮评估时被调用，可以传递评估相关的配置
    """
    return {"server_round": server_round}

# ============================================================================
# 自定义策略：扩展 FedAvg，增加精度监控功能
# ============================================================================

class AccuracyMonitorStrategy(FedAvg):
    """
    扩展 FedAvg 策略，增加精度下降监控
    
    功能：
    1. 继承 FedAvg 的模型参数聚合功能
    2. 在每轮训练后监控精度变化
    3. 如果精度下降超过阈值，将状态标记为 DEGRADED
    
    精度保护机制：
    - 首次评估时记录基线精度
    - 后续每轮计算精度下降百分比
    - 如果下降超过 allowed_drop_percent，标记为降级状态
    """
    
    def aggregate_fit(self, server_round, results, failures):
        """
        聚合训练结果，并监控精度变化
        
        参数:
            server_round: 当前轮次
            results: 客户端训练结果列表
            failures: 失败的客户端列表
        
        返回:
            (聚合后的模型参数, 聚合后的指标)
        
        流程：
        1. 调用父类方法进行 FedAvg 参数聚合
        2. 从聚合指标中提取平均精度
        3. 更新全局状态
        4. 检查精度是否下降超过阈值
        """
        # 调用父类方法进行 FedAvg 聚合（加权平均模型参数）
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)
        
        # 如果聚合成功且有指标，进行精度监控
        if aggregated_metrics:
            # 获取聚合后的平均精度
            avg_accuracy = aggregated_metrics.get("accuracy", 0.0)
            # 更新全局状态
            TRAINING_STATE["round"] = server_round
            TRAINING_STATE["accuracy"] = avg_accuracy
            
            # 检查精度下降
            if TRAINING_STATE["baseline_accuracy"] is None:
                # 首次评估：建立基线精度
                TRAINING_STATE["baseline_accuracy"] = avg_accuracy
                logger.info(f"Round {server_round}: Baseline accuracy = {avg_accuracy:.4f}")
            else:
                # 后续轮次：计算精度下降百分比
                baseline = TRAINING_STATE["baseline_accuracy"]
                drop = (baseline - avg_accuracy) / baseline * 100.0  # 下降百分比
                allowed = TRAINING_STATE["allowed_drop_percent"]
                
                if drop > allowed:
                    # 精度下降超过阈值：标记为降级状态
                    TRAINING_STATE["status"] = "DEGRADED"
                    logger.warning(f"Round {server_round}: Accuracy dropped {drop:.2f}% (allowed: {allowed}%)")
                else:
                    # 精度正常：保持运行状态
                    TRAINING_STATE["status"] = "RUNNING"
                    logger.info(f"Round {server_round}: Accuracy = {avg_accuracy:.4f} (drop: {drop:.2f}%)")
        
        return aggregated_parameters, aggregated_metrics
    
    def aggregate_evaluate(self, server_round, results, failures):
        """
        聚合评估结果
        
        参数:
            server_round: 当前轮次
            results: 客户端评估结果列表
            failures: 失败的客户端列表
        
        返回:
            (聚合后的损失, 聚合后的指标)
        
        说明:
            在每轮训练结束后，服务器会要求客户端评估模型
            此方法聚合所有客户端的评估结果
        """
        # 调用父类方法聚合评估结果
        aggregated_loss, aggregated_metrics = super().aggregate_evaluate(server_round, results, failures)
        
        # 更新全局状态中的精度信息
        if aggregated_metrics:
            avg_accuracy = aggregated_metrics.get("accuracy", 0.0)
            TRAINING_STATE["accuracy"] = avg_accuracy
            logger.info(f"Round {server_round} Evaluation: Accuracy = {avg_accuracy:.4f}")
        
        return aggregated_loss, aggregated_metrics

# ============================================================================
# 服务器启动函数
# ============================================================================

def start_server(
    model_type: str,
    num_rounds: int = 10,
    min_available_clients: int = 2,
    min_fit_clients: int = 2,
    baseline_accuracy: float = None,
    allowed_drop_percent: float = 5.0,
    port: int = 8080
):
    """
    启动 Flower Server
    
    参数:
        model_type: 模型类型（YOLO_V8, LSTM, UNET, RESNET, VISION_TRANSFORMER）
        num_rounds: 训练轮数（默认 10）
        min_available_clients: 最少可用客户端数（默认 2）
        min_fit_clients: 每轮最少参与训练的客户端数（默认 2）
        baseline_accuracy: 基线精度（可选，如果提供则使用，否则使用首次评估结果）
        allowed_drop_percent: 允许的精度下降百分比（默认 5.0%）
        port: 服务器监听端口（默认 8080）
    
    流程：
    1. 初始化全局状态
    2. 创建自定义策略（带精度监控）
    3. 配置服务器参数
    4. 启动服务器，等待客户端连接
    
    说明：
    - fraction_fit=1.0: 每轮选择 100% 的客户端参与训练
    - fraction_evaluate=1.0: 每轮选择 100% 的客户端参与评估
    - min_fit_clients: 至少需要这么多客户端才能开始训练
    - 服务器会一直运行直到完成所有轮次或手动停止
    """
    # 初始化全局状态
    TRAINING_STATE["baseline_accuracy"] = baseline_accuracy
    TRAINING_STATE["allowed_drop_percent"] = allowed_drop_percent
    TRAINING_STATE["status"] = "RUNNING"
    
    # 创建自定义策略（带精度监控的 FedAvg）
    strategy = AccuracyMonitorStrategy(
        fraction_fit=1.0,                    # 每轮选择 100% 客户端训练
        fraction_evaluate=1.0,               # 每轮选择 100% 客户端评估
        min_fit_clients=min_fit_clients,     # 最少训练客户端数
        min_evaluate_clients=min_fit_clients, # 最少评估客户端数（与训练数相同）
        min_available_clients=min_available_clients, # 最少可用客户端数
        on_fit_config_fn=fit_config,          # 训练配置函数
        on_evaluate_config_fn=evaluate_config, # 评估配置函数
        evaluate_metrics_aggregation_fn=weighted_average, # 指标聚合函数
    )
    
    # 创建服务器配置
    config = ServerConfig(num_rounds=num_rounds)
    
    # 记录启动信息
    logger.info(f"Starting Flower Server for {model_type} on port {port}")
    logger.info(f"Min clients: {min_fit_clients}, Rounds: {num_rounds}")
    
    # 启动服务器（阻塞运行，直到所有轮次完成）
    fl.server.start_server(
        server_address=f"0.0.0.0:{port}",  # 监听所有网络接口
        config=config,                      # 服务器配置
        strategy=strategy,                  # 训练策略
    )

# ============================================================================
# 主程序入口
# ============================================================================

if __name__ == "__main__":
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description="Flower Federated Learning Server")
    
    # 必需参数：模型类型
    parser.add_argument("--model-type", type=str, required=True,
                       choices=["YOLO_V8", "LSTM", "UNET", "RESNET", "VISION_TRANSFORMER"],
                       help="模型类型：YOLO_V8, LSTM, UNET, RESNET, VISION_TRANSFORMER")
    
    # 可选参数：训练轮数（默认 10）
    parser.add_argument("--num-rounds", type=int, default=10, 
                       help="训练轮数（默认：10）")
    
    # 可选参数：最少客户端数（默认 2）
    parser.add_argument("--min-clients", type=int, default=2, 
                       help="最少可用客户端数（默认：2）")
    
    # 可选参数：服务器端口（默认 8080）
    parser.add_argument("--port", type=int, default=8080, 
                       help="服务器监听端口（默认：8080）")
    
    # 可选参数：基线精度（可选）
    parser.add_argument("--baseline-accuracy", type=float, default=None, 
                       help="基线精度（如果提供，将使用此值；否则使用首次评估结果）")
    
    # 可选参数：允许精度下降百分比（默认 5.0%）
    parser.add_argument("--allowed-drop-percent", type=float, default=5.0, 
                       help="允许的精度下降百分比（默认：5.0）")
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 启动服务器
    start_server(
        model_type=args.model_type,
        num_rounds=args.num_rounds,
        min_available_clients=args.min_clients,
        min_fit_clients=args.min_clients,
        baseline_accuracy=args.baseline_accuracy,
        allowed_drop_percent=args.allowed_drop_percent,
        port=args.port
    )

