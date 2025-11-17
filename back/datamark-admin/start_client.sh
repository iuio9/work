#!/bin/bash
# ============================================================================
# Flower Client 启动脚本
# ============================================================================
# 用途：启动联邦学习客户端节点
# 使用方法：./start_client.sh [node_id] [server_address] [model_type]
# ============================================================================

# 默认参数
NODE_ID=${1:-"node-$(date +%s)"}
SERVER_ADDRESS=${2:-"localhost:8080"}
MODEL_TYPE=${3:-"RESNET"}

echo "============================================"
echo "启动 Flower Client"
echo "============================================"
echo "节点 ID: $NODE_ID"
echo "服务器地址: $SERVER_ADDRESS"
echo "模型类型: $MODEL_TYPE"
echo "============================================"

# 检查 Python 环境
if ! command -v python3 &> /dev/null; then
    echo "错误: 未找到 Python 3"
    exit 1
fi

# 检查依赖
echo "检查 Python 依赖..."
python3 -c "import flwr; import torch" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "警告: 缺少必要的 Python 库"
    echo "正在安装依赖..."
    pip3 install -r requirements.txt
fi

# 启动客户端
echo "启动客户端..."
python3 flower_client.py \
    --node-id "$NODE_ID" \
    --server-address "$SERVER_ADDRESS" \
    --model-type "$MODEL_TYPE"
