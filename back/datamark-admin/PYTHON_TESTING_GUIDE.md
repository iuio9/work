# Python环境测试指南

本指南帮助您逐步验证Python环境，确保系统能够正常运行。

## 📋 测试清单

我们提供了4个测试脚本，按以下顺序执行：

1. ✅ **环境检查** - `test_environment.py`
2. ✅ **数据集加载** - `test_dataset_loading.py`
3. ✅ **模型加载** - `test_model_loading.py`
4. ✅ **简单训练** - `test_simple_training.py`

## 🚀 快速开始

### 第一步：检查Python环境

```bash
cd /home/user/work/back/datamark-admin
python3 test_environment.py
```

**这个脚本会检查：**
- Python版本（需要 >= 3.8）
- PyTorch和CUDA是否安装
- 必需依赖包（transformers, peft, ultralytics等）
- GPU是否可用
- 张量运算是否正常
- 数据集和模型路径

**预期结果：**
- ✅ Python版本符合要求
- ✅ PyTorch已安装
- ⚠️ 部分可选包未安装（正常）
- ✅ GPU可用（如果有NVIDIA显卡）

**如果出现问题：**
```bash
# 安装PyTorch (CUDA版本)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 安装必需依赖
pip install -r requirements.txt
```

---

### 第二步：测试数据集加载

```bash
python3 test_dataset_loading.py
```

**这个脚本会检查：**
- CIFAR-10自动下载和加载
- CIFAR-100支持
- 自定义数据集格式
- 数据增强效果
- DataLoader性能

**预期结果：**
- ✅ CIFAR-10下载成功（第一次运行会下载，约170MB）
- ✅ 训练集50,000张，测试集10,000张
- ✅ DataLoader可以正常迭代

**如果出现问题：**
```bash
# 手动下载CIFAR-10（如果自动下载失败）
mkdir -p datasets
cd datasets
wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
tar -xzf cifar-10-python.tar.gz
cd ..
```

---

### 第三步：测试模型加载

```bash
python3 test_model_loading.py
```

**这个脚本会测试：**
- ResNet系列（18/34/50/101）
- Vision Transformer (ViT)
- YOLOv8系列
- UNet（分割）
- LSTM（时序）
- LoRA支持

**预期结果：**
- ✅ ResNet18/34/50/101 全部加载成功
- ✅ 可以执行前向传播
- ⚠️ 部分模型需要额外库（transformers, ultralytics等）

**如果出现问题：**
```bash
# 安装可选依赖
pip install transformers>=4.35.0
pip install ultralytics>=8.0.0
pip install segmentation-models-pytorch>=0.3.3
pip install peft>=0.7.0
```

---

### 第四步：运行简单训练

```bash
python3 test_simple_training.py
```

**这个脚本会：**
- 使用ResNet18在CIFAR-10上训练2个epoch
- 测试完整的训练循环
- 验证GPU加速
- 测试模型保存和加载

**预期结果：**
- ✅ 训练正常启动
- ✅ 每个epoch显示Loss和Accuracy
- ✅ 测试集评估正常
- ✅ 模型保存成功

**运行时间估算：**
- GPU (RTX 3090): 约2-3分钟
- GPU (GTX 1080): 约5-8分钟
- CPU: 约30-60分钟

**如果出现问题：**
```bash
# CUDA内存不足
# 修改脚本中的 BATCH_SIZE = 16 （默认32）

# CPU太慢
# 可以Ctrl+C中断，只要训练能启动就说明环境正常
```

---

## 📊 测试结果示例

### ✅ 完全通过（理想情况）

```
=======================================================================
🔍 大小模型协同训练系统 - Python环境测试
=======================================================================

✅ Python版本符合要求 (>= 3.8)
✅ PyTorch版本: 2.1.0
✅ CUDA可用
   CUDA版本: 11.8
   GPU数量: 1
   GPU 0: NVIDIA GeForce RTX 3090
   显存: 24.00 GB
✅ Transformers版本: 4.35.2
✅ PEFT版本: 0.7.1
✅ CPU张量运算正常
✅ GPU张量运算正常

测试完成！
```

### ⚠️ 部分通过（可接受）

```
✅ Python版本符合要求
✅ PyTorch版本: 2.1.0
⚠️  CUDA不可用，将使用CPU训练（速度会很慢）
⚠️  Transformers未安装（如需要ViT和Qwen2.5-VL才需要）
⚠️  PEFT未安装（如需要LoRA才需要）

说明：基础功能可用，可以训练ResNet/YOLOv8等模型
```

---

## 🛠️ 常见问题

### Q1: torch.cuda.is_available() 返回 False

**原因：**
- 安装了CPU版本的PyTorch
- NVIDIA驱动未安装
- CUDA版本不匹配

**解决：**
```bash
# 检查NVIDIA驱动
nvidia-smi

# 重新安装CUDA版本PyTorch
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Q2: CIFAR-10下载失败

**原因：**
- 网络连接问题
- 防火墙阻止

**解决：**
```bash
# 使用镜像源
export TORCH_HOME=/home/user/.cache/torch
mkdir -p datasets
# 手动下载后放入datasets文件夹
```

### Q3: 导入transformers失败

**原因：**
- 未安装或版本过低

**解决：**
```bash
pip install transformers>=4.35.0
# 如果网络慢，使用国内源
pip install transformers -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### Q4: GPU内存不足（OOM）

**原因：**
- batch_size太大
- 模型太大

**解决：**
```python
# 修改脚本中的配置
BATCH_SIZE = 16  # 降低批次大小
# 或者使用更小的模型
model = models.resnet18()  # 而不是resnet101
```

---

## 📝 测试通过后的下一步

### 1. 准备Qwen2.5-VL模型（可选）

如果需要使用大模型作为教师模型：

```bash
# 方案1: 使用Hugging Face下载
huggingface-cli download Qwen/Qwen2.5-VL-8B-Instruct --local-dir /home/user/models/qwen2.5-vl-8b

# 方案2: 使用ModelScope下载（国内推荐）
pip install modelscope
python3 << EOF
from modelscope import snapshot_download
model_dir = snapshot_download('Qwen/Qwen2.5-VL-8B-Instruct', cache_dir='/home/user/models')
EOF
```

### 2. 准备自定义数据集（可选）

如果使用自己的数据：

```bash
mkdir -p /home/user/datasets/my_dataset
cd /home/user/datasets/my_dataset

# 创建类别文件夹
mkdir class_1 class_2 class_3

# 放入图片
# class_1/*.jpg
# class_2/*.jpg
# class_3/*.jpg
```

### 3. 配置后端系统

```bash
cd /home/user/work/back/datamark-admin

# 修改 application.yml
# - 配置数据库连接
# - 配置文件存储路径
# - 配置模型路径

# 启动后端
mvn spring-boot:run
```

### 4. 配置前端系统

```bash
cd /home/user/work/front/data-mark-v3

# 安装依赖
npm install

# 启动开发服务器
npm run dev
```

### 5. 创建第一个训练任务

1. 打开浏览器访问前端 http://localhost:3000
2. 进入"模型蒸馏训练"页面
3. 点击"新建训练任务"
4. 填写配置：
   - 任务名称：测试训练
   - 教师模型：Qwen2.5-VL-8B（或留空使用ResNet预训练）
   - 学生模型：ResNet18
   - 数据集：选择已上传的CIFAR-10
   - 训练轮数：5
   - 批次大小：32
   - 学习率：0.001
5. 点击"开始训练"

---

## 🎯 性能基准参考

在不同硬件上的预期性能：

| 硬件配置 | ResNet18/CIFAR-10 | ResNet50/ImageNet | Qwen2.5-VL蒸馏 |
|---------|-------------------|-------------------|----------------|
| RTX 4090 (24GB) | 30s/epoch | 5min/epoch | 20min/epoch |
| RTX 3090 (24GB) | 45s/epoch | 8min/epoch | 30min/epoch |
| RTX 3060 (12GB) | 90s/epoch | 15min/epoch | 无法运行 |
| GTX 1080 Ti (11GB) | 120s/epoch | 20min/epoch | 无法运行 |
| CPU (16核) | 30min/epoch | 不推荐 | 不推荐 |

---

## ✅ 验证完成清单

完成以下所有项目后，您的环境已准备就绪：

- [ ] test_environment.py 运行通过
- [ ] test_dataset_loading.py 成功加载CIFAR-10
- [ ] test_model_loading.py ResNet系列加载成功
- [ ] test_simple_training.py 至少运行1个epoch
- [ ] GPU可用（可选，但强烈推荐）
- [ ] Qwen2.5-VL已下载（可选）
- [ ] 数据库已配置
- [ ] 后端服务可启动
- [ ] 前端服务可访问

---

## 📞 获取帮助

如果遇到问题：

1. 查看错误信息中的具体原因
2. 参考本文档的"常见问题"部分
3. 查看完整部署指南：`COMPLETE_DEPLOYMENT_GUIDE.md`
4. 检查requirements.txt中的依赖版本

祝您使用顺利！🎉
