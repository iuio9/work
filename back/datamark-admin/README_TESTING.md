# 大小模型协同训练系统 - Python环境测试

## 🎯 快速开始

如果您想先在Python环境下测试，请按以下顺序执行：

### 1️⃣ 环境检查（必须）

```bash
python3 test_environment.py
```

检查Python版本、PyTorch、CUDA、依赖包等是否正确安装。

**预计用时**: 10-30秒

---

### 2️⃣ 数据集测试（推荐）

```bash
python3 test_dataset_loading.py
```

测试CIFAR-10等数据集的加载，第一次运行会自动下载约170MB数据。

**预计用时**: 30秒-2分钟（首次需要下载）

---

### 3️⃣ 模型加载测试（推荐）

```bash
python3 test_model_loading.py
```

测试ResNet、ViT、YOLOv8、UNet、LSTM等模型是否能正常加载。

**预计用时**: 10-30秒

---

### 4️⃣ 简单训练测试（可选）

```bash
python3 test_simple_training.py
```

在CIFAR-10上训练ResNet18两个epoch，验证完整训练流程。

**预计用时**:
- GPU: 2-5分钟
- CPU: 30-60分钟（可Ctrl+C提前终止）

---

## 📚 详细文档

- **[Python测试指南](PYTHON_TESTING_GUIDE.md)** - 逐步测试说明、常见问题、性能基准
- **[完整部署指南](../../COMPLETE_DEPLOYMENT_GUIDE.md)** - 系统架构、完整安装、配置说明

---

## 🔍 测试脚本说明

| 脚本 | 用途 | 必需性 | 耗时 |
|-----|------|--------|------|
| `test_environment.py` | 检查Python环境和依赖 | ✅ 必须 | 10-30秒 |
| `test_dataset_loading.py` | 测试数据集加载 | ⭐ 推荐 | 30秒-2分钟 |
| `test_model_loading.py` | 测试模型加载 | ⭐ 推荐 | 10-30秒 |
| `test_simple_training.py` | 完整训练流程 | ⚪ 可选 | 2-60分钟 |

---

## ✅ 最小可运行配置

如果您只想快速验证环境是否可用：

```bash
# 步骤1: 检查环境
python3 test_environment.py

# 步骤2: 快速训练测试（可提前Ctrl+C终止）
python3 test_simple_training.py
```

只要能启动训练并看到第一个batch的输出，就说明环境配置成功。

---

## 🐛 常见问题

### ❌ 没有GPU，可以运行吗？

可以，但训练会很慢。建议：
- 使用小模型（ResNet18, YOLOv8-n）
- 减小batch_size到16或8
- 减少训练轮数

### ❌ CUDA不可用

检查：
```bash
nvidia-smi  # 检查驱动
python3 -c "import torch; print(torch.cuda.is_available())"
```

如果返回False，重新安装CUDA版本的PyTorch：
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### ❌ 依赖包缺失

```bash
pip install -r requirements.txt
```

### ❌ 网络下载失败

使用国内镜像源：
```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

---

## 📊 验收标准

完成测试后，您应该看到：

- ✅ Python >= 3.8
- ✅ PyTorch >= 2.0
- ✅ CUDA可用（如果有GPU）
- ✅ 能加载CIFAR-10数据集
- ✅ 能加载ResNet18模型
- ✅ 能运行至少1个训练epoch

达到以上标准后，可以继续配置完整系统。

---

## 🚀 下一步

测试通过后：

1. **配置数据库**: 参考 `COMPLETE_DEPLOYMENT_GUIDE.md` 第3.1节
2. **启动后端**: `mvn spring-boot:run`
3. **启动前端**: `cd ../../front/data-mark-v3 && npm run dev`
4. **创建训练任务**: 浏览器访问 http://localhost:3000

---

## 📞 技术支持

遇到问题？请查看：
- [PYTHON_TESTING_GUIDE.md](PYTHON_TESTING_GUIDE.md) - 详细测试步骤和问题排查
- [COMPLETE_DEPLOYMENT_GUIDE.md](../../COMPLETE_DEPLOYMENT_GUIDE.md) - 完整系统部署
- 错误信息和堆栈跟踪

---

**祝您使用顺利！** 🎉
