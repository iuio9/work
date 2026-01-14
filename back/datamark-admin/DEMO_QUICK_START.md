# 🚀 演示数据 - 30秒快速开始

## 一键导入（推荐）

```bash
cd /home/user/work/back/datamark-admin
./import_demo_data.sh
```

输入密码：`qczy1717`，然后按 `y` 确认。

**完成！** 🎉

---

## 手动导入（备选）

```bash
cd /home/user/work/back/datamark-admin
mysql -u root -pqczy1717 datamark < demo_data.sql
```

**完成！** 🎉

---

## 验证导入

```bash
mysql -u root -pqczy1717 datamark -e "SELECT task_id, task_name, status FROM md_training_task WHERE task_id LIKE 'DEMO_%';"
```

应该看到3条记录：
- ✅ DEMO_COMPLETED（已完成）
- 🔄 DEMO_RUNNING（运行中）
- ⏸️ DEMO_PAUSED（已暂停）

---

## 启动系统

```bash
# 后端
cd /home/user/work/back/datamark-admin
mvn spring-boot:run

# 前端（新终端）
cd /home/user/work/front/data-mark-v3
npm run dev
```

打开浏览器：http://localhost:3000

---

## 演示数据内容

### DEMO_COMPLETED - ✅ 已完成
- **教师**：Qwen2.5-VL-3B（3B参数）
- **学生**：ResNet18（11M参数）
- **数据集**：CIFAR-10
- **准确率**：76.98%
- **压缩比**：273倍
- **用途**：展示训练完成效果，用于自动标注

### DEMO_RUNNING - 🔄 运行中
- **教师**：ResNet50
- **学生**：YOLOv8-n
- **数据集**：COCO2017
- **进度**：42/100 epoch
- **准确率**：63.89%
- **用途**：展示实时训练监控

### DEMO_PAUSED - ⏸️ 已暂停
- **教师**：ViT-Large
- **学生**：MobileViT-Small
- **数据集**：ImageNet-1K
- **进度**：87/200 epoch
- **准确率**：81.23%
- **用途**：展示暂停/恢复功能

---

## 数据量

- ✅ 3个训练任务
- ✅ 179条训练历史（完整曲线）
- ✅ 2条模型评估
- ✅ 3个LoRA预设

---

## 删除演示数据

演示后清理：

```bash
mysql -u root -pqczy1717 datamark -e "
DELETE FROM md_training_task WHERE task_id LIKE 'DEMO_%';
DELETE FROM md_training_history WHERE task_id LIKE 'DEMO_%';
DELETE FROM md_model_evaluation WHERE task_id LIKE 'DEMO_%';
"
```

---

## 问题排查

### 看不到演示数据？

1. 检查后端是否启动：`curl http://localhost:9091/api/distillation/tasks`
2. 检查数据是否导入：`mysql -u root -pqczy1717 datamark -e "SELECT COUNT(*) FROM md_training_task WHERE task_id LIKE 'DEMO_%';"`
3. 清空浏览器缓存：Ctrl+Shift+Delete

### 导入失败？

1. 检查MySQL服务：`sudo systemctl status mysql`
2. 检查数据库是否存在：`mysql -u root -pqczy1717 -e "SHOW DATABASES;"`
3. 检查表是否存在：`mysql -u root -pqczy1717 datamark -e "SHOW TABLES;"`

---

## 📖 详细文档

完整指南请查看：[DEMO_SETUP_GUIDE.md](DEMO_SETUP_GUIDE.md)

---

**明天演示加油！你一定行的！** 💪
