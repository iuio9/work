# 数据库迁移指南

## 概述

本指南用于执行数据集ID字段类型转换的数据库迁移。此迁移将 `dataset_id` 字段从 `BIGINT` 类型改为 `VARCHAR(100)` 类型，以支持非数字的数据集ID（如 'dialogue-zh-v2'）。

## 修改内容

### 1. md_training_task 表修改
- ✅ 修改 `dataset_id` 字段：BIGINT → VARCHAR(100)
- ✅ 添加 `val_dataset_id` 字段：VARCHAR(100)
- ✅ 添加 `val_dataset_name` 字段：VARCHAR(255)
- ✅ 添加 `training_config` 字段：TEXT（用于存储高级训练配置JSON）
- ✅ 添加 `description` 字段：TEXT（任务描述）

### 2. md_model_evaluation 表修改
- ✅ 修改 `eval_dataset_id` 字段：BIGINT → VARCHAR(100)

## 执行步骤

### 方式一：使用MySQL命令行（推荐）

1. **连接到MySQL数据库**
   ```bash
   mysql -h localhost -u root -p
   ```
   输入密码：`20031217scz`

2. **执行迁移脚本**
   ```bash
   source /home/user/work/back/datamark-admin/update_dataset_id_to_string.sql
   ```

3. **验证修改**
   ```sql
   USE mark;
   DESCRIBE md_training_task;
   DESCRIBE md_model_evaluation;
   ```

### 方式二：直接执行SQL文件

```bash
mysql -h localhost -u root -p20031217scz < /home/user/work/back/datamark-admin/update_dataset_id_to_string.sql
```

### 方式三：使用数据库管理工具

如果你使用 Navicat、DataGrip、MySQL Workbench 等数据库管理工具：

1. 连接到数据库
   - Host: localhost
   - Port: 3306
   - Database: mark
   - Username: root
   - Password: 20031217scz

2. 打开 SQL 编辑器

3. 粘贴 `update_dataset_id_to_string.sql` 文件内容

4. 执行 SQL 脚本

## 执行后的操作

### 1. 重启后端服务

数据库迁移完成后，必须重启后端服务以使更改生效：

```bash
# 如果后端服务在运行，先停止
# 然后重新启动
cd /home/user/work/back/datamark-admin
mvn spring-boot:run
# 或者如果已经打包
java -jar target/datamark-admin.jar
```

### 2. 验证功能

重启后端服务后，测试以下功能：

1. **创建训练任务**
   - 打开前端：大小模型协同训练页面
   - 在 Tab1 配置模型（教师模型、学生模型、LoRA配置等）
   - 在 Tab2 创建训练任务
   - 使用字符串类型的数据集ID（如 'dialogue-zh-v2'）
   - 确认任务创建成功

2. **检查数据库记录**
   ```sql
   USE mark;
   SELECT task_id, task_name, dataset_id, val_dataset_id
   FROM md_training_task
   ORDER BY created_time DESC
   LIMIT 5;
   ```

## 回滚方案

如果迁移出现问题，可以使用以下SQL回滚：

```sql
USE mark;

-- 回滚 md_training_task 表
ALTER TABLE `md_training_task`
  MODIFY COLUMN `dataset_id` BIGINT(20) DEFAULT NULL COMMENT '训练数据集ID';

ALTER TABLE `md_training_task`
  DROP COLUMN IF EXISTS `val_dataset_id`;

ALTER TABLE `md_training_task`
  DROP COLUMN IF EXISTS `val_dataset_name`;

ALTER TABLE `md_training_task`
  DROP COLUMN IF EXISTS `training_config`;

-- 回滚 md_model_evaluation 表
ALTER TABLE `md_model_evaluation`
  MODIFY COLUMN `eval_dataset_id` BIGINT(20) DEFAULT NULL COMMENT '评估数据集ID';
```

**⚠️ 注意：回滚将导致已存储的字符串类型数据集ID丢失！**

## 常见问题

### Q1: 执行SQL时报错 "Table doesn't exist"
**A:** 确认数据库名称是否正确。本项目使用的数据库名是 `mark`，不是 `datamark`。

### Q2: 执行SQL时报错 "Duplicate column name"
**A:** 这表示某些字段已经存在。SQL脚本已经包含了检查逻辑，会跳过已存在的字段。这不是错误，可以忽略。

### Q3: 后端启动时报错 "Jackson deserialization error"
**A:** 这表示数据库迁移可能没有成功执行。请确保：
1. SQL脚本已正确执行
2. 所有字段类型已更新为 VARCHAR
3. 后端服务已重启

### Q4: 前端提交任务时仍然报 "not a valid Long value" 错误
**A:** 确保以下三步都已完成：
1. ✅ Java代码已修改（CreateTaskRequestDTO.java 和 MdTrainingTaskEntity.java）
2. ✅ 数据库schema已更新（执行本SQL脚本）
3. ✅ 后端服务已重启

## 技术背景

### 为什么要做这个修改？

原始设计中，`dataset_id` 字段使用 `BIGINT` 类型，只能存储数字ID。但实际使用中发现：

1. 某些数据集使用字符串ID（如 'dialogue-zh-v2', 'coco-2017'）
2. 前端直接发送字符串ID到后端
3. Jackson反序列化时报错：`Cannot deserialize value of type java.lang.Long from String "dialogue-zh-v2"`

### 解决方案

1. **前端**：保持发送字符串ID（无需修改）
2. **后端Java代码**：将 `Long` 类型改为 `String` 类型（已完成）
3. **数据库**：将 `BIGINT` 改为 `VARCHAR(100)`（本次迁移）

## 相关提交

- `f523b4a` - feat: 实现完整的训练任务管理功能
- `bfdb853` - fix: 将数据集ID字段类型从Long改为String
- `8337c59` - feat: 添加数据集ID字段类型转换的SQL迁移脚本
- `b1c9661` - fix: 修正SQL脚本中的数据库名称从datamark改为mark

## 联系支持

如果在执行迁移过程中遇到任何问题，请查看：
- SQL脚本：`/home/user/work/back/datamark-admin/update_dataset_id_to_string.sql`
- 原始schema：`/home/user/work/back/datamark-admin/model_distillation_schema.sql`
