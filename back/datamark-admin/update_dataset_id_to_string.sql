-- ============================================
-- 更新数据集ID字段类型：从 BIGINT 改为 VARCHAR
-- 以支持非数字的数据集ID（如 'dialogue-zh-v2'）
-- ============================================

USE mark;

-- 1. 修改 md_training_task 表的 dataset_id 字段类型
ALTER TABLE `md_training_task`
  MODIFY COLUMN `dataset_id` VARCHAR(100) DEFAULT NULL COMMENT '训练数据集ID（支持字符串类型）';

-- 2. 添加验证数据集ID字段（如果不存在）
-- 先检查字段是否存在，如果不存在则添加
SET @col_exists = 0;
SELECT COUNT(*) INTO @col_exists
FROM information_schema.COLUMNS
WHERE TABLE_SCHEMA = DATABASE()
  AND TABLE_NAME = 'md_training_task'
  AND COLUMN_NAME = 'val_dataset_id';

SET @query = IF(@col_exists = 0,
  'ALTER TABLE `md_training_task` ADD COLUMN `val_dataset_id` VARCHAR(100) DEFAULT NULL COMMENT ''验证数据集ID'' AFTER `dataset_name`',
  'SELECT ''Column val_dataset_id already exists'' AS Result');

PREPARE stmt FROM @query;
EXECUTE stmt;
DEALLOCATE PREPARE stmt;

-- 3. 添加验证数据集名称字段（如果不存在）
SET @col_exists = 0;
SELECT COUNT(*) INTO @col_exists
FROM information_schema.COLUMNS
WHERE TABLE_SCHEMA = DATABASE()
  AND TABLE_NAME = 'md_training_task'
  AND COLUMN_NAME = 'val_dataset_name';

SET @query = IF(@col_exists = 0,
  'ALTER TABLE `md_training_task` ADD COLUMN `val_dataset_name` VARCHAR(255) DEFAULT NULL COMMENT ''验证数据集名称'' AFTER `val_dataset_id`',
  'SELECT ''Column val_dataset_name already exists'' AS Result');

PREPARE stmt FROM @query;
EXECUTE stmt;
DEALLOCATE PREPARE stmt;

-- 4. 添加训练配置JSON字段（如果不存在）- 用于存储高级配置
SET @col_exists = 0;
SELECT COUNT(*) INTO @col_exists
FROM information_schema.COLUMNS
WHERE TABLE_SCHEMA = DATABASE()
  AND TABLE_NAME = 'md_training_task'
  AND COLUMN_NAME = 'training_config';

SET @query = IF(@col_exists = 0,
  'ALTER TABLE `md_training_task` ADD COLUMN `training_config` TEXT DEFAULT NULL COMMENT ''训练配置JSON（包含优化器、调度器等高级配置）'' AFTER `lora_dropout`',
  'SELECT ''Column training_config already exists'' AS Result');

PREPARE stmt FROM @query;
EXECUTE stmt;
DEALLOCATE PREPARE stmt;

-- 5. 修改 md_model_evaluation 表的 eval_dataset_id 字段类型
ALTER TABLE `md_model_evaluation`
  MODIFY COLUMN `eval_dataset_id` VARCHAR(100) DEFAULT NULL COMMENT '评估数据集ID（支持字符串类型）';

-- 显示修改结果
SELECT 'Database schema update completed successfully!' AS Result;
SHOW WARNINGS;
