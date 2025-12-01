-- ============================================
-- 大小模型协同训练（Model Distillation）数据库表
-- ============================================

-- 1. 大小模型协同训练任务表
CREATE TABLE IF NOT EXISTS `md_training_task` (
  `id` BIGINT(20) NOT NULL AUTO_INCREMENT COMMENT '主键ID',
  `task_id` VARCHAR(64) NOT NULL COMMENT '任务ID（唯一标识）',
  `task_name` VARCHAR(255) NOT NULL COMMENT '任务名称',
  `task_input_name` VARCHAR(255) DEFAULT NULL COMMENT '任务输入名称',

  -- 模型配置
  `teacher_model` VARCHAR(100) NOT NULL COMMENT '教师模型类型（llama2-7b, qwen-7b等）',
  `student_model` VARCHAR(100) NOT NULL COMMENT '学生模型类型（yolov5s, resnet50等）',

  -- 训练参数
  `total_epochs` INT(11) NOT NULL DEFAULT 50 COMMENT '总训练轮数',
  `current_epoch` INT(11) NOT NULL DEFAULT 0 COMMENT '当前训练轮数',
  `batch_size` INT(11) NOT NULL DEFAULT 32 COMMENT '批次大小',
  `learning_rate` DECIMAL(10, 8) NOT NULL DEFAULT 0.001 COMMENT '学习率',

  -- 知识蒸馏参数
  `temperature` DECIMAL(5, 2) NOT NULL DEFAULT 3.0 COMMENT '蒸馏温度',
  `alpha` DECIMAL(5, 2) NOT NULL DEFAULT 0.7 COMMENT '软标签权重',

  -- LoRA配置
  `lora_rank` INT(11) NOT NULL DEFAULT 16 COMMENT 'LoRA秩',
  `lora_alpha` INT(11) DEFAULT 32 COMMENT 'LoRA alpha参数',
  `lora_dropout` DECIMAL(5, 3) DEFAULT 0.05 COMMENT 'LoRA dropout率',

  -- 训练状态
  `status` VARCHAR(20) NOT NULL DEFAULT 'PENDING' COMMENT '任务状态：PENDING-待开始，RUNNING-运行中，PAUSED-暂停，COMPLETED-已完成，FAILED-失败，STOPPED-已停止',
  `progress` INT(11) NOT NULL DEFAULT 0 COMMENT '训练进度（0-100）',

  -- 训练结果
  `accuracy` DECIMAL(10, 4) DEFAULT NULL COMMENT '模型准确率（%）',
  `loss` DECIMAL(10, 6) DEFAULT NULL COMMENT '最终损失值',
  `best_accuracy` DECIMAL(10, 4) DEFAULT NULL COMMENT '最佳准确率',

  -- 数据集信息
  `dataset_id` BIGINT(20) DEFAULT NULL COMMENT '训练数据集ID',
  `dataset_name` VARCHAR(255) DEFAULT NULL COMMENT '数据集名称',

  -- 模型文件路径
  `model_path` VARCHAR(500) DEFAULT NULL COMMENT '训练完成的模型文件路径',
  `model_url` VARCHAR(500) DEFAULT NULL COMMENT '模型访问URL',
  `checkpoint_path` VARCHAR(500) DEFAULT NULL COMMENT '检查点文件路径',

  -- 错误信息
  `error_message` TEXT DEFAULT NULL COMMENT '错误信息',

  -- 元数据
  `start_time` DATETIME DEFAULT NULL COMMENT '开始训练时间',
  `end_time` DATETIME DEFAULT NULL COMMENT '结束训练时间',
  `duration` BIGINT(20) DEFAULT NULL COMMENT '训练时长（秒）',

  -- 系统字段
  `create_by` VARCHAR(64) DEFAULT NULL COMMENT '创建人',
  `create_time` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  `update_by` VARCHAR(64) DEFAULT NULL COMMENT '更新人',
  `update_time` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  `remark` VARCHAR(500) DEFAULT NULL COMMENT '备注',
  `del_flag` TINYINT(1) NOT NULL DEFAULT 0 COMMENT '删除标志（0-未删除，1-已删除）',

  PRIMARY KEY (`id`),
  UNIQUE KEY `uk_task_id` (`task_id`),
  KEY `idx_status` (`status`),
  KEY `idx_create_time` (`create_time`),
  KEY `idx_teacher_model` (`teacher_model`),
  KEY `idx_student_model` (`student_model`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='大小模型协同训练任务表';

-- 2. 训练历史记录表（记录每个epoch的训练详情）
CREATE TABLE IF NOT EXISTS `md_training_history` (
  `id` BIGINT(20) NOT NULL AUTO_INCREMENT COMMENT '主键ID',
  `task_id` VARCHAR(64) NOT NULL COMMENT '任务ID',
  `epoch` INT(11) NOT NULL COMMENT '训练轮次',

  -- 训练指标
  `train_loss` DECIMAL(10, 6) DEFAULT NULL COMMENT '训练损失',
  `train_accuracy` DECIMAL(10, 4) DEFAULT NULL COMMENT '训练准确率',
  `val_loss` DECIMAL(10, 6) DEFAULT NULL COMMENT '验证损失',
  `val_accuracy` DECIMAL(10, 4) DEFAULT NULL COMMENT '验证准确率',

  -- 蒸馏相关指标
  `distill_loss` DECIMAL(10, 6) DEFAULT NULL COMMENT '蒸馏损失',
  `hard_loss` DECIMAL(10, 6) DEFAULT NULL COMMENT '硬标签损失',
  `soft_loss` DECIMAL(10, 6) DEFAULT NULL COMMENT '软标签损失',

  -- 系统资源
  `gpu_usage` DECIMAL(5, 2) DEFAULT NULL COMMENT 'GPU使用率（%）',
  `memory_usage` DECIMAL(10, 2) DEFAULT NULL COMMENT '内存使用量（MB）',

  -- 时间信息
  `epoch_time` DECIMAL(10, 2) DEFAULT NULL COMMENT '本轮训练耗时（秒）',
  `record_time` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '记录时间',

  PRIMARY KEY (`id`),
  KEY `idx_task_id` (`task_id`),
  KEY `idx_epoch` (`epoch`),
  KEY `idx_record_time` (`record_time`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='训练历史记录表';

-- 3. LoRA配置预设表
CREATE TABLE IF NOT EXISTS `md_lora_preset` (
  `id` BIGINT(20) NOT NULL AUTO_INCREMENT COMMENT '主键ID',
  `preset_name` VARCHAR(100) NOT NULL COMMENT '预设名称',
  `preset_desc` VARCHAR(500) DEFAULT NULL COMMENT '预设描述',

  -- LoRA参数
  `lora_rank` INT(11) NOT NULL DEFAULT 16 COMMENT 'LoRA秩',
  `lora_alpha` INT(11) NOT NULL DEFAULT 32 COMMENT 'LoRA alpha参数',
  `lora_dropout` DECIMAL(5, 3) NOT NULL DEFAULT 0.05 COMMENT 'LoRA dropout率',
  `target_modules` VARCHAR(500) DEFAULT NULL COMMENT '目标模块（JSON数组）',

  -- 系统字段
  `create_by` VARCHAR(64) DEFAULT NULL COMMENT '创建人',
  `create_time` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  `update_time` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  `del_flag` TINYINT(1) NOT NULL DEFAULT 0 COMMENT '删除标志',

  PRIMARY KEY (`id`),
  UNIQUE KEY `uk_preset_name` (`preset_name`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='LoRA配置预设表';

-- 4. 模型评估结果表
CREATE TABLE IF NOT EXISTS `md_model_evaluation` (
  `id` BIGINT(20) NOT NULL AUTO_INCREMENT COMMENT '主键ID',
  `task_id` VARCHAR(64) NOT NULL COMMENT '任务ID',

  -- 评估指标
  `precision` DECIMAL(10, 4) DEFAULT NULL COMMENT '精确率',
  `recall` DECIMAL(10, 4) DEFAULT NULL COMMENT '召回率',
  `f1_score` DECIMAL(10, 4) DEFAULT NULL COMMENT 'F1分数',
  `map_50` DECIMAL(10, 4) DEFAULT NULL COMMENT 'mAP@0.5',
  `map_95` DECIMAL(10, 4) DEFAULT NULL COMMENT 'mAP@0.5:0.95',

  -- 评估数据集
  `eval_dataset_id` BIGINT(20) DEFAULT NULL COMMENT '评估数据集ID',
  `eval_dataset_name` VARCHAR(255) DEFAULT NULL COMMENT '评估数据集名称',
  `eval_sample_count` INT(11) DEFAULT NULL COMMENT '评估样本数量',

  -- 评估时间
  `eval_time` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '评估时间',

  -- 详细结果（JSON格式）
  `detailed_results` TEXT DEFAULT NULL COMMENT '详细评估结果（JSON）',

  PRIMARY KEY (`id`),
  KEY `idx_task_id` (`task_id`),
  KEY `idx_eval_time` (`eval_time`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='模型评估结果表';

-- 插入默认的LoRA预设
INSERT INTO `md_lora_preset` (`preset_name`, `preset_desc`, `lora_rank`, `lora_alpha`, `lora_dropout`, `target_modules`) VALUES
('标准配置', '适用于大多数场景的标准LoRA配置', 16, 32, 0.050, '["q_proj","v_proj"]'),
('高效配置', '低秩配置，训练速度快，适合快速实验', 8, 16, 0.050, '["q_proj","v_proj"]'),
('高精度配置', '高秩配置，模型性能更好，但训练时间较长', 32, 64, 0.030, '["q_proj","v_proj","k_proj","o_proj"]'),
('全参数配置', '覆盖所有线性层的LoRA配置', 16, 32, 0.050, '["q_proj","v_proj","k_proj","o_proj","gate_proj","up_proj","down_proj"]');
