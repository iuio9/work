-- ============================================
-- 添加训练配置扩展字段
-- 执行此脚本以支持前端的高级训练配置功能
-- ============================================

-- 1. 添加训练配置JSON字段（存储优化器、调度器、GPU等高级配置）
ALTER TABLE `md_training_task`
ADD COLUMN `training_config` TEXT COMMENT '训练高级配置（JSON格式：优化器、调度器、GPU、梯度累积等）' AFTER `lora_dropout`;

-- 2. 添加任务描述字段
ALTER TABLE `md_training_task`
ADD COLUMN `description` VARCHAR(500) COMMENT '任务描述' AFTER `task_name`;

-- 3. 添加验证数据集字段
ALTER TABLE `md_training_task`
ADD COLUMN `val_dataset_id` BIGINT(20) COMMENT '验证数据集ID' AFTER `dataset_id`;

ALTER TABLE `md_training_task`
ADD COLUMN `val_dataset_name` VARCHAR(255) COMMENT '验证数据集名称' AFTER `val_dataset_id`;

-- ============================================
-- training_config JSON 字段示例数据结构：
-- ============================================
-- {
--   "optimizer": "adamw",                 // 优化器类型
--   "lrScheduler": "cosine",              // 学习率调度器
--   "weightDecay": 0.01,                  // 权重衰减
--   "gradAccumSteps": 4,                  // 梯度累积步数
--   "maxGradNorm": 1.0,                   // 最大梯度范数（梯度裁剪）
--   "gpuDevices": [0, 1],                 // GPU设备列表
--   "autoSaveCheckpoint": true,           // 是否自动保存检查点
--   "checkpointInterval": 5,              // 检查点保存间隔
--   "teacherModelConfig": {               // 教师模型详细配置
--     "paramSize": "7B",
--     "modelPath": "meta-llama/Llama-2-7b-hf",
--     "quantization": "int8"
--   },
--   "studentModelConfig": {               // 学生模型详细配置
--     "paramSize": "350M",
--     "initMethod": "random",
--     "pretrainPath": ""
--   },
--   "loraAdvancedConfig": {               // LoRA高级配置
--     "targetModules": ["q_proj", "v_proj"],
--     "layers": "all",
--     "biasTrain": "none"
--   },
--   "distillationAdvancedConfig": {       // 知识蒸馏高级配置
--     "hardLabelWeight": 0.3,
--     "softLabelWeight": 0.7,
--     "lossType": "kl_div",
--     "intermediateLayers": false,
--     "attentionDistill": false
--   }
-- }
