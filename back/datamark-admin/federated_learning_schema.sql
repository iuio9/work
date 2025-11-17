-- ============================================================================
-- 联邦学习框架数据库表结构
-- ============================================================================
-- 功能：支持联邦学习的节点管理、训练任务管理、模型训练记录
-- 作者：AI Assistant
-- 日期：2025-01-20
-- ============================================================================

-- ============================================================================
-- 表1：联邦学习节点表 (fl_federated_node)
-- ============================================================================
-- 功能：存储联邦学习参与节点的信息
-- 说明：每个节点代表一个独立的训练客户端，拥有自己的数据集
-- ============================================================================
CREATE TABLE IF NOT EXISTS `fl_federated_node` (
  `id` BIGINT NOT NULL AUTO_INCREMENT COMMENT '主键ID',
  `node_id` VARCHAR(64) NOT NULL COMMENT '节点唯一标识（UUID）',
  `node_name` VARCHAR(128) DEFAULT NULL COMMENT '节点名称',
  `host` VARCHAR(128) NOT NULL COMMENT '节点主机地址',
  `port` INT NOT NULL COMMENT '节点端口号',
  `status` VARCHAR(32) DEFAULT 'INACTIVE' COMMENT '节点状态：ACTIVE/INACTIVE/DISCONNECTED/ERROR',
  `is_active` TINYINT(1) DEFAULT 0 COMMENT '是否活跃：0-否，1-是',
  `cpu_cores` INT DEFAULT NULL COMMENT 'CPU核心数',
  `memory_gb` DECIMAL(10,2) DEFAULT NULL COMMENT '内存大小（GB）',
  `gpu_count` INT DEFAULT 0 COMMENT 'GPU数量',
  `gpu_model` VARCHAR(128) DEFAULT NULL COMMENT 'GPU型号',
  `dataset_size` BIGINT DEFAULT 0 COMMENT '数据集样本数量',
  `metadata` TEXT DEFAULT NULL COMMENT '节点元数据（JSON格式）',
  `last_heartbeat_at` DATETIME DEFAULT NULL COMMENT '最后心跳时间',
  `registered_at` DATETIME DEFAULT CURRENT_TIMESTAMP COMMENT '注册时间',
  `updated_at` DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (`id`),
  UNIQUE KEY `uk_node_id` (`node_id`),
  KEY `idx_status` (`status`),
  KEY `idx_is_active` (`is_active`),
  KEY `idx_last_heartbeat` (`last_heartbeat_at`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='联邦学习节点表';

-- ============================================================================
-- 表2：联邦学习训练任务表 (fl_training_job)
-- ============================================================================
-- 功能：存储联邦学习训练任务的配置和状态
-- 说明：每个任务对应一次完整的联邦学习训练流程
-- ============================================================================
CREATE TABLE IF NOT EXISTS `fl_training_job` (
  `id` BIGINT NOT NULL AUTO_INCREMENT COMMENT '主键ID',
  `job_id` VARCHAR(64) NOT NULL COMMENT '任务唯一标识（UUID）',
  `job_name` VARCHAR(128) DEFAULT NULL COMMENT '任务名称',
  `model_type` VARCHAR(64) NOT NULL COMMENT '模型类型：YOLO_V8/LSTM/UNET/RESNET/VISION_TRANSFORMER',
  `status` VARCHAR(32) DEFAULT 'CREATED' COMMENT '任务状态：CREATED/RUNNING/COMPLETED/FAILED/STOPPED/DEGRADED',
  `strategy` VARCHAR(64) DEFAULT 'FedAvg' COMMENT '聚合策略：FedAvg/FedProx/FedAdam',

  -- 训练参数
  `num_rounds` INT DEFAULT 10 COMMENT '训练轮数',
  `min_clients` INT DEFAULT 2 COMMENT '最少客户端数',
  `min_fit_clients` INT DEFAULT 2 COMMENT '每轮最少训练客户端数',
  `min_evaluate_clients` INT DEFAULT 2 COMMENT '每轮最少评估客户端数',
  `fraction_fit` DECIMAL(5,2) DEFAULT 1.0 COMMENT '每轮参与训练的客户端比例',
  `fraction_evaluate` DECIMAL(5,2) DEFAULT 1.0 COMMENT '每轮参与评估的客户端比例',

  -- 超参数（JSON格式）
  `hyperparameters` TEXT DEFAULT NULL COMMENT '超参数配置（JSON）：学习率、批大小、本地轮次等',

  -- 参与节点
  `participant_node_ids` TEXT DEFAULT NULL COMMENT '参与节点ID列表（JSON数组）',
  `participant_count` INT DEFAULT 0 COMMENT '参与节点数量',

  -- 精度监控
  `baseline_accuracy` DECIMAL(10,6) DEFAULT NULL COMMENT '基线精度（首次评估）',
  `current_accuracy` DECIMAL(10,6) DEFAULT NULL COMMENT '当前精度',
  `best_accuracy` DECIMAL(10,6) DEFAULT NULL COMMENT '最佳精度',
  `allowed_drop_percent` DECIMAL(5,2) DEFAULT 5.0 COMMENT '允许精度下降百分比',

  -- 训练进度
  `current_round` INT DEFAULT 0 COMMENT '当前训练轮次',
  `total_training_time` BIGINT DEFAULT 0 COMMENT '总训练时间（秒）',

  -- Flower Server 配置
  `server_port` INT DEFAULT NULL COMMENT 'Flower Server端口',
  `server_address` VARCHAR(128) DEFAULT NULL COMMENT 'Flower Server地址',

  -- 全局模型参数（存储路径或序列化数据）
  `global_model_path` VARCHAR(512) DEFAULT NULL COMMENT '全局模型保存路径',

  -- 日志和描述
  `description` TEXT DEFAULT NULL COMMENT '任务描述',
  `error_message` TEXT DEFAULT NULL COMMENT '错误信息',

  -- 时间戳
  `created_at` DATETIME DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  `started_at` DATETIME DEFAULT NULL COMMENT '开始时间',
  `completed_at` DATETIME DEFAULT NULL COMMENT '完成时间',
  `updated_at` DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',

  PRIMARY KEY (`id`),
  UNIQUE KEY `uk_job_id` (`job_id`),
  KEY `idx_model_type` (`model_type`),
  KEY `idx_status` (`status`),
  KEY `idx_created_at` (`created_at`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='联邦学习训练任务表';

-- ============================================================================
-- 表3：训练轮次记录表 (fl_training_round)
-- ============================================================================
-- 功能：记录每一轮训练的详细信息和指标
-- 说明：用于追踪训练过程、分析训练效果
-- ============================================================================
CREATE TABLE IF NOT EXISTS `fl_training_round` (
  `id` BIGINT NOT NULL AUTO_INCREMENT COMMENT '主键ID',
  `job_id` VARCHAR(64) NOT NULL COMMENT '所属任务ID',
  `round_number` INT NOT NULL COMMENT '轮次编号（从1开始）',

  -- 参与节点信息
  `participated_nodes` TEXT DEFAULT NULL COMMENT '参与的节点ID列表（JSON数组）',
  `participated_count` INT DEFAULT 0 COMMENT '参与节点数量',

  -- 训练指标
  `avg_loss` DECIMAL(10,6) DEFAULT NULL COMMENT '平均损失',
  `avg_accuracy` DECIMAL(10,6) DEFAULT NULL COMMENT '平均精度',
  `min_accuracy` DECIMAL(10,6) DEFAULT NULL COMMENT '最低精度',
  `max_accuracy` DECIMAL(10,6) DEFAULT NULL COMMENT '最高精度',

  -- 节点级指标（JSON格式）
  `node_metrics` TEXT DEFAULT NULL COMMENT '各节点指标详情（JSON）',

  -- 时间统计
  `round_duration` BIGINT DEFAULT 0 COMMENT '本轮耗时（毫秒）',
  `aggregation_time` BIGINT DEFAULT 0 COMMENT '参数聚合耗时（毫秒）',

  -- 时间戳
  `started_at` DATETIME DEFAULT NULL COMMENT '开始时间',
  `completed_at` DATETIME DEFAULT NULL COMMENT '完成时间',
  `created_at` DATETIME DEFAULT CURRENT_TIMESTAMP COMMENT '记录创建时间',

  PRIMARY KEY (`id`),
  KEY `idx_job_round` (`job_id`, `round_number`),
  KEY `idx_created_at` (`created_at`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='训练轮次记录表';

-- ============================================================================
-- 表4：节点训练历史表 (fl_node_training_history)
-- ============================================================================
-- 功能：记录节点参与训练的历史
-- 说明：用于追踪单个节点的训练贡献和性能
-- ============================================================================
CREATE TABLE IF NOT EXISTS `fl_node_training_history` (
  `id` BIGINT NOT NULL AUTO_INCREMENT COMMENT '主键ID',
  `node_id` VARCHAR(64) NOT NULL COMMENT '节点ID',
  `job_id` VARCHAR(64) NOT NULL COMMENT '任务ID',
  `round_number` INT NOT NULL COMMENT '轮次编号',

  -- 训练指标
  `local_loss` DECIMAL(10,6) DEFAULT NULL COMMENT '本地训练损失',
  `local_accuracy` DECIMAL(10,6) DEFAULT NULL COMMENT '本地训练精度',
  `samples_count` INT DEFAULT 0 COMMENT '训练样本数',

  -- 时间统计
  `training_time` BIGINT DEFAULT 0 COMMENT '训练耗时（毫秒）',
  `upload_time` BIGINT DEFAULT 0 COMMENT '上传参数耗时（毫秒）',

  -- 状态
  `status` VARCHAR(32) DEFAULT 'SUCCESS' COMMENT '训练状态：SUCCESS/FAILED/TIMEOUT',
  `error_message` TEXT DEFAULT NULL COMMENT '错误信息',

  -- 时间戳
  `trained_at` DATETIME DEFAULT CURRENT_TIMESTAMP COMMENT '训练时间',

  PRIMARY KEY (`id`),
  KEY `idx_node_job` (`node_id`, `job_id`),
  KEY `idx_job_round` (`job_id`, `round_number`),
  KEY `idx_trained_at` (`trained_at`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='节点训练历史表';

-- ============================================================================
-- 表5：模型版本管理表 (fl_model_version)
-- ============================================================================
-- 功能：管理联邦学习训练产生的模型版本
-- 说明：保存每个任务的模型文件和元数据
-- ============================================================================
CREATE TABLE IF NOT EXISTS `fl_model_version` (
  `id` BIGINT NOT NULL AUTO_INCREMENT COMMENT '主键ID',
  `job_id` VARCHAR(64) NOT NULL COMMENT '所属任务ID',
  `version` VARCHAR(32) NOT NULL COMMENT '版本号',
  `round_number` INT DEFAULT NULL COMMENT '对应训练轮次',

  -- 模型信息
  `model_type` VARCHAR(64) NOT NULL COMMENT '模型类型',
  `model_path` VARCHAR(512) NOT NULL COMMENT '模型文件路径',
  `model_size` BIGINT DEFAULT 0 COMMENT '模型文件大小（字节）',
  `model_hash` VARCHAR(64) DEFAULT NULL COMMENT '模型文件哈希值（MD5）',

  -- 性能指标
  `accuracy` DECIMAL(10,6) DEFAULT NULL COMMENT '模型精度',
  `loss` DECIMAL(10,6) DEFAULT NULL COMMENT '模型损失',
  `f1_score` DECIMAL(10,6) DEFAULT NULL COMMENT 'F1分数',

  -- 标签和描述
  `is_best` TINYINT(1) DEFAULT 0 COMMENT '是否最佳模型',
  `is_deployed` TINYINT(1) DEFAULT 0 COMMENT '是否已部署',
  `description` TEXT DEFAULT NULL COMMENT '版本描述',

  -- 时间戳
  `created_at` DATETIME DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  `updated_at` DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',

  PRIMARY KEY (`id`),
  UNIQUE KEY `uk_job_version` (`job_id`, `version`),
  KEY `idx_model_type` (`model_type`),
  KEY `idx_is_best` (`is_best`),
  KEY `idx_created_at` (`created_at`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='模型版本管理表';

-- ============================================================================
-- 初始化索引说明
-- ============================================================================
-- 1. uk_node_id: 保证节点ID唯一性
-- 2. uk_job_id: 保证任务ID唯一性
-- 3. idx_status: 加速按状态查询
-- 4. idx_is_active: 加速活跃节点查询
-- 5. idx_last_heartbeat: 加速心跳超时检查
-- 6. idx_job_round: 加速按任务和轮次查询
-- 7. idx_node_job: 加速按节点和任务查询
-- ============================================================================
