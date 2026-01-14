-- ================================================================================
-- 大小模型协同训练 - 演示数据SQL
-- ================================================================================
-- 用途：为明天的演示准备完整的示例数据
-- 包含：3个训练任务（已完成、运行中、已停止）+ 训练历史 + 模型评估
-- 使用：执行此SQL即可看到完整的训练效果演示
-- ================================================================================

-- 删除旧的演示数据（如果存在）
DELETE FROM md_training_task WHERE task_id LIKE 'DEMO_%';
DELETE FROM md_training_history WHERE task_id LIKE 'DEMO_%';
DELETE FROM md_model_evaluation WHERE task_id LIKE 'DEMO_%';

-- ================================================================================
-- 任务1：已完成的Qwen2.5-VL蒸馏任务（用于展示训练完成后的效果）
-- ================================================================================
INSERT INTO md_training_task (
    task_id, task_name, teacher_model, student_model, dataset_id, dataset_name,
    total_epochs, batch_size, learning_rate, temperature, alpha,
    lora_enabled, lora_r, lora_alpha, lora_dropout, target_modules,
    status, current_epoch, progress,
    train_accuracy, train_loss, val_accuracy, val_loss, best_accuracy,
    model_save_path, model_url,
    start_time, end_time, duration,
    create_time, update_time, del_flag, description
) VALUES (
    'DEMO_COMPLETED',
    'Qwen2.5-VL → ResNet18 图像分类蒸馏（已完成）',
    'Qwen2.5-VL-3B-Instruct',
    'ResNet18',
    'DS001',
    'CIFAR-10图像分类数据集',
    50, 32, 0.001, 4.0, 0.7,
    1, 8, 16, 0.1, 'q_proj,v_proj',
    'COMPLETED', 50, 100,
    78.45, 0.6234, 76.32, 0.6891, 76.98,
    '/home/user/models/students/DEMO_COMPLETED/final_model.pth',
    'http://localhost:9091/models/DEMO_COMPLETED/final_model.pth',
    '2026-01-14 09:00:00', '2026-01-14 11:23:45', 8625,
    '2026-01-14 08:55:00', '2026-01-14 11:23:45', 0,
    '使用Qwen2.5-VL-3B作为教师模型，将知识蒸馏到ResNet18学生模型，在CIFAR-10数据集上达到76.98%的准确率，相比ResNet18单独训练提升5.2%。训练已成功完成，模型可用于自动标注。'
);

-- ================================================================================
-- 任务2：正在运行的YOLOv8目标检测蒸馏任务（用于展示实时训练监控）
-- ================================================================================
INSERT INTO md_training_task (
    task_id, task_name, teacher_model, student_model, dataset_id, dataset_name,
    total_epochs, batch_size, learning_rate, temperature, alpha,
    lora_enabled, lora_r, lora_alpha, lora_dropout, target_modules,
    status, current_epoch, progress,
    train_accuracy, train_loss, val_accuracy, val_loss, best_accuracy,
    start_time,
    create_time, update_time, del_flag, description
) VALUES (
    'DEMO_RUNNING',
    'ResNet50 → YOLOv8-n 目标检测蒸馏（运行中）',
    'ResNet50-Pretrained',
    'YOLOv8-n',
    'DS002',
    'COCO2017目标检测数据集',
    100, 16, 0.0001, 3.5, 0.6,
    1, 16, 32, 0.05, 'c2f,sppf',
    'RUNNING', 42, 42,
    65.23, 1.2345, 63.89, 1.3012, 64.56,
    '2026-01-14 10:30:00',
    '2026-01-14 10:25:00', NOW(), 0,
    '使用ResNet50作为教师模型，蒸馏到轻量级YOLOv8-n模型，用于边缘设备部署。当前训练进度42/100 epoch，已达到63.89%的验证准确率。'
);

-- ================================================================================
-- 任务3：已暂停的ViT蒸馏任务（用于展示暂停/恢复功能）
-- ================================================================================
INSERT INTO md_training_task (
    task_id, task_name, teacher_model, student_model, dataset_id, dataset_name,
    total_epochs, batch_size, learning_rate, temperature, alpha,
    lora_enabled, lora_r, lora_alpha, lora_dropout, target_modules,
    status, current_epoch, progress,
    train_accuracy, train_loss, val_accuracy, val_loss, best_accuracy,
    start_time,
    create_time, update_time, del_flag, description
) VALUES (
    'DEMO_PAUSED',
    'ViT-Large → MobileViT 图像分类蒸馏（已暂停）',
    'ViT-Large-224',
    'MobileViT-Small',
    'DS003',
    'ImageNet-1K分类数据集',
    200, 64, 0.0005, 4.5, 0.75,
    1, 12, 24, 0.1, 'qkv,mlp',
    'PAUSED', 87, 44,
    82.15, 0.5123, 80.67, 0.5789, 81.23,
    '2026-01-13 14:00:00',
    '2026-01-13 13:55:00', '2026-01-14 08:30:00', 0,
    'Vision Transformer大模型蒸馏到移动端MobileViT模型。训练到87 epoch时手动暂停，当前最佳准确率81.23%，可随时恢复训练。'
);

-- ================================================================================
-- 为任务1（已完成）插入训练历史数据 - 50个epoch的完整记录
-- ================================================================================
INSERT INTO md_training_history (task_id, epoch, train_accuracy, train_loss, val_accuracy, val_loss, learning_rate, gpu_utilization, memory_used, record_time, create_time)
SELECT
    'DEMO_COMPLETED',
    epoch_num,
    -- 训练准确率：从35%逐步提升到78.45%（模拟真实训练曲线）
    ROUND(35 + (78.45 - 35) * (1 - EXP(-epoch_num / 15.0)), 2),
    -- 训练损失：从2.5逐步下降到0.62（指数衰减）
    ROUND(2.5 * EXP(-epoch_num / 20.0) + 0.62, 4),
    -- 验证准确率：略低于训练准确率，有波动
    ROUND(33 + (76.32 - 33) * (1 - EXP(-epoch_num / 16.0)) + (RAND() - 0.5) * 2, 2),
    -- 验证损失：略高于训练损失
    ROUND(2.6 * EXP(-epoch_num / 19.0) + 0.69 + (RAND() - 0.5) * 0.1, 4),
    -- 学习率：使用余弦退火
    ROUND(0.001 * (1 + COS(PI() * epoch_num / 50.0)) / 2, 6),
    -- GPU使用率：85-95%之间波动
    ROUND(85 + RAND() * 10, 2),
    -- 显存使用：6-8GB之间
    ROUND(6.0 + RAND() * 2.0, 2),
    TIMESTAMPADD(MINUTE, epoch_num * 2 + 30, '2026-01-14 09:00:00'),
    TIMESTAMPADD(MINUTE, epoch_num * 2 + 30, '2026-01-14 09:00:00')
FROM (
    SELECT 1 AS epoch_num UNION SELECT 2 UNION SELECT 3 UNION SELECT 4 UNION SELECT 5 UNION
    SELECT 6 UNION SELECT 7 UNION SELECT 8 UNION SELECT 9 UNION SELECT 10 UNION
    SELECT 11 UNION SELECT 12 UNION SELECT 13 UNION SELECT 14 UNION SELECT 15 UNION
    SELECT 16 UNION SELECT 17 UNION SELECT 18 UNION SELECT 19 UNION SELECT 20 UNION
    SELECT 21 UNION SELECT 22 UNION SELECT 23 UNION SELECT 24 UNION SELECT 25 UNION
    SELECT 26 UNION SELECT 27 UNION SELECT 28 UNION SELECT 29 UNION SELECT 30 UNION
    SELECT 31 UNION SELECT 32 UNION SELECT 33 UNION SELECT 34 UNION SELECT 35 UNION
    SELECT 36 UNION SELECT 37 UNION SELECT 38 UNION SELECT 39 UNION SELECT 40 UNION
    SELECT 41 UNION SELECT 42 UNION SELECT 43 UNION SELECT 44 UNION SELECT 45 UNION
    SELECT 46 UNION SELECT 47 UNION SELECT 48 UNION SELECT 49 UNION SELECT 50
) AS epochs;

-- ================================================================================
-- 为任务2（运行中）插入训练历史数据 - 42个epoch的记录
-- ================================================================================
INSERT INTO md_training_history (task_id, epoch, train_accuracy, train_loss, val_accuracy, val_loss, learning_rate, gpu_utilization, memory_used, record_time, create_time)
SELECT
    'DEMO_RUNNING',
    epoch_num,
    ROUND(28 + (65.23 - 28) * (1 - EXP(-epoch_num / 18.0)), 2),
    ROUND(3.2 * EXP(-epoch_num / 25.0) + 1.23, 4),
    ROUND(26 + (63.89 - 26) * (1 - EXP(-epoch_num / 19.0)) + (RAND() - 0.5) * 1.5, 2),
    ROUND(3.3 * EXP(-epoch_num / 24.0) + 1.30 + (RAND() - 0.5) * 0.15, 4),
    ROUND(0.0001 * (1 + COS(PI() * epoch_num / 100.0)) / 2, 6),
    ROUND(88 + RAND() * 8, 2),
    ROUND(5.5 + RAND() * 1.5, 2),
    TIMESTAMPADD(MINUTE, epoch_num * 3 + 15, '2026-01-14 10:30:00'),
    TIMESTAMPADD(MINUTE, epoch_num * 3 + 15, '2026-01-14 10:30:00')
FROM (
    SELECT 1 AS epoch_num UNION SELECT 2 UNION SELECT 3 UNION SELECT 4 UNION SELECT 5 UNION
    SELECT 6 UNION SELECT 7 UNION SELECT 8 UNION SELECT 9 UNION SELECT 10 UNION
    SELECT 11 UNION SELECT 12 UNION SELECT 13 UNION SELECT 14 UNION SELECT 15 UNION
    SELECT 16 UNION SELECT 17 UNION SELECT 18 UNION SELECT 19 UNION SELECT 20 UNION
    SELECT 21 UNION SELECT 22 UNION SELECT 23 UNION SELECT 24 UNION SELECT 25 UNION
    SELECT 26 UNION SELECT 27 UNION SELECT 28 UNION SELECT 29 UNION SELECT 30 UNION
    SELECT 31 UNION SELECT 32 UNION SELECT 33 UNION SELECT 34 UNION SELECT 35 UNION
    SELECT 36 UNION SELECT 37 UNION SELECT 38 UNION SELECT 39 UNION SELECT 40 UNION
    SELECT 41 UNION SELECT 42
) AS epochs;

-- ================================================================================
-- 为任务3（已暂停）插入训练历史数据 - 87个epoch的记录
-- ================================================================================
INSERT INTO md_training_history (task_id, epoch, train_accuracy, train_loss, val_accuracy, val_loss, learning_rate, gpu_utilization, memory_used, record_time, create_time)
SELECT
    'DEMO_PAUSED',
    epoch_num,
    ROUND(42 + (82.15 - 42) * (1 - EXP(-epoch_num / 30.0)), 2),
    ROUND(2.8 * EXP(-epoch_num / 35.0) + 0.51, 4),
    ROUND(40 + (80.67 - 40) * (1 - EXP(-epoch_num / 32.0)) + (RAND() - 0.5) * 2.5, 2),
    ROUND(2.9 * EXP(-epoch_num / 33.0) + 0.58 + (RAND() - 0.5) * 0.12, 4),
    ROUND(0.0005 * (1 + COS(PI() * epoch_num / 200.0)) / 2, 6),
    ROUND(90 + RAND() * 7, 2),
    ROUND(10.0 + RAND() * 3.0, 2),
    TIMESTAMPADD(MINUTE, epoch_num * 5, '2026-01-13 14:00:00'),
    TIMESTAMPADD(MINUTE, epoch_num * 5, '2026-01-13 14:00:00')
FROM (
    SELECT n AS epoch_num
    FROM (
        SELECT @row := @row + 1 AS n
        FROM (SELECT 0 UNION SELECT 1 UNION SELECT 2 UNION SELECT 3 UNION SELECT 4 UNION SELECT 5 UNION SELECT 6 UNION SELECT 7 UNION SELECT 8 UNION SELECT 9) t1,
             (SELECT 0 UNION SELECT 1 UNION SELECT 2 UNION SELECT 3 UNION SELECT 4 UNION SELECT 5 UNION SELECT 6 UNION SELECT 7 UNION SELECT 8 UNION SELECT 9) t2,
             (SELECT @row := 0) r
    ) AS numbers
    WHERE n <= 87
) AS epochs;

-- ================================================================================
-- 为任务1（已完成）插入模型评估数据
-- ================================================================================
INSERT INTO md_model_evaluation (
    task_id, eval_type, eval_dataset,
    accuracy, precision_score, recall_score, f1_score,
    inference_time, model_size, compression_ratio,
    eval_time, notes, create_time
) VALUES
(
    'DEMO_COMPLETED',
    'FINAL',
    'CIFAR-10测试集',
    76.98, 77.23, 76.45, 76.84,
    8.5, 42.3, 273.0,
    '2026-01-14 11:30:00',
    '最终模型评估：在CIFAR-10测试集上达到76.98%准确率。相比教师模型Qwen2.5-VL（3B参数），学生模型ResNet18仅11M参数，压缩比273倍，推理速度提升150倍（8.5ms vs 1280ms）。模型已部署，可用于自动标注任务。',
    '2026-01-14 11:30:00'
),
(
    'DEMO_COMPLETED',
    'CHECKPOINT',
    'CIFAR-10验证集',
    75.12, 75.56, 74.89, 75.22,
    8.3, 42.3, 273.0,
    '2026-01-14 10:45:00',
    'Epoch 40 checkpoint评估：验证准确率75.12%，训练进度良好。',
    '2026-01-14 10:45:00'
);

-- ================================================================================
-- 插入LoRA预设配置（用于展示预设功能）
-- ================================================================================
INSERT INTO md_lora_preset (preset_name, lora_r, lora_alpha, lora_dropout, target_modules, description, is_default, create_time, del_flag)
VALUES
('高精度配置', 16, 32, 0.05, 'q_proj,v_proj,k_proj,o_proj', '适用于追求高精度的场景，参数较多，训练时间较长', 0, NOW(), 0),
('平衡配置', 8, 16, 0.1, 'q_proj,v_proj', '精度和效率的平衡配置，适合大多数场景（推荐）', 1, NOW(), 0),
('快速训练配置', 4, 8, 0.15, 'q_proj', '快速训练配置，参数少，训练速度快，适合快速验证', 0, NOW(), 0)
ON DUPLICATE KEY UPDATE preset_name=preset_name;

-- ================================================================================
-- 验证数据插入
-- ================================================================================
SELECT
    '演示数据已成功插入！' AS message,
    (SELECT COUNT(*) FROM md_training_task WHERE task_id LIKE 'DEMO_%') AS training_tasks,
    (SELECT COUNT(*) FROM md_training_history WHERE task_id LIKE 'DEMO_%') AS history_records,
    (SELECT COUNT(*) FROM md_model_evaluation WHERE task_id LIKE 'DEMO_%') AS evaluations;

-- ================================================================================
-- 查看演示数据
-- ================================================================================
SELECT
    task_id,
    task_name,
    teacher_model,
    student_model,
    status,
    CONCAT(current_epoch, '/', total_epochs) AS epoch_progress,
    CONCAT(progress, '%') AS progress_pct,
    best_accuracy,
    TIMESTAMPDIFF(SECOND, start_time, IFNULL(end_time, NOW())) / 60 AS duration_minutes
FROM md_training_task
WHERE task_id LIKE 'DEMO_%'
ORDER BY create_time DESC;

-- ================================================================================
-- 使用说明
-- ================================================================================
--
-- 1. 执行此SQL后，系统中会有3个演示训练任务：
--    - DEMO_COMPLETED: 已完成的任务，可用于展示训练结果和模型标注
--    - DEMO_RUNNING:   运行中的任务，可用于展示实时监控界面
--    - DEMO_PAUSED:    已暂停的任务，可用于展示暂停/恢复功能
--
-- 2. 前端展示效果：
--    - 任务列表：显示3个不同状态的任务
--    - 训练监控：Loss/Accuracy曲线完整显示
--    - 模型评估：显示评估指标和对比数据
--    - 自动标注：已完成的模型可选用于标注
--
-- 3. 演示重点：
--    - 大模型（Qwen2.5-VL 3B）→ 小模型（ResNet18 11M）压缩273倍
--    - 训练过程完整可视化（50个epoch的历史数据）
--    - 实时监控界面（GPU使用率、显存、Loss/Acc曲线）
--    - 模型评估结果（准确率76.98%，推理速度提升150倍）
--
-- 4. 删除演示数据（演示后清理）：
--    DELETE FROM md_training_task WHERE task_id LIKE 'DEMO_%';
--    DELETE FROM md_training_history WHERE task_id LIKE 'DEMO_%';
--    DELETE FROM md_model_evaluation WHERE task_id LIKE 'DEMO_%';
--
-- ================================================================================
