-- ================================================================================
-- 大小模型协同训练 - 演示数据SQL（修复版）
-- ================================================================================
-- 说明：完全匹配 model_distillation_schema.sql 的表结构
-- 使用：直接在MySQL中执行即可
-- ================================================================================

-- 删除旧的演示数据（如果存在）
DELETE FROM md_training_task WHERE task_id LIKE 'DEMO_%';
DELETE FROM md_training_history WHERE task_id LIKE 'DEMO_%';
DELETE FROM md_model_evaluation WHERE task_id LIKE 'DEMO_%';
DELETE FROM md_lora_preset WHERE preset_name IN ('演示-高精度配置', '演示-平衡配置', '演示-快速训练配置');

-- ================================================================================
-- 任务1：已完成的Qwen2.5-VL蒸馏任务
-- ================================================================================
INSERT INTO md_training_task (
    task_id, task_name, teacher_model, student_model,
    dataset_id, dataset_name,
    total_epochs, current_epoch, batch_size, learning_rate,
    temperature, alpha,
    lora_rank, lora_alpha, lora_dropout,
    status, progress,
    accuracy, loss, best_accuracy,
    model_path, model_url,
    start_time, end_time, duration,
    create_time, update_time, del_flag, remark
) VALUES (
    'DEMO_COMPLETED',
    'Qwen2.5-VL → ResNet18 图像分类蒸馏（已完成）',
    'Qwen2.5-VL-3B-Instruct',
    'ResNet18',
    'DS001',
    'CIFAR-10图像分类数据集',
    50, 50, 32, 0.001000,
    4.0, 0.7,
    8, 16, 0.100,
    'COMPLETED', 100,
    78.45, 0.623400, 76.98,
    '/home/user/models/students/DEMO_COMPLETED/final_model.pth',
    'http://localhost:9091/models/DEMO_COMPLETED/final_model.pth',
    '2026-01-14 09:00:00', '2026-01-14 11:23:45', 8625,
    '2026-01-14 08:55:00', '2026-01-14 11:23:45', 0,
    '使用Qwen2.5-VL-3B作为教师模型，将知识蒸馏到ResNet18学生模型，在CIFAR-10数据集上达到76.98%的准确率，相比ResNet18单独训练提升5.2%。训练已成功完成，模型可用于自动标注。压缩比273倍，推理速度提升150倍。'
);

-- ================================================================================
-- 任务2：正在运行的YOLOv8目标检测蒸馏任务
-- ================================================================================
INSERT INTO md_training_task (
    task_id, task_name, teacher_model, student_model,
    dataset_id, dataset_name,
    total_epochs, current_epoch, batch_size, learning_rate,
    temperature, alpha,
    lora_rank, lora_alpha, lora_dropout,
    status, progress,
    accuracy, loss, best_accuracy,
    start_time,
    create_time, update_time, del_flag, remark
) VALUES (
    'DEMO_RUNNING',
    'ResNet50 → YOLOv8-n 目标检测蒸馏（运行中）',
    'ResNet50-Pretrained',
    'YOLOv8-n',
    'DS002',
    'COCO2017目标检测数据集',
    100, 42, 16, 0.000100,
    3.5, 0.6,
    16, 32, 0.050,
    'RUNNING', 42,
    65.23, 1.234500, 64.56,
    '2026-01-14 10:30:00',
    '2026-01-14 10:25:00', NOW(), 0,
    '使用ResNet50作为教师模型，蒸馏到轻量级YOLOv8-n模型，用于边缘设备部署。当前训练进度42/100 epoch，已达到63.89%的验证准确率。'
);

-- ================================================================================
-- 任务3：已暂停的ViT蒸馏任务
-- ================================================================================
INSERT INTO md_training_task (
    task_id, task_name, teacher_model, student_model,
    dataset_id, dataset_name,
    total_epochs, current_epoch, batch_size, learning_rate,
    temperature, alpha,
    lora_rank, lora_alpha, lora_dropout,
    status, progress,
    accuracy, loss, best_accuracy,
    start_time,
    create_time, update_time, del_flag, remark
) VALUES (
    'DEMO_PAUSED',
    'ViT-Large → MobileViT 图像分类蒸馏（已暂停）',
    'ViT-Large-224',
    'MobileViT-Small',
    'DS003',
    'ImageNet-1K分类数据集',
    200, 87, 64, 0.000500,
    4.5, 0.75,
    12, 24, 0.100,
    'PAUSED', 44,
    82.15, 0.512300, 81.23,
    '2026-01-13 14:00:00',
    '2026-01-13 13:55:00', '2026-01-14 08:30:00', 0,
    'Vision Transformer大模型蒸馏到移动端MobileViT模型。训练到87 epoch时手动暂停，当前最佳准确率81.23%，可随时恢复训练。'
);

-- ================================================================================
-- 为任务1（已完成）插入训练历史数据 - 50个epoch的完整记录
-- ================================================================================
INSERT INTO md_training_history (
    task_id, epoch,
    train_accuracy, train_loss,
    val_accuracy, val_loss,
    gpu_usage, memory_usage,
    epoch_time, record_time
)
SELECT
    'DEMO_COMPLETED' AS task_id,
    seq AS epoch,
    -- 训练准确率：从35%逐步提升到78.45%
    ROUND(35 + (78.45 - 35) * (1 - EXP(-seq / 15.0)), 2) AS train_accuracy,
    -- 训练损失：从2.5逐步下降到0.62
    ROUND(2.5 * EXP(-seq / 20.0) + 0.62, 6) AS train_loss,
    -- 验证准确率：略低于训练准确率，有波动
    ROUND(33 + (76.32 - 33) * (1 - EXP(-seq / 16.0)) + (RAND() - 0.5) * 2, 2) AS val_accuracy,
    -- 验证损失：略高于训练损失
    ROUND(2.6 * EXP(-seq / 19.0) + 0.69 + (RAND() - 0.5) * 0.1, 6) AS val_loss,
    -- GPU使用率：85-95%之间波动
    ROUND(85 + RAND() * 10, 2) AS gpu_usage,
    -- 显存使用：6000-8000MB之间
    ROUND(6000 + RAND() * 2000, 2) AS memory_usage,
    -- epoch耗时：120-150秒
    ROUND(120 + RAND() * 30, 2) AS epoch_time,
    TIMESTAMPADD(MINUTE, seq * 3, '2026-01-14 09:00:00') AS record_time
FROM (
    SELECT @row := @row + 1 AS seq
    FROM (SELECT 0 UNION SELECT 1 UNION SELECT 2 UNION SELECT 3 UNION SELECT 4 UNION SELECT 5 UNION SELECT 6 UNION SELECT 7 UNION SELECT 8 UNION SELECT 9) t1,
         (SELECT 0 UNION SELECT 1 UNION SELECT 2 UNION SELECT 3 UNION SELECT 4) t2,
         (SELECT @row := 0) init
    WHERE @row < 50
) AS epochs;

-- ================================================================================
-- 为任务2（运行中）插入训练历史数据 - 42个epoch的记录
-- ================================================================================
INSERT INTO md_training_history (
    task_id, epoch,
    train_accuracy, train_loss,
    val_accuracy, val_loss,
    gpu_usage, memory_usage,
    epoch_time, record_time
)
SELECT
    'DEMO_RUNNING' AS task_id,
    seq AS epoch,
    ROUND(28 + (65.23 - 28) * (1 - EXP(-seq / 18.0)), 2) AS train_accuracy,
    ROUND(3.2 * EXP(-seq / 25.0) + 1.23, 6) AS train_loss,
    ROUND(26 + (63.89 - 26) * (1 - EXP(-seq / 19.0)) + (RAND() - 0.5) * 1.5, 2) AS val_accuracy,
    ROUND(3.3 * EXP(-seq / 24.0) + 1.30 + (RAND() - 0.5) * 0.15, 6) AS val_loss,
    ROUND(88 + RAND() * 8, 2) AS gpu_usage,
    ROUND(5500 + RAND() * 1500, 2) AS memory_usage,
    ROUND(180 + RAND() * 40, 2) AS epoch_time,
    TIMESTAMPADD(MINUTE, seq * 4, '2026-01-14 10:30:00') AS record_time
FROM (
    SELECT @row2 := @row2 + 1 AS seq
    FROM (SELECT 0 UNION SELECT 1 UNION SELECT 2 UNION SELECT 3 UNION SELECT 4 UNION SELECT 5 UNION SELECT 6 UNION SELECT 7 UNION SELECT 8 UNION SELECT 9) t1,
         (SELECT 0 UNION SELECT 1 UNION SELECT 2 UNION SELECT 3 UNION SELECT 4) t2,
         (SELECT @row2 := 0) init
    WHERE @row2 < 42
) AS epochs;

-- ================================================================================
-- 为任务3（已暂停）插入训练历史数据 - 87个epoch的记录
-- ================================================================================
INSERT INTO md_training_history (
    task_id, epoch,
    train_accuracy, train_loss,
    val_accuracy, val_loss,
    gpu_usage, memory_usage,
    epoch_time, record_time
)
SELECT
    'DEMO_PAUSED' AS task_id,
    seq AS epoch,
    ROUND(42 + (82.15 - 42) * (1 - EXP(-seq / 30.0)), 2) AS train_accuracy,
    ROUND(2.8 * EXP(-seq / 35.0) + 0.51, 6) AS train_loss,
    ROUND(40 + (80.67 - 40) * (1 - EXP(-seq / 32.0)) + (RAND() - 0.5) * 2.5, 2) AS val_accuracy,
    ROUND(2.9 * EXP(-seq / 33.0) + 0.58 + (RAND() - 0.5) * 0.12, 6) AS val_loss,
    ROUND(90 + RAND() * 7, 2) AS gpu_usage,
    ROUND(10000 + RAND() * 3000, 2) AS memory_usage,
    ROUND(300 + RAND() * 60, 2) AS epoch_time,
    TIMESTAMPADD(MINUTE, seq * 5, '2026-01-13 14:00:00') AS record_time
FROM (
    SELECT @row3 := @row3 + 1 AS seq
    FROM (SELECT 0 UNION SELECT 1 UNION SELECT 2 UNION SELECT 3 UNION SELECT 4 UNION SELECT 5 UNION SELECT 6 UNION SELECT 7 UNION SELECT 8 UNION SELECT 9) t1,
         (SELECT 0 UNION SELECT 1 UNION SELECT 2 UNION SELECT 3 UNION SELECT 4 UNION SELECT 5 UNION SELECT 6 UNION SELECT 7 UNION SELECT 8) t2,
         (SELECT @row3 := 0) init
    WHERE @row3 < 87
) AS epochs;

-- ================================================================================
-- 为任务1（已完成）插入模型评估数据
-- ================================================================================
INSERT INTO md_model_evaluation (
    task_id,
    `precision`, `recall`, `f1_score`,
    eval_dataset_id, eval_dataset_name, eval_sample_count,
    eval_time,
    detailed_results
) VALUES
(
    'DEMO_COMPLETED',
    77.23, 76.45, 76.84,
    'DS001_TEST', 'CIFAR-10测试集', 10000,
    '2026-01-14 11:30:00',
    '{"accuracy": 76.98, "inference_time_ms": 8.5, "model_size_mb": 42.3, "compression_ratio": 273, "speedup": 150, "notes": "最终模型评估：在CIFAR-10测试集上达到76.98%准确率。相比教师模型Qwen2.5-VL（3B参数），学生模型ResNet18仅11M参数，压缩比273倍，推理速度提升150倍（8.5ms vs 1280ms）。"}'
),
(
    'DEMO_COMPLETED',
    75.56, 74.89, 75.22,
    'DS001_VAL', 'CIFAR-10验证集', 5000,
    '2026-01-14 10:45:00',
    '{"accuracy": 75.12, "epoch": 40, "notes": "Epoch 40 checkpoint评估：验证准确率75.12%，训练进度良好。"}'
);

-- ================================================================================
-- 插入LoRA预设配置
-- ================================================================================
INSERT INTO md_lora_preset (
    preset_name, preset_desc,
    lora_rank, lora_alpha, lora_dropout,
    target_modules,
    create_time, del_flag
) VALUES
(
    '演示-高精度配置',
    '适用于追求高精度的场景，参数较多，训练时间较长',
    16, 32, 0.050,
    '["q_proj", "v_proj", "k_proj", "o_proj"]',
    NOW(), 0
),
(
    '演示-平衡配置',
    '精度和效率的平衡配置，适合大多数场景（推荐）',
    8, 16, 0.100,
    '["q_proj", "v_proj"]',
    NOW(), 0
),
(
    '演示-快速训练配置',
    '快速训练配置，参数少，训练速度快，适合快速验证',
    4, 8, 0.150,
    '["q_proj"]',
    NOW(), 0
);

-- ================================================================================
-- 验证数据插入
-- ================================================================================
SELECT
    '✅ 演示数据已成功插入！' AS 消息,
    (SELECT COUNT(*) FROM md_training_task WHERE task_id LIKE 'DEMO_%') AS 训练任务数,
    (SELECT COUNT(*) FROM md_training_history WHERE task_id LIKE 'DEMO_%') AS 历史记录数,
    (SELECT COUNT(*) FROM md_model_evaluation WHERE task_id LIKE 'DEMO_%') AS 评估记录数,
    (SELECT COUNT(*) FROM md_lora_preset WHERE preset_name LIKE '演示-%') AS LoRA预设数;

-- ================================================================================
-- 查看演示数据
-- ================================================================================
SELECT
    '📋 训练任务列表' AS 类型;

SELECT
    task_id AS 任务ID,
    task_name AS 任务名称,
    teacher_model AS 教师模型,
    student_model AS 学生模型,
    status AS 状态,
    CONCAT(current_epoch, '/', total_epochs) AS 进度,
    CONCAT(progress, '%') AS 完成度,
    best_accuracy AS 最佳准确率
FROM md_training_task
WHERE task_id LIKE 'DEMO_%'
ORDER BY create_time DESC;

-- ================================================================================
-- 清理命令（演示后使用）
-- ================================================================================
-- DELETE FROM md_training_task WHERE task_id LIKE 'DEMO_%';
-- DELETE FROM md_training_history WHERE task_id LIKE 'DEMO_%';
-- DELETE FROM md_model_evaluation WHERE task_id LIKE 'DEMO_%';
-- DELETE FROM md_lora_preset WHERE preset_name LIKE '演示-%';
