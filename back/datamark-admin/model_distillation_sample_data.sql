-- ============================================
-- 大小模型协同训练示例数据
-- ============================================

-- 插入示例训练任务
INSERT INTO `md_training_task` (
    `task_id`, `task_name`, `teacher_model`, `student_model`,
    `total_epochs`, `current_epoch`, `batch_size`, `learning_rate`,
    `temperature`, `alpha`, `lora_rank`, `lora_alpha`, `lora_dropout`,
    `status`, `progress`, `accuracy`, `loss`, `best_accuracy`,
    `model_path`, `model_url`,
    `start_time`, `end_time`, `duration`,
    `create_time`, `update_time`, `del_flag`
) VALUES
-- 已完成的任务
('TASK_001', '目标检测协同训练-YOLOv5', 'llama2-7b', 'yolov5s',
 50, 50, 32, 0.001,
 3.0, 0.7, 16, 32, 0.05,
 'COMPLETED', 100, 92.50, 0.0234, 93.20,
 '/models/distillation/TASK_001/final_model.pth', '/api/models/distillation/TASK_001',
 DATE_SUB(NOW(), INTERVAL 2 DAY), DATE_SUB(NOW(), INTERVAL 1 DAY), 7200,
 DATE_SUB(NOW(), INTERVAL 3 DAY), DATE_SUB(NOW(), INTERVAL 1 DAY), 0),

('TASK_002', '图像分类协同训练-ResNet', 'qwen-7b', 'resnet50',
 40, 40, 64, 0.0005,
 2.5, 0.6, 8, 16, 0.05,
 'COMPLETED', 100, 88.30, 0.0456, 89.10,
 '/models/distillation/TASK_002/final_model.pth', '/api/models/distillation/TASK_002',
 DATE_SUB(NOW(), INTERVAL 5 DAY), DATE_SUB(NOW(), INTERVAL 4 DAY), 5400,
 DATE_SUB(NOW(), INTERVAL 6 DAY), DATE_SUB(NOW(), INTERVAL 4 DAY), 0),

('TASK_003', '语义分割协同训练-UNet', 'llama2-13b', 'unet',
 60, 60, 16, 0.002,
 4.0, 0.8, 16, 32, 0.05,
 'COMPLETED', 100, 85.70, 0.0567, 87.40,
 '/models/distillation/TASK_003/final_model.pth', '/api/models/distillation/TASK_003',
 DATE_SUB(NOW(), INTERVAL 7 DAY), DATE_SUB(NOW(), INTERVAL 6 DAY), 10800,
 DATE_SUB(NOW(), INTERVAL 8 DAY), DATE_SUB(NOW(), INTERVAL 6 DAY), 0),

('TASK_005', '视觉Transformer协同训练', 'llama2-7b', 'vit',
 45, 45, 32, 0.0008,
 3.0, 0.65, 16, 32, 0.05,
 'COMPLETED', 100, 90.20, 0.0312, 91.50,
 '/models/distillation/TASK_005/final_model.pth', '/api/models/distillation/TASK_005',
 DATE_SUB(NOW(), INTERVAL 10 DAY), DATE_SUB(NOW(), INTERVAL 9 DAY), 8100,
 DATE_SUB(NOW(), INTERVAL 11 DAY), DATE_SUB(NOW(), INTERVAL 9 DAY), 0),

-- 正在运行的任务
('TASK_004', '序列预测协同训练-LSTM', 'qwen-14b', 'lstm',
 30, 18, 32, 0.001,
 3.5, 0.7, 8, 16, 0.05,
 'RUNNING', 60, 82.50, 0.0623, 83.20,
 NULL, NULL,
 DATE_SUB(NOW(), INTERVAL 3 HOUR), NULL, NULL,
 DATE_SUB(NOW(), INTERVAL 4 HOUR), NOW(), 0);

-- 插入训练历史记录（为TASK_001添加部分历史）
INSERT INTO `md_training_history` (
    `task_id`, `epoch`,
    `train_loss`, `train_accuracy`, `val_loss`, `val_accuracy`,
    `distill_loss`, `hard_loss`, `soft_loss`,
    `gpu_usage`, `memory_usage`, `epoch_time`, `record_time`
) VALUES
('TASK_001', 10, 0.4523, 78.50, 0.4821, 76.30, 0.4672, 0.3215, 0.1457, 85.30, 3456.50, 142.50, DATE_SUB(NOW(), INTERVAL 2 DAY)),
('TASK_001', 20, 0.3214, 84.20, 0.3456, 82.10, 0.3335, 0.2456, 0.0879, 86.20, 3512.30, 138.20, DATE_SUB(NOW(), INTERVAL 2 DAY)),
('TASK_001', 30, 0.2567, 88.70, 0.2789, 86.50, 0.2678, 0.1923, 0.0755, 87.10, 3587.60, 135.80, DATE_SUB(NOW(), INTERVAL 1 DAY)),
('TASK_001', 40, 0.1823, 91.30, 0.2012, 89.80, 0.1918, 0.1534, 0.0384, 88.50, 3645.20, 132.40, DATE_SUB(NOW(), INTERVAL 1 DAY)),
('TASK_001', 50, 0.0234, 92.50, 0.0456, 91.20, 0.0345, 0.0234, 0.0111, 89.20, 3698.40, 128.60, DATE_SUB(NOW(), INTERVAL 1 DAY));

-- 插入模型评估结果（为TASK_001添加评估）
INSERT INTO `md_model_evaluation` (
    `task_id`, `precision`, `recall`, `f1_score`, `map_50`, `map_95`,
    `eval_dataset_id`, `eval_dataset_name`, `eval_sample_count`,
    `eval_time`, `detailed_results`
) VALUES
('TASK_001', 93.50, 91.20, 92.33, 89.70, 85.40,
 101, 'COCO Validation Set', 5000,
 DATE_SUB(NOW(), INTERVAL 1 DAY),
 '{"class_accuracies": {"person": 95.2, "car": 92.1, "dog": 89.5}, "confusion_matrix": [[450, 30, 20], [25, 460, 15], [35, 20, 445]]}');
