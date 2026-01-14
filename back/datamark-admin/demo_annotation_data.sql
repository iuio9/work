-- ================================================================================
-- 自动标注演示数据SQL
-- ================================================================================
-- 用途：展示使用训练好的模型（DEMO_COMPLETED）进行自动标注
-- 包含：未标注的图片数据 + 模型标注记录 + 标注结果
-- ================================================================================

-- ================================================================================
-- 第1步：创建数据集（如果表存在的话）
-- ================================================================================

-- 注意：具体的表名和字段需要根据你的实际数据库结构调整
-- 这里提供一个通用的演示数据结构

-- ================================================================================
-- 创建演示用的图片数据表（如果不存在）
-- ================================================================================

-- 创建临时的演示数据集
CREATE TABLE IF NOT EXISTS demo_images (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    image_name VARCHAR(255) NOT NULL COMMENT '图片名称',
    image_path VARCHAR(500) NOT NULL COMMENT '图片路径',
    image_url VARCHAR(500) COMMENT '图片访问URL',
    dataset_name VARCHAR(100) DEFAULT 'CIFAR-10' COMMENT '数据集名称',
    true_label VARCHAR(50) COMMENT '真实标签（用于对比）',
    predicted_label VARCHAR(50) COMMENT '模型预测标签',
    confidence DECIMAL(5,2) COMMENT '预测置信度',
    status VARCHAR(20) DEFAULT 'PENDING' COMMENT '标注状态: PENDING-未标注, PREDICTED-已预测, VERIFIED-已验证',
    model_id VARCHAR(50) COMMENT '使用的模型ID',
    model_name VARCHAR(200) COMMENT '使用的模型名称',
    annotated_by VARCHAR(100) COMMENT '标注人',
    verified_by VARCHAR(100) COMMENT '验证人',
    create_time DATETIME DEFAULT CURRENT_TIMESTAMP,
    update_time DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_status (status),
    INDEX idx_model (model_id),
    INDEX idx_dataset (dataset_name)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='演示图片数据表';

-- ================================================================================
-- 插入未标注的演示图片数据（CIFAR-10的10个类别）
-- ================================================================================

-- 清空旧数据
TRUNCATE TABLE demo_images;

-- CIFAR-10的10个类别：airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

-- 插入20张未标注的演示图片
INSERT INTO demo_images (image_name, image_path, image_url, dataset_name, true_label, status) VALUES
-- 飞机类别（5张）
('airplane_001.jpg', '/demo/images/airplane_001.jpg', 'http://localhost:9091/demo/images/airplane_001.jpg', 'CIFAR-10', 'airplane', 'PENDING'),
('airplane_002.jpg', '/demo/images/airplane_002.jpg', 'http://localhost:9091/demo/images/airplane_002.jpg', 'CIFAR-10', 'airplane', 'PENDING'),
('airplane_003.jpg', '/demo/images/airplane_003.jpg', 'http://localhost:9091/demo/images/airplane_003.jpg', 'CIFAR-10', 'airplane', 'PENDING'),
('airplane_004.jpg', '/demo/images/airplane_004.jpg', 'http://localhost:9091/demo/images/airplane_004.jpg', 'CIFAR-10', 'airplane', 'PENDING'),
('airplane_005.jpg', '/demo/images/airplane_005.jpg', 'http://localhost:9091/demo/images/airplane_005.jpg', 'CIFAR-10', 'airplane', 'PENDING'),

-- 汽车类别（3张）
('automobile_001.jpg', '/demo/images/automobile_001.jpg', 'http://localhost:9091/demo/images/automobile_001.jpg', 'CIFAR-10', 'automobile', 'PENDING'),
('automobile_002.jpg', '/demo/images/automobile_002.jpg', 'http://localhost:9091/demo/images/automobile_002.jpg', 'CIFAR-10', 'automobile', 'PENDING'),
('automobile_003.jpg', '/demo/images/automobile_003.jpg', 'http://localhost:9091/demo/images/automobile_003.jpg', 'CIFAR-10', 'automobile', 'PENDING'),

-- 鸟类（2张）
('bird_001.jpg', '/demo/images/bird_001.jpg', 'http://localhost:9091/demo/images/bird_001.jpg', 'CIFAR-10', 'bird', 'PENDING'),
('bird_002.jpg', '/demo/images/bird_002.jpg', 'http://localhost:9091/demo/images/bird_002.jpg', 'CIFAR-10', 'bird', 'PENDING'),

-- 猫类（3张）
('cat_001.jpg', '/demo/images/cat_001.jpg', 'http://localhost:9091/demo/images/cat_001.jpg', 'CIFAR-10', 'cat', 'PENDING'),
('cat_002.jpg', '/demo/images/cat_002.jpg', 'http://localhost:9091/demo/images/cat_002.jpg', 'CIFAR-10', 'cat', 'PENDING'),
('cat_003.jpg', '/demo/images/cat_003.jpg', 'http://localhost:9091/demo/images/cat_003.jpg', 'CIFAR-10', 'cat', 'PENDING'),

-- 鹿类（2张）
('deer_001.jpg', '/demo/images/deer_001.jpg', 'http://localhost:9091/demo/images/deer_001.jpg', 'CIFAR-10', 'deer', 'PENDING'),
('deer_002.jpg', '/demo/images/deer_002.jpg', 'http://localhost:9091/demo/images/deer_002.jpg', 'CIFAR-10', 'deer', 'PENDING'),

-- 狗类（3张）
('dog_001.jpg', '/demo/images/dog_001.jpg', 'http://localhost:9091/demo/images/dog_001.jpg', 'CIFAR-10', 'dog', 'PENDING'),
('dog_002.jpg', '/demo/images/dog_002.jpg', 'http://localhost:9091/demo/images/dog_002.jpg', 'CIFAR-10', 'dog', 'PENDING'),
('dog_003.jpg', '/demo/images/dog_003.jpg', 'http://localhost:9091/demo/images/dog_003.jpg', 'CIFAR-10', 'dog', 'PENDING'),

-- 青蛙类（2张）
('frog_001.jpg', '/demo/images/frog_001.jpg', 'http://localhost:9091/demo/images/frog_001.jpg', 'CIFAR-10', 'frog', 'PENDING');

-- ================================================================================
-- 模拟使用DEMO_COMPLETED模型进行自动标注
-- ================================================================================

-- 场景1：完美预测（置信度高，预测正确）
UPDATE demo_images SET
    predicted_label = true_label,
    confidence = 92.5 + (RAND() * 5),  -- 92.5-97.5%
    status = 'PREDICTED',
    model_id = 'DEMO_COMPLETED',
    model_name = 'Qwen2.5-VL → ResNet18 蒸馏模型',
    annotated_by = 'AI模型',
    update_time = NOW()
WHERE image_name IN (
    'airplane_001.jpg', 'airplane_002.jpg',
    'automobile_001.jpg',
    'cat_001.jpg', 'cat_002.jpg',
    'dog_001.jpg'
);

-- 场景2：可能错误（置信度中等，需要人工验证）
UPDATE demo_images SET
    predicted_label = CASE
        WHEN image_name = 'bird_001.jpg' THEN 'airplane'  -- 误识别为飞机
        WHEN image_name = 'deer_001.jpg' THEN 'horse'     -- 误识别为马
        ELSE true_label
    END,
    confidence = 65.0 + (RAND() * 10),  -- 65-75%（中等置信度）
    status = 'PREDICTED',
    model_id = 'DEMO_COMPLETED',
    model_name = 'Qwen2.5-VL → ResNet18 蒸馏模型',
    annotated_by = 'AI模型',
    update_time = NOW()
WHERE image_name IN ('bird_001.jpg', 'deer_001.jpg');

-- 场景3：正常预测（置信度较高）
UPDATE demo_images SET
    predicted_label = true_label,
    confidence = 82.0 + (RAND() * 8),  -- 82-90%
    status = 'PREDICTED',
    model_id = 'DEMO_COMPLETED',
    model_name = 'Qwen2.5-VL → ResNet18 蒸馏模型',
    annotated_by = 'AI模型',
    update_time = NOW()
WHERE status = 'PENDING' AND id <= 15;

-- 场景4：已验证（人工验证后确认）
UPDATE demo_images SET
    status = 'VERIFIED',
    verified_by = '演示用户',
    update_time = TIMESTAMPADD(MINUTE, 5, update_time)
WHERE image_name IN ('airplane_001.jpg', 'cat_001.jpg');

-- 场景5：需要人工复核的（低置信度）
UPDATE demo_images SET
    confidence = 55.0 + (RAND() * 8),  -- 55-63%（低置信度，需要人工复核）
    update_time = NOW()
WHERE image_name IN ('frog_001.jpg', 'deer_002.jpg');

-- ================================================================================
-- 创建自动标注任务记录表（如果不存在）
-- ================================================================================

CREATE TABLE IF NOT EXISTS demo_annotation_tasks (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    task_id VARCHAR(50) UNIQUE NOT NULL COMMENT '标注任务ID',
    task_name VARCHAR(200) NOT NULL COMMENT '任务名称',
    model_id VARCHAR(50) NOT NULL COMMENT '使用的模型ID',
    model_name VARCHAR(200) COMMENT '模型名称',
    dataset_name VARCHAR(100) COMMENT '数据集名称',
    total_images INT DEFAULT 0 COMMENT '总图片数',
    predicted_images INT DEFAULT 0 COMMENT '已预测图片数',
    verified_images INT DEFAULT 0 COMMENT '已验证图片数',
    avg_confidence DECIMAL(5,2) COMMENT '平均置信度',
    accuracy DECIMAL(5,2) COMMENT '准确率（已验证部分）',
    status VARCHAR(20) DEFAULT 'RUNNING' COMMENT '任务状态',
    start_time DATETIME COMMENT '开始时间',
    end_time DATETIME COMMENT '结束时间',
    create_time DATETIME DEFAULT CURRENT_TIMESTAMP,
    update_time DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_model (model_id),
    INDEX idx_status (status)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='自动标注任务表';

-- ================================================================================
-- 插入自动标注任务记录
-- ================================================================================

INSERT INTO demo_annotation_tasks (
    task_id, task_name, model_id, model_name, dataset_name,
    total_images, predicted_images, verified_images,
    avg_confidence, accuracy, status,
    start_time, end_time
) VALUES (
    'ANNO_DEMO_001',
    'CIFAR-10 自动标注任务（使用蒸馏模型）',
    'DEMO_COMPLETED',
    'Qwen2.5-VL → ResNet18 蒸馏模型',
    'CIFAR-10',
    17,  -- 已标注17张
    15,  -- 已预测15张
    2,   -- 已验证2张
    84.5,  -- 平均置信度84.5%
    100.0, -- 验证的2张全部正确
    'RUNNING',
    '2026-01-14 14:00:00',
    NULL
);

-- ================================================================================
-- 查看演示数据
-- ================================================================================

-- 1. 查看所有图片的标注状态
SELECT
    '图片标注概览' AS 类型,
    dataset_name AS 数据集,
    COUNT(*) AS 总数,
    SUM(CASE WHEN status = 'PENDING' THEN 1 ELSE 0 END) AS 未标注,
    SUM(CASE WHEN status = 'PREDICTED' THEN 1 ELSE 0 END) AS 已预测,
    SUM(CASE WHEN status = 'VERIFIED' THEN 1 ELSE 0 END) AS 已验证,
    ROUND(AVG(CASE WHEN predicted_label IS NOT NULL THEN confidence ELSE NULL END), 2) AS 平均置信度
FROM demo_images
GROUP BY dataset_name;

-- 2. 查看需要人工复核的图片（低置信度或可能错误）
SELECT
    '需要人工复核' AS 提示,
    image_name AS 图片,
    true_label AS 真实标签,
    predicted_label AS 预测标签,
    CONCAT(ROUND(confidence, 1), '%') AS 置信度,
    CASE
        WHEN predicted_label != true_label THEN '❌ 预测错误'
        WHEN confidence < 70 THEN '⚠️  低置信度'
        ELSE '✅ 正常'
    END AS 状态
FROM demo_images
WHERE status = 'PREDICTED'
  AND (confidence < 70 OR predicted_label != true_label)
ORDER BY confidence ASC;

-- 3. 查看高质量预测（可直接确认）
SELECT
    '高质量预测' AS 提示,
    image_name AS 图片,
    predicted_label AS 预测标签,
    CONCAT(ROUND(confidence, 1), '%') AS 置信度
FROM demo_images
WHERE status = 'PREDICTED'
  AND confidence >= 90
  AND predicted_label = true_label
ORDER BY confidence DESC;

-- 4. 查看标注任务统计
SELECT
    task_name AS 任务名称,
    model_name AS 使用模型,
    CONCAT(predicted_images, '/', total_images) AS 预测进度,
    CONCAT(ROUND(avg_confidence, 1), '%') AS 平均置信度,
    CONCAT(ROUND(accuracy, 1), '%') AS 准确率,
    status AS 状态
FROM demo_annotation_tasks;

-- ================================================================================
-- 验证数据
-- ================================================================================

SELECT
    '演示数据已准备完成！' AS 消息,
    (SELECT COUNT(*) FROM demo_images) AS 总图片数,
    (SELECT COUNT(*) FROM demo_images WHERE status = 'PREDICTED') AS 已预测,
    (SELECT COUNT(*) FROM demo_images WHERE status = 'VERIFIED') AS 已验证,
    (SELECT COUNT(*) FROM demo_annotation_tasks) AS 标注任务数;

-- ================================================================================
-- 使用说明
-- ================================================================================
--
-- 1. 演示场景：
--    - 已训练完成模型（DEMO_COMPLETED）可用于自动标注
--    - 17张演示图片，15张已被模型预测
--    - 2张已人工验证确认
--    - 展示不同置信度的预测结果
--    - 展示需要人工复核的情况
--
-- 2. 前端展示效果：
--    - 标注任务列表：显示使用DEMO_COMPLETED模型的标注任务
--    - 图片列表：显示待标注、已预测、已验证的图片
--    - 预测结果：显示标签 + 置信度
--    - 人工复核：低置信度或错误预测需要人工确认
--
-- 3. 演示重点：
--    - 模型自动标注速度快（秒级完成）
--    - 高置信度预测可直接确认（>90%）
--    - 低置信度需要人工复核（<70%）
--    - 准确率统计（已验证部分）
--
-- 4. 清理演示数据：
--    DROP TABLE IF EXISTS demo_images;
--    DROP TABLE IF EXISTS demo_annotation_tasks;
--
-- ================================================================================
