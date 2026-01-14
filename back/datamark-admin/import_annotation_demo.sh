#!/bin/bash
#=============================================================================
# 标注演示数据一键导入脚本
#=============================================================================
# 用途：自动导入标注演示数据，展示完整的"训练→标注"流程
# 使用：chmod +x import_annotation_demo.sh && ./import_annotation_demo.sh
#=============================================================================

set -e

echo "=========================================="
echo "🏷️  自动标注演示数据导入工具"
echo "=========================================="
echo ""

# 配置
DB_USER="root"
DB_PASSWORD="qczy1717"
DB_NAME="datamark"
DB_HOST="localhost"
DB_PORT="3306"
SQL_FILE="demo_annotation_data.sql"

# 检查SQL文件
if [ ! -f "$SQL_FILE" ]; then
    echo "❌ 找不到 $SQL_FILE"
    exit 1
fi

echo "📋 将导入以下演示数据："
echo "   • 17张待标注图片（CIFAR-10）"
echo "   • 1个自动标注任务（使用DEMO_COMPLETED模型）"
echo "   • 15张已预测图片（不同置信度）"
echo "   • 2张已验证图片"
echo ""

# 检查MySQL连接
echo "🔍 检查MySQL连接..."
if ! mysql -u "$DB_USER" -p"$DB_PASSWORD" -h "$DB_HOST" -P "$DB_PORT" -e "SELECT 1;" > /dev/null 2>&1; then
    echo "❌ 无法连接到MySQL"
    exit 1
fi
echo "✅ MySQL连接成功"
echo ""

# 询问是否继续
read -p "是否继续导入？(y/N) " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ 已取消"
    exit 0
fi

# 导入数据
echo ""
echo "🚀 开始导入标注演示数据..."
echo ""

if mysql -u "$DB_USER" -p"$DB_PASSWORD" -h "$DB_HOST" -P "$DB_PORT" "$DB_NAME" < "$SQL_FILE"; then
    echo ""
    echo "=========================================="
    echo "✅ 标注演示数据导入成功！"
    echo "=========================================="
    echo ""

    # 验证结果
    echo "📊 验证导入结果："
    mysql -u "$DB_USER" -p"$DB_PASSWORD" -h "$DB_HOST" -P "$DB_PORT" "$DB_NAME" -e "
        SELECT
            '图片数据' AS 数据类型,
            COUNT(*) AS 总数,
            SUM(CASE WHEN status='PENDING' THEN 1 ELSE 0 END) AS 未标注,
            SUM(CASE WHEN status='PREDICTED' THEN 1 ELSE 0 END) AS 已预测,
            SUM(CASE WHEN status='VERIFIED' THEN 1 ELSE 0 END) AS 已验证
        FROM demo_images
        UNION ALL
        SELECT
            '标注任务',
            COUNT(*),
            NULL,
            NULL,
            NULL
        FROM demo_annotation_tasks;
    "

    echo ""
    echo "📝 标注任务详情："
    mysql -u "$DB_USER" -p"$DB_PASSWORD" -h "$DB_HOST" -P "$DB_PORT" "$DB_NAME" -e "
        SELECT
            task_name AS 任务名称,
            model_name AS 使用模型,
            CONCAT(predicted_images, '/', total_images) AS 预测进度,
            CONCAT(ROUND(avg_confidence, 1), '%') AS 平均置信度
        FROM demo_annotation_tasks;
    "

    echo ""
    echo "🎯 需要重点关注的图片（低置信度或错误预测）："
    mysql -u "$DB_USER" -p"$DB_PASSWORD" -h "$DB_HOST" -P "$DB_PORT" "$DB_NAME" -e "
        SELECT
            image_name AS 图片,
            predicted_label AS 预测,
            CONCAT(ROUND(confidence, 1), '%') AS 置信度,
            CASE
                WHEN predicted_label != true_label THEN '❌ 错误'
                WHEN confidence < 70 THEN '⚠️  低置信'
                ELSE '✅ 正常'
            END AS 状态
        FROM demo_images
        WHERE status='PREDICTED'
          AND (confidence < 70 OR predicted_label != true_label)
        ORDER BY confidence;
    "

    echo ""
    echo "=========================================="
    echo "🎉 全部完成！"
    echo "=========================================="
    echo ""
    echo "📚 下一步："
    echo "   1. 查看详细演示指南：cat ANNOTATION_DEMO_GUIDE.md"
    echo "   2. 启动系统查看标注功能"
    echo "   3. 展示完整的 训练→标注 工作流程"
    echo ""

else
    echo ""
    echo "❌ 导入失败"
    exit 1
fi
