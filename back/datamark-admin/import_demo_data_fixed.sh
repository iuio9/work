#!/bin/bash
#=============================================================================
# 演示数据一键导入脚本（修复版）
#=============================================================================
# 用途：导入与实际表结构完全匹配的演示数据
# 使用：chmod +x import_demo_data_fixed.sh && ./import_demo_data_fixed.sh
#=============================================================================

set -e

echo "=========================================="
echo "🎯 大小模型协同训练 - 演示数据导入工具（修复版）"
echo "=========================================="
echo ""

# 配置
DB_USER="root"
DB_PASSWORD="20031217scz"
DB_NAME="mark"
DB_HOST="localhost"
DB_PORT="3306"
SQL_FILE="demo_data_fixed.sql"

# 检查SQL文件
if [ ! -f "$SQL_FILE" ]; then
    echo "❌ 错误：找不到 $SQL_FILE 文件"
    echo "   请确保在 /home/user/work/back/datamark-admin 目录下执行此脚本"
    exit 1
fi

echo "📋 配置信息："
echo "   数据库：$DB_NAME"
echo "   用户：$DB_USER"
echo "   SQL文件：$SQL_FILE"
echo ""

# 询问是否继续
echo "⚠️  此操作将："
echo "   1. 删除所有以 DEMO_ 开头的旧演示数据"
echo "   2. 导入新的演示数据（3个任务 + 179条历史 + 2条评估）"
echo ""
read -p "是否继续？(y/N) " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ 已取消导入"
    exit 0
fi

# 开始导入
echo ""
echo "🚀 开始导入演示数据..."
echo ""

# 执行SQL
if mysql -u "$DB_USER" -p"$DB_PASSWORD" -h "$DB_HOST" -P "$DB_PORT" "$DB_NAME" < "$SQL_FILE"; then
    echo ""
    echo "=========================================="
    echo "✅ 演示数据导入成功！"
    echo "=========================================="
    echo ""

    # 验证导入结果
    echo "📊 验证导入结果："
    mysql -u "$DB_USER" -p"$DB_PASSWORD" -h "$DB_HOST" -P "$DB_PORT" "$DB_NAME" -e "
        SELECT
            '训练任务' AS 数据类型,
            COUNT(*) AS 数量
        FROM md_training_task
        WHERE task_id LIKE 'DEMO_%'
        UNION ALL
        SELECT
            '训练历史',
            COUNT(*)
        FROM md_training_history
        WHERE task_id LIKE 'DEMO_%'
        UNION ALL
        SELECT
            '模型评估',
            COUNT(*)
        FROM md_model_evaluation
        WHERE task_id LIKE 'DEMO_%'
        UNION ALL
        SELECT
            'LoRA预设',
            COUNT(*)
        FROM md_lora_preset
        WHERE preset_name LIKE '演示-%';
    "

    echo ""
    echo "📝 演示任务列表："
    mysql -u "$DB_USER" -p"$DB_PASSWORD" -h "$DB_HOST" -P "$DB_PORT" "$DB_NAME" -e "
        SELECT
            task_id AS 任务ID,
            task_name AS 任务名称,
            status AS 状态,
            CONCAT(current_epoch, '/', total_epochs) AS 进度,
            best_accuracy AS 准确率
        FROM md_training_task
        WHERE task_id LIKE 'DEMO_%'
        ORDER BY create_time DESC;
    "

    echo ""
    echo "=========================================="
    echo "🎉 全部完成！"
    echo "=========================================="
    echo ""
    echo "📚 下一步操作："
    echo "   1. 重启后端：mvn spring-boot:run"
    echo "   2. 刷新前端页面（Ctrl+Shift+R 清除缓存）"
    echo "   3. 进入"模型蒸馏训练"页面查看演示数据"
    echo ""
    echo "   如果看不到数据，请检查："
    echo "   - 后端日志中是否有错误"
    echo "   - 浏览器控制台是否有API调用失败"
    echo "   - 数据库连接是否正确"
    echo ""

else
    echo ""
    echo "=========================================="
    echo "❌ 导入失败"
    echo "=========================================="
    echo ""
    echo "请检查："
    echo "1. MySQL服务是否运行：sudo systemctl status mysql"
    echo "2. 数据库是否存在：mysql -u $DB_USER -p$DB_PASSWORD -e 'SHOW DATABASES;'"
    echo "3. 表结构是否存在：mysql -u $DB_USER -p$DB_PASSWORD $DB_NAME -e 'SHOW TABLES;'"
    echo ""
    exit 1
fi
