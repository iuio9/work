#!/bin/bash
#=============================================================================
# 演示数据一键导入脚本
#=============================================================================
# 用途：自动导入演示数据到数据库，无需手动操作
# 使用：chmod +x import_demo_data.sh && ./import_demo_data.sh
#=============================================================================

set -e  # 遇到错误立即退出

echo "=========================================="
echo "🎯 大小模型协同训练 - 演示数据导入工具"
echo "=========================================="
echo ""

# 配置（根据你的实际配置修改）
DB_USER="root"
DB_PASSWORD="qczy1717"
DB_NAME="datamark"
DB_HOST="localhost"
DB_PORT="3306"
SQL_FILE="demo_data.sql"

# 检查SQL文件是否存在
if [ ! -f "$SQL_FILE" ]; then
    echo "❌ 错误：找不到 $SQL_FILE 文件"
    echo "   请确保在 /home/user/work/back/datamark-admin 目录下执行此脚本"
    exit 1
fi

echo "📋 配置信息："
echo "   数据库：$DB_NAME"
echo "   用户：$DB_USER"
echo "   主机：$DB_HOST:$DB_PORT"
echo "   SQL文件：$SQL_FILE"
echo ""

# 检查MySQL是否可访问
echo "🔍 检查MySQL连接..."
if ! mysql -u "$DB_USER" -p"$DB_PASSWORD" -h "$DB_HOST" -P "$DB_PORT" -e "SELECT 1;" > /dev/null 2>&1; then
    echo "❌ 无法连接到MySQL"
    echo "   请检查："
    echo "   1. MySQL服务是否启动：sudo systemctl status mysql"
    echo "   2. 用户名密码是否正确"
    echo "   3. 数据库是否存在：mysql -u root -p -e 'SHOW DATABASES;'"
    exit 1
fi
echo "✅ MySQL连接成功"
echo ""

# 询问是否继续
echo "⚠️  警告：此操作将："
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
if mysql -u "$DB_USER" -p"$DB_PASSWORD" -h "$DB_HOST" -P "$DB_PORT" "$DB_NAME" < "$SQL_FILE" 2>&1 | tee /tmp/import_log.txt; then
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
        WHERE task_id LIKE 'DEMO_%';
    "

    echo ""
    echo "📝 演示任务列表："
    mysql -u "$DB_USER" -p"$DB_PASSWORD" -h "$DB_HOST" -P "$DB_PORT" "$DB_NAME" -e "
        SELECT
            task_id AS 任务ID,
            task_name AS 任务名称,
            status AS 状态,
            CONCAT(current_epoch, '/', total_epochs) AS 进度,
            best_accuracy AS 最佳准确率
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
    echo "   1. 启动后端：mvn spring-boot:run"
    echo "   2. 启动前端：cd ../../front/data-mark-v3 && npm run dev"
    echo "   3. 打开浏览器：http://localhost:3000"
    echo "   4. 进入"模型蒸馏训练"页面查看演示数据"
    echo ""
    echo "📖 详细说明请查看：DEMO_SETUP_GUIDE.md"
    echo ""

else
    echo ""
    echo "=========================================="
    echo "❌ 导入失败"
    echo "=========================================="
    echo ""
    echo "错误日志已保存到：/tmp/import_log.txt"
    echo ""
    echo "常见问题排查："
    echo "   1. 检查数据库表是否存在："
    echo "      mysql -u $DB_USER -p$DB_PASSWORD -e 'USE $DB_NAME; SHOW TABLES;'"
    echo ""
    echo "   2. 检查表结构是否正确："
    echo "      mysql -u $DB_USER -p$DB_PASSWORD -e 'DESC $DB_NAME.md_training_task;'"
    echo ""
    echo "   3. 查看详细错误："
    echo "      cat /tmp/import_log.txt"
    echo ""
    exit 1
fi
