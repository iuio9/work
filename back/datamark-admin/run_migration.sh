#!/bin/bash

# ============================================
# 数据库迁移执行脚本
# ============================================

echo "=================================="
echo "数据集ID字段类型转换 - 数据库迁移"
echo "=================================="
echo ""

# 数据库配置
DB_HOST="localhost"
DB_PORT="3306"
DB_NAME="mark"
DB_USER="root"
DB_PASS="20031217scz"
SQL_FILE="/home/user/work/back/datamark-admin/update_dataset_id_to_string.sql"

echo "📋 迁移内容："
echo "  ✓ 修改 md_training_task.dataset_id: BIGINT → VARCHAR(100)"
echo "  ✓ 添加 md_training_task.val_dataset_id: VARCHAR(100)"
echo "  ✓ 添加 md_training_task.val_dataset_name: VARCHAR(255)"
echo "  ✓ 添加 md_training_task.training_config: TEXT"
echo "  ✓ 修改 md_model_evaluation.eval_dataset_id: BIGINT → VARCHAR(100)"
echo ""
echo "数据库信息："
echo "  Host: localhost"
echo "  Port: 3306"
echo "  Database: mark"
echo "  Username: root"
echo ""

# 询问用户是否继续
read -p "是否执行数据库迁移？(y/n): " confirm
if [[ $confirm != "y" && $confirm != "Y" ]]; then
    echo "❌ 迁移已取消"
    exit 0
fi

echo "📝 正在执行数据库迁移..."

# 执行SQL脚本
mysql -h localhost -u root -p20031217scz mark < update_dataset_id_to_string.sql

if [ $? -eq 0 ]; then
    echo "✅ 数据库迁移成功！"
    echo ""
    echo "📋 下一步操作："
    echo "1. 重启后端服务（Spring Boot应用）"
    echo "2. 测试创建训练任务功能"
    echo "3. 使用字符串类型的数据集ID进行测试（如 'dialogue-zh-v2'）"
    echo ""
    echo "详细说明请参考：DATABASE_MIGRATION_GUIDE.md"
else
    echo "数据库迁移失败！请检查错误信息并参考 DATABASE_MIGRATION_GUIDE.md"
    exit 1
fi
