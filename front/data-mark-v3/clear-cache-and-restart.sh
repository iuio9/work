#!/bin/bash

echo "================================"
echo "清除 Vite 缓存并重启开发服务器"
echo "================================"

# 停止当前运行的开发服务器
echo "1. 停止开发服务器（如果正在运行）..."
pkill -f "vite" || true

# 清除 Vite 缓存
echo "2. 清除 Vite 缓存..."
rm -rf node_modules/.vite
rm -rf node_modules/.cache
rm -rf dist
rm -rf .vite

# 清除浏览器缓存提示
echo "3. 请在浏览器中执行以下操作："
echo "   - 打开开发者工具（F12）"
echo "   - 右键点击刷新按钮"
echo "   - 选择 '清空缓存并硬性重新加载'"
echo "   或者按 Ctrl+Shift+R (Windows/Linux) / Cmd+Shift+R (Mac)"

# 等待用户确认
echo ""
echo "按回车键继续重启开发服务器..."
read

# 重启开发服务器
echo "4. 重启开发服务器..."
npm run dev
