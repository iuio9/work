-- 检查联邦学习菜单是否存在
USE datamark;

-- 1. 检查菜单表中是否有联邦学习菜单
SELECT '=== 检查菜单是否存在 ===' AS step;
SELECT * FROM qczy_menu WHERE menu_name = 'federated-learning';

-- 2. 检查最大菜单ID（如果上面查询结果为空，说明菜单还没添加）
SELECT '=== 当前最大菜单ID ===' AS step;
SELECT MAX(id) as max_menu_id FROM qczy_menu;

-- 3. 检查管理员角色是否有联邦学习菜单权限
SELECT '=== 检查角色菜单关联 ===' AS step;
SELECT rm.*, m.menu_name
FROM qczy_role_menu rm
LEFT JOIN qczy_menu m ON rm.menu_id = m.id
WHERE rm.role_id = 1 AND m.menu_name = 'federated-learning';

-- 4. 列出所有顶级菜单（parent_id=0）
SELECT '=== 所有顶级菜单 ===' AS step;
SELECT id, menu_name, icon, sort, hide_in_menu
FROM qczy_menu
WHERE parent_id = 0
ORDER BY sort;
