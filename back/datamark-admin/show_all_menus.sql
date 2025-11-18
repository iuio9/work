-- 查看数据库中所有菜单和页面
USE datamark;

-- 1. 查看所有顶级菜单（parent_id=0）
SELECT
    id,
    menu_name AS '菜单名称',
    web_path AS 'URL路径',
    icon AS '图标',
    type AS '类型',
    sort AS '排序',
    hide_in_menu AS '是否隐藏'
FROM qczy_menu
WHERE parent_id = 0 AND is_deleted = 0
ORDER BY sort;

SELECT '======================================' AS '';

-- 2. 查看所有菜单（包括子菜单）
SELECT
    id,
    parent_id AS '父ID',
    menu_name AS '菜单名称',
    web_path AS 'URL路径',
    component AS '组件',
    type AS '类型',
    sort AS '排序'
FROM qczy_menu
WHERE is_deleted = 0
ORDER BY parent_id, sort
LIMIT 50;

SELECT '======================================' AS '';

-- 3. 查看admin角色（role_id=1）有哪些菜单权限
SELECT
    rm.menu_id AS '菜单ID',
    m.menu_name AS '菜单名称',
    m.web_path AS 'URL路径'
FROM qczy_role_menu rm
LEFT JOIN qczy_menu m ON rm.menu_id = m.id
WHERE rm.role_id = 1
ORDER BY m.sort
LIMIT 20;
