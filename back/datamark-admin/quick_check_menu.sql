-- 快速检查联邦学习菜单是否已添加
USE datamark;
SELECT COUNT(*) as menu_exists FROM qczy_menu WHERE menu_name = 'federated-learning';
