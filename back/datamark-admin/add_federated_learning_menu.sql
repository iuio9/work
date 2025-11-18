-- ============================================================================
-- 添加联邦学习菜单到数据库
-- ============================================================================
-- 功能：将联邦学习菜单添加到qczy_menu表，并分配给管理员角色
-- 使用方法：mysql -u root -p datamark < add_federated_learning_menu.sql
-- ============================================================================

USE datamark;

-- 插入联邦学习菜单（一级菜单）
INSERT INTO `qczy_menu` VALUES (
  66,                                    -- id
  0,                                     -- parent_id (0表示顶级菜单)
  'federated-learning',                  -- menu_name
  'carbon:machine-learning-model',       -- icon (机器学习图标)
  'carbon:machine-learning-model',       -- local_icon
  NULL,                                  -- permissions
  'route.federated-learning',            -- i18nKey
  '/federated-learning',                 -- web_path
  'layout.base$view.federated-learning', -- component
  NULL,                                  -- active_menu
  1,                                     -- type (1表示菜单)
  8,                                     -- sort (排序为8，在boxpulse之后)
  NOW(),                                 -- create_time
  NOW(),                                 -- update_time
  0,                                     -- hide_in_menu (0表示显示)
  0,                                     -- is_deleted (0表示未删除)
  NULL                                   -- href
);

-- 将联邦学习菜单分配给管理员角色（role_id=1）
INSERT INTO `qczy_role_menu` VALUES (1, 66);

-- 显示插入结果
SELECT '✅ 联邦学习菜单添加成功！' AS status;
SELECT * FROM qczy_menu WHERE id = 66;

-- 提示信息
SELECT '请重新登录系统以加载新菜单' AS notice;
