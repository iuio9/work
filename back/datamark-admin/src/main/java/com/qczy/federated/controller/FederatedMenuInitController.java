package com.qczy.federated.controller;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.web.bind.annotation.*;

import java.util.HashMap;
import java.util.Map;

/**
 * 联邦学习菜单初始化控制器
 * 用于将联邦学习菜单添加到数据库
 */
@RestController
@RequestMapping("/api/federated/init")
public class FederatedMenuInitController {

    @Autowired
    private JdbcTemplate jdbcTemplate;

    /**
     * 初始化联邦学习菜单
     * 访问方式: POST http://localhost:8081/api/federated/init/menu
     */
    @PostMapping("/menu")
    public Map<String, Object> initFederatedMenu() {
        Map<String, Object> result = new HashMap<>();

        try {
            // 1. 检查菜单是否已存在
            Integer count = jdbcTemplate.queryForObject(
                "SELECT COUNT(*) FROM qczy_menu WHERE menu_name = 'federated-learning'",
                Integer.class
            );

            if (count != null && count > 0) {
                result.put("success", true);
                result.put("message", "联邦学习菜单已存在，无需重复添加");
                result.put("menuExists", true);
                return result;
            }

            // 2. 插入联邦学习菜单（一级菜单）
            String insertMenuSql = "INSERT INTO `qczy_menu` (" +
                "id, parent_id, menu_name, icon, local_icon, permissions, i18nKey, " +
                "web_path, component, active_menu, type, sort, create_time, update_time, " +
                "hide_in_menu, is_deleted, href" +
                ") VALUES (" +
                "66, 0, 'federated-learning', 'carbon:machine-learning-model', " +
                "'carbon:machine-learning-model', NULL, 'route.federated-learning', " +
                "'/federated-learning', 'layout.base$view.federated-learning', NULL, " +
                "1, 8, NOW(), NOW(), 0, 0, NULL" +
                ")";

            jdbcTemplate.execute(insertMenuSql);

            // 3. 将联邦学习菜单分配给管理员角色（role_id=1）
            String insertRoleMenuSql = "INSERT INTO `qczy_role_menu` (role_id, menu_id) VALUES (1, 66)";
            jdbcTemplate.execute(insertRoleMenuSql);

            // 4. 验证插入结果
            Map<String, Object> menuInfo = jdbcTemplate.queryForMap(
                "SELECT id, menu_name, web_path, icon, sort FROM qczy_menu WHERE id = 66"
            );

            result.put("success", true);
            result.put("message", "✅ 联邦学习菜单添加成功！请重新登录系统以加载新菜单");
            result.put("menuExists", false);
            result.put("menuInfo", menuInfo);

        } catch (Exception e) {
            result.put("success", false);
            result.put("message", "添加菜单失败: " + e.getMessage());
            result.put("error", e.getClass().getName());
        }

        return result;
    }

    /**
     * 检查联邦学习菜单状态
     * 访问方式: GET http://localhost:8081/api/federated/init/check
     */
    @GetMapping("/check")
    public Map<String, Object> checkFederatedMenu() {
        Map<String, Object> result = new HashMap<>();

        try {
            // 1. 检查菜单是否存在
            Integer menuCount = jdbcTemplate.queryForObject(
                "SELECT COUNT(*) FROM qczy_menu WHERE menu_name = 'federated-learning'",
                Integer.class
            );

            boolean menuExists = menuCount != null && menuCount > 0;
            result.put("menuExists", menuExists);

            if (menuExists) {
                // 2. 获取菜单完整信息（包含component字段用于诊断）
                Map<String, Object> menuInfo = jdbcTemplate.queryForMap(
                    "SELECT id, parent_id, menu_name, web_path, component, icon, i18nKey, sort, hide_in_menu, is_deleted FROM qczy_menu WHERE menu_name = 'federated-learning'"
                );
                result.put("menuInfo", menuInfo);

                // 3. 检查角色菜单关联
                Integer roleMenuCount = jdbcTemplate.queryForObject(
                    "SELECT COUNT(*) FROM qczy_role_menu WHERE role_id = 1 AND menu_id = 66",
                    Integer.class
                );
                result.put("roleMenuAssigned", roleMenuCount != null && roleMenuCount > 0);
            }

            result.put("success", true);
            result.put("message", menuExists ? "联邦学习菜单已存在" : "联邦学习菜单未添加");

        } catch (Exception e) {
            result.put("success", false);
            result.put("message", "检查失败: " + e.getMessage());
            result.put("error", e.getClass().getName());
        }

        return result;
    }

    /**
     * 删除联邦学习菜单（仅用于测试/重置）
     * 访问方式: DELETE http://localhost:8081/api/federated/init/menu
     */
    @DeleteMapping("/menu")
    public Map<String, Object> removeFederatedMenu() {
        Map<String, Object> result = new HashMap<>();

        try {
            // 1. 删除角色菜单关联
            jdbcTemplate.execute("DELETE FROM qczy_role_menu WHERE menu_id = 66");

            // 2. 删除菜单记录
            jdbcTemplate.execute("DELETE FROM qczy_menu WHERE id = 66");

            result.put("success", true);
            result.put("message", "联邦学习菜单已删除");

        } catch (Exception e) {
            result.put("success", false);
            result.put("message", "删除失败: " + e.getMessage());
            result.put("error", e.getClass().getName());
        }

        return result;
    }
}
