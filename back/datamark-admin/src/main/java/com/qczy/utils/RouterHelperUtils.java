package com.qczy.utils;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.core.toolkit.CollectionUtils;
import com.qczy.mapper.ButtonPermissionMapper;
import com.qczy.mapper.RoleButtonMapper;
import com.qczy.mapper.RoleMapper;
import com.qczy.mapper.UserMapper;
import com.qczy.model.entity.ButtonPermissionEntity;
import com.qczy.model.entity.MenuEntity;
import com.qczy.model.entity.RoleButtonEntity;
import com.qczy.model.entity.UserEntity;
import com.qczy.model.response.MenuResponse;
import com.qczy.model.response.MetaResponse;
import com.qczy.model.response.RouterResponse;
import com.qczy.service.MenuService;
import com.qczy.service.RoleService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;
import org.springframework.util.StringUtils;


import java.util.*;
import java.util.stream.Collectors;

/**
 * 根据菜单数据构建路由的工具类
 */
@Component
public class RouterHelperUtils {


    @Autowired
    private ButtonPermissionMapper buttonPermissionMapper;
    @Autowired
    private RoleMapper roleMapper;
    @Autowired
    private UserMapper userMapper;
    @Autowired
    private CurrentLoginUserUtils currentLoginUserUtils;
    @Autowired
    private RoleService roleService;
    @Autowired
    private RoleButtonMapper roleButtonMapper;

    /**
     * 根据菜单构建路由
     *
     * @param menus
     * @return
     */
    public List<RouterResponse> buildRouters(List<MenuEntity> menus) {
        // 获取当前登录的用户
        Integer userId = currentLoginUserUtils.getCurrentLoginUserId();
        // 当前用户所拥有的权限
        UserEntity userEntity = userMapper.selectById(userId);
        String userRoles = userEntity.getUserRoles();
        String[] roleIds = userRoles.split(",");
        List<Integer> buttonIds = new ArrayList<>();
        for (String roleId : roleIds) {
            List<RoleButtonEntity> list
                    = roleButtonMapper.selectList(new LambdaQueryWrapper<RoleButtonEntity>().eq(RoleButtonEntity::getRoleId, roleId));
            if (CollectionUtils.isEmpty(list)) {
                continue;
            }
            list.stream().map(RoleButtonEntity::getButtonId).forEach(buttonIds::add);
        }
        if (!CollectionUtils.isEmpty(buttonIds)) {
            buttonIds = buttonIds.stream().distinct().collect(Collectors.toList());
        }


        List<RouterResponse> routers = new LinkedList<RouterResponse>();
        for (MenuEntity menu : menus) {
            RouterResponse router = new RouterResponse();
            // router.setHidden(false);
            // router.setAlwaysShow(false);
            router.setName(menu.getMenuName());
            router.setPath(getRouterPath(menu));
            router.setComponent(menu.getComponent());
            router.setActiveMenu(menu.getActiveMenu());
            router.setMeta(new MetaResponse(menu.getMenuName(), menu.getIcon(), menu.getSort(), menu.getI18nKey(), menu.getActiveMenu(), menu.getLocalIcon(),
                    menu.getType() != null && menu.getType() == 2,menu.getHref()));
            List<MenuEntity> children = menu.getChildren();

            List<ButtonPermissionEntity> buttonList = buttonPermissionMapper.selectList(
                    new LambdaQueryWrapper<ButtonPermissionEntity>()
                            .eq(ButtonPermissionEntity::getParentId, menu.getId())
            );

            if (!CollectionUtils.isEmpty(buttonList)) {
                List<ButtonPermissionEntity> buttonList1 = new ArrayList<>();
                for (ButtonPermissionEntity buttonPermissionEntity : buttonList) {
                    for (Integer buttonId : buttonIds) {
                        if (buttonPermissionEntity.getId().equals(buttonId)) {
                            buttonList1.add(buttonPermissionEntity);
                        }
                    }
                }

                router.setButtonPermission(buttonList1);
            }






/*
            List<ButtonPermissionEntity> buttonList1 = buttonPermissionMapper.selectList(
                    new LambdaQueryWrapper<ButtonPermissionEntity>()
                            .eq(ButtonPermissionEntity::getParentId, menu.getId())
            );


            if (!CollectionUtils.isEmpty(buttonList1)) {
                List<ButtonPermissionEntity> buttonList2 = new ArrayList<>();
                // 查询角色都有哪些值
                UserEntity user = userMapper.selectById(userId);
                for (String roleIdStr : user.getUserRoles().split(",")) {
                    Integer roleId = Integer.parseInt(roleIdStr);
                    MenuResponse roleMenu = roleService.getRoleMenu(roleId);
                    for (ButtonPermissionEntity buttonPermissionEntity : buttonList1) {
                        List<Integer> menuIds = roleMenu.getMenuIds();
                        for (Integer menuId : menuIds) {
                            if (buttonPermissionEntity.getId().equals(menuId)) {
                                buttonList2.add(buttonPermissionEntity);
                            }
                        }
                    }
                }
                router.setButtonPermission(buttonList2);
            }*/




            //如果当前是菜单，需将按钮对应的路由加载出来，如：“角色授权”按钮对应的路由在“系统管理”下面
            if (menu.getType().intValue() == 1) {
                List<MenuEntity> hiddenMenuList = children.stream().filter(item -> !StringUtils.isEmpty(item.getComponent())).collect(Collectors.toList());
                for (MenuEntity hiddenMenu : hiddenMenuList) {
                    RouterResponse hiddenRouter = new RouterResponse();
                    //  hiddenRouter.setHidden(true);
                    hiddenRouter.setName(hiddenMenu.getMenuName());
                    //  hiddenRouter.setAlwaysShow(false);
                    hiddenRouter.setPath(getRouterPath(hiddenMenu));
                    hiddenRouter.setActiveMenu(menu.getActiveMenu());
                    hiddenRouter.setComponent(hiddenMenu.getComponent());
                    // hiddenRouter.setPermissions(hiddenMenu.getPermissions());


                   /* List<ButtonPermissionEntity> buttonList2 = buttonPermissionMapper.selectList(
                            new LambdaQueryWrapper<ButtonPermissionEntity>()
                                    .eq(ButtonPermissionEntity::getParentId, menu.getId())
                    );

                    if (!CollectionUtils.isEmpty(buttonList1)) {
                        List<ButtonPermissionEntity> buttonList3 = new ArrayList<>();
                        // 查询角色都有哪些值
                        UserEntity user = userMapper.selectById(userId);
                        for (String roleIdStr : user.getUserRoles().split(",")) {
                            Integer roleId = Integer.parseInt(roleIdStr);
                            MenuResponse roleMenu = roleService.getRoleMenu(roleId);
                            for (ButtonPermissionEntity buttonPermissionEntity : buttonList2) {
                                List<Integer> menuIds = roleMenu.getMenuIds();
                                for (Integer menuId : menuIds) {
                                    if (buttonPermissionEntity.getId().equals(menuId)) {
                                        buttonList3.add(buttonPermissionEntity);
                                    }
                                }
                            }
                        }
                        hiddenRouter.setButtonPermission(buttonList3);
                    }


*/


                    if (!CollectionUtils.isEmpty(buttonList)) {
                        List<ButtonPermissionEntity> buttonList3 = new ArrayList<>();
                        for (ButtonPermissionEntity buttonPermissionEntity : buttonList) {
                            for (Integer buttonId : buttonIds) {
                                if (buttonPermissionEntity.getId().equals(buttonId)) {
                                    buttonList3.add(buttonPermissionEntity);
                                }
                            }
                        }

                        hiddenRouter.setButtonPermission(buttonList3);
                    }





                    hiddenRouter.setMeta(new MetaResponse(hiddenMenu.getMenuName(), hiddenMenu.getIcon(), hiddenMenu.getSort(), hiddenMenu.getI18nKey(),
                            hiddenMenu.getActiveMenu(), menu.getLocalIcon(), hiddenMenu.getType() != null && hiddenMenu.getType() == 2,menu.getHref()));
                    routers.add(hiddenRouter);
                }
            } else {
                if (!CollectionUtils.isEmpty(children)) {
                    if (children.size() > 0) {
                        //  router.setAlwaysShow(true);
                    }
                    router.setChildren(buildRouters(children));
                }
            }
            routers.add(router);
        }
        return routers;
    }


    /**
     * 获取路由地址
     *
     * @param menu 菜单信息
     * @return 路由地址
     */
    public static String getRouterPath(MenuEntity menu) {
        String routerPath = menu.getWebPath();
        if (menu.getParentId().intValue() != 0) {
            routerPath = menu.getWebPath();
        }
        return routerPath;
    }


    public List<ButtonPermissionEntity> optimizeCode(Integer userId, Integer menuId) {
        // 查询父菜单下的所有按钮权限
        List<ButtonPermissionEntity> buttonList1 = buttonPermissionMapper.selectList(
                new LambdaQueryWrapper<ButtonPermissionEntity>()
                        .eq(ButtonPermissionEntity::getParentId, menuId)
        );

        if (CollectionUtils.isEmpty(buttonList1)) {
            return null;
        }

        // 查询用户信息
        UserEntity user = userMapper.selectById(userId);
        if (user == null || user.getUserRoles() == null) {
            return null;
        }

        // 存储所有角色对应的菜单 ID 集合
        Set<Integer> allMenuIds = new HashSet<>();
        // 拆分用户角色 ID 字符串
        String[] roleIdStrs = user.getUserRoles().split(",");
        for (String roleIdStr : roleIdStrs) {
            try {
                Integer roleId = Integer.parseInt(roleIdStr);
                // 获取角色对应的菜单响应
                MenuResponse roleMenu = roleService.getRoleMenu(roleId);
                if (roleMenu != null && CollectionUtils.isNotEmpty(roleMenu.getMenuIds())) {
                    allMenuIds.addAll(roleMenu.getMenuIds());
                }
            } catch (NumberFormatException e) {
                // 处理角色 ID 解析异常
                e.printStackTrace();
            }
        }

        // 过滤出符合条件的按钮权限
        List<ButtonPermissionEntity> buttonList2 = new ArrayList<>();
        for (ButtonPermissionEntity buttonPermissionEntity : buttonList1) {
            if (allMenuIds.contains(buttonPermissionEntity.getId())) {
                buttonList2.add(buttonPermissionEntity);
            }
        }

        return buttonList2;
    }


}
