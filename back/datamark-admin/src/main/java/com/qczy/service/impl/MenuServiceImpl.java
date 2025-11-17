package com.qczy.service.impl;


import cn.hutool.core.convert.Convert;
import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.core.toolkit.CollectionUtils;
import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import com.qczy.common.constant.SystemConstant;
import com.qczy.mapper.ButtonPermissionMapper;
import com.qczy.mapper.MenuMapper;
import com.qczy.model.entity.ButtonPermissionEntity;
import com.qczy.model.entity.MenuEntity;
import com.qczy.model.response.RouterResponse;
import com.qczy.model.response.TreeMenuButtonResponse;
import com.qczy.model.response.TreeMenuResponse;
import com.qczy.service.MenuService;
import com.qczy.utils.RouterHelperUtils;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.*;
import java.util.stream.Collectors;

/**
 * @Author: hh
 * @Version: 1.0
 * @Date: 2024/7/26 10:02
 * @Description:
 */
@Service
public class MenuServiceImpl extends ServiceImpl<MenuMapper, MenuEntity> implements MenuService {

    @Autowired
    MenuMapper menuMapper;

    @Autowired
    private ButtonPermissionMapper buttonPermissionMapper;

    @Autowired
    private RouterHelperUtils routerHelperUtils;


    @Override
    public List<RouterResponse> getUserMenuList(String userRoles) {

        List<MenuEntity> menuEntities = new ArrayList<>();
        List<MenuEntity> menuEntityList = new ArrayList<>();
        List<MenuEntity> newMenuEntityList = new ArrayList<>();
        //先根据角色查询所拥有的权限信息
        if (userRoles.split(",").length > 1) {
            for (String roleId : userRoles.split(",")) {
                menuEntityList = menuMapper.getUserMenuList(Integer.parseInt(roleId));
                newMenuEntityList.addAll(menuEntityList);
            }
        } else {
            newMenuEntityList = menuMapper.getUserMenuList(Integer.parseInt(userRoles));
        }


        if (!CollectionUtils.isEmpty(newMenuEntityList)) {
            //  newMenuEntityList.addAll(menuMapper.selectList(new LambdaQueryWrapper<MenuEntity>().eq(MenuEntity::getParentId, 0)));
            //变成树形结构
            newMenuEntityList = newMenuEntityList.stream().distinct().collect(Collectors.toList());
            List<MenuEntity> userMenuSelect = getUserMenuSelect(newMenuEntityList);
            List<MenuEntity> menuEntities1 = userMenuSelect.stream().distinct().collect(Collectors.toList());

            //转换成前端需要的格式并返回
            return routerHelperUtils.buildRouters(menuEntities1);
        }
        //变成树形结构
        List<MenuEntity> userMenuSelect = getUserMenuSelect(menuEntities);
        System.out.println(userMenuSelect);
        //转换成前端需要的格式并返回
        return routerHelperUtils.buildRouters(userMenuSelect);
    }

    public List<MenuEntity> getUserMenuSelect(List<MenuEntity> menuEntityList) {
        List<MenuEntity> menuEntities = new ArrayList<>();
        //判断集合是否为空，如果为空，直接返回
        if (CollectionUtils.isEmpty(menuEntityList)) {
            return null;
        }
        //根节点为0,找下面的子节点
        for (MenuEntity menuEntity : menuEntityList) {
            if (menuEntity.getParentId() == 0) {
                menuEntities.add(children(menuEntity, menuEntityList));
            }
        }
        return menuEntities;
    }

    //执行递归操作
    private static MenuEntity children(MenuEntity menu, List<MenuEntity> menuList) {
        menu.setChildren(new ArrayList<MenuEntity>());
        for (MenuEntity menus : menuList) {
            //拿着父节点的id去查询所有的子节点       id == parentId
            if (menu.getId().equals(menus.getParentId())) {
                if (menu.getChildren() == null) {
                    menu.setChildren(new ArrayList<MenuEntity>());
                }
                //以此循环执行，直到找不到子节点结束...
                menu.getChildren().add(children(menus, menuList));
            }
        }
        return menu;
    }


    /**
     * 查询666
     *
     * @param id 666主键
     * @return
     */
    @Override
    public MenuEntity selectMenuById(Integer id) {
        return menuMapper.selectMenuById(id);
    }

    /**
     * 查询列表
     *
     * @param menu
     * @return
     */
    @Override
    public List<MenuEntity> selectMenuList(MenuEntity menu) {
        return menuMapper.selectMenuList(menu);
    }

    /**
     * 新增
     *
     * @param menu
     * @return 结果
     */
    @Override
    public int insertMenu(MenuEntity menu) {
        menu.setCreateTime(new Date());
        return menuMapper.insertMenu(menu);
    }

    /**
     * 修改
     *
     * @param menu
     * @return 结果
     */
    @Override
    public int updateMenu(MenuEntity menu) {
        menu.setUpdateTime(new Date());
        return menuMapper.updateMenu(menu);
    }

    /**
     * 批量删除
     *
     * @param ids 需要删除的主键
     * @return 结果
     */
    @Override
    public int deleteMenuByIds(String ids) {
        return menuMapper.deleteMenuByIds(Convert.toStrArray(ids));
    }

    /**
     * 删除信息
     *
     * @param id 主键
     * @return 结果
     */
    @Override
    public int deleteMenuById(Integer id) {
        return menuMapper.deleteMenuById(id);
    }

    @Override
    public List<TreeMenuResponse> getTreeMenuList() {

        List<TreeMenuResponse> treeMenuList = new ArrayList<>();
               /* .eq(MenuEntity::getType, SystemConstant.TYPE_DIR)
                        .eq(MenuEntity::getIsDeleted, SystemConstant.SYSTEM_NO_DISABLE)
                        .or()
                        .eq(MenuEntity::getType, SystemConstant.TYPE_MENU)
                        .eq(MenuEntity::getIsDeleted, SystemConstant.SYSTEM_NO_DISABLE)*/

        List<MenuEntity> menuList = menuMapper.selectList(
                null

        );

        //判断集合是否为空，如果为空，直接返回
        if (CollectionUtils.isEmpty(menuList)) {
            return null;
        }
        List<TreeMenuResponse> list = new ArrayList<>();
        for (MenuEntity menuEntity : menuList) {
            TreeMenuResponse request = new TreeMenuResponse();
            // 不显示登录按钮
            if (menuEntity.getMenuName().equals("login") ||
                    menuEntity.getMenuName().equals("data-ano_groupmanage")
                    || menuEntity.getMenuName().equals("data-ano_mulanotask")
                    || menuEntity.getMenuName().equals("data-thirdparty_modeloperate")
                    ) {
                continue;
            }
            request.setId(menuEntity.getId());
            request.setPId(menuEntity.getParentId());
            request.setLabel(menuEntity.getMenuName());
            request.setHideInMenu(menuEntity.getHideInMenu());
            // request.setPermissions(menuEntity.getPermissions());
       /*     List<ButtonPermissionEntity> buttonList = buttonPermissionMapper.selectList(
                    new LambdaQueryWrapper<ButtonPermissionEntity>()
                            .eq(ButtonPermissionEntity::getParentId, menuEntity.getId())
            );

            if (!CollectionUtils.isEmpty(buttonList)) {
                request.setButtonPermission(buttonList);
            }*/

            list.add(request);
        }
        // 判断是否是根节点，根节点递归寻找字节点
        for (TreeMenuResponse request : list) {
            if (request.getPId() == 0) {
                treeMenuList.add(children1(request, list));
            }
        }
        return treeMenuList;
    }

    @Override
    public List<TreeMenuButtonResponse> getMenuButtonTree() {
        List<ButtonPermissionEntity> list = buttonPermissionMapper.selectList(null);
        if (CollectionUtils.isEmpty(list)) {
            return null;
        }
        // 根据数据集 parent_id 分组
        // 使用TreeMap对parentId进行自然排序（升序）
        Map<Integer, List<ButtonPermissionEntity>> map = list.stream()
                .collect(Collectors.groupingBy(
                        ButtonPermissionEntity::getParentId,
                        TreeMap::new, // 指定使用TreeMap作为结果Map
                        Collectors.toList()
                ));

        List<TreeMenuButtonResponse> treeMenuList = new ArrayList<>();

        for (Map.Entry<Integer, List<ButtonPermissionEntity>> entity : map.entrySet()) {
            // key 为 菜单id
            TreeMenuButtonResponse buttonResponse = new TreeMenuButtonResponse();
            buttonResponse.setMenuName(entity.getValue().get(0).getMenuName());
            buttonResponse.setButtonPermissions(entity.getValue());
            treeMenuList.add(buttonResponse);
        }
        return treeMenuList;
    }


    //执行递归操作
    private static TreeMenuResponse children1(TreeMenuResponse menu, List<TreeMenuResponse> menuList) {
        menu.setChildren(new ArrayList<TreeMenuResponse>());
        for (TreeMenuResponse menus : menuList) {
            //拿着父节点的id去查询所有的子节点       id == parentId
            if (menu.getId().equals(menus.getPId())) {
                if (menu.getChildren() == null) {
                    menu.setChildren(new ArrayList<TreeMenuResponse>());
                }
                //以此循环执行，直到找不到子节点结束...
                menu.getChildren().add(children1(menus, menuList));
            }
        }
        return menu;
    }


}
