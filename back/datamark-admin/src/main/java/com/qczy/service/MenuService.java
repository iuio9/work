package com.qczy.service;

import com.qczy.model.entity.MenuEntity;
import com.qczy.model.response.RouterResponse;
import com.qczy.model.response.TreeMenuButtonResponse;
import com.qczy.model.response.TreeMenuResponse;

import java.util.List;

/**
 * @Author: hh
 * @Version: 1.0
 * @Date: 2024/7/26 10:16
 * @Description:
 */
public interface MenuService {
    List<RouterResponse> getUserMenuList(String userRoles);


    /**
     * 查询666
     *
     * @param id 主键
     * @return
     */
    public MenuEntity selectMenuById(Integer id);

    /**
     * 查询666列表
     *
     * @param menu
     * @return 集合
     */
    public List<MenuEntity> selectMenuList(MenuEntity menu);

    /**
     * 新增666
     *
     * @param menu
     * @return 结果
     */
    public int insertMenu(MenuEntity menu);

    /**
     * 修改666
     *
     * @param menu
     * @return 结果
     */
    public int updateMenu(MenuEntity menu);

    /**
     * 批量删除
     *
     * @param ids 需要删除的主键集合
     * @return 结果
     */
    public int deleteMenuByIds(String ids);

    /**
     * 删除
     *
     * @param id 666主键
     * @return 结果
     */
    public int deleteMenuById(Integer id);

    /**
     * 构建树形菜单
     */
    List<TreeMenuResponse> getTreeMenuList();

    /**
     * 构建树形按钮菜单
     */
    List<TreeMenuButtonResponse>  getMenuButtonTree();

}
