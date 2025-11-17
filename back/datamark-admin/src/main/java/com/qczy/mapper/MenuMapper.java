package com.qczy.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.qczy.model.entity.MenuEntity;

import java.util.List;

/**
 * @Author: hh
 * @Version: 1.0
 * @Date: 2024/7/24 10:19
 * @Description:
 */
public interface MenuMapper extends BaseMapper<MenuEntity> {

    List<MenuEntity> getUserMenuList(Integer id);


    /**
     * 查询单个菜单
     *
     * @param id 主键
     * @return MenuEntity
     */
    public MenuEntity selectMenuById(Integer id);

    /**
     * 查询列表
     *
     * @param menu
     * @return List<MenuEntity>
     */
    public List<MenuEntity> selectMenuList(MenuEntity menu);

    /**
     * 新增
     *
     * @param menu
     * @return 结果
     */
    public int insertMenu(MenuEntity menu);

    /**
     * 修改
     *
     * @param menu
     * @return 结果
     */
    public int updateMenu(MenuEntity menu);

    /**
     * 删除
     *
     * @param id 主键
     * @return 结果
     */
    public int deleteMenuById(Integer id);

    /**
     * 批量删除
     *
     * @param ids 需要删除的数据主键集合
     * @return 结果
     */
    public int deleteMenuByIds(String[] ids);
}
