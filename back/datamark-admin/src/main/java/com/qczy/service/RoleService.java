package com.qczy.service;

import com.baomidou.mybatisplus.core.metadata.IPage;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.baomidou.mybatisplus.extension.service.IService;
import com.qczy.model.entity.RoleEntity;
import com.qczy.model.entity.UserEntity;
import com.qczy.model.request.RoleMenuRequest;
import com.qczy.model.request.UserRequest;
import com.qczy.model.response.MenuResponse;

import java.util.List;
import java.util.Set;

/**
 * @Author: hh
 * @Version: 1.0
 * @Date: 2024/7/26 10:02
 * @Description:
 */
public interface RoleService extends IService<RoleEntity> {


    List<RoleEntity> getRoleSelect();


    /**
     * 查询【请填写功能名称】
     *
     * @param id 【请填写功能名称】主键
     * @return 【请填写功能名称】
     */
    public RoleEntity selectRoleById(Integer id);

    /**
     * 查询【请填写功能名称】列表
     *
     * @param pageParam 分页信息，【请填写功能名称】
     * @param role      查询参数，【请填写功能名称】
     * @return 【请填写功能名称】集合
     */
    IPage<RoleEntity> selectRoleList(Page<RoleEntity> pageParam, RoleEntity role);

    /**
     * 新增【请填写功能名称】
     *
     * @param role 【请填写功能名称】
     * @return 结果
     */
    public int insertRole(RoleEntity role);

    /**
     * 修改【请填写功能名称】
     *
     * @param role 【请填写功能名称】
     * @return 结果
     */
    public int updateRole(RoleEntity role);

    /**
     * 批量删除【请填写功能名称】
     *
     * @param ids 需要删除的【请填写功能名称】主键集合
     * @return 结果
     */
    public int deleteRoleByIds(String ids);

    /**
     * 删除【请填写功能名称】信息
     *
     * @param id 【请填写功能名称】主键
     * @return 结果
     */
    public int deleteRoleById(Integer id);


    int MyDeleteAll(int[] ids);

    // 新增菜单权限
    int addRoleMenu(RoleMenuRequest request);
    // 新增按钮权限
    int addRoleButton(RoleMenuRequest request);

    // 回显菜单
    MenuResponse getRoleMenu(Integer roleId);
    // 回显按钮
    MenuResponse getRoleButton(Integer roleId);

    int getRoleAndUserCount(int[] ids);


    /**
     * 根据角色名查询数量
     */
    int getByRoleNameCount(String roleName);

    /**
     * 根据id 角色名查询数量
     */
    int getByRoleNameCount(Integer roleId, String roleName);

    Set<Integer> isManyTask();


}
