package com.qczy.service.impl;

import cn.hutool.core.convert.Convert;
import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.core.metadata.IPage;
import com.baomidou.mybatisplus.core.toolkit.CollectionUtils;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import com.qczy.common.constant.SystemConstant;
import com.qczy.mapper.*;
import com.qczy.model.entity.*;
import com.qczy.model.request.RoleMenuRequest;
import com.qczy.model.response.MenuResponse;
import com.qczy.service.RoleService;
import com.qczy.service.UserService;
import com.qczy.utils.CurrentLoginUserUtils;
import org.apache.commons.lang3.StringUtils;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import org.springframework.util.ObjectUtils;

import java.util.*;
import java.util.stream.Collectors;


/**
 * @Author: hh
 * @Version: 1.0
 * @Date: 2024/7/26 10:02
 * @Description:
 */
@Service
public class RoleServiceImpl extends ServiceImpl<RoleMapper, RoleEntity> implements RoleService {

    @Autowired
    private RoleMapper roleMapper;

    @Autowired
    private RoleMenuMapper roleMenuMapper;

    @Autowired
    private UserMapper userMapper;

    @Autowired
    private MenuMapper menuMapper;

    @Autowired
    private CurrentLoginUserUtils currentLoginUserUtils;

    @Autowired
    private RoleButtonMapper roleButtonMapper;


    @Override
    public List<RoleEntity> getRoleSelect() {
        return roleMapper.selectList(
                new LambdaQueryWrapper<RoleEntity>()
                        .eq(RoleEntity::getStatus, SystemConstant.SYSTEM_NO_FREEZE)
                        .eq(RoleEntity::getIsDeleted, SystemConstant.SYSTEM_NO_DISABLE)
        );
    }

    /**
     * 查询【请填写功能名称】
     *
     * @param id 【请填写功能名称】主键
     * @return 【请填写功能名称】
     */
    @Override
    public RoleEntity selectRoleById(Integer id) {
        return roleMapper.selectRoleById(id);
    }


    /**
     * 查询【请填写功能名称】列表
     *
     * @param pageParam 分页信息，【请填写功能名称】
     * @param request   查询参数，【请填写功能名称】
     * @return 【请填写功能名称】
     */
    @Override
    public IPage<RoleEntity> selectRoleList(Page<RoleEntity> pageParam, RoleEntity request) {
        IPage<RoleEntity> dataPage = roleMapper.selectRoleList(pageParam, request);
        for (RoleEntity data : dataPage.getRecords()) {
            data.setMulStatusArray(convertStringToList(data.getMulStatus()));
        }
        return dataPage;
    }

    public List<Integer> convertStringToList(String input) {
        return Arrays.stream(input.split(","))
                .map(String::trim)
                .map(Integer::parseInt)
                .collect(Collectors.toList());
    }


    /**
     * 新增【请填写功能名称】
     *
     * @param role 【请填写功能名称】
     * @return 结果
     */
    @Override
    public int insertRole(RoleEntity role) {
        role.setCreateTime(new Date());
        role.setIsDeleted(SystemConstant.SYSTEM_NO_DISABLE);
        role.setIsAllowDeletion(SystemConstant.YES_DISABLE_DATA);
        role.setMulStatus(role.getMulStatusArray().stream().map(Object::toString).collect(Collectors.joining(",")));
        return roleMapper.insertRole(role);
    }

    /**
     * 修改【请填写功能名称】
     *
     * @param role 【请填写功能名称】
     * @return 结果
     */
    @Override
    public int updateRole(RoleEntity role) {
        role.setUpdateTime(new Date());
        return roleMapper.updateRole(role);
    }

    /**
     * 批量删除【请填写功能名称】
     *
     * @param ids 需要删除的【请填写功能名称】主键
     * @return 结果
     */
    @Override
    public int deleteRoleByIds(String ids) {
        return roleMapper.deleteRoleByIds(Convert.toStrArray(ids));
    }

    /**
     * 删除【请填写功能名称】信息
     *
     * @param id 【请填写功能名称】主键
     * @return 结果
     */
    @Override
    public int deleteRoleById(Integer id) {
        return roleMapper.deleteRoleById(id);
    }

    @Override
    public int MyDeleteAll(int[] ids) {
        for (int id : ids) {
            RoleEntity role = roleMapper.selectById(id);
            System.out.println(role.getRoleName());
            role.setIsDeleted(SystemConstant.SYSTEM_YES_DISABLE);
            roleMapper.updateById(role);
        }
        return 1;
    }

    @Override
    public int addRoleMenu(RoleMenuRequest request) {

        RoleEntity role = roleMapper.selectById(request.getRoleId());
        if (ObjectUtils.isEmpty(role)) {
            throw new RuntimeException("角色查询失败！");
        }

        // 判断角色是否以前绑定过，如果绑定过，先删除以前的数据
        List<RoleMenuEntity> roleMenuList = roleMenuMapper.selectList(new LambdaQueryWrapper<RoleMenuEntity>().eq(RoleMenuEntity::getRoleId, role.getId()));
        if (!CollectionUtils.isEmpty(roleMenuList)) {
            for (RoleMenuEntity roleMenu : roleMenuList) {
                roleMenuMapper.delete(
                        new LambdaQueryWrapper<RoleMenuEntity>()
                                .eq(RoleMenuEntity::getRoleId, roleMenu.getRoleId())
                                .eq(RoleMenuEntity::getMenuId, roleMenu.getMenuId())
                );
            }
        }


/*        // 进行数据绑定
        for (Integer menuId : request.getMenuIds()) {
            RoleMenuEntity roleMenu = new RoleMenuEntity();
            roleMenu.setRoleId(role.getId());
            roleMenu.setMenuId(menuId);
            roleMenuMapper.insert(roleMenu);

            if (menuId == 101) {
                roleMenu.setRoleId(role.getId());
                roleMenu.setMenuId(17);
                roleMenuMapper.insert(roleMenu);
            }

            if (menuId == 102) {
                roleMenu.setRoleId(role.getId());
                roleMenu.setMenuId(18);
                roleMenuMapper.insert(roleMenu);
            }

        }*/

        // 默认新增以下按钮
        roleMenuMapper.insert(new RoleMenuEntity(role.getId(), 1));
        roleMenuMapper.insert(new RoleMenuEntity(role.getId(), 17));
        roleMenuMapper.insert(new RoleMenuEntity(role.getId(), 18));
        roleMenuMapper.insert(new RoleMenuEntity(role.getId(), 60));
        roleMenuMapper.insert(new RoleMenuEntity(role.getId(), 61));
        roleMenuMapper.insert(new RoleMenuEntity(role.getId(), 65));

        // 进行新增
        for (Integer menuId : request.getMenuIds()) {
            RoleMenuEntity roleMenu = new RoleMenuEntity();
            roleMenu.setRoleId(role.getId());
            roleMenu.setMenuId(menuId);
            roleMenuMapper.insert(roleMenu);
        }

        return 1;
    }

    @Override
    public int addRoleButton(RoleMenuRequest request) {
        RoleEntity role = roleMapper.selectById(request.getRoleId());
        if (ObjectUtils.isEmpty(role)) {
            throw new RuntimeException("角色查询失败！");
        }

        // 判断角色是否以前绑定过，如果绑定过，先删除以前的数据
        List<RoleButtonEntity> roleButtonEntities = roleButtonMapper.selectList(new LambdaQueryWrapper<RoleButtonEntity>().eq(RoleButtonEntity::getRoleId, request.getRoleId()));
        if (!CollectionUtils.isEmpty(roleButtonEntities)) {
            // 进行删除
            for (RoleButtonEntity roleButtonEntity : roleButtonEntities) {
                roleButtonMapper.delete(
                        new LambdaQueryWrapper<RoleButtonEntity>()
                                .eq(RoleButtonEntity::getRoleId, roleButtonEntity.getRoleId())
                                .eq(RoleButtonEntity::getButtonId, roleButtonEntity.getButtonId())
                );
            }
        }

        // 进行新增
        for (Integer buttonId : request.getMenuIds()) {
            RoleButtonEntity roleButtonEntity = new RoleButtonEntity();
            roleButtonEntity.setRoleId(role.getId());
            roleButtonEntity.setButtonId(buttonId);
            roleButtonMapper.insert(roleButtonEntity);
        }
        return 1;
    }


    @Override
    public MenuResponse getRoleMenu(Integer roleId) {
        MenuResponse menuResponse = new MenuResponse();
        // 判断角色是否以前绑定过，如果绑定过，先删除以前的数据
        List<RoleMenuEntity> roleMenuList = roleMenuMapper.selectList(new LambdaQueryWrapper<RoleMenuEntity>().eq(RoleMenuEntity::getRoleId, roleId));
        if (!CollectionUtils.isEmpty(roleMenuList)) {

            List<Integer> menuIds = roleMenuList.stream()
                    .filter(roleMenuEntity -> !(roleMenuEntity.getMenuId() == 17) && !(roleMenuEntity.getMenuId() == 18))
                    .map(RoleMenuEntity::getMenuId)
                    .collect(Collectors.toList());
            menuResponse.setMenuIds(menuIds);
            // menuResponse.setMenuIds(roleMenuList.stream().map(RoleMenuEntity::getMenuId).collect(Collectors.toList()));
        }


        return menuResponse;
    }

    @Override
    public MenuResponse getRoleButton(Integer roleId) {
        MenuResponse menuResponse = new MenuResponse();
        List<RoleButtonEntity> roleButtonList = roleButtonMapper.selectList(new LambdaQueryWrapper<RoleButtonEntity>().eq(RoleButtonEntity::getRoleId, roleId));
        if (!CollectionUtils.isEmpty(roleButtonList)) {
            menuResponse.setMenuIds(
                    roleButtonList
                            .stream()
                            .map(RoleButtonEntity::getButtonId)
                            .collect(Collectors.toList())
            );
        }
        return menuResponse;
    }

    @Override
    public int getRoleAndUserCount(int[] ids) {
        return userMapper.selectCount(
                new LambdaQueryWrapper<UserEntity>()
                        .like(UserEntity::getUserRoles, StringUtils.join(ids, ','))
                        .eq(UserEntity::getIsDeleted, SystemConstant.SYSTEM_NO_DISABLE)
        );
    }

    @Override
    public int getByRoleNameCount(String roleName) {
        return roleMapper.selectCount(
                new LambdaQueryWrapper<RoleEntity>()
                        .eq(RoleEntity::getRoleName, roleName)
                        .eq(RoleEntity::getIsDeleted, SystemConstant.SYSTEM_NO_DISABLE));
    }

    @Override
    public int getByRoleNameCount(Integer roleId, String roleName) {
        return roleMapper.selectCount(
                new LambdaQueryWrapper<RoleEntity>()
                        .ne(RoleEntity::getId, roleId)
                        .eq(RoleEntity::getRoleName, roleName)
                        .eq(RoleEntity::getIsDeleted, SystemConstant.SYSTEM_NO_DISABLE));
    }

    @Override
    public Set<Integer> isManyTask() {
        // 获取当前登录用户
        Integer userId = currentLoginUserUtils.getCurrentLoginUserId();
        UserEntity user = userMapper.selectById(userId);
        Set<Integer> list = new HashSet<>();
        for (String roleStr : user.getUserRoles().split(",")) {
            Integer roleId = Integer.parseInt(roleStr);
            RoleEntity role = roleMapper.selectById(roleId);
            for (String number : role.getMulStatus().split(",")) {
                list.add(Integer.parseInt(number));
            }
        }
        return list;
    }
}
