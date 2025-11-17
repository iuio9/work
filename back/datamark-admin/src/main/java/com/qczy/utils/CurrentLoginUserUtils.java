package com.qczy.utils;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.mysql.cj.x.protobuf.MysqlxCrud;
import com.qczy.common.constant.SystemConstant;
import com.qczy.mapper.DeptMapper;
import com.qczy.mapper.RoleMapper;
import com.qczy.mapper.UserMapper;
import com.qczy.model.entity.DeptEntity;
import com.qczy.model.entity.RoleEntity;
import com.qczy.model.entity.UserEntity;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.security.core.Authentication;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.stereotype.Component;
import org.springframework.util.CollectionUtils;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

/**
 * @Author: hh
 * @Version: 1.0
 * @Date: 2024/8/8 15:42
 * @Description:
 */
@Component
public class CurrentLoginUserUtils {
    private static final Logger log = LoggerFactory.getLogger(CurrentLoginUserUtils.class);

    @Autowired
    private RoleMapper roleMapper;

    /**
     * 获取当前用户登录的id
     */

    public Integer getCurrentLoginUserId() {
        Authentication authentication = SecurityContextHolder.getContext().getAuthentication();
        UserEntity user = BeanUtils.getBean(UserMapper.class).selectOne(
                new LambdaQueryWrapper<UserEntity>()
                        .eq(UserEntity::getUserName, authentication.getName())
                        .eq(UserEntity::getIsDeleted, SystemConstant.SYSTEM_NO_DISABLE)
        );
        if (user == null) {
            throw new RuntimeException("后端异常，获取不到当前用户！");
        }
        return user.getId();
    }


    /**
     * 获取当前登录用户的部门id
     */

    public String getCurrentLoginDeptIds() {
        Authentication authentication = SecurityContextHolder.getContext().getAuthentication();
        UserEntity user = BeanUtils.getBean(UserMapper.class).selectOne(
                new LambdaQueryWrapper<UserEntity>()
                        .eq(UserEntity::getUserName, authentication.getName())
                        .eq(UserEntity::getIsDeleted, SystemConstant.SYSTEM_NO_DISABLE)
        );
        if (user == null) {
            throw new RuntimeException("后端异常，获取不到当前用户！");
        }

        if (user.getId() == 1) { //管理员
            List<DeptEntity> list = BeanUtils.getBean(DeptMapper.class).selectList(null);
            if (CollectionUtils.isEmpty(list)) {
                return "'1'";
            } else {
                List<Integer> ids = list.stream().map(DeptEntity::getId).collect(Collectors.toList());
                StringBuilder sb = new StringBuilder();
                for (Integer id : ids) {
                    sb.append("'").append(id).append("',");
                }
                return sb.deleteCharAt(sb.length() - 1).toString();
            }


        }
        return user.getDeptIds();
    }


    // 获取当前（领导 + 管理员）和申请者人员的id权限
    public boolean getCurrentLoginUserModelIds() {
        Authentication authentication = SecurityContextHolder.getContext().getAuthentication();
        UserEntity user = BeanUtils.getBean(UserMapper.class).selectOne(
                new LambdaQueryWrapper<UserEntity>()
                        .eq(UserEntity::getUserName, authentication.getName())
                        .eq(UserEntity::getIsDeleted, SystemConstant.SYSTEM_NO_DISABLE)
        );
        if (user == null) {
            throw new RuntimeException("后端异常，获取不到当前用户！");
        }
        // 获取角色
        for (String roleIdStr : user.getUserRoles().split(",")) {
            Integer roleId = Integer.parseInt(roleIdStr);
            RoleEntity roleEntity = roleMapper.selectById(roleId);
            if (roleEntity == null || StringUtils.isEmpty(roleEntity.getRoleName())) {
                continue;
            }
            if (roleEntity.getRoleName().equals("管理员") || roleEntity.getRoleName().equals("模型评估-领导")) {
                return true;
            }
        }
        return false;
    }


}
