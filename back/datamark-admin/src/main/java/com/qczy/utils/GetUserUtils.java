package com.qczy.utils;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.qczy.common.constant.SystemConstant;
import com.qczy.model.entity.UserEntity;
import com.qczy.service.UserService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

/**
 * @Author: hh
 * @Version: 1.0
 * @Date: 2024/8/5 11:35
 * @Description:
 */
@Component
public class GetUserUtils {


    @Autowired
    private UserService userService;


    public UserEntity getUser(String username) {
        return userService.getOne(
                new LambdaQueryWrapper<UserEntity>()
                        .eq(UserEntity::getUserName, username)
                        .eq(UserEntity::getIsDeleted, SystemConstant.SYSTEM_NO_DISABLE)
        );

    }


}
