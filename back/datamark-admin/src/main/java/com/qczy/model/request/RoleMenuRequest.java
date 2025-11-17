package com.qczy.model.request;

import lombok.Data;

import java.util.List;

/**
 * @Author: hh
 * @Version: 1.0
 * @Date: 2024/8/2 10:34
 * @Description:
 */
@Data
public class RoleMenuRequest {

    // 用户id
    private Integer roleId;

    // 菜单权限
    private List<Integer> menuIds;




}
