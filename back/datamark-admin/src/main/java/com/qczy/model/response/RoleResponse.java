package com.qczy.model.response;

import lombok.Data;


/**
 * @Author: hh
 * @Version: 1.0
 * @Date: 2024/7/26 11:06
 * @Description:
 */
@Data
public class RoleResponse {

    /**
     * 角色名称
     */
    private String roleName;
    /**
     * 角色编码
     */
    private String roleCode;
    /**
     * 角色描述
     */
    private String roleDesc;
}

