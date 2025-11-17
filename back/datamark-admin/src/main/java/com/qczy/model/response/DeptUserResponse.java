package com.qczy.model.response;

import lombok.Data;

import java.util.List;

/**
 * @Author: hh
 * @Version: 1.0
 * @Date: 2025/2/26 14:02
 * @Description:
 */
@Data
public class DeptUserResponse {

    // 部门id
    private int deptId;
    // 部门名称
    private String deptName;
    // 排序
    private Integer sort;
    // 用户列表
    private List<UserListResponse> userList;

}


