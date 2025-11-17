package com.qczy.model.response;

import lombok.Data;

import java.util.List;

/**
 * @Author: hh
 * @Version: 1.0
 * @Date: 2024/8/2 11:24
 * @Description:
 */
@Data
public class MenuResponse {

    // 菜单权限
    private List<Integer> menuIds;

}
