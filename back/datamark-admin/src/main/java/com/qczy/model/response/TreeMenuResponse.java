package com.qczy.model.response;

import com.qczy.model.entity.ButtonPermissionEntity;
import lombok.Data;

import java.util.List;

/**
 * @Author: hh
 * @Version: 1.0
 * @Date: 2024/8/1 16:31
 * @Description:
 */
@Data
public class TreeMenuResponse {
    private Integer id;

    private String label;

    private Integer pId;

    private List<TreeMenuResponse> children;

    private List<ButtonPermissionEntity> buttonPermission;

    //private String permissions;

    private Integer hideInMenu;

}
