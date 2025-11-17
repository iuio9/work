package com.qczy.model.response;

import com.qczy.model.entity.ButtonPermissionEntity;
import lombok.Data;

import java.util.List;

/**
 * @Author: hh
 * @Version: 1.0
 * @Date: 2025/5/29 11:20
 * @Description:
 */
@Data
public class TreeMenuButtonResponse {


    // 菜单名称
    private String menuName;


    private List<ButtonPermissionEntity> buttonPermissions;





}
