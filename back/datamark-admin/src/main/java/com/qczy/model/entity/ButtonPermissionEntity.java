package com.qczy.model.entity;

import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;

import java.io.Serializable;

/**
 * @Author: hh
 * @Version: 1.0
 * @Date: 2025/3/6 10:05
 * @Description:
 */
@Data
@TableName("qczy_button_permission")
public class ButtonPermissionEntity implements Serializable {


    /**
     * id
     */
    private Integer id;

    /**
     * 菜单名称
     */
    private String menuName;

    /**
     * 父id
     */
    private Integer parentId;

    /**
     * 按钮名称
     */
    private String buttonName;

    /**
     * 权限
     */
    private String permission;

    /**
     * 排序
     */
    private Integer sort;


}
