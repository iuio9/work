package com.qczy.model.entity;

import com.baomidou.mybatisplus.annotation.TableField;
import com.baomidou.mybatisplus.annotation.TableName;
import com.qczy.common.base.BaseEntity;
import io.swagger.annotations.ApiModelProperty;
import lombok.Data;

import javax.validation.constraints.NotNull;
import java.io.Serializable;

/**
* 
* @TableName qczy_role_menu
*/
@Data
@TableName("qczy_role_menu")
public class RoleMenuEntity implements Serializable {

    /**
    * 角色id
    */
    @NotNull(message="[角色id]不能为空")
    @ApiModelProperty("角色id")
    private Integer roleId;
    /**
    * 菜单id
    */
    @NotNull(message="[菜单id]不能为空")
    @ApiModelProperty("菜单id")
    private Integer menuId;


    // 前端额外传入的字段（这个字段没用）
    @TableField(exist = false)
    private Integer index;

    public RoleMenuEntity(Integer roleId, Integer menuId) {
        this.roleId = roleId;
        this.menuId = menuId;
    }

    public RoleMenuEntity() {
    }
}
