package com.qczy.model.entity;

import com.baomidou.mybatisplus.annotation.*;
import com.qczy.common.base.BaseEntity;
import io.swagger.annotations.ApiModelProperty;
import lombok.Data;
import org.hibernate.validator.constraints.Length;

import javax.validation.constraints.NotNull;
import javax.validation.constraints.Size;
import java.io.Serializable;
import java.util.Date;
import java.util.List;

/**
 * @TableName qczy_role
 */
@Data
@TableName("qczy_role")
public class RoleEntity implements Serializable {


    //自增id
    @TableId(type = IdType.AUTO)
    private Integer Id;

    /**
     * 角色名称
     */
    @Size(max = 20, message = "编码长度不能超过20")
    @ApiModelProperty("角色名称")
    @Length(max = 20, message = "编码长度不能超过20")
    private String roleName;
    /**
     * 角色编码
     */
    @Size(max = 50, message = "编码长度不能超过50")
    @ApiModelProperty("角色编码")
    @Length(max = 50, message = "编码长度不能超过50")
    private String roleCode;
    /**
     * 角色描述
     */
    @Size(max = 50, message = "编码长度不能超过50")
    @ApiModelProperty("角色描述")
    @Length(max = 50, message = "编码长度不能超过50")
    private String roleDesc;
    /**
     * 菜单状态：0->启用、1->禁用
     */
    @ApiModelProperty("菜单状态：1->启用、2->禁用")
    private Integer status;

    /**
     * 多人标注权限：多人标注权限：1. 发起任务  2. 标注员 3.审核员
     */
    @ApiModelProperty("多人标注权限：1. 发起任务  2. 标注员 3.审核员")
    private String mulStatus;

    /**
     * 是否允许删除：1-> 不允许 、 2->允许
     */
    private Integer isAllowDeletion;

    @ApiModelProperty(value = "创建时间")
    @TableField(fill = FieldFill.INSERT)
    private Date createTime;


    @ApiModelProperty(value = "更新时间")
    @TableField(fill = FieldFill.INSERT_UPDATE)
    private Date updateTime;


    @ApiModelProperty(value = "是否删除(1:已删除，0:未删除)")
    private Integer IsDeleted;

    // 前端额外传入的字段（这个字段没用）
    @TableField(exist = false)
    private Integer index;


    /**
     * 多人标注权限：多人标注权限：1. 发起任务  2. 标注员 3.审核员
     */
    @ApiModelProperty("多人标注权限：1. 发起任务  2. 标注员 3.审核员")
    @TableField(exist = false)
    private List<Integer> mulStatusArray;
}
