package com.qczy.model.entity;

import com.baomidou.mybatisplus.annotation.*;
import com.fasterxml.jackson.annotation.JsonFormat;
import com.qczy.common.base.BaseEntity;
import io.swagger.annotations.ApiModelProperty;
import lombok.Data;
import lombok.NonNull;
import org.springframework.format.annotation.DateTimeFormat;

import javax.validation.constraints.NotBlank;
import java.io.Serializable;
import java.util.Date;

/**
 * @Author: hh
 * @Version: 1.0
 * @Date: 2024/7/15 14:00
 * @Description:
 */
@Data
@TableName("qczy_user")
public class UserEntity implements Serializable {


    //自增id
    @TableId(type = IdType.AUTO)
    private Integer Id;

    @ApiModelProperty(value = "用户名")
    @TableField("user_name")
    @NotBlank(message = "用户名不能为空")
    private String  userName;

    @ApiModelProperty(value = "密码")
    @TableField("password")
    @NotBlank(message = "密码不能为空")
    private String  Password;

    @ApiModelProperty(value = "昵称")
    @NotBlank(message = "昵称不能为空")
    @TableField("nick_name")
    private String  nickName;

    @ApiModelProperty(value = "性别：1->男、2->女")
    @TableField("user_gender")
    private Integer userGender;

    @ApiModelProperty(value = "邮箱")
    @TableField("user_email")
    private String  userEmail;

    @ApiModelProperty(value = "手机号")
    @TableField("user_phone")
    private String userPhone;

    @ApiModelProperty(value = "状态：1->正常、2->禁用")
    @TableField("status")
    private Integer status;

    @ApiModelProperty(value = "角色id")
    @TableField("user_roles")
    private String userRoles;

    /** 部门id */
    @TableField("dept_ids")
    private String deptIds;

    /** 是否允许删除：1-> 不允许 、 2->允许 */
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

    //额外传入的字段
    @TableField(exist = false)
    private Integer isHide;




}
