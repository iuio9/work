package com.qczy.model.entity;

import com.baomidou.mybatisplus.annotation.FieldFill;
import com.baomidou.mybatisplus.annotation.TableField;
import com.baomidou.mybatisplus.annotation.TableName;
import com.fasterxml.jackson.annotation.JsonFormat;
import io.swagger.annotations.ApiModelProperty;
import lombok.Data;
import org.springframework.format.annotation.DateTimeFormat;

import java.io.Serializable;
import java.util.Date;

/**
 * @Author: hh
 * @Version: 1.0
 * @Date: 2024/8/7 18:40
 * @Description:
 */
@Data
@TableName("qczy_data_father")
public class DataFatherEntity implements Serializable {

    /** 自增id */
    @ApiModelProperty(value = "自增id")
    private Integer id;

    /** 数据集组id */
    @ApiModelProperty(value = "数据集组id")
    private String groupId;

    /** 数据集组名称 */
    @ApiModelProperty(value = "数据集组名称")
    private String groupName;

    /** 用户id */
    @ApiModelProperty(value = "用户id")
    private Integer userId;

    /** 创建时间 */
    @ApiModelProperty(value = "创建时间")
    @TableField(fill = FieldFill.INSERT)
    private Date createTime;

    /** 更新时间 */
    @ApiModelProperty(value = "更新时间")
    @TableField(fill = FieldFill.INSERT_UPDATE)
    private Date updateTime;


    /** 数据级（类型）id */
    @ApiModelProperty(value = "数据级（类型）id")
    private Integer dataTypeId;
/*
    *//** 是否为第三方数据  0：不是 、 1：是*//*
    private Integer isThirdParty;*/


    // 前端额外传入的字段（这个字段没用）
    @TableField(exist = false)
    private Integer index;
}
