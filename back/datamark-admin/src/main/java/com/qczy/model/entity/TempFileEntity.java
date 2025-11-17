package com.qczy.model.entity;


import com.baomidou.mybatisplus.annotation.FieldFill;
import com.baomidou.mybatisplus.annotation.TableField;
import com.baomidou.mybatisplus.annotation.TableName;
import io.swagger.annotations.ApiModelProperty;
import lombok.Data;

import java.io.Serializable;
import java.util.Date;

/**
 * @Author: hh
 * @Version: 1.0
 * @Date: 2024/8/7 19:04
 * @Description:
 */
@Data
@TableName("qczy_temp_file")
public class TempFileEntity implements Serializable {

    /** 自增id */
    @ApiModelProperty(value = "自增id")
    private Integer id;

    /** 文件名称 */
    @ApiModelProperty(value = "文件名称")
    private String fdName;

    /** 文件类型：(jpg、png、txt...) */
    @ApiModelProperty(value = "文件类型")
    private String fdType;

    /** 文件后缀 */
    @ApiModelProperty(value = "文件后缀")
    private String fdSuffix;

    /** 文件临时路径 */
    @ApiModelProperty(value = "文件临时路径")
    private String fdTempPath;

    /** 文件访问路径 */
    @ApiModelProperty(value = "文件访问路径")
    private String fdAccessPath;

    /** 文件大小 */
    @ApiModelProperty(value = "文件大小")
    private String fdSize;

    /** 创建时间 */
    @ApiModelProperty(value = "创建时间")
    @TableField(fill = FieldFill.INSERT)
    private Date createTime;

    /** 更新时间 */
    @ApiModelProperty(value = "更新时间")
    @TableField(fill = FieldFill.INSERT_UPDATE)
    private Date updateTime;


    /** 宽 */
    @ApiModelProperty(value = "宽")
    private Integer width;

    /** 高 */
    @ApiModelProperty(value = "高")
    private Integer height;


    // 前端额外传入的字段（这个字段没用）
    @TableField(exist = false)
    private Integer index;
}
