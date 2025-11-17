package com.qczy.model.entity;

import com.baomidou.mybatisplus.annotation.FieldFill;
import com.baomidou.mybatisplus.annotation.TableField;
import com.baomidou.mybatisplus.annotation.TableName;
import com.fasterxml.jackson.annotation.JsonFormat;
import io.swagger.annotations.ApiModelProperty;
import lombok.Data;

import java.io.Serializable;
import java.util.Date;

/**
 * @Author: hh
 * @Version: 1.0
 * @Date: 2024/8/9 12:00
 * @Description:
 */
@Data
@TableName("qczy_data_import_log")
public class DataImportLogEntity implements Serializable {

    /** 自增id */
    @ApiModelProperty(value = "自增id")
    private Integer id;

    /** 数据集id */
    @ApiModelProperty(value = "数据集id")
    private String sonId;

    /** 文件大小 */
    @ApiModelProperty(value = "文件大小")
    private String fileSize;

    /** 文件id */
    @ApiModelProperty(value = "文件id")
    private String fileIds;

    /** 用户id */
    @ApiModelProperty(value = "用户id")
    private Integer userId;

    /** 导入开始时间 */
    @ApiModelProperty(value = "导入开始时间")
    private Date importStartTime;

    /** 导入结束时间 */
    @ApiModelProperty(value = "导入结束时间")
    private Date importEndTime;

    /** 导入状态 */
    @ApiModelProperty(value = "导入状态")
    private Integer status;


    /** 创建时间 */
    @ApiModelProperty(value = "创建时间")
    @TableField(fill = FieldFill.INSERT)
    private Date createTime;


    // 前端额外传入的字段（这个字段没用）
    @TableField(exist = false)
    private Integer index;
}
