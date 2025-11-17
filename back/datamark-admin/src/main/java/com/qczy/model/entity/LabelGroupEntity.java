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
 * @Date: 2024/8/19 14:39
 * @Description:
 */
@Data
@TableName("qczy_label_group")
public class LabelGroupEntity implements Serializable {

    /**
     * 标签组id
     */
    @ApiModelProperty(value = "标签组id")
    private Integer id;

    /**
     * 标签组名称
     */
    @ApiModelProperty(value = "标签组名称")
    private String labelGroupName;

    /**
     * 标签组英文
     */
    @ApiModelProperty(value = "标签组名称英文")
    private String englishLabelGroupName;

    /**
     * 标签组描述
     */
    @ApiModelProperty(value = "标签组描述")
    private String labelGroupDesc;

    @ApiModelProperty(value = "创建时间")
    @DateTimeFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss", timezone = "GMT+8")
    @TableField(fill = FieldFill.INSERT)
    private Date createTime;


    @ApiModelProperty(value = "更新时间")
    @DateTimeFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss", timezone = "GMT+8")
    @TableField(fill = FieldFill.INSERT_UPDATE)
    private Date updateTime;

    // 前端额外传入的字段（这个字段没用）
    @TableField(exist = false)
    private Integer index;

}
