package com.qczy.model.entity;

import com.baomidou.mybatisplus.annotation.FieldFill;
import com.baomidou.mybatisplus.annotation.TableField;
import com.baomidou.mybatisplus.annotation.TableName;
import com.fasterxml.jackson.annotation.JsonFormat;
import com.qczy.common.excel.ExcelImport;
import io.swagger.annotations.ApiModelProperty;
import lombok.Data;
import org.springframework.format.annotation.DateTimeFormat;

import javax.validation.constraints.NotBlank;
import java.util.Date;

/**
 * @Author: hh
 * @Version: 1.0
 * @Date: 2024/8/19 15:16
 * @Description:
 */
@Data
@TableName("qczy_label")
public class LabelEntity {

    /** 标签id */
    @ApiModelProperty(value = "标签id")
    private Integer id;


    /** 标签id */
    @ApiModelProperty(value = "唯一id")
    @ExcelImport("标签编码")
    private String onlyId;

    /** 标签组id */
    @ApiModelProperty(value = "标签组id")
    private Integer labelGroupId;

    /** 标签名称 */
    @ApiModelProperty(value = "标签名称")
    @NotBlank(message = "标签名称不能为空")
    @ExcelImport("标签名")
    private String labelName;

    /** 英文标签名称 */
    @ApiModelProperty(value = "英文标签名称")
    @ExcelImport("标签编码")
    private String englishLabelName;

    /** 标签排序 */
    @ApiModelProperty(value = "标签排序")
    private Integer labelSort;

    /** 标签颜色 */
    @ApiModelProperty(value = "标签颜色")
    @NotBlank(message = "标签颜色不能为空")
    @ExcelImport("颜色")
    private String labelColor;

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


    @TableField(exist = false)
    private String sonId;
}
