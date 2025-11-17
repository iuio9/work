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
 * @Date: 2024/8/7 22:38
 * @Description:
 */
@Data
@TableName("qczy_file")
public class FileEntity implements Serializable {

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
    @ApiModelProperty(value = "文件正式路径")
    private String fdPath;

    /** 文件访问路径 */
    @ApiModelProperty(value = "文件访问路径")
    private String fdAccessPath;

    /** 前端文件访问路径 **/
    @ApiModelProperty(value = "前端文件访问路径")
    private String httpFilePath;

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

    /** 宽 */
    @ApiModelProperty(value = "mark宽")
    private Integer operateWidth;

    /** 高 */
    @ApiModelProperty(value = "mark高")
    private Integer operateHeight;

    /** 文件状态 */
    @ApiModelProperty(value = "文件状态->0:原始图片、1:训练结果图片")
    private Integer fileStatus;

    /** 文件状态 */
    @ApiModelProperty(value = "任务编号 -> 哪次任务生成的文件")
    private String taskId;






    // 前端额外传入的字段（这个字段没用）
    @TableField(exist = false)
    private Integer index;

}
