package com.qczy.model.entity;

import com.baomidou.mybatisplus.annotation.FieldFill;
import com.baomidou.mybatisplus.annotation.TableField;
import com.baomidou.mybatisplus.annotation.TableName;
import io.swagger.annotations.ApiModelProperty;
import lombok.Data;

import java.util.Date;

/**
 * @Author: hh
 * @Version: 1.0
 * @Date: 2024/8/7 21:43
 * @Description:
 */
@Data
@TableName("qczy_data_son")
public class DataSonEntity {

    /** 自增id */
    @ApiModelProperty(value = "自增id")
    private Integer id;

    /** 父id */
    @ApiModelProperty(value = "父id")
    private String fatherId;

    /** 数据集id */
    @ApiModelProperty(value = "数据集id")
    private String sonId;

    /** 版本 */
    @ApiModelProperty(value = "版本")
    private Integer version;


    /** 标注类型 */
    @ApiModelProperty(value = "标注类型")
    private Integer anoType;


    /** 标注类型 */
    @ApiModelProperty(value = "进度条")
    private Integer isSocket;



    /** 标注状态（待定） */
    @ApiModelProperty(value = "标注状态（进度）")
    private String status;

    /** 备注 */
    @ApiModelProperty(value = "备注")
    private String remark;

    /** 用户id */
    @ApiModelProperty(value = "用户id")
    private Integer userId;

    /** 文件id */
    @ApiModelProperty(value = "文件id")
    private String fileIds;

    /** 创建时间 */
    @ApiModelProperty(value = "创建时间")
    @TableField(fill = FieldFill.INSERT)
    private Date createTime;

    /** 更新时间 */
    @ApiModelProperty(value = "更新时间")
    @TableField(fill = FieldFill.INSERT_UPDATE)
    private Date updateTime;

    /** 是否为多人标注 */
    @ApiModelProperty(value = "是否为多人标注")
    private Integer isMany;

    // 前端额外传入的字段（这个字段没用）
    @TableField(exist = false)
    private Integer index;

    /**
     *  标签类型   group：标签组 、 single：标签
     */
    private String tagSelectionMode;

}
