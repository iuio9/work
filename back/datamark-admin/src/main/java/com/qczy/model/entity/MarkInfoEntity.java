package com.qczy.model.entity;

import com.baomidou.mybatisplus.annotation.FieldFill;
import com.baomidou.mybatisplus.annotation.TableField;
import com.baomidou.mybatisplus.annotation.TableName;
import io.swagger.annotations.ApiModelProperty;
import lombok.Data;
import com.baomidou.mybatisplus.annotation.FieldStrategy;


import java.io.Serializable;
import java.text.DecimalFormat;
import java.util.Date;

/**
 * @Author: hh
 * @Version: 1.0
 * @Date: 2024/8/21 10:20
 * @Description:
 */
@Data
@TableName("qczy_mark_info")
public class MarkInfoEntity implements Serializable {

    /** id */
    @ApiModelProperty(value = "id")
    private Integer id;

    /** 数据集id */
    @ApiModelProperty(value = "数据集id")
    private String sonId;

    /** 文件id */
    @ApiModelProperty(value = "文件id")
    private Integer fileId;

    /** 标注文件id */
    @ApiModelProperty(value = "标注文件id")
    // 设置更新策略为 ALWAYS，始终更新该字段，即使值为 null
    //@TableField(updateStrategy = FieldStrategy.ALWAYS)
    private Integer markFileId;


    /** 标签id */
    @ApiModelProperty(value = "标签id")
    private String labels;

    /** 标注信息 */
    @ApiModelProperty(value = "标注信息")
    private String markInfo;


    /** label标注信息 */
    @ApiModelProperty(value = "label标注信息")
    private String labelMarkInfo;

    /** 是否为无效数据->(0：无效、1：有效) */
    @ApiModelProperty(value = "是否为无效数据->(0：无效、1：有效)")
    private Integer isInvalid;

    /** 创建时间 */
    @ApiModelProperty(value = "创建时间")
    @TableField(fill = FieldFill.INSERT)
    private Date createTime;

    /** 更新时间 */
    @ApiModelProperty(value = "更新时间")
    @TableField(fill = FieldFill.INSERT_UPDATE)
    private Date updateTime;

    // 前端额外传入的字段（这个字段没用）
    @TableField(exist = false)
    private Integer index;


    private Integer operateWidth;

    private Integer operateHeight;
    @ApiModelProperty(value = "宽")
    private Integer width;
    @ApiModelProperty(value = "长")
    private  Integer height;

    /**
     * 用户id
     */
    @TableField(exist = false)
    private Integer markUserId;

    public static void main(String[] args) {
        double result = (double) 10 /(double)3;
        DecimalFormat df = new DecimalFormat("#.00"); // 保留两位小数
        System.out.println(df.format(result)); // 输出：3.33

    }


}
