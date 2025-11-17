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
 * @Date: 2024/8/2 14:55
 * @Description:
 */
@Data
@TableName("qczy_dict_type")
public class DictTypeEntity implements Serializable {


    @ApiModelProperty(value = "id")
    private Integer id;

    @ApiModelProperty(value = "字典名称")
    private String dictName;

    /** 状态（1正常 2停用） */
    private Integer status;


    /** 是否允许删除：1-> 不允许 、 2->允许 */
    private Integer isAllowDeletion;

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

    @ApiModelProperty(value = "备注")
    private String remark;

    // 前端额外传入的字段（这个字段没用）
    @TableField(exist = false)
    private Integer index;

}
