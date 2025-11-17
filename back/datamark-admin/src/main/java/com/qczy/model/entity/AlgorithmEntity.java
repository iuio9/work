package com.qczy.model.entity;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableField;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import io.swagger.annotations.ApiModelProperty;
import lombok.Data;

import java.io.Serializable;
import java.util.Date;
import java.util.List;
import java.util.Map;

/**
 * @Author: hh
 * @Version: 1.0
 * @Date: 2024/8/21 10:20
 * @Description:
 */
@Data
@TableName("qczy_algorithm")
public class AlgorithmEntity implements Serializable {

    /** id */
    @ApiModelProperty(value = "id")
    @TableId(type = IdType.AUTO)
    private Integer id;

    /** 模型id */
    @ApiModelProperty(value = "模型id")
    private String modelId;

    @ApiModelProperty(value = "任务Id")
    private String curTaskId;

    /** 算法名称 */
    @ApiModelProperty(value = "算法名称")
    private String algorithmName;

    /** 算法描述 */
    @ApiModelProperty(value = "算法描述")
    private String algorithmDesc;

    @ApiModelProperty(value = "训练类型")
    private String trainType;

    /** 算法请求地址 */
    @ApiModelProperty(value = "算法请求地址")
    private String url;

    /** 算法请求参数 */
    @ApiModelProperty(value = "算法请求参数")
    private String params;

    /** 算法响应参数 */
    @ApiModelProperty(value = "算法响应参数")
    private String responseParams;

    /** 算法请求类型 */
    @ApiModelProperty(value = "算法请求类型")
    private String requestType;
    @ApiModelProperty(value = "示意图处理前")
    private String beforeUrl;

    @ApiModelProperty(value = "示意图处理后")
    private String afterUrl;

    // 前端额外传入的字段（这个字段没用）
    @TableField(exist = false)
    private Integer index;


    /** 算法请求参数 */
    @ApiModelProperty(value = "算法请求参数")
    @TableField(exist = false)
    private List<Map> paramsMap;

    /** 算法请求参数 */
    @ApiModelProperty(value = "算法响应参数")
    @TableField(exist = false)
    private List<Map> responseParamsMap;

    /** 关联字典id */
    @ApiModelProperty(value = "关联字典id")
    @TableField(exist = false)
    private String dircId;


    @ApiModelProperty(value = "数据增强-图像算子操作")
    private String operate;
    @ApiModelProperty(value = "数据增强-图像算子操作mode")
    @TableField(exist = false)
    private String operateModel;
    @ApiModelProperty(value = "数据增强-图像算子操作(生成数量)")
    @TableField(exist = false)
    private String numAugmentations;

}
