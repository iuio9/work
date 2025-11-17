package com.qczy.model.entity;

import com.baomidou.mybatisplus.annotation.*;
import com.fasterxml.jackson.annotation.JsonFormat;
import io.swagger.annotations.ApiModelProperty;
import lombok.Data;
import org.springframework.format.annotation.DateTimeFormat;

import java.io.Serializable;
import java.util.Date;

/**
 * @Author: hh
 * @Version: 1.0
 * @Date: 2024/8/21 10:20
 * @Description:
 */
@Data
@TableName("qczy_algorithm_model")
public class AlgorithmModelEntity implements Serializable {

    /** id */
    @ApiModelProperty(value = "id")
    @TableId(type = IdType.AUTO)
    private Integer modelId;

    /** 模型名称 */
    @ApiModelProperty(value = "模型名称")
    private String modelName;

    /** 模型业务类型 */
    @ApiModelProperty(value = "模型业务类型")
    private String modelBizType;

    /** 模型地址 */
    @ApiModelProperty(value = "模型地址")
    private String modelUrl;

    /** 模型参数 */
    @ApiModelProperty(value = "模型参数")
    private String modelParams;

    /** 模型环境参数 */
    @ApiModelProperty(value = "模型环境参数")
    private String modelEnvParams;

    /** 模型请求类型 */
    @ApiModelProperty(value = "模型请求类型")
    private String modelReqType;

    /** 模型请求类型 */
    @ApiModelProperty(value = "模型描述")
    private String modelDesc;


    /** 模型请求类型 */
    @ApiModelProperty(value = "训练地址")
    private String trainUrl;

    /** 模型请求类型 */
    @ApiModelProperty(value = "控制台进度文件地址")
    private String trainConsole;

    /** 模型请求类型 */
    @ApiModelProperty(value = "是否可以刪除")
    private String isDelete;

    /** 训练状态 */
    @ApiModelProperty(value = "训练状态")
    private String trainStat;
    /** 模型版本 */
    @ApiModelProperty(value = "模型版本")
    private String modelVersion;

    /** 当前训练任务id */
    @ApiModelProperty(value = "当前训练任务id")
    private String trainTaskId;


    /** 当前训练任务id */
    @ApiModelProperty(value = "评估列表")
    private String assessLst;

    @ApiModelProperty(value = "创建时间")
    @DateTimeFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss", timezone = "GMT+8")
    @TableField(fill = FieldFill.INSERT)
    private Date createTime;

    /** 更新时间 */
    @ApiModelProperty(value = "更新时间")
    @DateTimeFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss", timezone = "GMT+8")
    @TableField(fill = FieldFill.INSERT_UPDATE)
    private Date updateTime;

    // 前端额外传入的字段（这个字段没用）
    @TableField(exist = false)
    private Integer index;
}
