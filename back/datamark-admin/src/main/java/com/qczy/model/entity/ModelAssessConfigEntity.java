package com.qczy.model.entity;

import com.baomidou.mybatisplus.annotation.TableName;
import io.swagger.annotations.ApiModelProperty;
import lombok.Data;

/**
 * @Author: hh
 * @Version: 1.0
 * @Date: 2025/6/5 10:59
 * @Description:
 */
@TableName("qczy_model_assess_config")
@Data
public class ModelAssessConfigEntity {

    /** id */
    @ApiModelProperty(value = "id")
    private Integer id;


    /** 评估任务id */
    @ApiModelProperty(value = "评估任务id")
    private Integer assessTaskId;


    /** 数据集id */
    @ApiModelProperty(value = "数据集id")
    private String sonId;


    /** 模型接口地址 */
    @ApiModelProperty(value = "模型接口地址")
    private String modelAddress;



    /** 模型传输方式(请求类型) */
    @ApiModelProperty(value = "模型传输方式(请求类型)")
    private Integer requestType;


    /** 模型参数文件名 */
    @ApiModelProperty(value = "模型参数文件名")
    private String modelFileName;


    /** 模型参数 */
    @ApiModelProperty(value = "模型参数")
    private String modelParams;


    /** 模型参数 */
    @ApiModelProperty(value = "模型参数地址")
    private String modelParamsPath;


    /** 评估描述 */
    @ApiModelProperty(value = "评估描述")
    private String assessDesc;


    /** 评估描述 */
    @ApiModelProperty(value = "评估指标")
    private String assessTarget;


    /** 评估描述 */
    @ApiModelProperty(value = "评估图表")
    private String assessChart;


    /** 评估描述 */
    @ApiModelProperty(value = "错误信息")
    private String errorMessage;

    private String labelMap;

    private String assessTargetMap;

    // 模型识别类型
    private String modelClass;

}
