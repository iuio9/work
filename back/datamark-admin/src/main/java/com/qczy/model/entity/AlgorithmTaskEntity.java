package com.qczy.model.entity;

import com.baomidou.mybatisplus.annotation.*;
import com.fasterxml.jackson.annotation.JsonFormat;
import io.swagger.annotations.ApiModelProperty;
import lombok.Data;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.format.annotation.DateTimeFormat;

import java.beans.Transient;
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
@TableName("qczy_algorithm_task")
public class AlgorithmTaskEntity implements Serializable {

    /** id */
    @ApiModelProperty(value = "id")
    @TableId(type = IdType.AUTO)
    private Long taskId;

    /** 用户输入任务名称 */
    @ApiModelProperty(value = "用户输入任务名称")
    private String taskInputName;

    @ApiModelProperty(value = "模型名称")
    private String modelName;
    @ApiModelProperty(value = "训练版本")
    private String modelVersion;


    /** 数据集id */
    @ApiModelProperty(value = "数据集id")
    private String dataSetId;

    @ApiModelProperty(value = "输出到数据集ID")
    private String datasetOutId;

    @ApiModelProperty(value = "版本")
    private String version;

    /** 綁定模型id */
    @ApiModelProperty(value = "綁定模型id")
    private String modelId;

    /** 綁定算法id */
    @ApiModelProperty(value = "綁定模型id")
    private String algorithmId;

    /** 任务状态 */
    @ApiModelProperty(value = "任务状态")
    private String  taskStat;

    /** 异常描述 */
    @ApiModelProperty(value = "异常描述")
    private String  taskException;

    /** 任务名称 */
    @ApiModelProperty(value = "任务类型")
    private String  taskName;

    /** 任务进度 */
    @ApiModelProperty(value = "任务进度")
    private String taskProgress;


    /** 是否训练 */
    @ApiModelProperty(value = "是否训练")
    private String isTrain;

    @ApiModelProperty(value = "算法任务描述")
    private String taskDesc;

    /** 是否训练 */
    @ApiModelProperty(value = "训练地址")
    private String trainUrl;

    @ApiModelProperty(value = "文件控制台地址")
    private String trainConsole;

    @ApiModelProperty(value = "评估地址")
    private String assessUrl;

    @ApiModelProperty(value = "pid列表")
    private String pid;


    @ApiModelProperty(value = "训练类型")
    private String trainType;
    @ApiModelProperty(value = "评估状态")
    private String isAssess;

    @ApiModelProperty(value = "记录功能类型")
    private String recordType;

    /** 创建时间 */
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


    @ApiModelProperty(value = "文件Id")
    @TableField(exist = false)
    private String filePath;

    @ApiModelProperty(value = "算法执行参数")
    @TableField(exist = false)
    Map<String,Object> params;

    @ApiModelProperty(value = "增强算法执行参数")
    @TableField(exist = false)
    List<Map<String,Object>> paramsLst;

    @TableField(exist = false)
    @ApiModelProperty(hidden = true)
    private List<FileEntity> result;

    @TableField(exist = false)
    private String groupVName;

    @TableField(exist = false)
    private String taskTimeStart;

    @TableField(exist = false)
    private String taskTimeEnd;

    @TableField(exist = false)
    private String testResult;


    //请选择标注类型-数据类型
    @TableField(exist = false)
    private String dataEnhanceType;
    @TableField(exist = false)
    //请选择标注类型-标注类型
    private String dataEnhanceMarkType;
    //请选择数据集-数据输入
    @TableField(exist = false)
    private String datasetId;
    //请选择数据集-选择标签
    @TableField(exist = false)
    private List datasetTags;
    //请选择数据集-增强区域
    @TableField(exist = false)
    private String datasetEnhanceArea;
    @TableField(exist = false)
    // 请选择算子处理策略（串行叠加:0  并行遍历:1）
    private String  dataEnhanceTactics;

    // 前端额外传入的字段（这个字段没用）
    @TableField(exist = false)
    private Integer index;

    @TableField(exist = false)
    private String dataSetName;
    @TableField(exist = false)
    private String dataSetTotal;



}
