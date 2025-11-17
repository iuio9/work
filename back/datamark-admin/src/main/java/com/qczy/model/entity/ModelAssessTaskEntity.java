package com.qczy.model.entity;

import com.baomidou.mybatisplus.annotation.TableField;
import com.baomidou.mybatisplus.annotation.TableName;
import io.swagger.annotations.ApiModelProperty;
import lombok.Data;

import java.util.Date;

/**
 * @Author: hh
 * @Version: 1.0
 * @Date: 2025/6/5 10:42
 * @Description:
 */
@TableName("qczy_model_assess_task")
@Data
public class ModelAssessTaskEntity {

    /** id */
    @ApiModelProperty(value = "id")
    private Integer id;


    /** 任务名称 */
    @ApiModelProperty(value = "任务名称")
    private String taskName;


    /** 任务类型
     1：测试
     2：评估
     * */
    @ApiModelProperty(value = "任务类型")
    private Integer taskType;


    /** 任务状态
     1：待执行
     2：执行中
     3：已完成
     4：任务失败
     5：终止
     6：继续
     * */
    @ApiModelProperty(value = "任务状态")
    private Integer taskStatus;


    /** 任务进度 */
    @ApiModelProperty(value = "任务进度")
    private String taskProgress;


    /** 模型基础信息id */
    @ApiModelProperty(value = "模型基础信息id")
    private Integer modelBaseId;


    /** 任务描述 */
    @ApiModelProperty(value = "任务描述")
    private String taskDesc;

    /** 评估结果 */
    @ApiModelProperty(value = "评估结果")
    private String taskResult;



    /** 任务版本 */
    @ApiModelProperty(value = "任务版本")
    private String taskVersion;


    /** 版本描述 */
    @ApiModelProperty(value = "版本描述")
    private String versionDesc;


    /** 用户id */
    @ApiModelProperty(value = "用户id")
    private Integer userId;


    /** 创建时间 */
    @ApiModelProperty(value = "创建时间")
    private Date createTime;

    @TableField(exist = false)
    private String sign;
}
