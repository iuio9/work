package com.qczy.distillation.model.entity;

import com.baomidou.mybatisplus.annotation.*;
import io.swagger.annotations.ApiModel;
import io.swagger.annotations.ApiModelProperty;
import lombok.Data;

import java.math.BigDecimal;
import java.time.LocalDateTime;

/**
 * 模型评估结果实体类
 *
 * 对应数据库表：md_model_evaluation
 * 功能：存储模型评估结果和指标
 *
 * @author AI Assistant
 * @date 2025-01-25
 */
@Data
@TableName("md_model_evaluation")
@ApiModel(value = "模型评估结果实体", description = "存储模型评估结果")
public class MdModelEvaluationEntity {

    /**
     * 主键ID（自增）
     */
    @TableId(value = "id", type = IdType.AUTO)
    @ApiModelProperty(value = "主键ID")
    private Long id;

    /**
     * 任务ID
     */
    @TableField("task_id")
    @ApiModelProperty(value = "任务ID")
    private String taskId;

    // ========== 评估指标 ==========

    /**
     * 精确率
     */
    @TableField("precision")
    @ApiModelProperty(value = "精确率")
    private BigDecimal precision;

    /**
     * 召回率
     */
    @TableField("recall")
    @ApiModelProperty(value = "召回率")
    private BigDecimal recall;

    /**
     * F1分数
     */
    @TableField("f1_score")
    @ApiModelProperty(value = "F1分数")
    private BigDecimal f1Score;

    /**
     * mAP@0.5
     */
    @TableField("map_50")
    @ApiModelProperty(value = "mAP@0.5")
    private BigDecimal map50;

    /**
     * mAP@0.5:0.95
     */
    @TableField("map_95")
    @ApiModelProperty(value = "mAP@0.5:0.95")
    private BigDecimal map95;

    // ========== 评估数据集 ==========

    /**
     * 评估数据集ID
     */
    @TableField("eval_dataset_id")
    @ApiModelProperty(value = "评估数据集ID")
    private Long evalDatasetId;

    /**
     * 评估数据集名称
     */
    @TableField("eval_dataset_name")
    @ApiModelProperty(value = "评估数据集名称")
    private String evalDatasetName;

    /**
     * 评估样本数量
     */
    @TableField("eval_sample_count")
    @ApiModelProperty(value = "评估样本数量")
    private Integer evalSampleCount;

    // ========== 评估时间 ==========

    /**
     * 评估时间
     */
    @TableField(value = "eval_time", fill = FieldFill.INSERT)
    @ApiModelProperty(value = "评估时间")
    private LocalDateTime evalTime;

    // ========== 详细结果 ==========

    /**
     * 详细评估结果（JSON格式）
     */
    @TableField("detailed_results")
    @ApiModelProperty(value = "详细评估结果（JSON格式）")
    private String detailedResults;
}
