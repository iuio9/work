package com.qczy.distillation.model.entity;

import com.baomidou.mybatisplus.annotation.*;
import io.swagger.annotations.ApiModel;
import io.swagger.annotations.ApiModelProperty;
import lombok.Data;

import java.math.BigDecimal;
import java.time.LocalDateTime;

/**
 * 训练历史记录实体类
 *
 * 对应数据库表：md_training_history
 * 功能：记录每个epoch的训练详情和指标
 *
 * @author AI Assistant
 * @date 2025-01-25
 */
@Data
@TableName("md_training_history")
@ApiModel(value = "训练历史记录实体", description = "记录每个epoch的训练详情")
public class MdTrainingHistoryEntity {

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

    /**
     * 训练轮次
     */
    @TableField("epoch")
    @ApiModelProperty(value = "训练轮次")
    private Integer epoch;

    // ========== 训练指标 ==========

    /**
     * 训练损失
     */
    @TableField("train_loss")
    @ApiModelProperty(value = "训练损失")
    private BigDecimal trainLoss;

    /**
     * 训练准确率
     */
    @TableField("train_accuracy")
    @ApiModelProperty(value = "训练准确率")
    private BigDecimal trainAccuracy;

    /**
     * 验证损失
     */
    @TableField("val_loss")
    @ApiModelProperty(value = "验证损失")
    private BigDecimal valLoss;

    /**
     * 验证准确率
     */
    @TableField("val_accuracy")
    @ApiModelProperty(value = "验证准确率")
    private BigDecimal valAccuracy;

    // ========== 蒸馏相关指标 ==========

    /**
     * 蒸馏损失
     */
    @TableField("distill_loss")
    @ApiModelProperty(value = "蒸馏损失")
    private BigDecimal distillLoss;

    /**
     * 硬标签损失
     */
    @TableField("hard_loss")
    @ApiModelProperty(value = "硬标签损失")
    private BigDecimal hardLoss;

    /**
     * 软标签损失
     */
    @TableField("soft_loss")
    @ApiModelProperty(value = "软标签损失")
    private BigDecimal softLoss;

    // ========== 系统资源 ==========

    /**
     * GPU使用率（%）
     */
    @TableField("gpu_usage")
    @ApiModelProperty(value = "GPU使用率（%）")
    private BigDecimal gpuUsage;

    /**
     * 内存使用量（MB）
     */
    @TableField("memory_usage")
    @ApiModelProperty(value = "内存使用量（MB）")
    private BigDecimal memoryUsage;

    // ========== 时间信息 ==========

    /**
     * 本轮训练耗时（秒）
     */
    @TableField("epoch_time")
    @ApiModelProperty(value = "本轮训练耗时（秒）")
    private BigDecimal epochTime;

    /**
     * 记录时间
     */
    @TableField(value = "record_time", fill = FieldFill.INSERT)
    @ApiModelProperty(value = "记录时间")
    private LocalDateTime recordTime;
}
