package com.qczy.distillation.model.entity;

import com.baomidou.mybatisplus.annotation.*;
import io.swagger.annotations.ApiModel;
import io.swagger.annotations.ApiModelProperty;
import lombok.Data;

import java.math.BigDecimal;
import java.time.LocalDateTime;

/**
 * 大小模型协同训练任务实体类
 *
 * 对应数据库表：md_training_task
 * 功能：存储大小模型协同训练（知识蒸馏）任务的完整信息
 *
 * @author AI Assistant
 * @date 2025-01-25
 */
@Data
@TableName("md_training_task")
@ApiModel(value = "大小模型协同训练任务实体", description = "存储模型蒸馏训练任务信息")
public class MdTrainingTaskEntity {

    /**
     * 主键ID（自增）
     */
    @TableId(value = "id", type = IdType.AUTO)
    @ApiModelProperty(value = "主键ID")
    private Long id;

    /**
     * 任务ID（唯一标识）
     */
    @TableField("task_id")
    @ApiModelProperty(value = "任务ID（唯一标识）")
    private String taskId;

    /**
     * 任务名称
     */
    @TableField("task_name")
    @ApiModelProperty(value = "任务名称")
    private String taskName;

    /**
     * 任务描述
     */
    @TableField("description")
    @ApiModelProperty(value = "任务描述")
    private String description;

    /**
     * 任务输入名称
     */
    @TableField("task_input_name")
    @ApiModelProperty(value = "任务输入名称")
    private String taskInputName;

    // ========== 模型配置 ==========

    /**
     * 教师模型类型（llama2-7b, qwen-7b等）
     */
    @TableField("teacher_model")
    @ApiModelProperty(value = "教师模型类型")
    private String teacherModel;

    /**
     * 学生模型类型（yolov5s, resnet50等）
     */
    @TableField("student_model")
    @ApiModelProperty(value = "学生模型类型")
    private String studentModel;

    // ========== 训练参数 ==========

    /**
     * 总训练轮数
     */
    @TableField("total_epochs")
    @ApiModelProperty(value = "总训练轮数")
    private Integer totalEpochs;

    /**
     * 当前训练轮数
     */
    @TableField("current_epoch")
    @ApiModelProperty(value = "当前训练轮数")
    private Integer currentEpoch;

    /**
     * 批次大小
     */
    @TableField("batch_size")
    @ApiModelProperty(value = "批次大小")
    private Integer batchSize;

    /**
     * 学习率
     */
    @TableField("learning_rate")
    @ApiModelProperty(value = "学习率")
    private BigDecimal learningRate;

    // ========== 知识蒸馏参数 ==========

    /**
     * 蒸馏温度
     */
    @TableField("temperature")
    @ApiModelProperty(value = "蒸馏温度")
    private BigDecimal temperature;

    /**
     * 软标签权重
     */
    @TableField("alpha")
    @ApiModelProperty(value = "软标签权重")
    private BigDecimal alpha;

    // ========== LoRA配置 ==========

    /**
     * LoRA秩
     */
    @TableField("lora_rank")
    @ApiModelProperty(value = "LoRA秩")
    private Integer loraRank;

    /**
     * LoRA alpha参数
     */
    @TableField("lora_alpha")
    @ApiModelProperty(value = "LoRA alpha参数")
    private Integer loraAlpha;

    /**
     * LoRA dropout率
     */
    @TableField("lora_dropout")
    @ApiModelProperty(value = "LoRA dropout率")
    private BigDecimal loraDropout;

    /**
     * 训练高级配置（JSON格式）
     * 包含：优化器、调度器、GPU配置、梯度配置、模型详细配置等
     */
    @TableField("training_config")
    @ApiModelProperty(value = "训练高级配置（JSON格式）")
    private String trainingConfig;

    // ========== 训练状态 ==========

    /**
     * 任务状态：PENDING-待开始，RUNNING-运行中，PAUSED-暂停，COMPLETED-已完成，FAILED-失败，STOPPED-已停止
     */
    @TableField("status")
    @ApiModelProperty(value = "任务状态")
    private String status;

    /**
     * 训练进度（0-100）
     */
    @TableField("progress")
    @ApiModelProperty(value = "训练进度（0-100）")
    private Integer progress;

    // ========== 训练结果 ==========

    /**
     * 模型准确率（%）
     */
    @TableField("accuracy")
    @ApiModelProperty(value = "模型准确率（%）")
    private BigDecimal accuracy;

    /**
     * 最终损失值
     */
    @TableField("loss")
    @ApiModelProperty(value = "最终损失值")
    private BigDecimal loss;

    /**
     * 最佳准确率
     */
    @TableField("best_accuracy")
    @ApiModelProperty(value = "最佳准确率")
    private BigDecimal bestAccuracy;

    // ========== 数据集信息 ==========

    /**
     * 训练数据集ID（支持字符串类型以兼容非数字ID）
     */
    @TableField("dataset_id")
    @ApiModelProperty(value = "训练数据集ID")
    private String datasetId;

    /**
     * 数据集名称
     */
    @TableField("dataset_name")
    @ApiModelProperty(value = "数据集名称")
    private String datasetName;

    /**
     * 验证数据集ID（支持字符串类型以兼容非数字ID）
     */
    @TableField("val_dataset_id")
    @ApiModelProperty(value = "验证数据集ID")
    private String valDatasetId;

    /**
     * 验证数据集名称
     */
    @TableField("val_dataset_name")
    @ApiModelProperty(value = "验证数据集名称")
    private String valDatasetName;

    // ========== 模型文件路径 ==========

    /**
     * 训练完成的模型文件路径
     */
    @TableField("model_path")
    @ApiModelProperty(value = "训练完成的模型文件路径")
    private String modelPath;

    /**
     * 模型访问URL
     */
    @TableField("model_url")
    @ApiModelProperty(value = "模型访问URL")
    private String modelUrl;

    /**
     * 检查点文件路径
     */
    @TableField("checkpoint_path")
    @ApiModelProperty(value = "检查点文件路径")
    private String checkpointPath;

    // ========== 错误信息 ==========

    /**
     * 错误信息
     */
    @TableField("error_message")
    @ApiModelProperty(value = "错误信息")
    private String errorMessage;

    // ========== 元数据 ==========

    /**
     * 开始训练时间
     */
    @TableField("start_time")
    @ApiModelProperty(value = "开始训练时间")
    private LocalDateTime startTime;

    /**
     * 结束训练时间
     */
    @TableField("end_time")
    @ApiModelProperty(value = "结束训练时间")
    private LocalDateTime endTime;

    /**
     * 训练时长（秒）
     */
    @TableField("duration")
    @ApiModelProperty(value = "训练时长（秒）")
    private Long duration;

    // ========== 系统字段 ==========

    /**
     * 创建人
     */
    @TableField("create_by")
    @ApiModelProperty(value = "创建人")
    private String createBy;

    /**
     * 创建时间
     */
    @TableField(value = "create_time", fill = FieldFill.INSERT)
    @ApiModelProperty(value = "创建时间")
    private LocalDateTime createTime;

    /**
     * 更新人
     */
    @TableField("update_by")
    @ApiModelProperty(value = "更新人")
    private String updateBy;

    /**
     * 更新时间
     */
    @TableField(value = "update_time", fill = FieldFill.INSERT_UPDATE)
    @ApiModelProperty(value = "更新时间")
    private LocalDateTime updateTime;

    /**
     * 备注
     */
    @TableField("remark")
    @ApiModelProperty(value = "备注")
    private String remark;

    /**
     * 删除标志（0-未删除，1-已删除）
     */
    @TableField("del_flag")
    @ApiModelProperty(value = "删除标志（0-未删除，1-已删除）")
    private Integer delFlag;
}
