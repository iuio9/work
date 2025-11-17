package com.qczy.federated.model.entity;

import com.baomidou.mybatisplus.annotation.*;
import lombok.Data;

import java.math.BigDecimal;
import java.time.LocalDateTime;

/**
 * 联邦学习训练任务实体类
 *
 * 对应数据库表：fl_training_job
 * 功能：存储联邦学习训练任务的完整信息
 *
 * @author AI Assistant
 * @date 2025-01-20
 */
@Data
@TableName("fl_training_job")
public class TrainingJobEntity {

    /**
     * 主键ID（自增）
     */
    @TableId(value = "id", type = IdType.AUTO)
    private Long id;

    /**
     * 任务唯一标识（UUID）
     */
    @TableField("job_id")
    private String jobId;

    /**
     * 任务名称
     */
    @TableField("job_name")
    private String jobName;

    /**
     * 模型类型：YOLO_V8/LSTM/UNET/RESNET/VISION_TRANSFORMER
     */
    @TableField("model_type")
    private String modelType;

    /**
     * 任务状态：CREATED/RUNNING/COMPLETED/FAILED/STOPPED/DEGRADED
     */
    @TableField("status")
    private String status;

    /**
     * 聚合策略：FedAvg/FedProx/FedAdam
     */
    @TableField("strategy")
    private String strategy;

    // ========== 训练参数 ==========

    /**
     * 训练轮数
     */
    @TableField("num_rounds")
    private Integer numRounds;

    /**
     * 最少客户端数
     */
    @TableField("min_clients")
    private Integer minClients;

    /**
     * 每轮最少训练客户端数
     */
    @TableField("min_fit_clients")
    private Integer minFitClients;

    /**
     * 每轮最少评估客户端数
     */
    @TableField("min_evaluate_clients")
    private Integer minEvaluateClients;

    /**
     * 每轮参与训练的客户端比例
     */
    @TableField("fraction_fit")
    private BigDecimal fractionFit;

    /**
     * 每轮参与评估的客户端比例
     */
    @TableField("fraction_evaluate")
    private BigDecimal fractionEvaluate;

    /**
     * 超参数配置（JSON格式）
     */
    @TableField("hyperparameters")
    private String hyperparameters;

    // ========== 参与节点 ==========

    /**
     * 参与节点ID列表（JSON数组）
     */
    @TableField("participant_node_ids")
    private String participantNodeIds;

    /**
     * 参与节点数量
     */
    @TableField("participant_count")
    private Integer participantCount;

    // ========== 精度监控 ==========

    /**
     * 基线精度（首次评估）
     */
    @TableField("baseline_accuracy")
    private BigDecimal baselineAccuracy;

    /**
     * 当前精度
     */
    @TableField("current_accuracy")
    private BigDecimal currentAccuracy;

    /**
     * 最佳精度
     */
    @TableField("best_accuracy")
    private BigDecimal bestAccuracy;

    /**
     * 允许精度下降百分比
     */
    @TableField("allowed_drop_percent")
    private BigDecimal allowedDropPercent;

    // ========== 训练进度 ==========

    /**
     * 当前训练轮次
     */
    @TableField("current_round")
    private Integer currentRound;

    /**
     * 总训练时间（秒）
     */
    @TableField("total_training_time")
    private Long totalTrainingTime;

    // ========== Flower Server 配置 ==========

    /**
     * Flower Server端口
     */
    @TableField("server_port")
    private Integer serverPort;

    /**
     * Flower Server地址
     */
    @TableField("server_address")
    private String serverAddress;

    /**
     * 全局模型保存路径
     */
    @TableField("global_model_path")
    private String globalModelPath;

    // ========== 日志和描述 ==========

    /**
     * 任务描述
     */
    @TableField("description")
    private String description;

    /**
     * 错误信息
     */
    @TableField("error_message")
    private String errorMessage;

    // ========== 时间戳 ==========

    /**
     * 创建时间
     */
    @TableField(value = "created_at", fill = FieldFill.INSERT)
    private LocalDateTime createdAt;

    /**
     * 开始时间
     */
    @TableField("started_at")
    private LocalDateTime startedAt;

    /**
     * 完成时间
     */
    @TableField("completed_at")
    private LocalDateTime completedAt;

    /**
     * 更新时间
     */
    @TableField(value = "updated_at", fill = FieldFill.INSERT_UPDATE)
    private LocalDateTime updatedAt;
}
