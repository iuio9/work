package com.qczy.distillation.model.dto;

import lombok.Data;

import java.math.BigDecimal;

/**
 * 创建训练任务请求DTO
 *
 * 接收前端提交的完整训练任务数据
 * 包含基础字段和高级配置
 *
 * @author AI Assistant
 * @date 2025-01-25
 */
@Data
public class CreateTaskRequestDTO {

    // ========== 基础信息 ==========

    /**
     * 任务名称
     */
    private String taskName;

    /**
     * 任务描述
     */
    private String description;

    // ========== 模型配置 ==========

    /**
     * 教师模型类型
     */
    private String teacherModel;

    /**
     * 学生模型类型
     */
    private String studentModel;

    // ========== 数据集 ==========

    /**
     * 训练数据集ID（支持字符串类型以兼容非数字ID）
     */
    private String datasetId;

    /**
     * 验证数据集ID（支持字符串类型以兼容非数字ID）
     */
    private String valDatasetId;

    // ========== 基础训练参数 ==========

    /**
     * 训练轮数
     */
    private Integer epochs;

    /**
     * 批次大小
     */
    private Integer batchSize;

    /**
     * 学习率
     */
    private BigDecimal learningRate;

    // ========== LoRA配置 ==========

    /**
     * LoRA Rank
     */
    private Integer loraRank;

    /**
     * LoRA Alpha
     */
    private Integer loraAlpha;

    /**
     * LoRA Dropout
     */
    private BigDecimal loraDropout;

    // ========== 知识蒸馏基础参数 ==========

    /**
     * 蒸馏温度
     */
    private BigDecimal temperature;

    /**
     * 软标签权重（alpha）
     */
    private BigDecimal alpha;

    // ========== 高级训练配置（这些会被序列化为JSON） ==========

    /**
     * 优化器类型
     */
    private String optimizer;

    /**
     * 学习率调度器
     */
    private String lrScheduler;

    /**
     * 权重衰减
     */
    private BigDecimal weightDecay;

    /**
     * 梯度累积步数
     */
    private Integer gradAccumSteps;

    /**
     * 最大梯度范数
     */
    private BigDecimal maxGradNorm;

    /**
     * GPU设备列表（以逗号分隔的字符串，如 "0,1,2"）
     */
    private String gpuDevices;

    /**
     * 是否自动保存检查点
     */
    private Boolean autoSaveCheckpoint;

    /**
     * 检查点保存间隔
     */
    private Integer checkpointInterval;

    // ========== 教师模型详细配置 ==========

    /**
     * 教师模型参数量
     */
    private String teacherParamSize;

    /**
     * 教师模型路径
     */
    private String teacherModelPath;

    /**
     * 教师模型量化方式
     */
    private String teacherQuantization;

    // ========== 学生模型详细配置 ==========

    /**
     * 学生模型参数量
     */
    private String studentParamSize;

    /**
     * 学生模型初始化方式
     */
    private String studentInitMethod;

    /**
     * 学生模型预训练路径
     */
    private String studentPretrainPath;

    // ========== LoRA高级配置 ==========

    /**
     * LoRA目标模块（以逗号分隔，如 "q_proj,v_proj"）
     */
    private String loraTargetModules;

    /**
     * LoRA应用层
     */
    private String loraLayers;

    /**
     * Bias训练策略
     */
    private String loraBiasTrain;

    // ========== 知识蒸馏高级配置 ==========

    /**
     * 硬标签权重
     */
    private BigDecimal hardLabelWeight;

    /**
     * 软标签权重
     */
    private BigDecimal softLabelWeight;

    /**
     * 蒸馏损失类型
     */
    private String distillLossType;

    /**
     * 是否启用中间层蒸馏
     */
    private Boolean intermediateLayers;

    /**
     * 是否启用注意力蒸馏
     */
    private Boolean attentionDistill;

    // ========== Qwen2.5-VL多模型配置 ==========

    /**
     * 学生模型类型（resnet, vit, yolov8, unet, lstm）
     */
    private String studentModelType;

    /**
     * 学生模型大小（resnet50, vit-base, s, medium等）
     */
    private String studentModelSize;

    /**
     * 任务类型（classification, detection, segmentation）
     */
    private String taskType;

    /**
     * 分类类别数
     */
    private Integer numClasses;

    /**
     * 图像尺寸
     */
    private Integer imageSize;

    /**
     * 蒸馏类型（feature, logit, hybrid）
     */
    private String distillationType;

    /**
     * 特征损失类型（mse, cosine）
     */
    private String featureLossType;

    /**
     * 是否启用特征对齐
     */
    private Boolean alignFeature;
}
