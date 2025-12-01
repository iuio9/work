package com.qczy.distillation.model.dto;

import lombok.Data;

import java.io.Serializable;
import java.math.BigDecimal;
import java.util.List;

/**
 * 训练配置DTO
 *
 * 用于存储前端提交的所有高级训练配置参数
 * 会被序列化为JSON存储在md_training_task表的training_config字段中
 *
 * @author AI Assistant
 * @date 2025-01-25
 */
@Data
public class TrainingConfigDTO implements Serializable {

    // ========== 优化器和调度器配置 ==========

    /**
     * 优化器类型：adamw, adam, sgd, adagrad等
     */
    private String optimizer;

    /**
     * 学习率调度器：cosine, linear, constant, polynomial等
     */
    private String lrScheduler;

    /**
     * 权重衰减（L2正则化）
     */
    private BigDecimal weightDecay;

    /**
     * 梯度累积步数
     */
    private Integer gradAccumSteps;

    /**
     * 最大梯度范数（梯度裁剪）
     */
    private BigDecimal maxGradNorm;

    // ========== 硬件配置 ==========

    /**
     * GPU设备列表，如 [0, 1, 2]
     */
    private List<Integer> gpuDevices;

    /**
     * 是否自动保存检查点
     */
    private Boolean autoSaveCheckpoint;

    /**
     * 检查点保存间隔（每N个epoch保存一次）
     */
    private Integer checkpointInterval;

    // ========== 教师模型详细配置 ==========

    /**
     * 教师模型详细配置
     */
    private TeacherModelConfig teacherModelConfig;

    @Data
    public static class TeacherModelConfig implements Serializable {
        /**
         * 参数量，如 "7B", "13B", "70B"
         */
        private String paramSize;

        /**
         * 模型文件路径或HuggingFace ID
         */
        private String modelPath;

        /**
         * 量化方式：none, int8, int4, fp16等
         */
        private String quantization;
    }

    // ========== 学生模型详细配置 ==========

    /**
     * 学生模型详细配置
     */
    private StudentModelConfig studentModelConfig;

    @Data
    public static class StudentModelConfig implements Serializable {
        /**
         * 参数量，如 "110M", "350M", "1.5B"
         */
        private String paramSize;

        /**
         * 初始化方式：random, xavier, kaiming, pretrained等
         */
        private String initMethod;

        /**
         * 预训练权重路径（可选）
         */
        private String pretrainPath;
    }

    // ========== LoRA高级配置 ==========

    /**
     * LoRA高级配置
     */
    private LoraAdvancedConfig loraAdvancedConfig;

    @Data
    public static class LoraAdvancedConfig implements Serializable {
        /**
         * 目标模块列表，如 ["q_proj", "v_proj", "k_proj", "o_proj"]
         */
        private List<String> targetModules;

        /**
         * LoRA应用的层，如 "all" 或 "0-11"
         */
        private String layers;

        /**
         * Bias训练策略：none, all, lora_only
         */
        private String biasTrain;
    }

    // ========== 知识蒸馏高级配置 ==========

    /**
     * 知识蒸馏高级配置
     */
    private DistillationAdvancedConfig distillationAdvancedConfig;

    @Data
    public static class DistillationAdvancedConfig implements Serializable {
        /**
         * 硬标签权重（真实标签损失权重）
         */
        private BigDecimal hardLabelWeight;

        /**
         * 软标签权重（教师输出损失权重）
         */
        private BigDecimal softLabelWeight;

        /**
         * 蒸馏损失类型：kl_div, mse, cosine等
         */
        private String lossType;

        /**
         * 是否启用中间层蒸馏
         */
        private Boolean intermediateLayers;

        /**
         * 是否启用注意力蒸馏
         */
        private Boolean attentionDistill;
    }
}
