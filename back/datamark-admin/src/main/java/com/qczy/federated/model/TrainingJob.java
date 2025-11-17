package com.qczy.federated.model;

import lombok.Data;

import java.time.Instant;
import java.util.List;
import java.util.Map;

@Data
public class TrainingJob {

    private String jobId;

    private ModelType modelType;

    // 超参数，如学习率、批大小、轮次等
    private Map<String, Object> hyperParameters;

    // 参与训练的节点
    private List<String> participantNodeIds;

    // 训练状态
    private String status;

    private Instant createdAt;

    private Instant updatedAt;

    // 全局参数（聚合结果），简化为 Map<String, double[]>
    private Map<String, double[]> globalParameters;

    // 基线精度（首次评估记录），和允许的最大下降百分比（默认 5%）
    private Double baselineAccuracy;

    private Double allowedDropPercent = 5.0;

    // 最近一次全局评估精度
    private Double lastGlobalAccuracy;

    // 当前训练轮次
    private int currentRound;
}





