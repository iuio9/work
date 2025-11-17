package com.qczy.federated.spi;

import com.qczy.federated.model.TrainingJob;

import java.util.Map;

public interface ModelAdapter {

    // 本地训练一个周期，返回更新后的参数
    Map<String, double[]> trainOneRound(TrainingJob job, String nodeId);

    // 应用全局参数到本地模型
    void applyGlobalParameters(Map<String, double[]> globalParams, TrainingJob job, String nodeId);

    // 评估精度（用于控制精度下降在 5% 以内）
    double evaluate(TrainingJob job, String nodeId);
}









