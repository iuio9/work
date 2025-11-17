package com.qczy.federated.adapters;

import com.qczy.federated.model.TrainingJob;
import com.qczy.federated.spi.ModelAdapter;

import java.util.HashMap;
import java.util.Map;

public class PlaceholderAdapter implements ModelAdapter {
    @Override
    public Map<String, double[]> trainOneRound(TrainingJob job, String nodeId) {
        // 占位实现：返回一个假参数
        Map<String, double[]> params = new HashMap<>();
        params.put("layer1.weight", new double[]{1.0});
        return params;
    }

    @Override
    public void applyGlobalParameters(Map<String, double[]> globalParams, TrainingJob job, String nodeId) {
        // 占位实现：不做事
    }

    @Override
    public double evaluate(TrainingJob job, String nodeId) {
        // 占位实现：返回固定精度
        return 0.9;
    }
}









