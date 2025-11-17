package com.qczy.federated.optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

public class FedAvgOptimizer {

    // 简化：用 Map<String, double[]> 代表模型参数
    public Map<String, double[]> aggregate(List<Map<String, double[]>> clientUpdates) {
        if (clientUpdates == null || clientUpdates.isEmpty()) {
            return null;
        }
        Map<String, double[]> base = clientUpdates.get(0);
        int n = clientUpdates.size();

        for (int i = 1; i < n; i++) {
            Map<String, double[]> update = clientUpdates.get(i);
            for (Map.Entry<String, double[]> e : base.entrySet()) {
                double[] w = e.getValue();
                double[] u = update.get(e.getKey());
                if (u == null) continue;
                for (int j = 0; j < w.length; j++) {
                    w[j] += u[j];
                }
            }
        }
        for (Map.Entry<String, double[]> e : base.entrySet()) {
            double[] w = e.getValue();
            for (int j = 0; j < w.length; j++) {
                w[j] /= n;
            }
        }
        return base;
    }
}









