package com.qczy.federated.adapters;

import com.qczy.federated.model.ModelType;
import com.qczy.federated.spi.ModelAdapter;

public class DefaultAdapterFactory {

    public static ModelAdapter forType(ModelType type) {
        // 实际项目中应分别实现对应模型的适配器
        // 这里返回一个占位实现，确保流程跑通
        return new PlaceholderAdapter();
    }
}









