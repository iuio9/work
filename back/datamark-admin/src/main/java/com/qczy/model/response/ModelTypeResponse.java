package com.qczy.model.response;

import com.qczy.model.entity.ModelBaseEntity;
import lombok.Data;

/**
 * @Author: hh
 * @Version: 1.0
 * @Date: 2025/6/5 14:01
 * @Description:
 */
@Data
public class ModelTypeResponse {

    // id
    private int id;

    // 模型名称
    private String modelName;

    // 新增构造函数
    public ModelTypeResponse(ModelBaseEntity entity) {
        this.id = entity.getId();
        this.modelName = entity.getModelName();
    }
}
