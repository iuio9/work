package com.qczy.service;

import com.baomidou.mybatisplus.extension.service.IService;
import com.qczy.model.entity.ModelBaseEntity;
import com.qczy.model.request.DeleteRequest;

/**
 * @Author: hh
 * @Version: 1.0
 * @Date: 2025/5/16 11:14
 * @Description:
 */
public interface ModelBaseService extends IService<ModelBaseEntity> {
    /**
     *  删除模型
     */
    int delModel(DeleteRequest request);

    /**
     *  判断当前模型是否有任务正在使用
     */
    boolean isModelUse(Integer modelId);

    int deleteFile(DeleteRequest request);

    // 判断模型名称是否存在
    boolean isModelNameRepeat(String modelName, Integer id);
}
