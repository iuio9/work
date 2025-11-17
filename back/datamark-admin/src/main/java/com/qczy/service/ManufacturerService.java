package com.qczy.service;

import com.qczy.model.entity.ModelAssessTaskEntity;

/**
 * @Author: hh
 * @Version: 1.0
 * @Date: 2025/6/12 11:07
 * @Description:
 */
public interface ManufacturerService {

    /**
     * 开始调用厂商
     */
    void startManufacturer(ModelAssessTaskEntity modelAssessTaskEntity);

    /**
     * 暂停调用厂商
     */
    void stopManufacturer(ModelAssessTaskEntity modelAssessTaskEntity);

    /**
     * 结束调用厂商
     */
    void endManufacturer(ModelAssessTaskEntity modelAssessTaskEntity);


    /**
     *  判断是否执行过该任务 [调用厂商过程]
     */
    boolean isExecuteTask(ModelAssessTaskEntity modelAssessTaskEntity);

    /**
     * 完成对接
     */
    int finishContact(ModelAssessTaskEntity modelAssessTaskEntity);

}
