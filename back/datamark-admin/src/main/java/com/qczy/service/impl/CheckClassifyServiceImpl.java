package com.qczy.service.impl;

import cn.hutool.core.util.ObjectUtil;
import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.qczy.common.constant.TaskRecordTypeConstants;
import com.qczy.model.entity.AlgorithmEntity;
import com.qczy.model.entity.AlgorithmTaskEntity;
import com.qczy.model.request.BWTestRequest;
import com.qczy.model.request.CheckAndClassify;
import com.qczy.service.AlgorithmService;
import com.qczy.service.AlgorithmTaskService;
import com.qczy.service.CheckClassifyService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.Date;

@Service
public class CheckClassifyServiceImpl implements CheckClassifyService {

    @Autowired
    AlgorithmTaskService algorithmTaskService;

    @Autowired
    AlgorithmService algorithmService;

    public void check(CheckAndClassify checkAndClassify){
        AlgorithmEntity algorithmConfig = getAlgorithmConfig("第三方模型-检测");
        orderTask(checkAndClassify,algorithmConfig);
    }
    public void classify(CheckAndClassify checkAndClassify){
        AlgorithmEntity algorithmConfig = getAlgorithmConfig("第三方模型-分类");
        orderTask(checkAndClassify,algorithmConfig);
    }
    public AlgorithmEntity getAlgorithmConfig(String taskName){
        AlgorithmEntity config = algorithmService.getOne(new LambdaQueryWrapper<AlgorithmEntity>()
                .eq(AlgorithmEntity::getAlgorithmName, taskName));
        return config;
    }
    //装配并调用任务
    public void orderTask(CheckAndClassify checkAndClassify,AlgorithmEntity algorithm){
        AlgorithmTaskEntity algorithmTaskEntity = new AlgorithmTaskEntity();
        algorithmTaskEntity.setTaskName(algorithm.getAlgorithmName());
        algorithmTaskEntity.setTaskDesc(algorithm.getAlgorithmDesc());
        algorithmTaskEntity.setTaskInputName(checkAndClassify.getTaskName());
        insertTaskRecord(algorithmTaskEntity);



    }

    public void startCheckClassify(CheckAndClassify checkAndClassify){
        check(checkAndClassify);
        classify(checkAndClassify);
    }
    private void insertTaskRecord(AlgorithmTaskEntity taskEntity) {
        taskEntity.setCreateTime(new Date());
        algorithmTaskService.addTaskInfo(taskEntity);
    }

}
