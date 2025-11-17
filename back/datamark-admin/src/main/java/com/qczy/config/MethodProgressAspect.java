package com.qczy.config;

/**
 * @author ：gwj
 * @date ：Created in 2024-08-24 15:36
 * @description：
 * @modified By：
 * @version: $
 */
import cn.hutool.json.JSONUtil;
import com.alibaba.fastjson.JSON;
import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.core.conditions.query.QueryWrapper;
import com.qczy.common.constant.AssessConstants;
import com.qczy.common.constant.BizConstants;
import com.qczy.common.constant.ThirdAssessConstants;
import com.qczy.handler.MyWebSocketHandler;
import com.qczy.mapper.DataSonMapper;
import com.qczy.mapper.ModelAssessTaskMapper;
import com.qczy.model.entity.AlgorithmTaskEntity;
import com.qczy.model.entity.DataSonEntity;
import com.qczy.model.entity.ModelAssessTaskEntity;
import com.qczy.service.AlgorithmTaskService;
import com.qczy.service.DataSonService;
import com.qczy.task.ProgressListener;
import org.aspectj.lang.ProceedingJoinPoint;
import org.aspectj.lang.annotation.Around;
import org.aspectj.lang.annotation.Aspect;
import org.aspectj.lang.annotation.Pointcut;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.scheduling.annotation.Async;
import org.springframework.stereotype.Component;

import java.io.IOException;
import java.util.Date;
import java.util.HashMap;
import java.util.Map;

@Aspect
@Component
public class MethodProgressAspect {

    private static final Logger logger = LoggerFactory.getLogger(MethodProgressAspect.class);

    @Autowired
    AlgorithmTaskService algorithmTaskService;

    @Autowired
    ModelAssessTaskMapper modelAssessTaskMapper;

    @Autowired
    DataSonService dataSonService;

    @Autowired
    MyWebSocketHandler myWebSocketHandler;

    // 定义切入点，拦截标注了 @MonitorProgress 的方法
    @Pointcut("@annotation(com.qczy.common.annotation.MonitorProgress)")
    public void monitorProgressPointcut() {
        // 切入点签名
    }

    // 使用 @Around 环绕通知来监控方法进度
    @Around("monitorProgressPointcut()")
    @Async
    public Object monitorProgress(ProceedingJoinPoint joinPoint) throws Throwable {
        long startTime = System.currentTimeMillis();
        logger.info("Method {} started at {}", joinPoint.getSignature(), startTime);

        // 执行目标方法
        Object result;
        try {
            ProgressContext.setProgressListener(new ProgressListener() {
                @Override
                public void onProgress(AlgorithmTaskEntity taskEntity, int progress) {
                        // 处理进度更新，例如记录日志或发送消息
                    System.out.println("Progress for " + taskEntity.getDataSetId() + ": " + progress + "%");
                    taskEntity.setTaskStat("进行中");
                    taskEntity.setTaskProgress( progress + "%");
                    myWebSocketHandler.sendMessageToUser(BizConstants.TASK_PROGRESS,taskEntity.getTaskId()+"", JSONUtil.toJsonStr(taskEntity));
                    LambdaQueryWrapper<DataSonEntity> updateWrapper = new LambdaQueryWrapper<DataSonEntity>()
                            .eq(DataSonEntity::getSonId, taskEntity.getDataSetId());
                    if(progress==100){
                        taskEntity.setTaskStat("结束");
//                        myWebSocketHandler.disconnectUser(BizConstants.TASK_PROGRESS,taskEntity.getTaskId()+"");
                    }
                    algorithmTaskService.updateById(taskEntity);
                }

                @Override
                public void onAssessProgress(ModelAssessTaskEntity taskEntity, int progressPercentage) {
                    taskEntity.setTaskStatus(ThirdAssessConstants.IN_PROGRESS);
                    taskEntity.setTaskProgress( progressPercentage + "%");
                    myWebSocketHandler.sendMessageToUser(BizConstants.ASSESS_PROGRESS,taskEntity.getId()+"", JSONUtil.toJsonStr(taskEntity));
                    if(progressPercentage==100){
                        taskEntity.setTaskStatus(ThirdAssessConstants.COMPLETED);
//                        myWebSocketHandler.disconnectUser(BizConstants.ASSESS_PROGRESS,taskEntity.getId()+"");
                    }
                    modelAssessTaskMapper.updateById(taskEntity);
                }
            });
            result = joinPoint.proceed();
        } catch (Throwable throwable) {
            logger.error("Method {} threw an exception: {}", joinPoint.getSignature(), throwable.getMessage());
            throw throwable;
        }

        long endTime = System.currentTimeMillis();
        logger.info("Method {} finished at {}, taking {} ms", joinPoint.getSignature(), endTime, (endTime - startTime));

        return result;
    }
}

