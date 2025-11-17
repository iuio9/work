package com.qczy.service;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.baomidou.mybatisplus.extension.service.IService;
import com.qczy.model.entity.AlgorithmModelEntity;
import com.qczy.model.entity.AlgorithmTaskEntity;
import com.qczy.model.entity.DataSonEntity;
import com.qczy.model.entity.FileEntity;
import com.qczy.model.request.SegmentParams;
import com.qczy.model.request.TrainEntity;

import java.util.List;
import java.util.Map;

/**
 * @author ：gwj
 * @date ：Created in 2024-08-27 10:44
 * @description：
 * @modified By：
 * @version: $
 */
public interface AlgorithmTaskService  extends IService<AlgorithmTaskEntity> {

    boolean addTaskInfo(AlgorithmTaskEntity taskEntity);
    boolean editTaskInfo(AlgorithmTaskEntity taskEntity);

    boolean delTaskInfo(AlgorithmTaskEntity taskEntity);

    AlgorithmTaskEntity getDataDetails(AlgorithmTaskEntity taskEntity);

    Page<AlgorithmTaskEntity> getTaskPage(Page<AlgorithmTaskEntity> pageParam, AlgorithmTaskEntity algorithmTaskEntity);

    List<AlgorithmTaskEntity> getTaskList(AlgorithmTaskEntity taskEntity);

    Map<String, Object> startSegment(SegmentParams segmentParams);

    void submitTaskInfo(AlgorithmTaskEntity algorithmTaskEntity);

    Page<FileEntity> getTaskResult(Page<FileEntity> pageParam,AlgorithmTaskEntity algorithmTaskEntity);
    Page<AlgorithmTaskEntity> getAssessLst(AlgorithmModelEntity modelEnity,Page<AlgorithmTaskEntity> pageParam);
}

