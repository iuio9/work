package com.qczy.task;

import com.qczy.common.constant.AssessConstants;
import com.qczy.model.entity.AlgorithmTaskEntity;
import com.qczy.model.entity.DataSonEntity;
import com.qczy.model.entity.ModelAssessTaskEntity;

/**
 * @author ：gwj
 * @date ：Created in 2024-08-24 15:44
 * @description：1. 创建进度监听接口
 * @modified By：
 * @version: $
 */
public interface ProgressListener {
    void onProgress(AlgorithmTaskEntity taskEntity, int progressPercentage);
    void onAssessProgress(ModelAssessTaskEntity taskEntity, int progressPercentage);
}
