package com.qczy.service;

import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.baomidou.mybatisplus.extension.service.IService;
import com.qczy.model.entity.AlgorithmModelEntity;
import com.qczy.model.entity.AlgorithmTaskEntity;
import com.qczy.model.request.DeleteRequest;

import java.util.List;

/**
 * @author ：gwj
 * @date ：Created in 2024-08-27 10:44
 * @description：
 * @modified By：
 * @version: $
 */
public interface AlgorithmModelService extends IService<AlgorithmModelEntity> {

    boolean addModelInfo(AlgorithmModelEntity taskEntity);
    boolean updateModelInfo(AlgorithmModelEntity taskEntity);


    boolean delModelInfo(DeleteRequest taskEntity);

    AlgorithmModelEntity getModelDetails(AlgorithmModelEntity taskEntity);

    Page<AlgorithmModelEntity> getModelPage(Page<AlgorithmModelEntity> pageParam, AlgorithmModelEntity algorithmModelEntity);

    List<AlgorithmModelEntity> getModelList(AlgorithmModelEntity taskEntity);

}
