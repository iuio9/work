package com.qczy.service;

import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.baomidou.mybatisplus.extension.service.IService;
import com.qczy.model.entity.AlgorithmEntity;

import java.util.List;

/**
 * @author ：gwj
 * @date ：Created in 2024-08-27 10:44
 * @description：
 * @modified By：
 * @version: $
 */
public interface AlgorithmService extends IService<AlgorithmEntity> {

    boolean addExampleInfo(AlgorithmEntity taskEntity);
    boolean editExampleInfo(AlgorithmEntity taskEntity);

    boolean delExampleInfo(AlgorithmEntity taskEntity);

    AlgorithmEntity getExampleDetails(AlgorithmEntity taskEntity);

    Page<AlgorithmEntity> getExamplePage(Page<AlgorithmEntity> pageParam, AlgorithmEntity algorithmEntity);

    List<AlgorithmEntity> getExampleList(AlgorithmEntity taskEntity);

}
