package com.qczy.service.impl;

import cn.hutool.core.util.ObjectUtil;
import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.core.conditions.query.QueryWrapper;
import com.baomidou.mybatisplus.core.metadata.IPage;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import com.qczy.mapper.AlgorithmModelMapper;
import com.qczy.model.entity.AlgorithmModelEntity;
import com.qczy.model.entity.AlgorithmModelEntity;
import com.qczy.model.entity.AlgorithmTaskEntity;
import com.qczy.model.request.DeleteRequest;
import com.qczy.service.AlgorithmModelService;
import io.swagger.models.auth.In;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

/**
 * @author ：gwj
 * @date ：Created in 2024-08-27 11:09
 * @description：
 * @modified By：
 * @version: $
 */
@Service
public class AlgorithmModelServiceImpl extends ServiceImpl<AlgorithmModelMapper,AlgorithmModelEntity> implements AlgorithmModelService {

    @Autowired
    AlgorithmModelMapper algorithmModelMapper;
    @Autowired
    AlgorithmModelMapper algorithmTaskMapper;
    LambdaQueryWrapper<AlgorithmModelEntity> getQueryConditon(AlgorithmModelEntity modelEntity){
        LambdaQueryWrapper<AlgorithmModelEntity> algorithmModelEntityLambdaQueryWrapper = new LambdaQueryWrapper<AlgorithmModelEntity>()
                .eq(ObjectUtil.isNotEmpty(modelEntity.getModelId()),AlgorithmModelEntity::getModelId, modelEntity.getModelId())
                .eq(ObjectUtil.isNotEmpty(modelEntity.getModelName()),AlgorithmModelEntity::getModelName, modelEntity.getModelName())
                .eq(ObjectUtil.isNotEmpty(modelEntity.getModelBizType()),AlgorithmModelEntity::getModelBizType, modelEntity.getModelBizType())
                .eq(ObjectUtil.isNotEmpty(modelEntity.getModelReqType()),AlgorithmModelEntity::getModelReqType, modelEntity.getModelReqType());
        return algorithmModelEntityLambdaQueryWrapper;
    }


    private LambdaQueryWrapper<AlgorithmModelEntity> updateConditon(AlgorithmModelEntity algorithmModelEntity){
        LambdaQueryWrapper<AlgorithmModelEntity> algorithmModelEntityLambdaQueryWrapper = new LambdaQueryWrapper<AlgorithmModelEntity>()
                .eq(AlgorithmModelEntity::getModelId, algorithmModelEntity.getModelId());
        return algorithmModelEntityLambdaQueryWrapper;
    }

    @Override
    public boolean addModelInfo(AlgorithmModelEntity modelEnity) {
        int insert = algorithmModelMapper.insert(modelEnity);
        return insert > 0;
    }

    @Override
    public boolean updateModelInfo(AlgorithmModelEntity modelEntity) {
        int update = algorithmModelMapper.update(modelEntity,updateConditon(modelEntity));
        return update > 0;
    }

    @Override
    public boolean delModelInfo(DeleteRequest deleteRequest) {
        int rows = 0;
        for (int id : deleteRequest.getIds()) {

            int i = algorithmModelMapper.deleteById(id);
            rows+=i;
        }
        return rows == deleteRequest.getIds().length;
    }



    @Override
    public AlgorithmModelEntity getModelDetails(AlgorithmModelEntity modelEnity) {
        return algorithmModelMapper.selectOne(getQueryConditon(modelEnity));
    }

    @Override
    public Page<AlgorithmModelEntity> getModelPage(Page<AlgorithmModelEntity> pageParam, AlgorithmModelEntity modelEnity) {
        return algorithmModelMapper.selectPage(pageParam,getQueryConditon(modelEnity));
    }

    @Override
    public List<AlgorithmModelEntity> getModelList(AlgorithmModelEntity modelEnity) {
        return algorithmModelMapper.selectList(getQueryConditon(modelEnity));
    }


}
