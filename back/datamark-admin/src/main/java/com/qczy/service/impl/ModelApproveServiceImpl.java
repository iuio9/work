package com.qczy.service.impl;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.qczy.mapper.ModelBaseMapper;
import com.qczy.model.entity.ModelBaseEntity;
import com.qczy.model.response.ModelTypeResponse;
import com.qczy.service.ModelApproveService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import org.springframework.util.CollectionUtils;

import java.util.Collections;
import java.util.List;
import java.util.stream.Collectors;

/**
 * @Author: hh
 * @Version: 1.0
 * @Date: 2025/5/22 11:18
 * @Description:
 */
@Service
public class ModelApproveServiceImpl implements ModelApproveService {


    @Autowired
    private ModelBaseMapper modelBaseMapper;

    @Override
    public int pass(Integer id) {
        ModelBaseEntity modelBaseEntity = modelBaseMapper.selectById(id);
        if (modelBaseEntity == null) {
            return 0;
        }
        modelBaseEntity.setApplyForStatus(3);
        modelBaseEntity.setApproveStatus(2);
        return modelBaseMapper.updateById(modelBaseEntity);
    }

    @Override
    public int notPass(Integer id) {
        ModelBaseEntity modelBaseEntity = modelBaseMapper.selectById(id);
        if (modelBaseEntity == null) {
            return 0;
        }
        modelBaseEntity.setApplyForStatus(4);
        modelBaseEntity.setApproveStatus(3);
        return modelBaseMapper.updateById(modelBaseEntity);
    }

    @Override
    public List<ModelTypeResponse> getModelTypeList(Integer modelWay) {
        List<ModelBaseEntity> modelBaseList = modelBaseMapper.selectList(
                new LambdaQueryWrapper<ModelBaseEntity>()
                        .eq(ModelBaseEntity::getModelWay, modelWay)
                        .eq(ModelBaseEntity::getApplyForStatus, 3) //审核通过
        );

        return modelBaseList.stream()
                .map(ModelTypeResponse::new)
                .collect(Collectors.toList());

    }
}
