package com.qczy.service.impl;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.core.conditions.query.QueryWrapper;
import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import com.qczy.mapper.ModelAssessTaskMapper;
import com.qczy.mapper.ModelBaseMapper;
import com.qczy.mapper.ModelCodeMapper;
import com.qczy.mapper.ModelConfigureMapper;
import com.qczy.model.entity.ModelAssessTaskEntity;
import com.qczy.model.entity.ModelBaseEntity;
import com.qczy.model.entity.ModelCodeEntity;
import com.qczy.model.entity.ModelConfigureEntity;
import com.qczy.model.request.DeleteRequest;
import com.qczy.service.ModelBaseService;
import com.qczy.utils.StringUtils;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import org.springframework.util.CollectionUtils;

import java.io.File;
import java.util.List;
import java.util.stream.Collectors;

/**
 * @Author: hh
 * @Version: 1.0
 * @Date: 2025/5/16 11:14
 * @Description:
 */
@Service
public class ModelBaseServiceImpl extends ServiceImpl<ModelBaseMapper, ModelBaseEntity> implements ModelBaseService {

    @Autowired
    private ModelBaseMapper modelBaseMapper;

    @Autowired
    private ModelAssessTaskMapper modelAssessTaskMapper;

    @Autowired
    private ModelConfigureMapper modelConfigureMapper;
    @Autowired
    private ModelCodeMapper modelCodeMapper;


    @Override
    public int delModel(DeleteRequest request) {
        return modelBaseMapper.deleteById(request.getId());
    }

    @Override
    public boolean isModelUse(Integer modelId) {
        return modelAssessTaskMapper.selectCount(
                new LambdaQueryWrapper<ModelAssessTaskEntity>()
                        .eq(ModelAssessTaskEntity::getModelBaseId, modelId)
        ) > 0;
    }

    @Override
    public int deleteFile(DeleteRequest request) {
        ModelConfigureEntity modelConfigureEntity = modelConfigureMapper.selectOne(
                new LambdaQueryWrapper<ModelConfigureEntity>()
                        .eq(ModelConfigureEntity::getModelBaseId, request.getModelId())
        );

        if (modelConfigureEntity != null && !StringUtils.isEmpty(request.getServerKey())) {
            switch (request.getServerKey()) {
                case "modelInterfaceDesc":
                    if (!StringUtils.isEmpty(modelConfigureEntity.getModelInterfaceDesc())) {
                        new File(modelConfigureEntity.getModelInterfaceDesc()).delete();
                        modelConfigureEntity.setModelInterfaceDesc("");
                    }
                    break;
                case "modelCase":
                    if (!StringUtils.isEmpty(modelConfigureEntity.getModelCase())) {
                        new File(modelConfigureEntity.getModelCase()).delete();
                        modelConfigureEntity.setModelCase("");
                    }
                    break;
                case "modelAlgorithmCode":
                    if (!StringUtils.isEmpty(modelConfigureEntity.getModelAlgorithmCode())) {
                        new File(modelConfigureEntity.getModelAlgorithmCode()).delete();
                        modelConfigureEntity.setModelAlgorithmCode("");
                        //清空当前模型的编码
                        List<ModelCodeEntity> list = modelCodeMapper.selectList(
                                new LambdaQueryWrapper<ModelCodeEntity>()
                                        .eq(ModelCodeEntity::getModelBaseId, modelConfigureEntity.getModelBaseId())
                        );
                        if (!CollectionUtils.isEmpty(list)) {
                            List<Integer> ids = list.stream().map(ModelCodeEntity::getId).collect(Collectors.toList());
                            modelCodeMapper.deleteBatchIds(ids);
                        }
                    }
                    break;
                case "testCase":
                    if (!StringUtils.isEmpty(modelConfigureEntity.getTestCase())) {
                        new File(modelConfigureEntity.getTestCase()).delete();
                        modelConfigureEntity.setTestCase("");
                    }
                    break;
                case "modelTrainCode":
                    if (!StringUtils.isEmpty(modelConfigureEntity.getModelTrainCode())) {
                        new File(modelConfigureEntity.getModelTrainCode()).delete();
                        modelConfigureEntity.setModelTrainCode("");
                    }
                    break;
            }

            // 只有当modelConfigureEntity不为null时才更新
            return modelConfigureMapper.updateById(modelConfigureEntity);
        }

        // 如果modelConfigureEntity为null或serverKey为空，返回0表示未更新任何记录
        return 1;
    }

    @Override
    public boolean isModelNameRepeat(String modelName, Integer id) {
        if (StringUtils.isEmpty(modelName)) {
            return false;
        }
        QueryWrapper<ModelBaseEntity> queryWrapper = new QueryWrapper<>();
        queryWrapper.eq("model_name", modelName);
        if (id != null) {
            queryWrapper.ne("id", id);
        }
        return modelBaseMapper.selectCount(queryWrapper) > 0;
    }

}
