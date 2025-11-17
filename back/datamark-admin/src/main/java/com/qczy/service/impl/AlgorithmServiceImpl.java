package com.qczy.service.impl;

import cn.hutool.core.util.ObjectUtil;
import cn.hutool.json.JSONUtil;
import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import com.qczy.common.result.Result;
import com.qczy.mapper.AlgorithmMapper;
import com.qczy.model.entity.AlgorithmEntity;
import com.qczy.service.AlgorithmService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

/**
 * @author ：gwj
 * @date ：Created in 2024-08-27 11:09
 * @description：
 * @modified By：
 * @version: $
 */
@Service
public class AlgorithmServiceImpl extends ServiceImpl<AlgorithmMapper, AlgorithmEntity> implements AlgorithmService {

    @Autowired
    AlgorithmMapper algorithmMapper;
    LambdaQueryWrapper<AlgorithmEntity> getQueryConditon(AlgorithmEntity algorithmEntity){
        LambdaQueryWrapper<AlgorithmEntity> algorithmEntityLambdaQueryWrapper = new LambdaQueryWrapper<AlgorithmEntity>()
                .eq(ObjectUtil.isNotEmpty(algorithmEntity.getId()),AlgorithmEntity::getId, algorithmEntity.getId())
                .eq(ObjectUtil.isNotEmpty(algorithmEntity.getModelId()),AlgorithmEntity::getModelId, algorithmEntity.getModelId())
                .eq(ObjectUtil.isNotEmpty(algorithmEntity.getAlgorithmName()),AlgorithmEntity::getAlgorithmName, algorithmEntity.getAlgorithmName());
        return algorithmEntityLambdaQueryWrapper;
    }


    @Override
    public boolean addExampleInfo(AlgorithmEntity entity) {
        int insert = algorithmMapper.insert(entity);
        return insert > 0;
    }

    @Override
    public boolean editExampleInfo(AlgorithmEntity entity) {
        int update = algorithmMapper.updateById(entity);
        return update > 0;
    }

    @Override
    public boolean delExampleInfo(AlgorithmEntity entity) {
        int del = algorithmMapper.deleteById(entity);
        return del > 0;
    }

    @Override
    public AlgorithmEntity getExampleDetails(AlgorithmEntity entity) {
        return algorithmMapper.selectOne(getQueryConditon(entity));
    }

    @Override
    public Page<AlgorithmEntity> getExamplePage(Page<AlgorithmEntity> pageParam, AlgorithmEntity entity) {
        return algorithmMapper.selectPage(pageParam,getQueryConditon(entity));
    }

    @Override
    public List<AlgorithmEntity> getExampleList(AlgorithmEntity entity) {
        List<AlgorithmEntity> algorithmEntities = algorithmMapper.selectList(getQueryConditon(entity));
        for (AlgorithmEntity algorithmEntity : algorithmEntities) {
            String params = algorithmEntity.getParams();
            String responseParams = algorithmEntity.getResponseParams();
            if (JSONUtil.isJsonArray(params)) {
                List<Map> maps = JSONUtil.toList(JSONUtil.parseArray(params), Map.class);
                // 使用 Stream 筛选出 map 中键为 "type" 且值为 "text" 的 map
                List<Map> filteredList = maps.stream()
                        .filter(map -> "text".equals(map.get("type")) || "select".equals(map.get("type")))  // 筛选条件
                        .collect(Collectors.toList());

                algorithmEntity.setParamsMap(filteredList);

            }
            if (JSONUtil.isJsonArray(responseParams)) {
                List<Map> maps = JSONUtil.toList(JSONUtil.parseArray(responseParams), Map.class);
                // 使用 Stream 筛选出 map 中键为 "type" 且值为 "text或者select" 的 map
                List<Map> filteredList = maps.stream()
                        .filter(map -> "text".equals(map.get("type")) || "select".equals(map.get("type")))  // 筛选条件
                        .collect(Collectors.toList());

                algorithmEntity.setResponseParamsMap(filteredList);

            }
        }

        return algorithmEntities;
    }
}
