package com.qczy.service.impl;


import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.qczy.common.annotation.MonitorProgress;
import com.qczy.mapper.AlgorithmMapper;
import com.qczy.model.entity.*;

import com.qczy.service.StyleConvertService;
import com.qczy.utils.HttpUtil;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.scheduling.annotation.Async;
import org.springframework.stereotype.Service;

/**
 * @author ：gwj
 * @date ：Created in 2024-08-24 16:19
 * @description：
 * @modified By：
 * @version: $
 */
@Service
public class StyleConvertServiceImpl implements StyleConvertService {

    private final HttpUtil httpUtil;
    public StyleConvertServiceImpl(HttpUtil httpUtil) {
        this.httpUtil = httpUtil;
    }

    @Autowired
    AlgorithmMapper algorithmMapper;


    @Override
    @Async
    @MonitorProgress
    public String startStyleConvert(AlgorithmTaskEntity algorithmTaskEntity) {
        String algorithmId = algorithmTaskEntity.getAlgorithmId();
        AlgorithmEntity algorithmEntity = algorithmMapper.selectOne(new LambdaQueryWrapper<AlgorithmEntity>().eq(AlgorithmEntity::getId, algorithmId));
        String post = httpUtil.post(algorithmEntity.getUrl(), algorithmEntity.getParams());
        return post;
    }

}
