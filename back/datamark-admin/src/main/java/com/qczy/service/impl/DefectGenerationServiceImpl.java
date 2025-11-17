package com.qczy.service.impl;

import cn.hutool.json.JSONObject;
import cn.hutool.json.JSONUtil;
import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.qczy.common.annotation.MonitorProgress;
import com.qczy.config.ProgressContext;
import com.qczy.mapper.AlgorithmMapper;
import com.qczy.mapper.MarkInfoMapper;
import com.qczy.model.entity.*;
import com.qczy.model.response.DataDetailsResponse;
import com.qczy.service.AlgorithmModelService;
import com.qczy.service.AlgorithmTaskService;
import com.qczy.service.DefectGenerationService;
import com.qczy.service.FileService;
import com.qczy.task.ProgressListener;
import com.qczy.utils.HttpUtil;
import com.qczy.utils.JsonUtil;
import com.qczy.utils.ModelUtil;
import com.qczy.utils.TaskUtil;
import io.swagger.models.auth.In;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.scheduling.annotation.Async;
import org.springframework.stereotype.Service;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * @author ：gwj
 * @date ：Created in 2024-08-27 15:31
 * @description：
 * @modified By：
 * @version: $
 */
@Service
public class DefectGenerationServiceImpl implements DefectGenerationService {

    private final HttpUtil httpUtil;
    public DefectGenerationServiceImpl(HttpUtil httpUtil) {
        this.httpUtil = httpUtil;
    }

    @Autowired
    AlgorithmMapper algorithmMapper;

    @Override
    @Async
    @MonitorProgress
    public String startDefectGen(AlgorithmTaskEntity algorithmTaskEntity) {
        String algorithmId = algorithmTaskEntity.getAlgorithmId();
        AlgorithmEntity algorithmEntity = algorithmMapper.selectOne(new LambdaQueryWrapper<AlgorithmEntity>().eq(AlgorithmEntity::getId, algorithmId));
        String post = httpUtil.post(algorithmEntity.getUrl(), algorithmEntity.getParams());
        return post;
    }


}
