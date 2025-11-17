package com.qczy.service.impl;

import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import com.qczy.mapper.ModelConfigureMapper;
import com.qczy.model.entity.ModelConfigureEntity;
import com.qczy.service.ModelConfigureService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

/**
 * @Author: hh
 * @Version: 1.0
 * @Date: 2025/5/16 11:16
 * @Description:
 */
@Service
public class ModelConfigureServiceImpl extends ServiceImpl<ModelConfigureMapper, ModelConfigureEntity> implements ModelConfigureService {

    @Autowired
    private ModelConfigureMapper modelConfigureMapper;

}
