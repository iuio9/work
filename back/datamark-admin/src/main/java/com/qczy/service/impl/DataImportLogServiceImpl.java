package com.qczy.service.impl;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.baomidou.mybatisplus.core.metadata.IPage;
import com.baomidou.mybatisplus.core.toolkit.ObjectUtils;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import com.qczy.mapper.DataImportLogMapper;
import com.qczy.mapper.FileMapper;
import com.qczy.model.entity.DataImportLogEntity;
import com.qczy.model.entity.FileEntity;
import com.qczy.model.response.DataImportLogResponse;
import com.qczy.model.response.DataResponse;
import com.qczy.model.response.FileDetailsResponse;
import com.qczy.service.DataImportLogService;
import com.qczy.utils.StringUtils;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.Collections;
import java.util.List;

/**
 * @Author: hh
 * @Version: 1.0
 * @Date: 2024/8/9 15:42
 * @Description:
 */
@Service
public class DataImportLogServiceImpl extends ServiceImpl<DataImportLogMapper, DataImportLogEntity> implements DataImportLogService {

    @Autowired
    private DataImportLogMapper dataImportLogMapper;

    @Autowired
    private FileMapper fileMapper;


    @Override
    public List<DataImportLogResponse> selectImportList(long sonId) {
        return dataImportLogMapper.selectImportList(sonId);
    }

    @Override
    public IPage<FileDetailsResponse> selectImportFileList(Page<FileDetailsResponse> pageParam, Integer id) {
        DataImportLogEntity importLogEntity = dataImportLogMapper.selectById(id);
        if (ObjectUtils.isEmpty(importLogEntity)) {
            throw new RuntimeException("对象数据不存在！");
        }
        if (!StringUtils.isEmpty(importLogEntity.getFileIds())){
            return dataImportLogMapper.selectImportFileList(pageParam,importLogEntity.getFileIds());
        }
        return null;
    }
}
