package com.qczy.service.impl;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.qczy.common.excel.ExcelUtils;
import com.qczy.controller.dataupload.MarkFileUploadController;
import com.qczy.mapper.ModelCodeMapper;
import com.qczy.model.entity.LabelEntity;
import com.qczy.model.entity.ModelCodeEntity;
import com.qczy.service.ModelCodeService;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import org.springframework.util.CollectionUtils;

import java.io.File;
import java.util.Collections;
import java.util.List;

@Service
public class ModelCodeServiceImpl implements ModelCodeService {


    private static final Logger log = LoggerFactory.getLogger(MarkFileUploadController.class);


    @Autowired
    private ModelCodeMapper modelCodeMapper;


    @Override
    public void analysisXlsxCode(String filePath, Integer modelId) {
        try {
            File file = new File(filePath);
            if (!file.exists()) {
                return;
            }

            // 读取 Excel 文件内容
            List<ModelCodeEntity> codeEntityList = ExcelUtils.readFile(file, ModelCodeEntity.class);
            for (ModelCodeEntity modelCodeEntity : codeEntityList) {
                System.out.println(modelCodeEntity);
            }
            if (CollectionUtils.isEmpty(codeEntityList)) {
                return;
            }
            for (ModelCodeEntity modelCodeEntity : codeEntityList) {
                modelCodeEntity.setModelBaseId(modelId);
            }

            // 执行批量新增
            modelCodeMapper.batchInsertModelCode(codeEntityList);


        } catch (Exception e) {
            // 只打印异常就行
            log.error("标签导入失败：{}", e.getMessage());
        }


    }

    @Override
    public Page<ModelCodeEntity> modelCodeList(Page<ModelCodeEntity> pageParam, Integer modelId) {
       /* return modelCodeMapper.selectList(
                new LambdaQueryWrapper<ModelCodeEntity>()
                        .eq(ModelCodeEntity::getModelBaseId, modelId)
        );*/
        return modelCodeMapper.modelCodeList(pageParam,modelId);
    }
}
