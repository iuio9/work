package com.qczy.service;

import com.baomidou.mybatisplus.core.metadata.IPage;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.qczy.model.entity.ModelCodeEntity;

import java.util.List;

public interface ModelCodeService {

    /**
     *  根据文件路径解析xlsx文件
     */
    void analysisXlsxCode(String filePath,Integer modelId);

    /**
     *  根据模型id获取算法编码
     */
    IPage<ModelCodeEntity> modelCodeList(Page<ModelCodeEntity> pageParam,Integer modelId);

}
