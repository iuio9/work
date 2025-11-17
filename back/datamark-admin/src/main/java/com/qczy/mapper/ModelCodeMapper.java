package com.qczy.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.qczy.model.entity.ModelCodeEntity;
import org.apache.ibatis.annotations.Select;

import java.util.List;


public interface ModelCodeMapper extends BaseMapper<ModelCodeEntity> {

    /**
     *  批量新增
     * @param codeEntityList 集合
     */
    void batchInsertModelCode(List<ModelCodeEntity> codeEntityList);



    Page<ModelCodeEntity> modelCodeList(Page<ModelCodeEntity> pageParam, Integer modelId);
}
