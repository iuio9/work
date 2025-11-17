package com.qczy.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.baomidou.mybatisplus.core.metadata.IPage;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.qczy.model.entity.ModelAssessTaskEntity;
import com.qczy.model.response.ModelAssessResponse;
import com.qczy.model.response.ModelReportResponse;
import org.apache.ibatis.annotations.Param;

/**
 * @Author: hh
 * @Version: 1.0
 * @Date: 2025/6/5 10:57
 * @Description:
 */
public interface ModelAssessTaskMapper extends BaseMapper<ModelAssessTaskEntity> {

    IPage<ModelAssessResponse> listPage(Page<ModelAssessResponse> pageParam,@Param("req") ModelAssessTaskEntity modelAssessTaskEntity);

    /**
     *
     */
    IPage<ModelReportResponse> reportListPage(Page<ModelReportResponse> pageParam);
}
