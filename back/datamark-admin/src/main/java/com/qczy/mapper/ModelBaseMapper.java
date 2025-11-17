package com.qczy.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.baomidou.mybatisplus.core.metadata.IPage;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.qczy.model.entity.ModelBaseEntity;
import com.qczy.model.request.ModelApplyForRequest;
import com.qczy.model.request.ModelApplyForRequestParam;
import com.qczy.model.response.ModelApplyForListResponse;
import org.apache.ibatis.annotations.Param;

import java.util.List;

/**
 * @Author: hh
 * @Version: 1.0
 * @Date: 2025/5/16 11:01
 * @Description:
 */
public interface ModelBaseMapper extends BaseMapper<ModelBaseEntity> {

    /**
     * 申请列表查询
     */
    IPage<ModelApplyForListResponse> list(Page<ModelApplyForListResponse> pageParam
            , @Param("param") ModelApplyForRequestParam requestParam
            , @Param("isAdminLogin") boolean isAdminLogin
            , @Param("userId") Integer userId);

    /**
     * 审批列表查询
     */
    IPage<ModelApplyForListResponse> approveList(Page<ModelApplyForListResponse> pageParam, @Param("param") ModelApplyForRequestParam requestParam);
}
