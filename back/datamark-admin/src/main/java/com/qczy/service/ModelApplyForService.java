package com.qczy.service;

import com.baomidou.mybatisplus.core.metadata.IPage;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.baomidou.mybatisplus.extension.service.IService;
import com.qczy.common.result.Result;
import com.qczy.model.entity.ModelBaseEntity;
import com.qczy.model.entity.UserEntity;
import com.qczy.model.request.ModelApplyForRequest;
import com.qczy.model.request.ModelApplyForRequestParam;
import com.qczy.model.request.ModelBackFillRequest;
import com.qczy.model.response.ModelApplyForListResponse;

import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import java.util.List;

/**
 * @Author: hh
 * @Version: 1.0
 * @Date: 2025/5/16 11:24
 * @Description:
 */
public interface ModelApplyForService  {


    /**
     * 新增模型
     */
    ModelApplyForListResponse addModel(ModelApplyForRequest request);


    /**
     * 第三方模型申请列表
     */
    IPage<ModelApplyForListResponse> list(Page<ModelApplyForListResponse> pageParam, ModelApplyForRequestParam requestParam);

    /**
     *  第三方模型审批列表
     */
    IPage<ModelApplyForListResponse> approveList(Page<ModelApplyForListResponse> pageParam, ModelApplyForRequestParam requestParam);

    /**
     *  编辑
     */

    int editModel(ModelApplyForRequest request);

    /**
     * 生成pdf
     */
    void generatePad(Integer modelId, HttpServletRequest request, HttpServletResponse response);

    /**
     * 模型详情
     */
    ModelApplyForRequest modelDetails(Integer modelId);

    /**
     * 模型回填
     */
    int modelBackFill(ModelBackFillRequest request);

    /**
     *  提交审批
     */
    int submitApprove(ModelBaseEntity modelBaseEntity);


}
