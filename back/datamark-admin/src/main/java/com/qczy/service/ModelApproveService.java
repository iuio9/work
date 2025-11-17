package com.qczy.service;

import com.qczy.model.response.ModelTypeResponse;

import java.util.List;

/**
 * @Author: hh
 * @Version: 1.0
 * @Date: 2025/5/22 11:18
 * @Description:
 */
public interface ModelApproveService {

    /**
     * 审批通
     */
    int pass(Integer id);


    /**
     * 审不批通
     */
    int notPass(Integer id);



    /**
     * 根据类型获取模型列表
     */
    List<ModelTypeResponse> getModelTypeList(Integer modelWay);
}
