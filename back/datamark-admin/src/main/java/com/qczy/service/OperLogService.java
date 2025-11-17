package com.qczy.service;

import com.baomidou.mybatisplus.core.metadata.IPage;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.qczy.model.entity.OperLogEntity;
import com.qczy.model.entity.RoleEntity;
import com.qczy.model.request.OperLogRequest;

/**
 * @Author: hh
 * @Version: 1.0
 * @Date: 2024/8/5 14:36
 * @Description:
 */
public interface OperLogService {


    /**
     * 查询【请填写功能名称】列表
     *
     * @param pageParam 分页信息，【请填写功能名称】
     * @param request   查询参数，【请填写功能名称】
     * @return 【请填写功能名称】集合
     */
    IPage<OperLogEntity> selectOperLogList(Page<OperLogEntity> pageParam, OperLogRequest request);


    OperLogEntity getOperLog(Integer operId);
}
