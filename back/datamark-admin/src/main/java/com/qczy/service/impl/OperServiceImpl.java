package com.qczy.service.impl;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.core.metadata.IPage;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.qczy.mapper.OperLogMapper;
import com.qczy.model.entity.OperLogEntity;
import com.qczy.model.entity.RoleEntity;
import com.qczy.model.request.OperLogRequest;
import com.qczy.service.OperLogService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

/**
 * @Author: hh
 * @Version: 1.0
 * @Date: 2024/8/5 14:37
 * @Description:
 */
@Service
public class OperServiceImpl implements OperLogService {

    @Autowired
    private OperLogMapper operLogMapper;


    /**
     * 查询【请填写功能名称】列表
     *
     * @param pageParam 分页信息，【请填写功能名称】
     * @param request   查询参数，【请填写功能名称】
     * @return 【请填写功能名称】
     */
    @Override
    public IPage<OperLogEntity> selectOperLogList(Page<OperLogEntity> pageParam, OperLogRequest request) {
        return operLogMapper.selectOperLogList(pageParam, request);
    }

    @Override
    public OperLogEntity getOperLog(Integer operId) {
        return operLogMapper.selectOne(
                new LambdaQueryWrapper<OperLogEntity>().eq(OperLogEntity::getOperId, operId));

    }


}
