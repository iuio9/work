package com.qczy.service.impl;

import com.baomidou.mybatisplus.core.metadata.IPage;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.qczy.mapper.LoginLogMapper;
import com.qczy.mapper.OperLogMapper;
import com.qczy.model.entity.LoginLogEntity;
import com.qczy.model.entity.OperLogEntity;
import com.qczy.service.LoginLogService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

/**
 * @Author: hh
 * @Version: 1.0
 * @Date: 2024/8/5 17:31
 * @Description:
 */
@Service
public class LoginLogServiceImpl implements LoginLogService {

    @Autowired
    private LoginLogMapper loginLogMapper;

    @Override
    public IPage<OperLogEntity> selectLoginLogList(Page<OperLogEntity> pageParam, LoginLogEntity loginLog) {
        return loginLogMapper.selectLoginLogList(pageParam,loginLog);
    }
}
