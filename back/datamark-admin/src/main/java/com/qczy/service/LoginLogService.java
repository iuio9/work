package com.qczy.service;

import com.baomidou.mybatisplus.core.metadata.IPage;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.qczy.model.entity.LoginLogEntity;
import com.qczy.model.entity.OperLogEntity;

/**
 * @Author: hh
 * @Version: 1.0
 * @Date: 2024/8/5 17:30
 * @Description:
 */
public interface LoginLogService {


    IPage<OperLogEntity> selectLoginLogList(Page<OperLogEntity> pageParam, LoginLogEntity loginLog);
}
