package com.qczy.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.baomidou.mybatisplus.core.metadata.IPage;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.qczy.model.entity.LoginLogEntity;
import com.qczy.model.entity.OperLogEntity;
import org.apache.ibatis.annotations.Param;

/**
 * @Author: hh
 * @Version: 1.0
 * @Date: 2024/8/5 16:49
 * @Description:
 */
public interface LoginLogMapper extends BaseMapper<LoginLogEntity> {

    IPage<OperLogEntity> selectLoginLogList(Page<OperLogEntity> pageParam, @Param("req") LoginLogEntity loginLog);
}
