package com.qczy.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.baomidou.mybatisplus.core.metadata.IPage;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.qczy.model.entity.DictDataEntity;
import com.qczy.model.entity.OperLogEntity;
import com.qczy.model.request.OperLogRequest;
import org.apache.ibatis.annotations.Param;

/**
 * @Author: hh
 * @Version: 1.0
 * @Date: 2024/8/5 12:25
 * @Description:
 */
public interface OperLogMapper extends BaseMapper<OperLogEntity> {


    IPage<OperLogEntity> selectOperLogList(Page<OperLogEntity> pageParam, @Param("req") OperLogRequest request);

}
