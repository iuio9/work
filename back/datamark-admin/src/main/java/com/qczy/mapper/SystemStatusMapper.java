package com.qczy.mapper;


import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.qczy.model.entity.ComputerInfoEntity;
import com.qczy.model.entity.SystemStatusEntity;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;

import java.util.List;
import java.util.Map;

@Mapper
public interface SystemStatusMapper extends BaseMapper<SystemStatusEntity> {
    public List<Map> statisticalBar(@Param("type") String type);
    public List<Map> statisticalPie(@Param("type") String type);
}

