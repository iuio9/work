package com.qczy.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.baomidou.mybatisplus.core.metadata.IPage;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.qczy.model.entity.AlgorithmTaskEntity;
import com.qczy.model.entity.DictDataEntity;
import org.apache.ibatis.annotations.Param;

/**
 * @Author: hh
 * @Version: 1.0
 * @Date: 2024/8/2 16:52
 * @Description:
 */
public interface AlgorithmTaskMapper extends BaseMapper<AlgorithmTaskEntity> {


    Page<AlgorithmTaskEntity> getTaskPage(Page<AlgorithmTaskEntity> pageParam, @Param("req") AlgorithmTaskEntity taskEntity);
}
