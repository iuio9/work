package com.qczy.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.baomidou.mybatisplus.core.metadata.IPage;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.qczy.model.entity.ManyMarkEntity;
import com.qczy.model.response.ManyCreateListResponse;
import org.apache.ibatis.annotations.Param;

/**
 * @Author: hh
 * @Version: 1.0
 * @Date: 2025/2/26 16:28
 * @Description:
 */
public interface ManyMarkMapper extends BaseMapper<ManyMarkEntity> {

    IPage<ManyCreateListResponse> getMyCreateTaskList(Page<ManyCreateListResponse> pageParam, @Param("userId") Integer userId);
}
