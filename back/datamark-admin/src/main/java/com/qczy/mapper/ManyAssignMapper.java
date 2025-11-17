package com.qczy.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.baomidou.mybatisplus.core.metadata.IPage;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.qczy.model.entity.ManyAssignEntity;
import com.qczy.model.response.ManyReceiveListResponse;
import com.qczy.model.response.ViewProgressResponse;
import org.apache.ibatis.annotations.Param;

import java.util.List;

/**
 * @Author: hh
 * @Version: 1.0
 * @Date: 2025/2/27 9:49
 * @Description:
 */
public interface ManyAssignMapper extends BaseMapper<ManyAssignEntity> {

    IPage<ViewProgressResponse> viewProgress(Page<ViewProgressResponse> pageParam, @Param("id") Integer id);

    IPage<ManyReceiveListResponse> getMyReceiveList(Page<ManyReceiveListResponse> pageParam, @Param("userId") Integer userId);

    // 批量插入 ManyAssignEntity
    void insertBatch(List<ManyAssignEntity> entities);

}
