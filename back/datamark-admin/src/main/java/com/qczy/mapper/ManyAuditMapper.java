package com.qczy.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.baomidou.mybatisplus.core.metadata.IPage;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.qczy.model.entity.ManyAuditEntity;
import com.qczy.model.response.ExamineTeamInfoResponse;
import com.qczy.model.response.ManyAuditDetailsResponse;
import com.qczy.model.response.ManyReceiveListResponse;
import org.apache.ibatis.annotations.Param;

import java.util.List;

/**
 * @Author: hh
 * @Version: 1.0
 * @Date: 2025/3/17 15:58
 * @Description:
 */
public interface ManyAuditMapper extends BaseMapper<ManyAuditEntity> {

    IPage<ManyAuditDetailsResponse> examineDetails(Page<ManyAuditDetailsResponse> pageParam, @Param("taskId") Integer taskId);

    IPage<ManyReceiveListResponse> myExamineTaskList(Page<ManyReceiveListResponse> pageParam,@Param("userId") Integer userId);

    ExamineTeamInfoResponse examineTeamInfo(Integer id);


    // 批量修改
    int batchUpdateManyAuditEntities(List<ManyAuditEntity> entities);

    // 批量查询
    List<ManyAuditEntity> selectManyAuditByIds(List<Integer> ids);

}
