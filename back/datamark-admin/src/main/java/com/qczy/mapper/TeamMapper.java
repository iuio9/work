package com.qczy.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.baomidou.mybatisplus.core.metadata.IPage;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.qczy.model.entity.TeamEntity;
import com.qczy.model.response.TeamUserResponse;
import org.apache.ibatis.annotations.Param;

import java.util.List;

/**
 * @Author: hh
 * @Version: 1.0
 * @Date: 2025/2/24 9:54
 * @Description:
 */
public interface TeamMapper extends BaseMapper<TeamEntity> {
    IPage<TeamEntity> selectTeamList(Page<TeamEntity> pageParam,@Param("req") TeamEntity request);

    int deleteTeamGroupByIds(int[] ids);

    List<TeamUserResponse> getByTaskIdTeamList(Integer taskId);
    List<TeamUserResponse> getByTaskIdTeamList1(Integer taskId);
}
