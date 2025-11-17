package com.qczy.service;

import com.baomidou.mybatisplus.core.metadata.IPage;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.baomidou.mybatisplus.extension.service.IService;
import com.qczy.model.entity.TeamEntity;
import com.qczy.model.response.ManyCreateListResponse;
import com.qczy.model.response.ViewProgressResponse;

import java.util.List;

/**
 * @Author: hh
 * @Version: 1.0
 * @Date: 2025/2/24 9:59
 * @Description:
 */
public interface TeamService extends IService<TeamEntity> {

    // 查询
    IPage<TeamEntity> selectTeamList(Page<TeamEntity> pageParam, TeamEntity request);

    // 新增
    int insertTeam(TeamEntity team);

    // 修改
    int updateTeam(TeamEntity team);

    // 删除
    int deleteTeamGroupByIds(int[] ids);

    // 我发起的任务-查看进度
    IPage<ViewProgressResponse> viewProgress(Page<ViewProgressResponse> pageParam, Integer id);

     // 检测当前团队，是否在执行多人任务
    boolean isManyTask(Integer id);

    // 判断成员是否重复
    boolean hasNoDuplicates(List<Integer> list);
}
