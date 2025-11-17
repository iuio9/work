package com.qczy.service.impl;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.core.conditions.query.QueryWrapper;
import com.baomidou.mybatisplus.core.metadata.IPage;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import com.qczy.mapper.*;
import com.qczy.model.entity.ManyAssignEntity;
import com.qczy.model.entity.ManyMarkEntity;
import com.qczy.model.entity.TeamEntity;
import com.qczy.model.entity.TeamUserEntity;
import com.qczy.model.request.AddTeamUser;
import com.qczy.model.response.ViewProgressResponse;
import com.qczy.service.TeamService;
import com.qczy.utils.CurrentLoginUserUtils;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.util.CollectionUtils;
import org.springframework.util.ObjectUtils;

import java.util.ArrayList;
import java.util.Date;
import java.util.List;
import java.util.stream.Collectors;

/**
 * @Author: hh
 * @Version: 1.0
 * @Date: 2025/2/24 9:59
 * @Description:
 */
@Service
public class TeamServiceImpl extends ServiceImpl<TeamMapper, TeamEntity> implements TeamService {

    @Autowired
    private TeamMapper teamMapper;
    @Autowired
    private CurrentLoginUserUtils currentLoginUserUtils;
    @Autowired
    private TeamUserMapper teamUserMapper;
    @Autowired
    private ManyMarkMapper manyMarkMapper;
    @Autowired
    private ManyAssignMapper manyAssignMapper;
    @Autowired
    private UserMapper userMapper;
    @Autowired
    private MarkInfoMapper markInfoMapper;


    @Override
    public IPage<ViewProgressResponse> viewProgress(Page<ViewProgressResponse> pageParam, Integer id) {
        ManyMarkEntity manyMarkEntity = manyMarkMapper.selectById(id);
        if (ObjectUtils.isEmpty(manyMarkEntity)) {
            return null;
        }
        Integer teamId = manyMarkEntity.getTeamId();
        TeamEntity teamEntity = teamMapper.selectById(teamId);
        if (ObjectUtils.isEmpty(teamEntity)) {
            return null;
        }

        return manyAssignMapper.viewProgress(pageParam, id);
    }

    @Override
    public boolean isManyTask(Integer id) {
        TeamEntity teamEntity = teamMapper.selectById(id);
        if (ObjectUtils.isEmpty(teamEntity)) {
            return true;
        }

        /*return manyMarkMapper.selectCount(
                new LambdaQueryWrapper<ManyMarkEntity>()
                        .ne(ManyMarkEntity::getTaskState, 8)
                        .eq(ManyMarkEntity::getTeamId, teamEntity.getId())
                        .or()
                        .eq(ManyMarkEntity::getAuditTeamId, teamEntity.getId())
        ) == 0;*/
        return manyMarkMapper.selectCount(
                new LambdaQueryWrapper<ManyMarkEntity>()
                        // 核心条件：存在任何状态不是8的任务
                        .ne(ManyMarkEntity::getTaskState, 8)
                        // 关联当前团队的任务（团队ID或审核团队ID匹配）
                        .and(wrapper -> wrapper
                                .eq(ManyMarkEntity::getTeamId, teamEntity.getId())
                                .or()
                                .eq(ManyMarkEntity::getAuditTeamId, teamEntity.getId())
                        )
        ) == 0;
    }


    @Override
    public IPage<TeamEntity> selectTeamList(Page<TeamEntity> pageParam, TeamEntity request) {
        IPage<TeamEntity> teamList = teamMapper.selectTeamList(pageParam, request);
        if (CollectionUtils.isEmpty(teamList.getRecords())) {
            return null;
        }
        for (TeamEntity teamEntity : teamList.getRecords()) {
            List<TeamUserEntity> userEntityList = teamUserMapper.selectList(
                    new LambdaQueryWrapper<TeamUserEntity>()
                            .eq(TeamUserEntity::getTeamId, teamEntity.getId())
            );

            List<AddTeamUser> data = new ArrayList<>();
            for (TeamUserEntity teamUserEntity : userEntityList) {
                AddTeamUser addTeamUser = new AddTeamUser();
                addTeamUser.setUserId(teamUserEntity.getUserId());
                addTeamUser.setRemark(teamUserEntity.getRemark());
                data.add(addTeamUser);
            }

            teamEntity.setUserList(data);
        }
        return teamList;
    }

    @Transactional
    @Override
    public int insertTeam(TeamEntity team) {
        team.setCreateTime(new Date());
        // 获取创建人id
        team.setCreator(currentLoginUserUtils.getCurrentLoginUserId());
        int result = teamMapper.insert(team);
        List<AddTeamUser> userList = team.getUserList();
        for (AddTeamUser addTeamUser : userList) {
            TeamUserEntity teamUserEntity = new TeamUserEntity();
            teamUserEntity.setTeamId(team.getId());
            teamUserEntity.setUserId(addTeamUser.getUserId());
            teamUserEntity.setRemark(addTeamUser.getRemark());
            teamUserMapper.insert(teamUserEntity);
        }
        return result;
    }


    @Transactional
    @Override
    public int updateTeam(TeamEntity team) {
        team.setUpdateTime(new Date());
        // 获取创建人id
        team.setCreator(currentLoginUserUtils.getCurrentLoginUserId());
        int result = teamMapper.updateById(team);
        // 先删除在新增
        List<TeamUserEntity> teamUserEntities = teamUserMapper.selectList(
                new LambdaQueryWrapper<TeamUserEntity>()
                        .eq(TeamUserEntity::getTeamId, team.getId()));
        List<Integer> teamUserIds =
                teamUserEntities
                        .stream()
                        .map(TeamUserEntity::getId)
                        .collect(Collectors.toList());
        if (!CollectionUtils.isEmpty(teamUserIds)) {
            teamUserMapper.deleteBatchIds(teamUserIds);
        }


        // 再次执行新增
        List<AddTeamUser> userList = team.getUserList();
        for (AddTeamUser addTeamUser : userList) {
            TeamUserEntity teamUserEntity = new TeamUserEntity();
            teamUserEntity.setTeamId(team.getId());
            teamUserEntity.setUserId(addTeamUser.getUserId());
            teamUserEntity.setRemark(addTeamUser.getRemark());
            teamUserMapper.insert(teamUserEntity);
        }
        return result;
    }


    // 判断成员是否重复
    public boolean hasNoDuplicates(List<Integer> list) {
        for (int i = 0; i < list.size(); i++) {
            for (int j = i + 1; j < list.size(); j++) {
                if (list.get(i).equals(list.get(j))) {
                    return false;
                }
            }
        }
        return true;
    }


    @Override
    public int deleteTeamGroupByIds(int[] ids) {
        for (int id : ids) {
            List<Integer> teamUserIds = teamUserMapper.selectList(
                            new LambdaQueryWrapper<TeamUserEntity>()
                                    .eq(TeamUserEntity::getTeamId, id))
                    .stream()
                    .map(TeamUserEntity::getId)
                    .collect(Collectors.toList());
            teamUserMapper.deleteBatchIds(teamUserIds);
        }
        return teamMapper.deleteTeamGroupByIds(ids);
    }


}
