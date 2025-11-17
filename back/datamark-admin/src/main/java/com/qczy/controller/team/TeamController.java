package com.qczy.controller.team;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.core.metadata.IPage;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.qczy.common.result.Result;
import com.qczy.model.entity.TeamEntity;
import com.qczy.model.request.AddTeamUser;
import com.qczy.model.request.DeleteRequest;
import com.qczy.service.TeamService;
import io.swagger.annotations.ApiOperation;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.util.CollectionUtils;
import org.springframework.util.ObjectUtils;
import org.springframework.web.bind.annotation.*;

import java.util.List;
import java.util.stream.Collectors;

/**
 * @Author: hh
 * @Version: 1.0
 * @Date: 2025/2/24 9:58
 * @Description:
 */
@RequestMapping("/team")
@RestController
public class TeamController {

    @Autowired
    private TeamService teamService;


    /**
     * 查询【请填写功能名称】列表
     */
    @GetMapping("/list")
    public Result list(@RequestParam Integer page,
                       @RequestParam Integer limit,
                       @ModelAttribute TeamEntity request) {
        Page<TeamEntity> pageParam = new Page<>(page, limit);
        IPage<TeamEntity> teamList = teamService.selectTeamList(pageParam, request);
        return Result.ok(teamList);
    }

    @GetMapping("/getTeamList")
    public Result getTeamList(Integer teamType) {
        return Result.ok(teamService.list(new LambdaQueryWrapper<TeamEntity>().eq(TeamEntity::getTeamType, teamType)));
    }


    /**
     * 新增保存【请填写功能名称】
     */
    @PostMapping("/add")
    public Result addSave(@RequestBody TeamEntity team) {
        if (CollectionUtils.isEmpty(team.getUserList())) {
            return Result.fail("团队成员不能为空！");
        }
        List<Integer> list = team.getUserList().stream().map(AddTeamUser::getUserId).collect(Collectors.toList());
        if (!teamService.hasNoDuplicates(list)){
            return Result.fail("团队成员不能重复！");
        }
            return Result.ok(teamService.insertTeam(team));
    }


    /**
     * 修改保存【请填写功能名称】
     */

    @PostMapping("/edit")
    public Result editSave(@RequestBody TeamEntity team) {
        if (CollectionUtils.isEmpty(team.getUserList())) {
            return Result.fail("团队成员不能为空！");
        }
        List<Integer> list = team.getUserList().stream().map(AddTeamUser::getUserId).collect(Collectors.toList());
        if (!teamService.hasNoDuplicates(list)){
            return Result.fail("团队成员不能重复！");
        }
        if (!teamService.isManyTask(team.getId())) {
            return Result.fail("当前团队正在执行多人标注任务，任务结束后重试！");
        }
        return Result.ok(teamService.updateTeam(team));
    }


    /**
     * 删除【请填写功能名称】
     */
    @DeleteMapping("/remove")
    @ApiOperation("删除")
    public Result remove(@RequestBody DeleteRequest request) {
        if (ObjectUtils.isEmpty(request) && request.getIds() == null) {
            return Result.fail("请传入要删除的数据！");
        }

        for (int id : request.getIds()) {
            if (!teamService.isManyTask(id)) {
                return Result.fail("当前团队正在执行多人标注任务，任务结束后重试！");
            }
        }
        return Result.ok(teamService.deleteTeamGroupByIds(request.getIds()));
    }


}
