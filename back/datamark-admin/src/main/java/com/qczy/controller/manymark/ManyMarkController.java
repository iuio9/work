package com.qczy.controller.manymark;

import com.baomidou.mybatisplus.core.metadata.IPage;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.qczy.common.result.Result;
import com.qczy.model.request.DeleteRequest;
import com.qczy.model.request.ManyMarkTaskRequest;
import com.qczy.model.request.TaskShiftRequest;
import com.qczy.model.response.ManyCreateListResponse;
import com.qczy.model.response.ManyReceiveListResponse;
import com.qczy.model.response.ViewProgressResponse;
import com.qczy.service.ManyMarkService;
import com.qczy.service.TeamService;
import io.swagger.annotations.Api;
import io.swagger.annotations.ApiOperation;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.util.ObjectUtils;
import org.springframework.web.bind.annotation.*;

/**
 * @Author: hh
 * @Version: 1.0
 * @Date: 2025/2/26 16:28
 * @Description:
 */
@RestController
@RequestMapping("/manyMark")
@Api(tags = "多人标注")
public class ManyMarkController {

    @Autowired
    private ManyMarkService manyMarkService;
    @Autowired
    private TeamService teamService;


    /**
     * 创建多人标注任务
     */
    @ApiOperation("创建多人标注任务")
    @PostMapping("/addManyMarkTask")
    public Result addManyMarkTask(@RequestBody ManyMarkTaskRequest request) {
        if (manyMarkService.isDataSet(request.getSonId())) {
            return Result.fail("当前数据集数据不符合要求，请检查数据集后重试！");
        }
        return Result.ok(manyMarkService.addManyMarkTask(request));
    }

    /**
     * 我发起的任务-列表
     */
    @ApiOperation("我发起的任务-列表")
    @GetMapping("getMyCreateTaskList")
    public Result getMyCreateTaskList(@RequestParam Integer page,
                                      @RequestParam Integer limit) {
        Page<ManyCreateListResponse> pageParam = new Page<>(page, limit);
        IPage<ManyCreateListResponse> manyCreateList = manyMarkService.getMyCreateTaskList(pageParam);
        return Result.ok(manyCreateList);
    }

    /**
     * 我发起的任务-查看进度
     */
    @ApiOperation("我发起的任务-查看进度")
    @GetMapping("/viewProgress")
    public Result viewProgress(
            @RequestParam Integer page,
            @RequestParam Integer limit,
            Integer id) {
        if (id == null) {
            return Result.fail("id不能为空！");
        }
        Page<ViewProgressResponse> pageParam = new Page<>(page, limit);
        IPage<ViewProgressResponse> viewProgressList = teamService.viewProgress(pageParam, id);
        return Result.ok(viewProgressList);
    }


    /**
     * 我发起的任务-结束任务
     */
    @ApiOperation("我发起的任务-结束任务")
    @GetMapping("/endTask")
    public Result endTask(Integer id) {
        if (id == null) {
            return Result.fail("任务id不能为空！");
        }
        return Result.ok(manyMarkService.endTask(id));
    }


    /**
     * 我发起的任务-删除
     */
    @ApiOperation("我发起的任务-删除")
    @DeleteMapping("/deleteTask")
    public Result deleteTask(Integer id) {
        if (id == null) {
            return Result.fail("任务id不能为空！");
        }
        return Result.ok(manyMarkService.deleteTask(id));
    }


    /**
     * 我接收的任务 - 列表
     */
    @ApiOperation("我接收的任务 - 列表")
    @GetMapping("/getMyReceiveList")
    public Result getMyReceiveList(@RequestParam Integer page,
                                   @RequestParam Integer limit) {
        Page<ManyReceiveListResponse> pageParam = new Page<>(page, limit);
        IPage<ManyReceiveListResponse> manyReceiveList = manyMarkService.getMyReceiveList(pageParam);
        return Result.ok(manyReceiveList);
    }

    /**
     * 根据任务获取团队成员
     */
    @ApiOperation("根据任务id获取团队成员")
    @GetMapping("/getByTaskIdTeamList")
    public Result getByTaskIdTeamList(Integer taskId,Integer teamType) {
        return Result.ok(manyMarkService.getByTaskIdTeamList(taskId,teamType));
    }

    /**
     * 查看任务-结束任务
     */
    @ApiOperation("查看任务-结束任务")
    @GetMapping("/endUserTask")
    public Result endUserTask(Integer id) {
        if (id == null) {
            return Result.fail("id不能为空！");
        }
        int result = manyMarkService.endUserTask(id);
        if (result > 0) {
            return Result.ok(1);
        }else {
            return Result.fail("任务结束失败！");
        }


    }


    /**
     * 任务转交
     */
    @ApiOperation("任务转交")
    @PostMapping("/taskShift")
    private Result taskShift(@RequestBody TaskShiftRequest request) {
        if (request.getCurrentUserId() == null) {
            return Result.fail("当前用户id不能为空！");
        }
        if (request.getShiftId() == null) {
            return Result.fail("转交用户id不能为空！");
        }

        if (!manyMarkService.isRelayed(request)) {
            return Result.fail("当前用户状态不可转交，请更换用户后重试！");
        }

        return Result.ok(manyMarkService.taskShift(request));
    }

    /**
     * 撤回
     */
    @ApiOperation("撤回")
    @DeleteMapping("/withdraw")
    public Result withdraw(@RequestBody DeleteRequest request) {
        if (!ObjectUtils.isEmpty(request) && request.getIds() != null) {
            return Result.ok(manyMarkService.withdraw(request));
        }
        return Result.ok();
    }

    /**
     * 每人分配数量
     */
    @GetMapping("/allocationNum")
    public Result allocationNum(String sonId, String teamId) {
        return Result.ok(manyMarkService.allocationNum(sonId, teamId));
    }


}
