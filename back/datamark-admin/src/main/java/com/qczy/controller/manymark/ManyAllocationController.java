package com.qczy.controller.manymark;

import com.baomidou.mybatisplus.core.metadata.IPage;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.qczy.common.result.Result;
import com.qczy.model.request.ExamineReturnRequest;
import com.qczy.model.request.TaskShiftRequest;
import com.qczy.model.response.ManyAuditDetailsResponse;
import com.qczy.model.response.ManyReceiveListResponse;
import com.qczy.service.ManyAllocationService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;

import javax.validation.Valid;

/**
 * @Author: hh
 * @Version: 1.0
 * @Date: 2025/3/17 15:05
 * @Description: 分配任务
 */
@RequestMapping("/many/allocation")
@RestController
public class ManyAllocationController {


    @Autowired
    private ManyAllocationService manyAllocationService;

    // 审核团队基础信息
    @GetMapping("/examineTeamInfo")
    public Result examineTeamInfo(Integer id) {
        return Result.ok(manyAllocationService.examineTeamInfo(id));
    }

    // 审核人员分配的列表
    @GetMapping("/examineTeamList")
    public Result examineTeamList(Integer id) {
        return Result.ok(manyAllocationService.examineTeamList(id));
    }


    // 分配审核任务
    @GetMapping("/distributionExamine")
    public Result distributionExamine(Integer taskId) {
        if (taskId == null) {
            return Result.fail("任务id不能为空！");
        }
        if (manyAllocationService.submitTaskShift(taskId)) {
            System.out.println("调用了该方法~");
            return Result.ok(1);
        }
        int i = manyAllocationService.distributionExamine(taskId);
        if (i > 0) {
            return Result.ok(1);
        } else {
            return Result.fail("分配失败！");
        }
    }

    // 确认分配审核
    @GetMapping("/confirmAudit")
    public Result confirmAudit(Integer id) {
        if (id == null) {
            return Result.fail("任务id不能为空！");
        }
        if (manyAllocationService.submitTaskShift(id)) {
            System.out.println("调用了该方法~");
            return Result.ok(1);
        }
        return Result.ok(manyAllocationService.confirmAudit(id));
    }


    // 审核任务转交
    @PostMapping("/examineTaskShift")
    public Result examineTaskShift(@RequestBody TaskShiftRequest request) {
        if (request == null) {
            return Result.fail("参数不能为空！");
        }
        if (manyAllocationService.examineTaskShift(request) == 1) {
            return Result.ok(1);
        } else {
            return Result.fail("转交失败！");
        }
    }


    // 审核详情
    @GetMapping("/examineDetails")
    public Result examineDetails(@RequestParam Integer page,
                                 @RequestParam Integer limit,
                                 Integer taskId) {
        if (taskId == null) {
            return Result.fail("任务id不能为空！");
        }
        Page<ManyAuditDetailsResponse> pageParam = new Page<>(page, limit);
        IPage<ManyAuditDetailsResponse> examineDetailsList = manyAllocationService.examineDetails(pageParam, taskId);
        return Result.ok(examineDetailsList);
    }


    // 我的审核列表
    @GetMapping("/myExamineTaskList")
    public Result myExamineTaskList(@RequestParam Integer page,
                                    @RequestParam Integer limit) {
        Page<ManyReceiveListResponse> pageParam = new Page<>(page, limit);
        IPage<ManyReceiveListResponse> receiveList = manyAllocationService.myExamineTaskList(pageParam);
        return Result.ok(receiveList);
    }

    // 我的审核 - 提交任务
    @GetMapping("/submitExamineTask")
    public Result submitExamineTask(@RequestParam Integer id) {
        if (id == null) {
            return Result.fail("不能为空！");
        }
        System.out.println("标注端调用了接口！！");
        return Result.ok(manyAllocationService.submitExamineTask(id));
    }


    // 我的审核 - 提交任务提示语
    @GetMapping("/submitExamineTaskPrompt")
    public Result submitExamineTaskPrompt(@RequestParam Integer id) {
        if (id == null) {
            return Result.fail("不能为空！");
        }
        return Result.ok(manyAllocationService.submitExamineTaskPrompt(id));
    }


    // 审核退回
    @PostMapping("/examineReturn")
    public Result examineReturn(@RequestBody ExamineReturnRequest request) {
        if (request == null) {
            return Result.fail("参数对象不能为空！");
        }
        if (request.getTaskId() == null) {
            return Result.fail("任务id不能为空！");
        }

        if (!manyAllocationService.isExamineStatus(request)){
            return Result.fail("当前存在审核员状态不可退回，请重新选择！");
        }

        int result = manyAllocationService.examineReturn(request);
        if (result > 0) {
            return Result.ok(1);
        } else {
            return Result.fail("退回审核失败！");
        }
    }


    // 审核通过
    @GetMapping("/approved")
    public Result approved(@RequestParam Integer taskId) {
        if (taskId == null) {
            return Result.fail("任务id不能为空！");
        }

        if (!manyAllocationService.isExamineSubmit(taskId)) {
            return Result.fail("当前状态不可提交，等待所有审核人员提交完成后重试！");
        }
        int result = manyAllocationService.approved(taskId);

        if (result > 0) {
            return Result.ok(1);
        } else {
            return Result.fail("审核通过失败！");
        }
    }


}
