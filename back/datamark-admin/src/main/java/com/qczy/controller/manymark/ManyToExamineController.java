package com.qczy.controller.manymark;


import com.qczy.common.result.Result;
import com.qczy.model.entity.ManyFileEntity;
import com.qczy.service.ManyToExamineService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;


/**
 * @Author: hh
 * @Version: 1.0
 * @Date: 2025/3/12 10:49
 * @Description: 多人标注审核
 */
@RestController
@RequestMapping("/many/ToExamine")
public class ManyToExamineController {

    @Autowired
    private ManyToExamineService manyExamineService;


    /**
     * 验收是否通过
     */
    @PostMapping("fileIsApprove")
    public Result isApprove(@RequestBody ManyFileEntity manyFileEntity) {
        if (manyExamineService.isApprove(manyFileEntity) == 1) {
            return Result.ok(1);
        } else {
            return Result.fail("审核失败！");
        }
    }


    /**
     * 验收完成
     */
    @GetMapping("/verifyComplete")
    public Result verifyComplete(Integer taskId, Integer verifyState) {
        if (taskId == null || verifyState == null) {
            return Result.fail("验收参数不能为空！");
        }
        return Result.ok(manyExamineService.verifyComplete(taskId, verifyState));
    }


    /**
     * 打回任务
     */
    @GetMapping("/returnTask")
    public Result returnTask(Integer taskId, Integer returnState,Integer id) {
        if (taskId == null || returnState == null || id == null) {
            return Result.fail("打回参数不能为空！");
        }
        if (!manyExamineService.returnTaskState(taskId, returnState,id)) {
            return Result.fail("没有符合要打回任务的数据！");
        }
        return Result.ok(manyExamineService.returnTask(taskId, returnState,id));
    }


    /**
     * 剩余验收通过
     */
    @GetMapping("/remainingApprove")
    public Result remainingApprove(Integer taskId, Integer id) {
        if (taskId == null) {
            return Result.fail("任务id不能为空！");
        }
        // 没有未验收数据
        if (!manyExamineService.isRemaining(taskId,id)) {
            return Result.fail("没有未验收数据！");
        }
        return Result.ok(manyExamineService.remainingApprove(taskId,id));
    }

    /**
     * 提交任务
     */
    @GetMapping("/submitTask")
    public Result submitTask(Integer id) {
        if (id == null) {
            return Result.fail("子任务id不能为空！");
        }
        System.out.println("审核端调用了接口！！");
        return Result.ok(manyExamineService.submitTask(id));
    }

    /**
     * 提交提示语
     */
    @GetMapping("/submitTaskPrompt")
    public Result submitTaskPrompt(Integer id) {
        if (id == null) {
            return Result.fail("子任务id不能为空！");
        }
        return Result.ok(manyExamineService.submitTaskPrompt(id));
    }


}
