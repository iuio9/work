package com.qczy.controller.modelEvaluation;

import com.baomidou.mybatisplus.core.metadata.IPage;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.qczy.common.result.Result;
import com.qczy.model.entity.ModelAssessTaskEntity;
import com.qczy.model.request.AssessTaskRequest;
import com.qczy.model.request.DeleteRequest;
import com.qczy.model.response.ModelAssessResponse;
import com.qczy.service.ManufacturerService;
import com.qczy.service.ModelAssessService;
import io.swagger.annotations.Api;
import io.swagger.annotations.ApiOperation;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;

import java.util.HashMap;
import java.util.Map;


/**
 * @Author: hh
 * @Version: 1.0
 * @Date: 2025/6/5 11:06
 * @Description:
 */
@RestController
@RequestMapping("/model/assess")
@Api(tags = "测试评估任务")
public class ModelAssessController {


    @Autowired
    private ModelAssessService modelAssessService;

    @Autowired
    private ManufacturerService manufacturerService;


    @PostMapping("/createAssessTask")
    @ApiOperation("创建评估任务")
    public Result createAssessTask(AssessTaskRequest request) {

        boolean isRepeat = modelAssessService.isTaskNameRepeat(request.getTaskName(), null);
        if (isRepeat) {
            return Result.fail("任务名称已存在！");
        }

        int result = modelAssessService.createAssessTask(request);
        if (result > 0) {
            return Result.ok(1);
        } else {
            return Result.fail("创建评估任务失败！");
        }
    }

    // 模型列表
    @GetMapping("/listPage")
    @ApiOperation("获取带分页带条件查询")
    public Result listPage(
            @RequestParam Integer page,
            @RequestParam Integer limit,
            @ModelAttribute ModelAssessTaskEntity modelAssessTaskEntity) {
        Page<ModelAssessResponse> pageParam = new Page<>(page, limit);
        IPage<ModelAssessResponse> modelList = modelAssessService.listPage(pageParam, modelAssessTaskEntity);
        return Result.ok(modelList);

    }

    // 模型任务详情
    @GetMapping("/taskDetail")
    @ApiOperation("模型任务详情")
    public Result taskDetails(@RequestParam Integer id) {
        if (id == null) {
            return Result.fail("任务id不能为空！");
        }
        return Result.ok(modelAssessService.taskDetails(id));
    }

    // 编辑
    @PostMapping("/editTask")
    @ApiOperation("编辑")
    public Result editTask(AssessTaskRequest request) {
        boolean isRepeat = modelAssessService.isTaskNameRepeat(request.getTaskName(), request.getId());
        if (isRepeat) {
            return Result.fail("任务名称已存在！");
        }
        int result = modelAssessService.editTask(request);
        if (result > 0) {
            return Result.ok(1);
        } else {
            return Result.fail("编辑评估任务失败！");
        }
    }


    /**
     * 开始调用厂商
     */
    @PostMapping("/startManufacturer")
    @ApiOperation("开始调用厂商")
    public Result startManufacturer(ModelAssessTaskEntity modelAssessTaskEntity) {
        manufacturerService.startManufacturer(modelAssessTaskEntity);
        Map<String, Object> data = new HashMap<>();
        data.put("taskId", modelAssessTaskEntity.getId());
        data.put("messageType:", "task_log");
        return Result.ok(data);
    }

    /**
     * 暂停调用厂商
     */
    @PostMapping("/stopManufacturer")
    @ApiOperation("暂停调用厂商")
    public Result stopManufacturer(ModelAssessTaskEntity modelAssessTaskEntity) {
        manufacturerService.stopManufacturer(modelAssessTaskEntity);
        return Result.ok("暂停调用厂商");
    }

    /**
     * 结束调用厂商
     */
    @PostMapping("/endManufacturer")
    @ApiOperation("结束调用厂商")
    public Result endManufacturer(ModelAssessTaskEntity modelAssessTaskEntity) {
        manufacturerService.endManufacturer(modelAssessTaskEntity);
        return Result.ok("结束调用厂商");
    }

    /**
     * 重新开始
     */
    @PostMapping("/restartManufacturer")
    @ApiOperation("重新调用厂商")
    public Result restartManufacturer(@RequestBody ModelAssessTaskEntity modelAssessTaskEntity) {
        manufacturerService.startManufacturer(modelAssessTaskEntity);
        return Result.ok(1);
    }

    /**
     * 完成对接
     */
    @PostMapping("/finishContact")
    @ApiOperation("完成对接")
    public Result finishContact(@RequestBody ModelAssessTaskEntity modelAssessTaskEntity) {
        if (!manufacturerService.isExecuteTask(modelAssessTaskEntity)) {
            return Result.fail("当前任务不能完成对接，请先执行完任务后重试！");
        }
        int result = manufacturerService.finishContact(modelAssessTaskEntity);
        if (result > 0) {
            return Result.ok(1);
        } else {
            return Result.fail("对接失败！");
        }
    }


    /**
     * 删除模型
     */
    @PostMapping("/delTask")
    public Result delTask(@RequestBody DeleteRequest request) {
        if (request == null || request.getId() == null) {
            return Result.fail("要删除的模型id不能为空！");
        }
        int result = modelAssessService.delTask(request);
        if (result > 0) {
            return Result.ok(1);
        } else {
            return Result.fail("删除失败！");
        }
    }

    /**
     * 从调试结果里面获取 请求路径和 请求方式
     */
    @PostMapping("/getModelDebugInfo")
    public Result getModelDebugInfo(@RequestBody AssessTaskRequest request) {
        if (request.getModelId() == null) {
            return Result.fail("模型id不能为空！");
        }
        return Result.ok(modelAssessService.getModelDebugInfo(request));
    }


    // 删除文件
    @PostMapping("/deleteFile")
    public Result deleteFile(@RequestBody DeleteRequest request) {
        // 获取模型配置
        int result = modelAssessService.deleteFile(request);
        if (result > 0) {
            return Result.ok(1);
        } else {
            return Result.ok("文件删除失败！");
        }
    }


}
