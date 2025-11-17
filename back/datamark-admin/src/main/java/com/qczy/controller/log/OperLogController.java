package com.qczy.controller.log;

import com.baomidou.mybatisplus.core.metadata.IPage;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.qczy.common.result.Result;
import com.qczy.model.entity.OperLogEntity;
import com.qczy.model.request.OperLogRequest;
import com.qczy.service.OperLogService;
import io.swagger.annotations.Api;
import io.swagger.annotations.ApiOperation;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;

/**
 * @Author: hh
 * @Version: 1.0
 * @Date: 2024/8/5 14:39
 * @Description:
 */
@RestController
@RequestMapping("/log/oper")
@Api(tags = "操作日志管理")
public class OperLogController {


    @Autowired
    private OperLogService operLogService;

    @GetMapping("/list")
    @ApiOperation("查询日志列表")
    public Result list(@RequestParam Integer page,
                       @RequestParam Integer limit,
                       @ModelAttribute OperLogRequest request) {
        Page<OperLogEntity> pageParam = new Page<>(page, limit);
        IPage<OperLogEntity> roleEntityList = operLogService.selectOperLogList(pageParam, request);
        return Result.ok(roleEntityList);
    }



    @GetMapping("/getOperLog")
    @ApiOperation("获取单个日志详情")
    public Result getOperLog(@RequestParam Integer operId) {
        return Result.ok(operLogService.getOperLog(operId));
    }


}
