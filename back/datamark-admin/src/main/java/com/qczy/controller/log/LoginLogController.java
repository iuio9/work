package com.qczy.controller.log;

import com.baomidou.mybatisplus.core.metadata.IPage;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.qczy.common.result.Result;
import com.qczy.model.entity.LoginLogEntity;
import com.qczy.model.entity.OperLogEntity;
import com.qczy.service.LoginLogService;
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
@RequestMapping("/log/login")
@Api(tags = "登录日志管理")
public class LoginLogController {


    @Autowired
    private LoginLogService loginLogService;

    @GetMapping("/list")
    @ApiOperation("查询登录日志列表")
    public Result list(@RequestParam Integer page,
                       @RequestParam Integer limit,
                       @ModelAttribute LoginLogEntity loginLog) {
        Page<OperLogEntity> pageParam = new Page<>(page, limit);
        IPage<OperLogEntity> roleEntityList = loginLogService.selectLoginLogList(pageParam, loginLog);
        return Result.ok(roleEntityList);
    }




}
