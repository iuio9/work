package com.qczy.controller.algorithm;

import com.qczy.common.result.Result;
import com.qczy.model.request.BWTestRequest;
import com.qczy.model.request.CheckAndClassify;
import com.qczy.service.CheckClassifyService;
import io.swagger.annotations.ApiOperation;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

//检测分类接口
@RestController
@RequestMapping("/algorithm/checkAndClassify")
public class CheckClassifyController {

    @Autowired
    private CheckClassifyService checkClassifyService;
    @PostMapping("/check")
    @ApiOperation("检测")
    public Result check(CheckAndClassify checkAndClassify) {

        return Result.ok();
    }

    @PostMapping("/classify")
    @ApiOperation("分类")
    public Result classify(CheckAndClassify checkAndClassify) {
        return Result.ok();
    }
}
