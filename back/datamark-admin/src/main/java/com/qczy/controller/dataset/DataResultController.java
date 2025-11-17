package com.qczy.controller.dataset;

import com.qczy.common.result.Result;
import com.qczy.model.request.SavaResultRequest;
import com.qczy.service.DataResultService;
import io.swagger.annotations.Api;
import io.swagger.annotations.ApiOperation;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.util.Map;

@RestController
@RequestMapping("/data/result")
@Api(tags = "数据集-结果保存 or 结果删除")
public class DataResultController {

    @Autowired
    private DataResultService dataResultService;


    @PostMapping("/saveResult")
    @ApiOperation("数据集结果保存")
    public Result savaResult(@RequestBody SavaResultRequest request) {
        Map<String, Object> result = dataResultService.savaResult(request);
        if (result == null) {
            return Result.fail("保存错误！");
        }
        int i = (int) result.get("status");
        if (i == 1){
            return Result.ok(result);
        }else {
            return Result.fail("保存失败！");
        }
    }


}
