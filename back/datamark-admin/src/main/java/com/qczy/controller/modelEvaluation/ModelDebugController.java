package com.qczy.controller.modelEvaluation;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.qczy.common.result.Result;
import com.qczy.model.request.DebugModelRequest;
import com.qczy.model.entity.ModelDebugLog;
import com.qczy.service.ModelDebugService;
import com.qczy.utils.StringUtils;
import io.swagger.annotations.Api;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;

import java.util.HashMap;
import java.util.Map;

/**
 * @Author: hh
 * @Version: 1.0
 * @Date: 2025/5/30 10:27
 * @Description:
 */
@RestController
@RequestMapping("/api")
@Api(tags = "模型调试")
public class ModelDebugController {
    private static final Logger logger = LoggerFactory.getLogger(ModelDebugController.class);

    @Autowired
    private ModelDebugService modelDebugService;


    // 模型调试
    @PostMapping("/debugModel")
    public Result debugModel(DebugModelRequest debugModelRequest) {

        long startTime = System.currentTimeMillis();
        boolean success = false;
        Integer statusCode = null;

        try {
            Map<String, Object> resultMap = new HashMap<>();
            Map<String, Object> serviceResult = modelDebugService.debugModel(debugModelRequest);
            long endTime = System.currentTimeMillis();


            // 获取HTTP状态码
            statusCode = (Integer) serviceResult.get("statusCode");

            // 只有状态码为200时才认为成功
            success = statusCode != null && statusCode == 200;

            resultMap.put("modelDebugLogEntity", serviceResult.get("modelDebugLogEntity"));
            serviceResult.remove("modelDebugLogEntity");
            resultMap.put("interfaceData", serviceResult);
            // 组织网络请求日志
            Map<String, Object> netData = new HashMap<>();
            netData.put("status", success ? "成功" : "失败");
            netData.put("timestamp", (endTime - startTime) + "ms");
            resultMap.put("netData", netData);
            return Result.ok(resultMap);

        } catch (Exception e) {
            // 异常处理
            Map<String, Object> errorResult = new HashMap<>();
            errorResult.put("status", false);
            errorResult.put("error", e.getMessage());
            return Result.fail(errorResult);
        }


    }


    // 保存测试结果
    @PostMapping("/savaDebugLog")
    public Result savaDebugLog(@RequestBody ModelDebugLog modelDebugLog) {
        if (StringUtils.isEmpty(modelDebugLog.getModelAddress()) || modelDebugLog.getRequestType() == null) {
            return Result.fail("请先进行调试！");
        }
        int result = modelDebugService.savaDebugLog(modelDebugLog);
        if (result > 0) {
            return Result.ok(1);
        } else {
            return Result.fail("保存失败！");
        }
    }


    // 一键调试
    @GetMapping("/oneClickDebugging")
    public Result oneClickDebugging(Integer id) {
        if (id == null) {
            return Result.fail("模型id不能为空！");
        }
        ModelDebugLog modelDebugLog = modelDebugService.getModelDebugLog(id);
        if (modelDebugLog == null) {
            return Result.fail("当前模型没有进行过测试！");
        }
        return Result.ok(modelDebugService.oneClickDebugging(modelDebugLog));
    }


    // 查询当前模型测试结果是否保存
    @GetMapping("/isSavaDebugLog")
    public Result isSavaDebugLog(Integer id) {
        if (id == null) {
            return Result.fail("模型id不能为空！");
        }
        int count = modelDebugService.count(
                new LambdaQueryWrapper<ModelDebugLog>()
                        .eq(ModelDebugLog::getModelBaseId, id)
        );

        return Result.ok(count > 0);
    }


}
