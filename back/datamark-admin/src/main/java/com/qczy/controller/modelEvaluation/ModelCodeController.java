package com.qczy.controller.modelEvaluation;

import com.baomidou.mybatisplus.core.metadata.IPage;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.qczy.common.result.Result;
import com.qczy.model.entity.ModelCodeEntity;
import com.qczy.model.request.ModelApplyForRequestParam;
import com.qczy.model.response.DataResponse;
import com.qczy.service.ModelCodeService;
import io.swagger.annotations.Api;
import io.swagger.annotations.ApiOperation;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/model/code")
@Api(tags = "算法编码")
public class ModelCodeController {
    @Autowired
    private ModelCodeService modelCodeService;

    /**
     * 根据模型id获取算法编码
     */
    @GetMapping("/getAlgorithmList")
    @ApiOperation("根据模型id获取算法编码")
    public Result getModelDebugInfo(Integer modelId,
                                    @RequestParam Integer page,
                                    @RequestParam Integer limit
    ) {
        if (modelId == null) {
            return Result.fail("模型id不能为空！");
        }

        Page<ModelCodeEntity> pageParam = new Page<>(page, limit);
        IPage<ModelCodeEntity> dataSetList = modelCodeService.modelCodeList(pageParam,modelId);
        return Result.ok(dataSetList);
    }


}
