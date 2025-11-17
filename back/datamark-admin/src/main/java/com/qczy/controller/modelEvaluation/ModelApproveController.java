package com.qczy.controller.modelEvaluation;

import com.baomidou.mybatisplus.core.metadata.IPage;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.qczy.common.result.Result;
import com.qczy.model.request.ModelApplyForRequestParam;
import com.qczy.model.response.ModelApplyForListResponse;
import com.qczy.service.ModelApplyForService;
import com.qczy.service.ModelApproveService;
import io.swagger.annotations.Api;
import io.swagger.annotations.ApiOperation;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;

/**
 * @Author: hh
 * @Version: 1.0
 * @Date: 2025/5/22 11:14
 * @Description:
 */
@RestController
@RequestMapping("/model/approve")
@Api(tags = "模型审批")
public class ModelApproveController {

    @Autowired
    private ModelApproveService modelApproveService;
    @Autowired
    private ModelApplyForService modelApplyForService;

    @GetMapping("/list")
    @ApiOperation("模型审批页面")
    public Result list(@RequestParam Integer page,
                       @RequestParam Integer limit,
                       @ModelAttribute ModelApplyForRequestParam requestParam) {
        Page<ModelApplyForListResponse> pageParam = new Page<>(page, limit);
        IPage<ModelApplyForListResponse> modelApplyForList = modelApplyForService.approveList(pageParam, requestParam);
        return Result.ok(modelApplyForList);

    }

    @GetMapping("/pass")
    public Result pass(Integer id) {
        if (id == null) {
            return Result.fail("id不能为空！");
        }
        int result = modelApproveService.pass(id);
        if (result > 0) {
            return Result.ok(1);
        }else {
            return Result.ok("审批通过失败！");
        }
    }

    @GetMapping("/notPass")
    public Result notPass(Integer id) {
        if (id == null) {
            return Result.fail("id不能为空！");
        }
        int result = modelApproveService.notPass(id);
        if (result > 0) {
            return Result.ok(1);
        }else {
            return Result.ok("审批不通过失败！");
        }
    }


    /**
     * 根据类型获取模型列表
     */
    @GetMapping("/getModelTypeList")
    public Result getModelTypeList(Integer modelWay) {
        if (modelWay == null) {
            return Result.fail("模型类型不能为空！");
        }
        return Result.ok(modelApproveService.getModelTypeList(modelWay));
    }
}
