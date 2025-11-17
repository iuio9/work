package com.qczy.controller.modelEvaluation;

import com.baomidou.mybatisplus.core.metadata.IPage;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.qczy.common.result.Result;
import com.qczy.mapper.ModelBaseMapper;
import com.qczy.model.entity.ModelBaseEntity;
import com.qczy.model.entity.ModelDebugLog;
import com.qczy.model.request.DeleteRequest;
import com.qczy.model.request.ModelApplyForRequest;
import com.qczy.model.request.ModelApplyForRequestParam;
import com.qczy.model.request.ModelBackFillRequest;
import com.qczy.model.response.ModelApplyForListResponse;
import com.qczy.service.ModelApplyForService;
import com.qczy.service.ModelBaseService;
import com.qczy.service.ModelDebugService;
import com.qczy.utils.StringUtils;
import io.swagger.annotations.Api;
import io.swagger.annotations.ApiOperation;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.validation.BindingResult;
import org.springframework.web.bind.annotation.*;

import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import javax.validation.Valid;
import java.util.HashMap;
import java.util.Map;
import java.util.Objects;

/**
 * @Author: hh
 * @Version: 1.0
 * @Date: 2025/5/16 11:22
 * @Description:
 */

@RestController
@RequestMapping("/model/evaluation")
@Api(tags = "第三方模型-申请、 审核模块")
public class ModelEvaluationController {

    @Autowired
    private ModelApplyForService modelApplyForService;
    @Autowired
    private ModelBaseService modelBaseService;
    @Autowired
    private ModelDebugService modelDebugService;


    @PostMapping("/addModel")
    @ApiOperation("新增模型")
    public Result addModel(@RequestBody @Valid ModelApplyForRequest request, BindingResult bindingResult) {
        if (bindingResult.hasErrors()) {
            return Result.fail(Objects.requireNonNull(bindingResult.getFieldError()).getDefaultMessage());
        }

        // 判断模型名称是否重复
        boolean isRepeat = modelBaseService.isModelNameRepeat(request.getModelName(), null);
        if (isRepeat) {
            return Result.fail("模型名称已存在！");
        }

        ModelApplyForListResponse response = modelApplyForService.addModel(request);
        if (response != null && response.getId() != null && response.getId() > 0) {
            return Result.ok(response);
        } else {
            return Result.fail("模型申请失败！");
        }

    }

    @PostMapping("/editModel")
    @ApiOperation("编辑")
    public Result editModel(@RequestBody ModelApplyForRequest request) {

        // 判断模型名称是否重复
        boolean isRepeat = modelBaseService.isModelNameRepeat(request.getModelName(), request.getId());
        if (isRepeat) {
            return Result.fail("模型名称已存在！");
        }
        int result = modelApplyForService.editModel(request);
        if (result > 0) {
            return Result.ok(1);
        } else {
            return Result.fail("模型编辑失败！");
        }

    }


    @GetMapping("/list")
    @ApiOperation("模型申请列表")
    public Result list(
            @RequestParam Integer page,
            @RequestParam Integer limit,
            @ModelAttribute ModelApplyForRequestParam requestParam) {
        Page<ModelApplyForListResponse> pageParam = new Page<>(page, limit);
        IPage<ModelApplyForListResponse> modelApplyForList = modelApplyForService.list(pageParam, requestParam);
        return Result.ok(modelApplyForList);
    }


    // 根据内容生成pdf
    @PostMapping("/generatePad")
    @ApiOperation("生成pdf")
    public void generatePad(@RequestBody ModelBaseEntity baseEntity, HttpServletRequest request, HttpServletResponse response) {
        modelApplyForService.generatePad(baseEntity.getId(), request, response);
    }


    // 模型详情
    @GetMapping("/modelDetails")
    @ApiOperation("模型详情")
    public Result modelDetails(@RequestParam Integer id) {
        return Result.ok(modelApplyForService.modelDetails(id));
    }


    // 模型附件（回填）
    @PostMapping("/modelBackFill")
    public Result modelBackFill(@RequestBody ModelBackFillRequest request) {
        if (request == null || request.getId() == null) {
            return Result.fail("模型id不能为空！");
        }
        int result = modelApplyForService.modelBackFill(request);
        if (result > 0) {
            return Result.ok(1);
        } else {
            return Result.fail("模型附件回填失败！");
        }
    }

    // 提交审批
    @GetMapping("/submitApprove")
    public Result submitApprove(@RequestParam Integer id) {
        if (id == null) {
            return Result.fail("模型id不能为空！");
        }

        ModelBaseEntity modelBaseEntity = modelBaseService.getById(id);
        if (modelBaseEntity == null) {
            return Result.fail("模型申请对象不存在！");
        }

        Integer applyForType = modelBaseEntity.getApplyForType(); // 申请类型
        // 判断是否是文本申请
        if (applyForType == null) {
            return Result.fail("提交审批失败！");
        }

        if (applyForType == 1) { //文本申请
            if (StringUtils.isEmpty(modelBaseEntity.getApplyForPdf())) {
                return Result.fail("请先上传pdf附件！");
            }
        }

        // 判断是否进行模型测试
        ModelDebugLog modelDebugLog = modelDebugService.getModelDebugLog(id);
        if (modelDebugLog == null) {
            return Result.fail("请先进行模型调试！");
        }
        // 判断是否调试成功
        if (modelDebugService.oneClickDebugging(modelDebugLog).equals("调试失败！")){
            return Result.fail("模型调试失败，请再次重试模型调试！");
        }
        int result = modelApplyForService.submitApprove(modelBaseEntity);
        if (result > 0) {
            return Result.ok(1);
        } else {
            return Result.fail("提交审批失败！");
        }
    }

    /**
     * 删除模型
     */
    @PostMapping("/delModel")
    public Result delModel(@RequestBody DeleteRequest request) {
        if (request == null || request.getId() == null) {
            return Result.fail("要删除的模型id不能为空！");
        }
        // 判断当前模型是否有任务正在使用，如果正在使用，则不可进行删除
        if (modelBaseService.isModelUse(request.getId())) {
            return Result.fail("当前模型正在使用，请先删除第三方评估任务后重试！");
        }

        int result = modelBaseService.delModel(request);
        if (result > 0) {
            return Result.ok(1);
        } else {
            return Result.fail("删除失败！");
        }
    }


    @PostMapping("/deleteFile")
    public Result deleteFile(@RequestBody DeleteRequest request) {
        // 获取模型配置
        int result = modelBaseService.deleteFile(request);
        if (result > 0) {
            return Result.ok(1);
        } else {
            return Result.ok("文件删除失败！");
        }
    }


}
