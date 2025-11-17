package com.qczy.controller.algorithm;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.qczy.common.constant.ThirdAssessConstants;
import com.qczy.common.result.Result;
import com.qczy.mapper.ModelAssessTaskMapper;
import com.qczy.model.entity.AlgorithmTaskEntity;
import com.qczy.model.entity.ModelAssessTaskEntity;
import com.qczy.model.request.SegmentParams;
import com.qczy.service.ThirdModelAssessService;
import com.qczy.utils.TrainUtil;
import io.swagger.annotations.Api;
import io.swagger.annotations.ApiOperation;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;

/**
 * @author ：gwj
 * @date ：Created in 2025-05-29 13:42
 * @description：
 * @modified By：
 * @version: $
 */
@RestController
@RequestMapping("/thirdModelAssess")
@Api(tags = "第三方模型评估")
public class ThirdModelAssessController {
    @Autowired
    ThirdModelAssessService thirdModelAssessService;
    @Autowired
    private ModelAssessTaskMapper modelAssessTaskMapper;

    @GetMapping("/restart")
    @ApiOperation("重新开始评估")
    public Result reStart(AlgorithmTaskEntity algorithmTaskEntity){
        termination(algorithmTaskEntity);
        thirdModelAssessService.uploadJsonData(algorithmTaskEntity);
        return Result.ok("评估开始");
    }

    @GetMapping("/start")
    @ApiOperation("开始评估")
    public Result start(AlgorithmTaskEntity algorithmTaskEntity){
        thirdModelAssessService.uploadJsonData(algorithmTaskEntity);
        return Result.ok("评估开始");
    }

    @GetMapping("/startClass")
    @ApiOperation("开始评估分类")
    public Result startClass(AlgorithmTaskEntity algorithmTaskEntity){
        //检测
        thirdModelAssessService.uploadJsonData(algorithmTaskEntity);
        //分类
        thirdModelAssessService.uploadJsonData(algorithmTaskEntity);
        return Result.ok("评估开始");
    }
    @GetMapping("/pause")
    @ApiOperation("暂停评估")
    public Result pause(AlgorithmTaskEntity algorithmTaskEntity){
        thirdModelAssessService.controlTask(algorithmTaskEntity.getTaskId(), ThirdAssessConstants.PAUSE);
        return Result.ok("暂停评估");
    }
    @GetMapping("/continue")
    @ApiOperation("继续评估")
    public Result continue123(AlgorithmTaskEntity algorithmTaskEntity){
        thirdModelAssessService.controlTask(algorithmTaskEntity.getTaskId(), ThirdAssessConstants.CONTINUE);
        return Result.ok("继续评估");
    }
    @GetMapping("/termination")
    @ApiOperation("终止评估")
    public Result termination(AlgorithmTaskEntity algorithmTaskEntity){
        thirdModelAssessService.controlTask(algorithmTaskEntity.getTaskId(), ThirdAssessConstants.TERMINATION);
        return Result.ok("终止评估");
    }

    @Autowired
    private TrainUtil trainUtil;
    @GetMapping("/viewResult")
    @ApiOperation("查看评估结果")
    public Result viewResult(AlgorithmTaskEntity algorithmTaskEntity){
//        ModelAssessTaskEntity modelAssessTaskEntity = modelAssessTaskMapper.selectOne(new LambdaQueryWrapper<ModelAssessTaskEntity>()
//                .eq(ModelAssessTaskEntity::getId, algorithmTaskEntity.getTaskId()));
//        try{
//            trainUtil.thirdAssess(modelAssessTaskEntity);
//        }catch (Exception e){
//            e.printStackTrace();
//        }
        return Result.ok(thirdModelAssessService.viewAssessResult(algorithmTaskEntity));
    }


}
