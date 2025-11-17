package com.qczy.controller.dataEnhancement;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.qczy.common.constant.TaskRecordTypeConstants;
import com.qczy.common.result.Result;
import com.qczy.mapper.AlgorithmMapper;
import com.qczy.mapper.AlgorithmTaskMapper;
import com.qczy.model.entity.AlgorithmEntity;
import com.qczy.model.entity.AlgorithmTaskEntity;
import com.qczy.model.request.DataEnhancementEntity;
import com.qczy.model.request.TrainEntity;
import com.qczy.service.AlgorithmService;
import com.qczy.service.AlgorithmTaskService;
import io.swagger.annotations.Api;
import io.swagger.annotations.ApiOperation;
import lombok.Data;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.util.ObjectUtils;
import org.springframework.web.bind.annotation.*;

import java.util.Date;
import java.util.HashMap;
import java.util.Map;

/**
 * @author ：gwj
 * @date ：Created in 2025-02-19 15:09
 * @description：
 * @modified By：
 * @version: $
 */
//数据增强
@RestController
@RequestMapping("/algorithm/dataEnhancementTask")
@Api(tags = "图像算子增强任务")
public class DataEnhancementController {

    @Autowired
    AlgorithmService algorithmService;
    @Autowired
    AlgorithmTaskService algorithmTaskService;
    @Autowired
    AlgorithmMapper algorithmMapper;

    @GetMapping("/getDataEnhancementLst")
    @ApiOperation("根据id获取当前数据集的导入记录）")
    public Result selectImportList(@RequestParam long  modelId) {
        AlgorithmEntity algorithmEntity = new AlgorithmEntity();
        algorithmEntity.setModelId(modelId+"");
        return Result.ok(algorithmService.getExampleList(algorithmEntity));
    }

    /**
     *  提交图像算子增强任务
     */
    @PostMapping("/submitTask")
    @ApiOperation("提交图像算子任务")
    public Result addTaskInfo(@RequestBody
                              DataEnhancementEntity dataEnhancement){

        if(ObjectUtils.isEmpty(dataEnhancement.getDatasetId())){
            return Result.fail("未选择输入数据集");
        }
       /* if(ObjectUtils.isEmpty(dataEnhancement.getDatasetTags())){
            return Result.fail("未选择输入数据集标签");
        }*/
        if(ObjectUtils.isEmpty(dataEnhancement.getDataEnhanceLst())){
            return Result.fail("未选择算子任务");
        }
        if(ObjectUtils.isEmpty(dataEnhancement.getDataEnhanceTactics())){
            return Result.fail("未选择算子策略");
        }
        AlgorithmTaskEntity algorithmTaskEntity = new AlgorithmTaskEntity();
        algorithmTaskEntity.setTaskInputName(dataEnhancement.getTaskInputName());
        algorithmTaskEntity.setModelId("9");
        algorithmTaskEntity.setDataSetId(dataEnhancement.getDatasetId());
        for (Map<String, Object> params : dataEnhancement.getDataEnhanceLst()) {
            algorithmTaskEntity.setAlgorithmId(ObjectUtils.isEmpty(algorithmTaskEntity.getAlgorithmId()) ? "": algorithmTaskEntity.getAlgorithmId() +"," + params.get("algorithmId").toString());
        }
        algorithmTaskEntity.setParamsLst( dataEnhancement.getDataEnhanceLst());

        algorithmTaskEntity.setTaskStat("开始");
        algorithmTaskEntity.setCreateTime(new Date());
//        String algorithmId = algorithmTaskEntity.getAlgorithmId();
//        AlgorithmEntity algorithmEntity = algorithmMapper.selectOne(new LambdaQueryWrapper<AlgorithmEntity>().eq(AlgorithmEntity::getId, algorithmId));
        algorithmTaskEntity.setIsTrain("0");
        algorithmTaskEntity.setRecordType(TaskRecordTypeConstants.DATAENHANCEMENT_TASK);
        algorithmTaskEntity.setDatasetOutId(dataEnhancement.getDatasetOutId());
        algorithmTaskEntity.setTaskName("图像算子增强");
        algorithmTaskEntity.setDataEnhanceType(dataEnhancement.getDataEnhanceType());
        algorithmTaskEntity.setDataEnhanceTactics(dataEnhancement.getDataEnhanceTactics());
        algorithmTaskEntity.setDataEnhanceMarkType(dataEnhancement.getDataEnhanceMarkType());
        algorithmTaskEntity.setDatasetEnhanceArea(dataEnhancement.getDatasetEnhanceArea());
        algorithmTaskEntity.setCreateTime(new Date());
        try{
            algorithmTaskService.submitTaskInfo(algorithmTaskEntity);
        }catch (Exception e){
            return Result.fail();
        }
        return Result.ok("图像算子任务已后台运行，任务编号为:"+algorithmTaskEntity.getTaskId());

    }

}
