package com.qczy.controller.algorithm;

import cn.hutool.core.bean.BeanUtil;
import cn.hutool.core.util.ObjectUtil;
import cn.hutool.json.JSONArray;
import cn.hutool.json.JSONObject;
import cn.hutool.json.JSONUtil;
import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.qczy.common.constant.AssessConstants;
import com.qczy.common.constant.TaskRecordTypeConstants;
import com.qczy.common.result.Result;
import com.qczy.mapper.AlgorithmMapper;
import com.qczy.model.entity.AlgorithmEntity;
import com.qczy.model.entity.AlgorithmModelEntity;
import com.qczy.model.entity.AlgorithmTaskEntity;
import com.qczy.model.entity.TrainingParams;
import com.qczy.model.request.TrainEntity;
import com.qczy.service.AlgorithmModelService;
import com.qczy.service.AlgorithmTaskService;
import com.qczy.service.impl.AlgorithmModelServiceImpl;
import com.qczy.utils.ModelUtil;
import com.qczy.utils.TaskUtil;
import com.qczy.utils.TrainUtil;
import com.qczy.utils.TrainingService;
import io.swagger.annotations.Api;
import io.swagger.annotations.ApiOperation;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;

import javax.validation.ConstraintViolation;
import javax.validation.ConstraintViolationException;
import java.util.*;

/**
 * @Author: gwj
 * @Version: 1.0
 * @Date: 2024/8/22 20:01
 * @Description:
 */
@RestController
@RequestMapping("/algorithm/model")
@Api(tags = "算法==模型训练")
public class ModelTrainController {

    @Autowired
    private AlgorithmTaskService algorithmTaskService;

    @Autowired
    TrainUtil trainUtil;
    @Autowired
    private AlgorithmMapper algorithmMapper;
    @Autowired
    TrainingService trainingService;
    @Autowired
    private ModelUtil modelUtil;
    @Autowired
    private AlgorithmModelService algorithmModelService;

    /**
     *  添加算法风格转化 （传递图片和选择算法）
     */
    @PostMapping("/trainStart")
    @ApiOperation("模型训练开始")
    public Result start(@RequestBody TrainEntity trainEntity){




        //需要一张图片和markInfo
        AlgorithmTaskEntity algorithmTaskEntity = new AlgorithmTaskEntity();
        algorithmTaskEntity.setModelId(trainEntity.getModelId());
        algorithmTaskEntity.setTaskInputName(trainEntity.getTaskInputName());
        algorithmTaskEntity.setDataSetId(trainEntity.getDatasetId());
        algorithmTaskEntity.setAlgorithmId(trainEntity.getAlgorithmId());
        algorithmTaskEntity.setParams(trainEntity.getAlgorithmParam());
        algorithmTaskEntity.setDatasetOutId(trainEntity.getDatasetOutId());
        algorithmTaskEntity.setTaskDesc(trainEntity.getTaskDesc());
        algorithmTaskEntity.setTaskStat("开始");
        algorithmTaskEntity.setIsTrain("1"); // 1 为训练任务 0 为非训练任务
        algorithmTaskEntity.setIsAssess("0");
        algorithmTaskEntity.setRecordType(TaskRecordTypeConstants.TRAIN_TASK);
        algorithmTaskEntity.setTrainType(trainEntity.getAlgorithmParam().get("mode").toString());
        AlgorithmEntity algorithmEntity = algorithmMapper.selectOne(new LambdaQueryWrapper<AlgorithmEntity>()
                .eq(AlgorithmEntity::getTrainType, algorithmTaskEntity.getTrainType())
        );
        algorithmTaskEntity.setTaskName(algorithmEntity.getAlgorithmName());
        algorithmTaskEntity.setCreateTime(new Date());
        algorithmTaskService.addTaskInfo(algorithmTaskEntity);
        Map<String, Object> params = algorithmTaskEntity.getParams();
        JSONArray trainParams = JSONUtil.parseArray(params.get("trainPrams"));

        Map<String, Object> stringObjectHashMap = new HashMap<>();
        for (Object trainParam : trainParams) {
            Map bean = BeanUtil.toBean(trainParam, Map.class);
            stringObjectHashMap.put(bean.get("key").toString(),bean.get("value"));
        }
        TrainingParams trainingParams = BeanUtil.toBean(stringObjectHashMap, TrainingParams.class);

        try{
            trainingService.validateTrainingParams(trainingParams);
        }catch (Exception e){
            return Result.fail( e.getMessage());
        }
        trainUtil.execTask(algorithmTaskEntity);
        return Result.ok(algorithmTaskEntity.getTaskId());
    }


    /**
     *  添加算法风格转化 （传递图片和选择算法）
     */
    @PostMapping("/trainAssess/{id}")
    @ApiOperation("模型评估")
    public Result assess(@PathVariable("id") Integer id){
        AlgorithmTaskEntity byId = algorithmTaskService.getById(id);
        if(ObjectUtil.isEmpty(byId)){
            return Result.fail("评估对象不存在");
        }
        return Result.ok(trainUtil.getAssess(byId));
    }

    /**
     *  添加算法风格转化 （传递图片和选择算法）
     */
    @PostMapping("/trainAssess/start")
    @ApiOperation("开始模型评估")
    public Result assessStart(@RequestBody TrainEntity trainEntity){
        //需要一张图片和markInfo
        AlgorithmTaskEntity algorithmTaskEntity = new AlgorithmTaskEntity();
        algorithmTaskEntity.setModelId(trainEntity.getModelId());
        algorithmTaskEntity.setTaskInputName(trainEntity.getTaskInputName());
        algorithmTaskEntity.setDataSetId(trainEntity.getDatasetId());
        algorithmTaskEntity.setAlgorithmId(trainEntity.getAlgorithmId());
        algorithmTaskEntity.setParams(trainEntity.getAlgorithmParam());
        algorithmTaskEntity.setDatasetOutId(trainEntity.getDatasetOutId());
        algorithmTaskEntity.setTaskDesc(trainEntity.getTaskDesc());
        algorithmTaskEntity.setTaskStat("开始");
        algorithmTaskEntity.setIsTrain("0"); // 1 为训练任务 0 为非训练任务
        algorithmTaskEntity.setIsAssess(AssessConstants.ASSESS_ING);
        algorithmTaskEntity.setRecordType(TaskRecordTypeConstants.ASSESSMENT_TASK);
//        algorithmTaskEntity.setTrainType(trainEntity.getAlgorithmParam().get("mode").toString());
//        AlgorithmEntity algorithmEntity = algorithmMapper.selectOne(new LambdaQueryWrapper<AlgorithmEntity>()
//                .eq(AlgorithmEntity::getTrainType, algorithmTaskEntity.getTrainType())
//        );
//        algorithmTaskEntity.setTaskName(algorithmEntity.getAlgorithmName());
        algorithmTaskEntity.setCreateTime(new Date());
        algorithmTaskService.addTaskInfo(algorithmTaskEntity);
        trainUtil.execAssess(algorithmTaskEntity);
        return Result.ok(algorithmTaskEntity.getTaskId());
    }

    @PostMapping("/trainStop/{id}")
    @ApiOperation("模型训练停止")
    public Result stop(@PathVariable("id") Integer id){
        if(ObjectUtil.isEmpty(id)){
            return Result.fail("任务ID为空");
        }
        AlgorithmTaskEntity byId = algorithmTaskService.getById(id);
        if(ObjectUtil.isNotEmpty(byId) && ObjectUtil.isNotEmpty(byId.getPid())){
            //需要一张图片和markInfo
            String result = trainUtil.trainStop(byId);
            return Result.ok(result);
        } else if (ObjectUtil.isNotEmpty(byId)) {
            if(!byId.getTaskStat().contains("结束")){
                byId.setTaskStat("结束(手动)");
            }
            algorithmTaskService.updateById(byId);
            return Result.ok("任务已结束");
        }else {
            return Result.fail("任务不存在");
        }

    }




}
