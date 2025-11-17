package com.qczy.controller.model;

import cn.hutool.core.util.ObjectUtil;
import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.core.conditions.query.QueryWrapper;
import com.baomidou.mybatisplus.core.metadata.IPage;
import com.baomidou.mybatisplus.extension.api.R;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.qczy.common.result.Result;
import com.qczy.mapper.AlgorithmTaskMapper;
import com.qczy.model.entity.AlgorithmModelEntity;
import com.qczy.model.entity.AlgorithmTaskEntity;
import com.qczy.model.entity.MarkInfoEntity;
import com.qczy.model.request.DataSonQueryRequest;
import com.qczy.model.request.DeleteRequest;
import com.qczy.service.AlgorithmModelService;
import com.qczy.service.AlgorithmTaskService;
import com.qczy.service.impl.AlgorithmModelServiceImpl;
import com.qczy.utils.StringUtils;
import com.qczy.utils.TrainUtil;
import io.swagger.annotations.Api;
import io.swagger.annotations.ApiOperation;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.security.core.parameters.P;
import org.springframework.web.bind.annotation.*;

import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

/**
 * @Author: hh
 * @Version: 1.0
 * @Date: 2024/8/22 20:01
 * @Description:
 */
@RestController
@RequestMapping("/algorithm/model")
@Api(tags = "算法模型管理")
public class AlgorithmModelController {

    @Autowired
    private AlgorithmModelService algorithmModelService;
    @Autowired
    private AlgorithmTaskService algorithmTaskService;
    @Autowired
    TrainUtil trainUtil;

    /**
     * 算法模型列表
     */
    @PostMapping("/getModelList")
    @ApiOperation("算法模型列表")
    public Result getModelList(
                                     @RequestBody AlgorithmModelEntity modelEntity) {
        return Result.ok(algorithmModelService.getModelList(modelEntity));
    }


    @PostMapping("/getModelPage")
    @ApiOperation("算法模型分页")
    public Result getModelPage(@RequestParam(value = "page", defaultValue = "1") int page,
                                     @RequestParam(value = "limit", defaultValue = "10") int limit,
                                     @ModelAttribute AlgorithmModelEntity modelEntity) {
        Page<AlgorithmModelEntity> pageParam = new Page<>(page, limit);

        return Result.ok(algorithmModelService.getModelPage(pageParam,modelEntity));
    }



    /**
     *  查询算法模型
     */
    @PostMapping("/getModelDetail")
    @ApiOperation("算法模型详情")
    public Result getModelDetail(@RequestBody AlgorithmModelEntity modelEntity) {
        if (ObjectUtil.isEmpty(modelEntity)) {
            return Result.fail("校验参数！");
        }

        return Result.ok(algorithmModelService.getModelDetails(modelEntity));
    }

    /**
     *  保存标注信息
     */
    @PostMapping("/addModel")
    @ApiOperation("添加算法模型记录")
    public Result addModel(@RequestBody AlgorithmModelEntity modelEntity){
        return Result.ok(algorithmModelService.addModelInfo(modelEntity));
    }

    /**
     *  修改标注信息
     */
    @PostMapping("/updateModel")
    @ApiOperation("添加算法模型记录")
    public Result updateModel(@RequestBody AlgorithmModelEntity modelEntity){
        return Result.ok(algorithmModelService.updateModelInfo(modelEntity));
    }

    /**
     *  保存标注信息
     */
    @PostMapping("/delModel")
    @ApiOperation("删除算法模型记录")
    public Result delModelInfo(@RequestBody DeleteRequest deleteRequest){
        if(ObjectUtil.isEmpty(deleteRequest)){
            return Result.fail("参数为空");
        }
        return Result.ok(algorithmModelService.delModelInfo(deleteRequest));
    }

    /**
     *  获取评估列表
     */
    @PostMapping("/getAssessLst")
    @ApiOperation("获取评估列表")
    public Result getAssessLst(@RequestParam(value = "page", defaultValue = "1") int page,
                               @RequestParam(value = "limit", defaultValue = "10") int limit,
                               @RequestBody AlgorithmModelEntity modelEntity){
        if(ObjectUtil.isEmpty(modelEntity)){
            return Result.fail("参数为空");
        }
        Page<AlgorithmTaskEntity> pageParam = new Page<>(page, limit);
        return Result.ok(algorithmTaskService.getAssessLst(modelEntity,pageParam));
    }


    /**
     *  当前任务评估详情
     */
    @PostMapping("/modelLastAssess")
    @ApiOperation("当前任务评估详情")
    public Result getAssessLst(
                               @RequestBody AlgorithmModelEntity modelEntity){
        if(ObjectUtil.isEmpty(modelEntity)){
            return Result.fail("参数为空");
        }
        LambdaQueryWrapper<AlgorithmTaskEntity> queryWrapper = new LambdaQueryWrapper<AlgorithmTaskEntity>();
        queryWrapper.eq(AlgorithmTaskEntity::getModelId, modelEntity.getModelId()) // 筛选已完成任务
                .orderByDesc(AlgorithmTaskEntity::getUpdateTime) // 按创建时间降序排列
                .last("LIMIT 1"); // 取最新一条
        AlgorithmTaskEntity one = algorithmTaskService.getOne(queryWrapper);
        return Result.ok(trainUtil.getAssess(one));
    }


    /**
     *  删除评估任务
     */
    @PostMapping("/delAssessTask")
    @ApiOperation("删除评估记录")
    public Result delAssessLst(@RequestBody AlgorithmTaskEntity algorithmTaskEntity){
        if(ObjectUtil.isEmpty(algorithmTaskEntity)){
            return Result.fail("参数为空");
        }
        boolean b = algorithmTaskService.delTaskInfo(algorithmTaskEntity);
        if(b){
            AlgorithmModelEntity one = algorithmModelService.getOne(new LambdaQueryWrapper<AlgorithmModelEntity>().eq(
                    AlgorithmModelEntity::getModelId, algorithmTaskEntity.getModelId()
            ));
            String assessLst = one.getAssessLst();
            if(ObjectUtil.isNotEmpty(assessLst)){
                List<String> list = Arrays.stream(assessLst.split(","))
                        .filter(s -> !s.equals(algorithmTaskEntity.getTaskId()+""))
                        .collect(Collectors.toList());
                // 将列表中的元素用逗号连接成新的字符串

                String result = String.join(",", list);
                one.setAssessLst(result);
                boolean b1 = algorithmModelService.updateById(one);
                return Result.ok(b1);

            }
        }
        return Result.ok(b);

    }





}
