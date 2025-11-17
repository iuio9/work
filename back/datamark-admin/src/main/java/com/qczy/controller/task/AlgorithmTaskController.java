package com.qczy.controller.task;

import cn.hutool.core.io.FileUtil;
import cn.hutool.core.util.ObjectUtil;
import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.qczy.common.constant.TaskRecordTypeConstants;
import com.qczy.common.result.Result;
import com.qczy.mapper.AlgorithmMapper;
import com.qczy.model.entity.*;
import com.qczy.model.request.TrainEntity;
import com.qczy.service.AlgorithmTaskService;
import com.qczy.service.DataFatherService;
import com.qczy.service.DataSonService;
import io.swagger.annotations.Api;
import io.swagger.annotations.ApiOperation;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.util.MultiValueMap;
import org.springframework.util.ObjectUtils;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.io.BufferedReader;
import java.io.FileWriter;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.Date;

/**
 * @Author: hh
 * @Version: 1.0
 * @Date: 2024/8/22 20:01
 * @Description:
 */
@RestController
@RequestMapping("/algorithm/task")
@Api(tags = "算法任务进度")
public class AlgorithmTaskController {




    @Autowired
    private AlgorithmTaskService algorithmTaskService;



    /**
     * 算法任务列表
     */
    @PostMapping("/getTaskList")
    @ApiOperation("算法任务列表")
    public Result getTaskList(
            @RequestBody AlgorithmTaskEntity algorithmTaskEntity) {
        return Result.ok( algorithmTaskService.getTaskList(algorithmTaskEntity));
    }


    /**
     * 算法任务列表
     */
    @PostMapping("/getTaskResult")
    @ApiOperation("处理结果获取")
    public Result getTaskResult(@RequestParam(value = "page", defaultValue = "1") int page,
                                @RequestParam(value = "limit", defaultValue = "10") int limit,
            @RequestBody AlgorithmTaskEntity algorithmTaskEntity) {
        if(ObjectUtil.isEmpty(algorithmTaskEntity.getTaskId()+"")){
            return Result.fail("任务编号不能为空");
        }

        if(ObjectUtil.isEmpty(algorithmTaskEntity.getDataSetId()+"")){
            return Result.fail("数据集编号不能为空");
        }
        Page<FileEntity> pageParam = new Page<>(page, limit);
        return Result.ok( algorithmTaskService.getTaskResult(pageParam,algorithmTaskEntity));
    }
    /**
     * 算法任务列表
     */
    @PostMapping("/getTaskPage")
    @ApiOperation("算法任务分页")
    public Result getTaskPage(@RequestParam(value = "page", defaultValue = "1") int page,
                                     @RequestParam(value = "limit", defaultValue = "10") int limit,
                                        @ModelAttribute AlgorithmTaskEntity algorithmTaskEntity) {
        Page<AlgorithmTaskEntity> pageParam = new Page<>(page, limit);
        if(ObjectUtil.isEmpty(algorithmTaskEntity.getRecordType())){
            algorithmTaskEntity.setRecordType("0");
        }
        return Result.ok( algorithmTaskService.getTaskPage(pageParam, algorithmTaskEntity));
    }


    @Autowired
    private DataSonService dataSonService;
    @Autowired
    private DataFatherService dataFatherService;
    /**
     *  查询算法任务
     */
    @PostMapping("/getTaskDetail")
    @ApiOperation("算法任务详情")
    public Result getDataDetails(@RequestBody AlgorithmTaskEntity  taskEntity) {
        if(ObjectUtil.isEmpty(taskEntity)){
            return Result.fail("校验参数");
        }
        AlgorithmTaskEntity details= algorithmTaskService.getDataDetails(taskEntity);
        DataSonEntity son = dataSonService.getOne(new LambdaQueryWrapper<DataSonEntity>().eq(
                DataSonEntity::getSonId, details.getDataSetId()
        ));
        DataFatherEntity father = dataFatherService.getOne(new LambdaQueryWrapper<DataFatherEntity>().eq(
                DataFatherEntity::getGroupId, son.getFatherId()
        ));

        details.setDataSetTotal(son.getFileIds().split(",").length+"");
        details.setDataSetName(father.getGroupName());
        return Result.ok(details);

    }

    /**
     *  保存标注信息
     */
    @PostMapping("/addTask")
    @ApiOperation("添加算法任务记录")
    public Result addTaskInfo(@RequestBody AlgorithmTaskEntity taskEntity){
        return Result.ok(algorithmTaskService.addTaskInfo(taskEntity));
    }

    /**
     *  保存标注信息
     */
    @PostMapping("/editTask")
    @ApiOperation("编辑算法任务记录")
    public Result editTaskInfo(@RequestBody AlgorithmTaskEntity taskEntity){
        return Result.ok(algorithmTaskService.editTaskInfo(taskEntity));
    }



    @Autowired
    AlgorithmMapper algorithmMapper;

    /**
     *  提交算法任务
     */
    @PostMapping("/submitTask")
    @ApiOperation("提交算法任务")
    public Result addTaskInfo(@RequestBody
    TrainEntity trainEntity){
//        if(ObjectUtils.isEmpty(trainEntity.getAlgorithmParam())){
//            return Result.fail("算法参数不能为空");
//        }
        AlgorithmTaskEntity algorithmTaskEntity = new AlgorithmTaskEntity();
        algorithmTaskEntity.setTaskInputName(trainEntity.getTaskInputName());
        algorithmTaskEntity.setModelId(trainEntity.getModelId());
        algorithmTaskEntity.setDataSetId(trainEntity.getDatasetId());
        algorithmTaskEntity.setAlgorithmId(trainEntity.getAlgorithmId());
        algorithmTaskEntity.setParams(trainEntity.getAlgorithmParam());
        algorithmTaskEntity.setTaskStat("开始");
        algorithmTaskEntity.setCreateTime(new Date());
        String algorithmId = algorithmTaskEntity.getAlgorithmId();
        AlgorithmEntity algorithmEntity = algorithmMapper.selectOne(new LambdaQueryWrapper<AlgorithmEntity>().eq(AlgorithmEntity::getId, algorithmId));
        algorithmTaskEntity.setIsTrain("0");
        algorithmTaskEntity.setRecordType(TaskRecordTypeConstants.REEASONING_TASK);
        algorithmTaskEntity.setDatasetOutId(trainEntity.getDatasetOutId());
        algorithmTaskEntity.setTaskName(algorithmEntity.getAlgorithmName());
        algorithmTaskEntity.setCreateTime(new Date());
        try{
             algorithmTaskService.submitTaskInfo(algorithmTaskEntity);
        }catch (Exception e){
            return Result.fail();
        }
        return Result.ok(algorithmEntity.getAlgorithmName()+"任务已后台运行，任务编号为:"+algorithmTaskEntity.getTaskId());

    }

    @PostMapping("/delTask")
    @ApiOperation("删除算法任务记录")
    public Result delTaskInfo(@RequestBody AlgorithmTaskEntity taskEntity){
        if(ObjectUtil.isEmpty(taskEntity.getTaskId())){
            return Result.fail("taskId不能为空");
        }
        return Result.ok(algorithmTaskService.delTaskInfo(taskEntity));
    }

    @PostMapping("/readFileToJson")
    @ApiOperation("读取文件json信息")
    public Result readFileToJson(@RequestParam("file") MultipartFile file){
        try {
            // 读取文件内容
            StringBuilder jsonContent = new StringBuilder();
            try (BufferedReader reader = new BufferedReader(
                    new InputStreamReader(file.getInputStream(), StandardCharsets.UTF_8))) {
                String line;
                while ((line = reader.readLine()) != null) {
                    jsonContent.append(line);
                }
            }

            // 将 JSON 字符串转换为 JSON 对象（使用 Jackson 或 Gson）
            // 示例使用 Jackson
            ObjectMapper mapper = new ObjectMapper();
            JsonNode jsonNode = mapper.readTree(jsonContent.toString());
            // 返回结果
            return Result.ok(jsonNode); // 假设 Result 类有一个 success 方法来返回数据
        } catch (Exception e) {
            e.printStackTrace();
            return Result.fail("读取文件失败"); // 假设 Result 类有一个 failure 方法处理错误
        }
    }




}
