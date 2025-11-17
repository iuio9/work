package com.qczy.service.impl;

import cn.hutool.core.util.ObjectUtil;
import cn.hutool.json.JSONUtil;
import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.core.conditions.query.QueryWrapper;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import com.qczy.common.constant.TaskRecordTypeConstants;
import com.qczy.common.result.Result;
import com.qczy.mapper.AlgorithmMapper;
import com.qczy.mapper.AlgorithmTaskMapper;
import com.qczy.mapper.AlgorithmTaskResultMapper;
import com.qczy.mapper.FileMapper;
import com.qczy.model.entity.*;
import com.qczy.model.request.SegmentParams;
import com.qczy.model.request.TrainEntity;
import com.qczy.service.AlgorithmService;
import com.qczy.service.AlgorithmTaskService;
import com.qczy.utils.HttpUtil;
import com.qczy.utils.TaskUtil;
import com.qczy.utils.URLUtils;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import java.io.UnsupportedEncodingException;
import java.net.URLEncoder;
import java.util.*;
import java.util.stream.Collectors;

/**
 * @author ：gwj
 * @date ：Created in 2024-08-27 11:09
 * @description：
 * @modified By：
 * @version: $
 */
@Service
public class AlgorithmTaskServiceImpl extends ServiceImpl<AlgorithmTaskMapper, AlgorithmTaskEntity> implements AlgorithmTaskService {

    private final HttpUtil httpUtil;

    @Autowired
    private FileMapper fileMapper;
    @Autowired
    private AlgorithmService algorithmServie;

    public AlgorithmTaskServiceImpl(HttpUtil httpUtil) {
        this.httpUtil = httpUtil;
    }

    @Autowired
    TaskUtil taskUtil;

    @Autowired
    AlgorithmTaskMapper algorithmTaskMapper;

    @Autowired
    AlgorithmTaskResultMapper algorithmTaskResultMapper;

    @Autowired
    AlgorithmMapper algorithmMapper;
    private LambdaQueryWrapper<AlgorithmTaskEntity> getQueryConditon(AlgorithmTaskEntity taskEntity){
        LambdaQueryWrapper<AlgorithmTaskEntity> algorithmTasklEntityLambdaQueryWrapper = new LambdaQueryWrapper<AlgorithmTaskEntity>()
//                .eq(ObjectUtil.isNotEmpty(taskEntity.getModelId()),AlgorithmTaskEntity::getModelId, taskEntity.getModelId())
                .eq(ObjectUtil.isNotEmpty(taskEntity.getTaskId()),AlgorithmTaskEntity::getTaskId, taskEntity.getTaskId())
//                .eq(ObjectUtil.isNotEmpty(taskEntity.getTaskStat()),AlgorithmTaskEntity::getTaskStat, taskEntity.getTaskStat())
                .eq(ObjectUtil.isNotEmpty(taskEntity.getTaskName()),AlgorithmTaskEntity::getTaskName, taskEntity.getTaskName())
                .orderByDesc(AlgorithmTaskEntity::getTaskId)
                .orderByDesc(AlgorithmTaskEntity::getCreateTime);
        return algorithmTasklEntityLambdaQueryWrapper;
    }

    private LambdaQueryWrapper<AlgorithmTaskEntity> updateConditon(AlgorithmTaskEntity taskEntity){
        LambdaQueryWrapper<AlgorithmTaskEntity> algorithmTasklEntityLambdaQueryWrapper = new LambdaQueryWrapper<AlgorithmTaskEntity>()
                .eq(AlgorithmTaskEntity::getTaskId, taskEntity.getTaskId());
        return algorithmTasklEntityLambdaQueryWrapper;
    }



    @Override
    public boolean addTaskInfo(AlgorithmTaskEntity taskEntity) {
        int insert = algorithmTaskMapper.insert(taskEntity);
        return insert > 0;
    }

    @Override
    public boolean editTaskInfo(AlgorithmTaskEntity taskEntity) {
        return algorithmTaskMapper.update(taskEntity,updateConditon(taskEntity)) > 0;
    }

    @Override
    public boolean delTaskInfo(AlgorithmTaskEntity taskEntity) {
        int del = algorithmTaskMapper.delete(updateConditon(taskEntity));
        return del > 0;
    }

    @Override
    public AlgorithmTaskEntity getDataDetails(AlgorithmTaskEntity taskEntity) {
        return algorithmTaskMapper.selectOne(getQueryConditon(taskEntity));
    }

    @Override
    public Page<AlgorithmTaskEntity> getTaskPage(Page<AlgorithmTaskEntity> pageParam, AlgorithmTaskEntity taskEntity) {
        return algorithmTaskMapper.getTaskPage(pageParam,taskEntity);
    }

    @Override
    public List<AlgorithmTaskEntity> getTaskList(AlgorithmTaskEntity taskEntity) {
        return algorithmTaskMapper.selectList(getQueryConditon(taskEntity));
    }




    @Override
    public Map<String, Object> startSegment(SegmentParams segmentParams) {
        //1. 执行分割
        AlgorithmEntity algorithmEntity = algorithmMapper.selectOne(new LambdaQueryWrapper<AlgorithmEntity>().eq(AlgorithmEntity::getId, 1));
        // 2.插入任务数据
        AlgorithmTaskEntity algorithmTaskEntity = new AlgorithmTaskEntity();
        algorithmTaskEntity.setTaskStat("进行中");
        algorithmTaskEntity.setTaskName(algorithmEntity.getAlgorithmName());
        Map<String,Object> map  =  new HashMap<>();
        String post =null;
        try{
            post =httpUtil.post(algorithmEntity.getUrl(), segmentParams);
            algorithmTaskEntity.setTaskStat("结束");
            map.put("segmentResult", JSONUtil.parseObj(post));


        }catch (Exception e){
            algorithmTaskEntity.setTaskStat("异常");
            map.put("segmentResult", JSONUtil.parseObj(post));
        }
        algorithmTaskEntity.setTaskProgress("100%");
        int rows = algorithmTaskMapper.insert(algorithmTaskEntity);

        //将任务参数和返回结果(json)进行保存
        Long taskId = algorithmTaskEntity.getTaskId();
        AlgorithmTaskResultEntity algorithmTaskResultEntity = new AlgorithmTaskResultEntity();
        algorithmTaskResultEntity.setTaskId(taskId);
        algorithmTaskResultEntity.setTaskParams(JSONUtil.toJsonStr(segmentParams));
        algorithmTaskResultEntity.setTaskResult(JSONUtil.toJsonStr(post));
        algorithmTaskResultMapper.insert(algorithmTaskResultEntity);
        if(rows > 0){
            map.put("taskId",algorithmTaskEntity.getTaskId()+"");
        }


        return map;
    }

    @Override
    public void submitTaskInfo(AlgorithmTaskEntity algorithmTaskEntity) {
        //需要一张图片和markInfo

        int insert = algorithmTaskMapper.insert(algorithmTaskEntity);
        if(insert > 0){
            System.out.println("新增任务");
        }
        if(algorithmTaskEntity.getRecordType().equalsIgnoreCase(TaskRecordTypeConstants.DATAENHANCEMENT_TASK)){
            taskUtil.execDtaaEnhancementTask(algorithmTaskEntity);
        }else{
            taskUtil.execTask(algorithmTaskEntity);
        }
//        return post;
    }

    @Value("${upload.formalPath}")
    private String formalPath;
    @Value("${file.accessAddress}")
    private String accessAddress;

    @Value("${file.accessAddress1}")
    private String accessAddress1;

    @Override
    public Page<FileEntity> getTaskResult(Page<FileEntity> pageParam,AlgorithmTaskEntity algorithmTaskEntity) {
        Long taskId = algorithmTaskEntity.getTaskId();

        Page<FileEntity> fileEntityPage =
                fileMapper.selectPage(pageParam, new LambdaQueryWrapper<FileEntity>().eq(FileEntity::getTaskId, taskId)
                        .ne(FileEntity::getFdSuffix,".json"));

        if (ObjectUtil.isNotEmpty(fileEntityPage.getRecords())) {
            for (FileEntity fileEntity : fileEntityPage.getRecords()) {
                String prefixToRemove = formalPath;
                String fdSuffix = fileEntity.getFdSuffix();

                // 使用 replaceFirst 方法去掉前缀
                String replace = URLUtils.encodeURL(accessAddress1 + fileEntity.getFdPath().replaceFirst("^" + prefixToRemove, ""));
                fileEntity.setHttpFilePath(replace);

            }
        }
        return fileEntityPage;
    }

    @Override
    public Page<AlgorithmTaskEntity> getAssessLst(AlgorithmModelEntity modelEnity,Page<AlgorithmTaskEntity> pageParam) {
        String assessLst = modelEnity.getAssessLst();

        Page<AlgorithmTaskEntity> algorithmModelEntities =null;
        if(ObjectUtil.isNotEmpty(assessLst)){
            List<Integer> integerList = Arrays.stream(assessLst.split(","))
                    .map(Integer::parseInt)  // 将每个字符串转换为整数
                    .collect(Collectors.toList());
            return algorithmTaskMapper.selectPage(pageParam, new QueryWrapper<AlgorithmTaskEntity>()
                    .in("task_id", integerList)
            );
        }
        return algorithmModelEntities;
    }

}
