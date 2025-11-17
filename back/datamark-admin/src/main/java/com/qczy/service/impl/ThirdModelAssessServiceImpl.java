package com.qczy.service.impl;

import cn.hutool.core.map.MapUtil;
import cn.hutool.core.util.ObjectUtil;
import cn.hutool.json.JSONArray;
import cn.hutool.json.JSONObject;
import cn.hutool.json.JSONUtil;
import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.qczy.common.annotation.MonitorProgress;
import com.qczy.common.constant.AssessConstants;
import com.qczy.common.constant.BizConstants;
import com.qczy.common.constant.ThirdAssessConstants;
import com.qczy.config.ProgressContext;
import com.qczy.handler.MyWebSocketHandler;
import com.qczy.mapper.AlgorithmMapper;
import com.qczy.mapper.AlgorithmTaskResultMapper;
import com.qczy.mapper.ModelAssessConfigMapper;
import com.qczy.mapper.ModelAssessTaskMapper;
import com.qczy.model.entity.*;
import com.qczy.model.request.AlgorithmParams;
import com.qczy.service.AlgorithmTaskService;
import com.qczy.service.DataSonService;
import com.qczy.service.ThirdModelAssessService;
import com.qczy.task.ProgressListener;
import com.qczy.utils.*;
import io.swagger.models.auth.In;
import org.aspectj.util.FileUtil;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import java.io.File;
import java.io.IOException;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * @author ：gwj
 * @date ：Created in 2025-05-29 14:00
 * @description：
 * @modified By：
 * @version: $
 */
@Service
public class ThirdModelAssessServiceImpl implements ThirdModelAssessService {
    @Autowired
    private AlgorithmTaskService algorithmTaskService;

    @Autowired
    private AlgorithmMapper algorithmMapper;


    @Value("${upload.formalPath}")
    private String formalPath;
    @Value("${file.accessAddress}")
    private String accessAddress;
    @Autowired
    ModelUtil modelUtil;

    @Autowired
    DataSonService dataSonService;

    @Autowired
    private AlgorithmTaskResultMapper algorithmTaskResultMapper;

    private final ConcurrentHashMap<Long, AtomicBoolean> taskControlFlags = new ConcurrentHashMap<>();
    @Autowired
    private ModelAssessConfigMapper modelAssessConfigMapper;
    @Autowired
    private ModelAssessTaskMapper modelAssessTaskMapper;
    @Autowired
    private TaskUtil taskUtil;

    @Autowired
    private TrainUtil trainUtil;
    @Autowired
    private MyWebSocketHandler myWebSocketHandler;

    @Override
    @MonitorProgress
    public void uploadClassifyJsonData(AlgorithmTaskEntity algorithmTaskEntity) {

    }

    @Override
    @MonitorProgress
    public void uploadJsonData(AlgorithmTaskEntity algorithmTaskEntity) {
        // 任务开始，设置为执行中状态
        ModelAssessTaskEntity modelAssessTaskEntity = modelAssessTaskMapper.selectOne(new LambdaQueryWrapper<ModelAssessTaskEntity>()
                .eq(ModelAssessTaskEntity::getId, algorithmTaskEntity.getTaskId()));

        updateTaskStatus(algorithmTaskEntity.getTaskId(), ThirdAssessConstants.IN_PROGRESS);
        taskControlFlags.put(algorithmTaskEntity.getTaskId(), new AtomicBoolean(true));

        try{
            uploadJsonByType(algorithmTaskEntity, modelAssessTaskEntity,27,"json","thirdJson");
            //将json目录下的数据label值提取到新目录
            uploadJsonByType(algorithmTaskEntity, modelAssessTaskEntity,28,"classifyJson","thirdClassifyJson");
            updateTaskStatus(algorithmTaskEntity.getTaskId(), ThirdAssessConstants.COMPLETED);
        }catch (Exception e){
            updateTaskStatus(algorithmTaskEntity.getTaskId(), ThirdAssessConstants.FAILED);
        }finally {
            taskControlFlags.remove(algorithmTaskEntity.getTaskId());
        }
    }



    private void uploadJsonByType(AlgorithmTaskEntity algorithmTaskEntity, ModelAssessTaskEntity modelAssessTaskEntity,Integer type,String sourceJsonChildDir,String thirdJsonChildDir) {

            ModelAssessConfigEntity modelAssessConfigEntity = modelAssessConfigMapper.selectOne(
                    new LambdaQueryWrapper<ModelAssessConfigEntity>().eq(
                            ModelAssessConfigEntity::getAssessTaskId, algorithmTaskEntity.getTaskId()
                    )
            );
            String sonId = modelAssessConfigEntity.getSonId();
            DataSonEntity one = dataSonService.getOne(new LambdaQueryWrapper<DataSonEntity>()
                    .eq(DataSonEntity::getSonId, sonId));
            AlgorithmEntity algorithmEntity = algorithmMapper.selectById(type);

            // 拿到数据集的目录
            String dataSetPath = formalPath + one.getFatherId() + "/v" + one.getVersion();
            String dataSetHttpPath = accessAddress + one.getFatherId() + "/v" + one.getVersion();

            // 获取json文件夹中的所有文件
            File jsonFolder = new File(dataSetPath + "/"+sourceJsonChildDir+"/");
            File[] jsonFilesArray = null;
            if (jsonFolder.exists() && jsonFolder.isDirectory()) {
                jsonFilesArray = jsonFolder.listFiles();
            }

            // 获取source文件夹中的所有文件
            File thirdJsonFolder = new File(dataSetPath + "/"+thirdJsonChildDir+"/");
            if (thirdJsonFolder.exists() && thirdJsonFolder.isDirectory()) {
                File[] thirdJsonFolderArray = thirdJsonFolder.listFiles();
                if (thirdJsonFolderArray != null) {
                    int i = 0;
                    for (File file : thirdJsonFolderArray) {
                        i++;
                        // 计算并报告进度百分比
                        int progress = (i * 100) / thirdJsonFolderArray.length;
                        //将进度保存到数据库
                        ProgressListener listener = ProgressContext.getProgressListener();
                        if (listener != null) {
                            listener.onAssessProgress(modelAssessTaskEntity,progress);
                        }
                        // 检查任务是否被终止或暂停
                        if (!taskControlFlags.getOrDefault(algorithmTaskEntity.getTaskId(), new AtomicBoolean(true)).get()) {
                            int currentStatus = getTaskStatus(algorithmTaskEntity.getTaskId());
                            if (ThirdAssessConstants.TERMINATION==currentStatus) {
                                return;
                            } else if (ThirdAssessConstants.PAUSE == currentStatus) {
                                // 暂停任务，等待恢复信号
                                waitForResume(algorithmTaskEntity.getTaskId());
                            }
                        }

                        if (file.isFile()) {
                            for (File file1 : jsonFilesArray) {
                                if (file1.isFile()) {
                                    if (removeFileExtension(file.getName()).equals(removeFileExtension(file1.getName()))) {
                                        // 放入map
                                        Map<String, Object> jsonAndImgMap = new HashMap<>();
                                        if(type==28){
                                            jsonAndImgMap.put("iou_threshold",1);
                                        }
                                        jsonAndImgMap.put("pre_path", URLUtils.encodeURL(dataSetHttpPath + "/thirdJson/" + file1.getName()));
                                        jsonAndImgMap.put("gt_path", URLUtils.encodeURL(dataSetHttpPath + "/json/" + file.getName()));
                                        jsonAndImgMap.put("task_id", algorithmTaskEntity.getTaskId());
                                        AlgorithmParams algorithmParams = new AlgorithmParams();
                                        algorithmParams.setParams(jsonAndImgMap);
                                        String model = null;
                                        try {
                                            model = modelUtil.model(algorithmEntity, algorithmParams);
                                        } catch (Exception e) {
                                            recordTaskParams(algorithmTaskEntity, e.getMessage());
                                        }
                                        recordTaskParams(algorithmTaskEntity, model);
                                    }
                                }
                            }
                        }
                    }
                }
            }
            // 任务正常完成
            if(type == 27){
                trainUtil.thirdCheckAssess(modelAssessTaskEntity);
            }else{
                trainUtil.thirdClassifyAssess(modelAssessTaskEntity);
            }
            one.setIsMany(0);
            dataSonService.updateById(one);
    }

    public void recordTaskParams( AlgorithmTaskEntity algorithmTaskEntity,String genResult){
        AlgorithmTaskResultEntity algorithmTaskResultEntity = new AlgorithmTaskResultEntity();
        algorithmTaskResultEntity.setTaskId(algorithmTaskEntity.getTaskId());
        algorithmTaskResultEntity.setTaskParams(JSONUtil.toJsonStr(algorithmTaskEntity.getParams()));
        algorithmTaskResultEntity.setTaskResult(JSONUtil.isJson(genResult)?JSONUtil.toJsonStr(genResult):genResult);
        algorithmTaskResultMapper.insert(algorithmTaskResultEntity);
    }

    // 控制任务状态的公共方法
    @Override
    public void controlTask(Long taskId, Integer status) {
        switch (status) {
            case ThirdAssessConstants.PAUSE:
                updateTaskStatus(taskId, ThirdAssessConstants.PAUSE);
                taskControlFlags.computeIfAbsent(taskId, k -> new AtomicBoolean(true)).set(false);
                break;
            case ThirdAssessConstants.CONTINUE:
                updateTaskStatus(taskId, ThirdAssessConstants.IN_PROGRESS);
                taskControlFlags.computeIfAbsent(taskId, k -> new AtomicBoolean(false)).set(true);
                synchronized (taskId.toString()) {
                    taskId.toString().notifyAll(); // 唤醒等待的线程
                }
                break;
            case ThirdAssessConstants.TERMINATION:
                updateTaskStatus(taskId, ThirdAssessConstants.TERMINATION);
                taskControlFlags.computeIfAbsent(taskId, k -> new AtomicBoolean(true)).set(false);
                break;
            default:
                // 忽略不支持的状态
        }
    }

    // 等待任务恢复的辅助方法
    private void waitForResume(Long taskId) {
        synchronized (taskId.toString()) {
            while (ThirdAssessConstants.PAUSE == getTaskStatus(taskId)) {
                try {
                    taskId.toString().wait(); // 线程进入等待状态
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                    return;
                }
            }
        }
    }

    // 更新任务状态的辅助方法
    private void updateTaskStatus(Long taskId, Integer status) {
        ModelAssessTaskEntity modelAssessTaskEntity = modelAssessTaskMapper.selectOne(new LambdaQueryWrapper<ModelAssessTaskEntity>()
                .eq(ModelAssessTaskEntity::getId, taskId.intValue()));
        if (modelAssessTaskEntity != null) {
            modelAssessTaskEntity.setTaskStatus(status);
            if(status == ThirdAssessConstants.TERMINATION){
                modelAssessTaskEntity.setTaskProgress("0%");
                modelAssessTaskEntity.setTaskStatus(ThirdAssessConstants.FAILED);
                myWebSocketHandler.disconnectUser(BizConstants.ASSESS_PROGRESS,taskId+"");
            }
            modelAssessTaskMapper.updateById(modelAssessTaskEntity);
        }
    }

    // 获取任务状态的辅助方法
    private int getTaskStatus(Long taskId) {
        ModelAssessTaskEntity modelAssessTaskEntity = modelAssessTaskMapper.selectOne(new LambdaQueryWrapper<ModelAssessTaskEntity>()
                .eq(ModelAssessTaskEntity::getId, taskId.intValue()));
        return modelAssessTaskEntity != null ? modelAssessTaskEntity.getTaskStatus() : null;
    }

    // 假设已存在的方法
    private String removeFileExtension(String fileName) {
        int dotIndex = fileName.lastIndexOf('.');
        return (dotIndex == -1) ? fileName : fileName.substring(0, dotIndex);
    }


    @Override
    public Map viewAssessResult(AlgorithmTaskEntity algorithmTaskEntity) {
        Long taskId = algorithmTaskEntity.getTaskId();
        ModelAssessTaskEntity modelAssessTaskEntity = modelAssessTaskMapper.selectOne(new LambdaQueryWrapper<ModelAssessTaskEntity>()
                .eq(ModelAssessTaskEntity::getId, taskId));
        JSONObject jsonObject = JSONUtil.parseObj(modelAssessTaskEntity.getTaskResult());
        Map map = JSONUtil.toBean(jsonObject, Map.class);
        return map;
    }


}
