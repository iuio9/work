package com.qczy.service.impl;

import cn.hutool.core.io.FileUtil;
import cn.hutool.core.io.IoUtil;
import cn.hutool.core.util.ObjectUtil;
import cn.hutool.http.HttpUtil;
import cn.hutool.json.JSONObject;
import cn.hutool.json.JSONUtil;
import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.qczy.common.constant.TaskRecordTypeConstants;
import com.qczy.mapper.TempFileMapper;
import com.qczy.model.entity.*;
import com.qczy.model.request.AlgorithmParams;
import com.qczy.model.request.BWTestRequest;
import com.qczy.model.response.DictSetTypeResponse;
import com.qczy.service.*;
import com.qczy.utils.ModelUtil;
import com.qczy.utils.URLUtils;
import org.apache.commons.compress.archivers.tar.TarArchiveEntry;
import org.apache.commons.compress.archivers.tar.TarArchiveOutputStream;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.scheduling.annotation.Async;
import org.springframework.stereotype.Service;

import java.io.*;
import java.util.*;
import java.util.stream.Collectors;

@Service
public class BlackAndWhiteTestServiceImpl implements BlackAndWhiteTestService {
    @Value("${upload.formalPath}")
    private String formalPath;
    @Value("${file.accessAddress}")
    private String httpPath;


    @Autowired
    private ModelUtil modelUtil;
    @Autowired
    private AlgorithmTaskService algorithmTaskService;

    @Autowired
    private TempFileMapper tempFileMapper;
    //获取文件
    private String getFiles(BWTestRequest bwTestRequest){
       try{
           String fileId = bwTestRequest.getFileId();
           TempFileEntity tempFileEntity = tempFileMapper.selectTempFileById(Integer.parseInt(fileId));
           String fdPath = tempFileEntity.getFdTempPath();
           bwTestRequest.setDiskPath(fdPath);
           fdPath = URLUtils.encodeURL(httpPath+ fdPath.replaceFirst("^" + formalPath, ""));
           return fdPath;

       }catch (Exception e){
           throw new RuntimeException("文件不存在");
       }

    }
    @Override
    @Async
    public void startTest(BWTestRequest bwTestRequest) {
        AlgorithmTaskEntity taskEntity = new AlgorithmTaskEntity();
        insertTaskRecord(bwTestRequest,taskEntity);
        //1. 收集页面文件
//        Map<String, List<FileEntity>> mapFiles =
//                bwTestRequest.getMapFiles();
//        String tarName = "BWTest_"+bwTestRequest.getTaskId()+".tar";
//        compressFileEntitiesToTar(mapFiles,formalPath+tarName);
        uploadTar(getFiles(bwTestRequest));
        toTest(bwTestRequest.getTestType(),taskEntity);
        endTask(taskEntity);
        String diskPath = bwTestRequest.getDiskPath();
        if (FileUtil.exist(diskPath)) {
            FileUtil.del(diskPath);
        }
    }

    @Override
    public AlgorithmTaskEntity searchResult(BWTestRequest bwTestRequest) {
        String taskId = bwTestRequest.getTaskId();
        AlgorithmTaskEntity algorithmTaskEntity = new AlgorithmTaskEntity();
        algorithmTaskEntity.setTaskId(Long.parseLong(taskId));

        AlgorithmTaskEntity byId = algorithmTaskService.getById(algorithmTaskEntity);
        if(ObjectUtil.isEmpty(byId)){
            throw new RuntimeException("该任务不存在");
        }
        if(!byId.getTaskStat().equalsIgnoreCase("结束")){
            throw new RuntimeException("测试任务未完成");
        }

        return byId;

    }

    @Override
    public List<DictDataEntity> getDictData() {
        List<DictDataEntity> list = dictDataService.list(
                new LambdaQueryWrapper<DictDataEntity>().eq(
                        DictDataEntity::getParentId, 123
                )
        );
        return Collections.emptyList();
    }

    @Autowired
    DictDataService dictDataService;
    @Autowired
    DataFatherService dataFatherService;
    @Autowired
    private DataSonService dataSonService;


    @Override
    public Map<String, List<FileEntity>> getBWFiles() {
        List<Integer> listofIds = new ArrayList<>();
        List<DataFatherEntity> fatherIds = dataFatherService.list(new LambdaQueryWrapper<DataFatherEntity>(
        ).in(
                DataFatherEntity::getDataTypeId, listofIds
        ));

        List<DataSonEntity> dataSonEntities = dataSonService.list(new LambdaQueryWrapper<DataSonEntity>(
        ).in(
                DataSonEntity::getFatherId, fatherIds
        ).eq(
                DataSonEntity::getVersion,1
        ));


//        dataSonEntities

        return Collections.emptyMap();
    }

    private void endTask(AlgorithmTaskEntity algorithmTaskEntity) {
        algorithmTaskEntity.setTaskStat("结束");
        algorithmTaskService.updateById(algorithmTaskEntity);
    }

    private void insertTaskRecord(BWTestRequest bwTestRequest,AlgorithmTaskEntity taskEntity) {
        taskEntity.setRecordType(TaskRecordTypeConstants.BLACKWHITETEST_TASK);
        taskEntity.setTaskStat("进行中");
        taskEntity.setTaskDesc(bwTestRequest.getTaskDesc());
        if(ObjectUtil.isNotEmpty(bwTestRequest.getTestType()) && bwTestRequest.getTestType().equalsIgnoreCase("1")){
            taskEntity.setTaskName("白盒测试");
        }else{
            taskEntity.setTaskName("黑盒测试");
        }
        taskEntity.setTaskInputName(bwTestRequest.getTaskName());
        taskEntity.setTaskInputName(bwTestRequest.getTaskName());
        taskEntity.setCreateTime(new Date());
        algorithmTaskService.addTaskInfo(taskEntity);
    }

    //2. 将页面所选参数的文件打包成tar 包
    public void compressFileEntitiesToTar(Map<String, List<FileEntity>> mapFiles, String outputTarPath) {
        try (FileOutputStream fos = new FileOutputStream(outputTarPath);
             BufferedOutputStream bos = new BufferedOutputStream(fos);
             TarArchiveOutputStream tarOut = new TarArchiveOutputStream(bos)) {

            // 设置长文件名支持
            tarOut.setLongFileMode(TarArchiveOutputStream.LONGFILE_GNU);

            // 遍历所有文件实体
            for (List<FileEntity> fileEntities : mapFiles.values()) {
                for (FileEntity fileEntity : fileEntities) {
                    String filePath = fileEntity.getFdPath(); // 假设FileEntity有getPath()方法
                    File file = new File(filePath);

                    if (file.exists()) {
                        // 使用原始文件名作为tar包内的名称
                        TarArchiveEntry entry = new TarArchiveEntry(file, file.getName());
                        tarOut.putArchiveEntry(entry);

                        try (FileInputStream fis = new FileInputStream(file);
                             BufferedInputStream bis = new BufferedInputStream(fis)) {
                             IoUtil.copy(bis, tarOut);
                        }

                        tarOut.closeArchiveEntry();
                        System.out.println("已添加: " + filePath);
                    } else {
                        System.out.println("警告: 文件不存在 - " + filePath);
                    }
                }
            }

            System.out.println("Tar包创建完成: " + outputTarPath);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    @Autowired
    private AlgorithmService algorithmService;
    //3. 调用下载接口
    private void uploadTar(String tarPath){
        //获取算法路径
        AlgorithmEntity algorithm = algorithmService.getById(30);
        algorithm.getUrl();
        Map<String,Object>  frontParams = new HashMap<>();
        frontParams.put("tar_path", tarPath);
        AlgorithmParams alParams = new AlgorithmParams();
        alParams.setParams(frontParams);
        try{
            modelUtil.model(algorithm, alParams);
        }catch (Exception e){
            System.out.println("黑白盒下载出错");
        }
        System.out.println("下载接口执行完毕");

    }
    //4. 将页面所选参数的(黑盒还是白盒选项)传递给测试接口
    private void toTest(String testType,AlgorithmTaskEntity algorithmTaskEntity){
        AlgorithmEntity algorithm = algorithmService.getById(31);
        Map<String,Object>  frontParams = new HashMap<>();
        frontParams.put("mode", testType);
        AlgorithmParams alParams = new AlgorithmParams();
        alParams.setParams(frontParams);
        System.out.println("开始黑白盒测试");
        try{
            String model = modelUtil.model(algorithm, alParams);
            algorithmTaskEntity.setTestResult(model);
        }catch (Exception e){
            System.out.println("黑白盒测试出错");
        }
        System.out.println("黑白盒测试完毕");
    }


}
