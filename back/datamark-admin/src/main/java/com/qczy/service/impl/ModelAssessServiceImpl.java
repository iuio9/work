package com.qczy.service.impl;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.core.conditions.query.QueryWrapper;
import com.baomidou.mybatisplus.core.metadata.IPage;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.qczy.common.generate.GenerateReportPdfForm;
import com.qczy.common.generate.GenerateWordByApplyNoForm;
import com.qczy.common.generate.GenerateWordByApplyNoForm1;
import com.qczy.common.generate.GenerateWordForm;
import com.qczy.mapper.*;
import com.qczy.model.entity.DataSonEntity;
import com.qczy.model.entity.ModelAssessConfigEntity;
import com.qczy.model.entity.ModelAssessTaskEntity;
import com.qczy.model.entity.ModelDebugLog;
import com.qczy.model.entity.domain.ModelMarkInfoEntity;
import com.qczy.model.request.AssessTaskRequest;
import com.qczy.model.request.DeleteRequest;
import com.qczy.model.response.ModelAssessResponse;
import com.qczy.model.response.ModelDebugLogResponse;
import com.qczy.model.response.ModelReportResponse;
import com.qczy.service.ModelAssessService;
import com.qczy.utils.CurrentLoginUserUtils;
import com.qczy.utils.MultipartFileUtils;
import com.qczy.utils.ParamsUtils;
import com.qczy.utils.StringUtils;
import org.jpedal.parser.shape.S;
import org.springframework.beans.BeanUtils;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.util.CollectionUtils;
import org.springframework.web.multipart.MultipartFile;

import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import java.io.File;
import java.io.IOException;
import java.util.Date;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

/**
 * @Author: hh
 * @Version: 1.0
 * @Date: 2025/6/5 11:21
 * @Description:
 */
@Service
public class ModelAssessServiceImpl implements ModelAssessService {


    @Autowired
    private ModelAssessTaskMapper modelAssessTaskMapper;

    @Autowired
    private ModelAssessConfigMapper modelAssessConfigMapper;

    @Autowired
    private CurrentLoginUserUtils currentLoginUserUtils;
    @Autowired
    private DataSonMapper dataSonMapper;
    @Autowired
    private GenerateReportPdfForm generateReportPdfForm;
    @Autowired
    private GenerateWordForm generateWordForm;
    @Autowired
    private ModelDebugLogMapper modelDebugLogMapper;
    @Autowired
    private GenerateWordByApplyNoForm generateWordByApplyNoForm;
    @Autowired
    private GenerateWordByApplyNoForm1 generateWordByApplyNoForm1;
    @Autowired
    private ModelMarkInfoMapper modelMarkInfoMapper;
    @Value("${upload.paramsJsonPath}")
    private String uploadParamsJsonPath;
    @Value("${upload.formalPath}")
    private String uploadFormalPath;


    @Override
    @Transactional //保持原子性
    public int createAssessTask(AssessTaskRequest request) {
        ModelAssessTaskEntity modelAssessTaskEntity = new ModelAssessTaskEntity();
        BeanUtils.copyProperties(request, modelAssessTaskEntity);
        modelAssessTaskEntity.setModelBaseId(request.getModelId());
        modelAssessTaskEntity.setTaskStatus(6); //TODO 待处理 、 准备调用厂商
        modelAssessTaskEntity.setTaskProgress("0%");
        modelAssessTaskEntity.setUserId(currentLoginUserUtils.getCurrentLoginUserId()); //获取当前用户id
        modelAssessTaskEntity.setCreateTime(new Date());
        int result = modelAssessTaskMapper.insert(modelAssessTaskEntity);
        if (result <= 0) {
            return 0;
        }
        ModelAssessConfigEntity modelAssessConfigEntity = new ModelAssessConfigEntity();
        BeanUtils.copyProperties(request, modelAssessConfigEntity);
        String params = analysisParams(request.getModelParamsFile(), modelAssessConfigEntity);
        if (!StringUtils.isEmpty(params))
            modelAssessConfigEntity.setModelParams(params);
        modelAssessConfigEntity.setAssessTaskId(modelAssessTaskEntity.getId()); // 填写模型任务id
        modelAssessConfigMapper.insert(modelAssessConfigEntity);

        // TODO 更改数据集状态，设置为不可操作
        DataSonEntity dataSonEntity = dataSonMapper.getDataSonBySonId(modelAssessConfigEntity.getSonId());
        if (dataSonEntity == null) {
            throw new RuntimeException("数据集不存在！");
        } else {
            dataSonEntity.setIsMany(1);
            // 删除一下旧的第三方评估目录
            // 目录
            String dir = uploadFormalPath + dataSonEntity.getFatherId() + "/v" + dataSonEntity.getVersion() + "/thirdJson/";
            deleteDirectory(dir);
            return dataSonMapper.updateById(dataSonEntity);
        }
    }

    @Override
    @Transactional
    public int editTask(AssessTaskRequest request) {
        ModelAssessTaskEntity modelAssessTaskEntity = modelAssessTaskMapper.selectById(request.getId());
        if (modelAssessTaskEntity == null) {
            return 0;
        }
        BeanUtils.copyProperties(request, modelAssessTaskEntity);
        modelAssessTaskEntity.setModelBaseId(request.getModelId());
        modelAssessTaskEntity.setTaskStatus(6); //TODO 待处理 、 准备调用厂商
        modelAssessTaskEntity.setTaskProgress("0%");
        modelAssessTaskEntity.setUserId(currentLoginUserUtils.getCurrentLoginUserId()); //获取当前用户id
        int result = modelAssessTaskMapper.updateById(modelAssessTaskEntity);
        if (result <= 0) {
            return 0;
        }
        ModelAssessConfigEntity modelAssessConfigEntity = modelAssessConfigMapper.selectOne(
                new LambdaQueryWrapper<ModelAssessConfigEntity>()
                        .eq(ModelAssessConfigEntity::getAssessTaskId, modelAssessTaskEntity.getId())
        );
        if (modelAssessConfigEntity == null) {
            return 0;
        }

        // 判断是否更换了数据集
        if (!StringUtils.isEmpty(modelAssessConfigEntity.getSonId())) {
            DataSonEntity targetDataSon = null;
            // 1.判断是否是之前的数据集
            if (!modelAssessConfigEntity.getSonId().equals(request.getSonId())) {
                // 现在是不相同的，直接释放之前的数据集状态，变成可操作
                DataSonEntity sourceDataSon = dataSonMapper.getDataSonBySonId(modelAssessConfigEntity.getSonId());
                if (sourceDataSon != null) {
                    sourceDataSon.setIsMany(0);
                    dataSonMapper.updateById(sourceDataSon);
                }
                // 设置新的数据集，并且锁定状态，不可操作
                targetDataSon = dataSonMapper.getDataSonBySonId(request.getSonId());
                if (targetDataSon == null) {
                    throw new RuntimeException("数据集不存在！");
                }
                targetDataSon.setIsMany(1);
                dataSonMapper.updateById(targetDataSon);
            } else {
                targetDataSon = dataSonMapper.getDataSonBySonId(modelAssessConfigEntity.getSonId());
            }
            // 2. 不管换不换数据集，都要删除一下第三方文件
            // 目录
            String dir = uploadFormalPath + targetDataSon.getFatherId() + "/v" + targetDataSon.getVersion() + "/thirdJson/";
            deleteDirectory(dir);
        }

        // 记录一下id
        Integer id = modelAssessConfigEntity.getId();
        BeanUtils.copyProperties(request, modelAssessConfigEntity);
        String params = analysisParams(request.getModelParamsFile(), modelAssessConfigEntity);
        if (!StringUtils.isEmpty(params))
            modelAssessConfigEntity.setModelParams(params);
        modelAssessConfigEntity.setAssessTaskId(modelAssessTaskEntity.getId()); // 填写模型任务id
        modelAssessConfigEntity.setId(id);

        return modelAssessConfigMapper.updateById(modelAssessConfigEntity);
    }

    @Override
    public IPage<ModelAssessResponse> listPage(Page<ModelAssessResponse> pageParam, ModelAssessTaskEntity modelAssessTaskEntity) {
        return modelAssessTaskMapper.listPage(pageParam, modelAssessTaskEntity);
    }

    @Override
    @Transactional
    public int delTask(DeleteRequest request) {
        ModelAssessTaskEntity modelAssessTaskEntity = modelAssessTaskMapper.selectById(request.getId());
        if (modelAssessTaskEntity == null) {
            return 0;
        }
        ModelAssessConfigEntity modelAssessConfigEntity = modelAssessConfigMapper.selectOne(
                new LambdaQueryWrapper<ModelAssessConfigEntity>()
                        .eq(ModelAssessConfigEntity::getAssessTaskId, modelAssessTaskEntity.getId())
        );
        if (modelAssessConfigEntity == null) {
            return 0;
        }

        // 批量删除模型标注信息数据
        List<ModelMarkInfoEntity> modelMarkInfoEntityList = modelMarkInfoMapper.selectList(
                new LambdaQueryWrapper<ModelMarkInfoEntity>()
                        .eq(ModelMarkInfoEntity::getTaskId, modelAssessTaskEntity.getId())
                        .eq(ModelMarkInfoEntity::getSonId, modelAssessConfigEntity.getSonId())
        );
        if (!CollectionUtils.isEmpty(modelMarkInfoEntityList)) {
            modelMarkInfoMapper.deleteBatchIds(modelMarkInfoEntityList.stream().map(ModelMarkInfoEntity::getId).collect(Collectors.toList()));
        }


        // 修改数据集状态
        if (!StringUtils.isEmpty(modelAssessConfigEntity.getSonId())) {
            DataSonEntity dataSonEntity = dataSonMapper.getDataSonBySonId(modelAssessConfigEntity.getSonId());
            if (dataSonEntity != null) {
                dataSonEntity.setIsMany(0);
                dataSonMapper.updateById(dataSonEntity);
            }

        }

        modelAssessConfigMapper.delete(
                new LambdaQueryWrapper<ModelAssessConfigEntity>()
                        .eq(ModelAssessConfigEntity::getAssessTaskId, request.getId())
        );

        return modelAssessTaskMapper.deleteById(modelAssessTaskEntity.getId());
    }

    @Override
    public IPage<ModelReportResponse> reportListPage(Page<ModelReportResponse> pageParam) {
        return modelAssessTaskMapper.reportListPage(pageParam);
    }

    @Override
    public void generateWord(Integer id, HttpServletRequest request, HttpServletResponse response) {
        generateWordForm.downloadWord(id, request, response);
    }

    @Override
    public void generateApplyNoWordZip(Integer id, HttpServletRequest request, HttpServletResponse response) {
        //generateWordByApplyNoForm.downloadWordZip(id,request,response);
        generateWordByApplyNoForm1.downloadWord(id, request, response);
    }

    @Override
    public int getTaskStatus(Integer id) {
        return modelAssessTaskMapper.selectById(id).getTaskStatus();
    }

    @Override
    public ModelDebugLogResponse getModelDebugInfo(AssessTaskRequest request) {
        List<ModelDebugLog> modelDebugLogList = modelDebugLogMapper.selectList(
                new LambdaQueryWrapper<ModelDebugLog>()
                        .eq(ModelDebugLog::getModelBaseId, request.getModelId())
        );
        if (CollectionUtils.isEmpty(modelDebugLogList)) {
            return null;
        }

        ModelDebugLogResponse modelDebugLogResponse = new ModelDebugLogResponse();
        modelDebugLogResponse.setModelAddress(modelDebugLogList.get(0).getModelAddress());
        modelDebugLogResponse.setRequestType(modelDebugLogList.get(0).getRequestType());
        return modelDebugLogResponse;
    }

    @Override
    public boolean isTaskNameRepeat(String taskName, Integer id) {
        if (StringUtils.isEmpty(taskName)) {
            return false;
        }
        QueryWrapper<ModelAssessTaskEntity> queryWrapper = new QueryWrapper<>();
        queryWrapper.eq("task_name", taskName);
        if (id != null) {
            queryWrapper.ne("id", id);
        }
        return modelAssessTaskMapper.selectCount(queryWrapper) > 0;
    }

    @Override
    public AssessTaskRequest taskDetails(Integer id) {
        AssessTaskRequest assessTaskRequest = new AssessTaskRequest();
        ModelAssessTaskEntity modelAssessTaskEntity = modelAssessTaskMapper.selectById(id);
        if (modelAssessTaskEntity == null) {
            return null;
        }
        //BeanUtils.copyProperties(modelAssessTaskEntity, assessTaskRequest);
        ModelAssessConfigEntity modelAssessConfigEntity = modelAssessConfigMapper.selectOne(
                new LambdaQueryWrapper<ModelAssessConfigEntity>()
                        .eq(ModelAssessConfigEntity::getAssessTaskId, modelAssessTaskEntity.getId())
        );
        if (modelAssessConfigEntity != null) {
            BeanUtils.copyProperties(modelAssessConfigEntity, assessTaskRequest);
        }
        BeanUtils.copyProperties(modelAssessTaskEntity, assessTaskRequest);
        assessTaskRequest.setModelId(modelAssessTaskEntity.getModelBaseId());
        return assessTaskRequest;
    }

    @Override
    public int deleteFile(DeleteRequest request) {
        ModelAssessConfigEntity modelAssessConfigEntity = modelAssessConfigMapper.selectOne(
                new LambdaQueryWrapper<ModelAssessConfigEntity>()
                        .eq(ModelAssessConfigEntity::getAssessTaskId, request.getId())
        );
        if (modelAssessConfigEntity != null && !StringUtils.isEmpty(request.getServerKey())) {
            if (request.getServerKey().equals("modelParamsFile")) {
                if (!StringUtils.isEmpty(modelAssessConfigEntity.getModelParamsPath())) {
                    new File(modelAssessConfigEntity.getModelParamsPath()).delete();
                    modelAssessConfigEntity.setModelParamsPath("");
                    modelAssessConfigEntity.setModelParams("");
                }
            }
            return modelAssessConfigMapper.updateById(modelAssessConfigEntity);
        }
        return 1;
    }


    /**
     * 解析参数文件（JSON/XLSX）并保存原始文件到本地，更新配置实体中的文件路径
     *
     * @param file                    上传的文件
     * @param modelAssessConfigEntity 模型评估配置实体
     * @return 解析后的JSON字符串
     * @throws RuntimeException 当文件无效、解析失败或保存失败时抛出
     */
    public String analysisParams(MultipartFile file, ModelAssessConfigEntity modelAssessConfigEntity) {
        // 1. 校验文件有效性
        if (file == null || file.isEmpty()) {
            return null;
        }

        try {
            // 2. 获取并校验文件名
            String originalFileName = file.getOriginalFilename();
            if (originalFileName == null || originalFileName.trim().isEmpty()) {
                throw new RuntimeException("无法获取有效的文件名");
            }

            // 3. 校验保存目录路径
            if (uploadParamsJsonPath == null || uploadParamsJsonPath.trim().isEmpty()) {
                throw new RuntimeException("文件保存目录未配置");
            }

            // 4. 解析文件（只支持JSON/XLSX）
            String fileSuffix = MultipartFileUtils.getFileExtensionWithoutDot(file);
            Map<String, Object> requestData;
            switch (fileSuffix.toLowerCase()) { // 忽略大小写，兼容 .JSON/.XLSX 等后缀
                case "json":
                    requestData = ParamsUtils.readJsonFile(file);
                    break;
                case "xlsx":
                    requestData = ParamsUtils.convertXlsxToMap(file);
                    break;
                default:
                    throw new RuntimeException("不支持的文件类型：" + fileSuffix + "，仅支持json、xlsx");
            }

            // 5. 转换为JSON字符串
            ObjectMapper objectMapper = new ObjectMapper();
            String result = objectMapper.writeValueAsString(requestData);

            // 6. 保存原始文件到本地（处理重名）
            File dir = new File(uploadParamsJsonPath);
            if (!dir.exists() && !dir.mkdirs()) { // 确保目录创建成功
                throw new RuntimeException("无法创建文件保存目录：" + uploadParamsJsonPath);
            }

            // 处理文件名（无扩展名时直接加时间戳）
            int extIndex = originalFileName.lastIndexOf(".");
            String fileNameWithoutExt = (extIndex == -1) ? originalFileName : originalFileName.substring(0, extIndex);
            String ext = (extIndex == -1) ? "" : originalFileName.substring(extIndex);

            // 构建目标路径（避免重名）
            String localFilePath = uploadParamsJsonPath + fileNameWithoutExt + ext;
            File destFile = new File(localFilePath);
            if (destFile.exists()) {
                localFilePath = uploadParamsJsonPath + fileNameWithoutExt + "_" + System.currentTimeMillis() + ext;
                destFile = new File(localFilePath);
            }

            // 保存文件（transferTo可能因权限/路径问题失败，需捕获）
            file.transferTo(destFile);

            // 7. 删除旧文件（校验旧路径有效性）
            String oldFilePath = modelAssessConfigEntity.getModelParamsPath();
            if (oldFilePath != null && !oldFilePath.trim().isEmpty()) {
                File oldFile = new File(oldFilePath);
                if (oldFile.exists())
                    oldFile.delete();
            }

            // 8. 更新新路径
            modelAssessConfigEntity.setModelParamsPath(localFilePath);

            return result;

        } catch (IOException e) {
            // 区分异常场景，添加具体信息
            throw new RuntimeException("文件处理失败（解析或保存错误）：" + e.getMessage(), e);
        }
    }


    /**
     * 递归删除指定目录下的所有文件和文件夹，包括目录本身
     *
     * @param directoryPath 要删除的目录路径
     */
    public static void deleteDirectory(String directoryPath) {
        // 创建File对象
        File directory = new File(directoryPath);

        // 检查目录是否存在
        if (!directory.exists()) {
            System.out.println("目录不存在: " + directoryPath);
            return;
        }

        // 检查是否为目录
        if (!directory.isDirectory()) {
            System.out.println("不是有效的目录: " + directoryPath);
            return;
        }

        // 获取目录下的所有文件和子目录
        File[] files = directory.listFiles();
        if (files != null) {
            for (File file : files) {
                // 如果是子目录，递归删除
                if (file.isDirectory()) {
                    deleteDirectory(file.getAbsolutePath());
                } else {
                    // 如果是文件，直接删除
                    boolean isDeleted = file.delete();
                    if (!isDeleted) {
                        System.out.println("无法删除文件: " + file.getAbsolutePath());
                    }
                }
            }
        }

        // 最后删除空目录
        boolean isDirDeleted = directory.delete();
        if (!isDirDeleted) {
            System.out.println("无法删除目录: " + directoryPath);
        }
    }


}
