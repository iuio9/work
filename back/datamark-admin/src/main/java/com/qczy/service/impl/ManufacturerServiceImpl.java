package com.qczy.service.impl;

import com.alibaba.fastjson.JSON;
import com.alibaba.fastjson.JSONArray;
import com.alibaba.fastjson.JSONObject;
import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.core.toolkit.CollectionUtils;
import com.fasterxml.jackson.annotation.JsonInclude;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.qczy.common.label.JsonParser;
import com.qczy.handler.TaskWebSocketHandler;
import com.qczy.mapper.*;
import com.qczy.model.entity.*;
import com.qczy.model.entity.domain.ModelMarkInfoEntity;
import com.qczy.service.ManufacturerService;
import com.qczy.utils.FileDownloadUtils;
import com.qczy.utils.HttpUtil2;
import com.qczy.utils.MultipartFileUtils;
import com.qczy.utils.StringUtils;
import okhttp3.*;
import org.apache.commons.io.FileUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.scheduling.annotation.Async;
import org.springframework.stereotype.Service;
import org.springframework.util.ObjectUtils;

import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.DirectoryStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.stream.Collectors;

@Service
public class ManufacturerServiceImpl implements ManufacturerService {
    private static final Logger log = LoggerFactory.getLogger(ManufacturerServiceImpl.class);

    private final ConcurrentHashMap<Integer, AtomicBoolean> taskControlFlags = new ConcurrentHashMap<>();
    private final ObjectMapper objectMapper = new ObjectMapper();
    private final OkHttpClient client = new OkHttpClient();

    // 常量定义：任务类型
    private static final int TASK_TYPE_CLASSIFY = 1;      // 分类任务
    private static final int TASK_TYPE_DETECTION = 2;     // 目标检测任务

    @Autowired
    private ModelAssessTaskMapper modelAssessTaskMapper;
    @Autowired
    private ModelAssessConfigMapper modelAssessConfigMapper;

    @Value("${upload.address}")
    private String uploadAddress;

    @Autowired
    private FileMapper fileMapper;

    @Autowired
    private DataSonMapper dataSonMapper;

    @Autowired
    private HttpUtil2 httpUtil2;

    @Value("${upload.formalPath}")
    private String uploadFormalPath;

    @Autowired
    private FileDownloadUtils fileDownloadUtils;
    @Autowired
    private MarkInfoMapper markInfoMapper;
    @Autowired
    private DataSonLabelMapper dataSonLabelMapper;
    @Autowired
    private LabelMapper labelMapper;
    @Autowired
    private ModelMarkInfoMapper modelMarkInfoMapper;

    @Override
    @Async
    public void startManufacturer(ModelAssessTaskEntity modelAssessTaskEntity) {
        Integer taskId = modelAssessTaskEntity.getId();
        String taskIdStr = taskId.toString();

        log.info("开始执行任务: {}", taskId);
        sendMessage(taskIdStr, "task_log", "开始执行任务...");

        try {
            // 获取任务对象
            ModelAssessTaskEntity taskEntity = modelAssessTaskMapper.selectById(taskId);
            if (taskEntity == null) {
                String msg = "未找到任务配置: taskId=" + taskId;
                log.warn(msg);
                sendMessage(taskIdStr, "task_log", msg);
                return;
            }
            // 任务类型
            Integer taskType = taskEntity.getTaskType();  // 1: 分类任务 、 2：目标检测

            // 1. 获取任务配置
            ModelAssessConfigEntity modelAssessConfigEntity = modelAssessConfigMapper.selectOne(
                    new LambdaQueryWrapper<ModelAssessConfigEntity>()
                            .eq(ModelAssessConfigEntity::getAssessTaskId, taskId)
            );
            if (modelAssessConfigEntity == null) {
                String msg = "未找到任务配置: taskId=" + taskId;
                log.warn(msg);
                sendMessage(taskIdStr, "task_log", msg);
                return;
            }

            // 2. 验证配置参数
            String manufacturerAddress = modelAssessConfigEntity.getModelAddress();
            if (StringUtils.isEmpty(manufacturerAddress)) {
                String msg = "厂商地址为空: taskId=" + taskId;
                log.warn(msg);
                sendMessage(taskIdStr, "task_log", msg);
                return;
            }

            String requestTypeStr = "POST";

            // 3. 获取数据集信息
            DataSonEntity dataSonEntity = dataSonMapper.getDataSonBySonId(modelAssessConfigEntity.getSonId());
            if (dataSonEntity == null || StringUtils.isEmpty(dataSonEntity.getFileIds())) {
                String msg = "数据集为空: taskId=" + taskId + ", sonId=" + modelAssessConfigEntity.getSonId();
                log.warn(msg);
                sendMessage(taskIdStr, "task_log", msg);
                return;
            }

            // 4. 开始任务
            taskControlFlags.put(taskId, new AtomicBoolean(true));

            String startMsg = "------------------------------------开始调用厂商API------------------------------------------";
            log.info(startMsg);
            sendMessage(taskIdStr, "task_log", startMsg);

            String addressMsg = "taskId=" + taskId + ", 厂商地址=" + manufacturerAddress;
            log.info(addressMsg);
            sendMessage(taskIdStr, "task_log", addressMsg);

            // 5. 获取文件列表
            List<FileEntity> fileList = fileMapper.getFileListBySonId(dataSonEntity.getSonId());
            if (CollectionUtils.isEmpty(fileList)) {
                String msg = "文件列表为空: taskId=" + taskId + ", sonId=" + dataSonEntity.getSonId();
                log.warn(msg);
                sendMessage(taskIdStr, "task_log", msg);
                taskControlFlags.remove(taskId);
                return;
            }

            int successCount = 0;
            int failCount = 0;
            int totalFiles = fileList.size();

            // 8. 构建请求参数（每个文件单独创建Map，避免参数覆盖）
            Map<String, Object> params = new HashMap<>();
            // 判断有没有上传的参数，如果有，则一起发送
            if (!StringUtils.isEmpty(modelAssessConfigEntity.getModelParams())) {
                // 解析
                Map<String, Object> responseJson = objectMapper.readValue(modelAssessConfigEntity.getModelParams(), Map.class);
                params.putAll(responseJson);
            }

            // 6. 逐个处理文件
            for (int i = 0; i < fileList.size(); i++) {
                FileEntity fileEntity = fileList.get(i);
                int progress = (int) ((i + 1) * 100.0 / totalFiles);

                // 检查任务是否已停止
                if (!taskControlFlags.getOrDefault(taskId, new AtomicBoolean(false)).get()) {
                    String msg = "任务已停止: taskId=" + taskId + ", 文件ID=" + fileEntity.getId();
                    log.info(msg);
                    sendMessage(taskIdStr, "task_log", msg);
                    break;
                }

                // 更新进度
                sendMessage(taskIdStr, "progress", progress);

                try {
                    // 7. 转换文件为Base64
                    String imgBase64 = MultipartFileUtils.imagePathToBase64(fileEntity.getFdPath());
                    if (imgBase64 == null) {
                        String msg = "文件转换失败: 文件ID=" + fileEntity.getId() + ", 文件名=" + fileEntity.getFdName();
                        log.error(msg);
                        sendMessage(taskIdStr, "task_log", msg);
                        failCount++;
                        continue;
                    }

                    // 每个文件替换 base64
                    if (StringUtils.isEmpty(modelAssessConfigEntity.getModelFileName())) {
                        params.put("image_base64", imgBase64);
                    } else {
                        params.put(modelAssessConfigEntity.getModelFileName(), imgBase64);
                    }


                    // 构建JSON请求
                    Request request = httpUtil2.buildHttpRequest(
                            manufacturerAddress,
                            "POST",
                            params
                    );
                    // 10. 发送请求并处理响应
                    try (Response response = client.newCall(request).execute()) {
                        if (!response.isSuccessful()) {
                            String msg = "HTTP请求失败: 文件ID=" + fileEntity.getId() + ", 状态码=" + response.code();
                            log.error(msg);
                            sendMessage(taskIdStr, "task_log", msg);
                            failCount++;
                            continue;
                        }

                        // 11. 解析响应
                        String responseBody = response.body().string();
                        try {
                            Map<String, Object> responseJson = objectMapper.readValue(responseBody, Map.class);
                            Object resultData = responseJson.get("data");
                            successCount++;

                            // 12. 处理成功结果
                            if (taskType == 1) {
                                classificationTaskResult(taskId, fileEntity, resultData, dataSonEntity, taskIdStr, modelAssessConfigEntity.getLabelMap());
                            } else if (taskType == 2) {
                                processResult(taskId, fileEntity, resultData, dataSonEntity, taskIdStr, modelAssessConfigEntity.getLabelMap());
                            }


                        } catch (Exception e) {
                            String msg = "解析响应失败: 文件ID=" + fileEntity.getId() + ", 文件名=" + fileEntity.getFdName();
                            log.error(msg, e);
                            sendMessage(taskIdStr, "task_log", msg + ", 错误: " + e.getMessage());
                            failCount++;
                        }
                    }
                } catch (Exception e) {
                    String msg = "处理文件异常: 文件ID=" + fileEntity.getId() + ", 文件名=" + fileEntity.getFdName();
                    log.error(msg, e);
                    sendMessage(taskIdStr, "task_log", msg + ", 错误: " + e.getMessage());
                    failCount++;

                    String errorMsg = "------------------------------------调用失败------------------------------------------";
                    log.error(errorMsg);
                    sendMessage(taskIdStr, "task_log", errorMsg);
                }
            }

            // 13. 任务完成，清理状态
            taskControlFlags.remove(taskId);

            String completeMsg = "厂商调用完成: taskId=" + taskId + ", 成功=" + successCount + ", 失败=" + failCount;
            log.info(completeMsg);
            sendMessage(taskIdStr, "task_log", completeMsg);
            sendMessage(taskIdStr, "complete", true);

            String successMsg = "------------------------------------任务执行完毕！------------------------------------------";
            log.info(successMsg);
            sendMessage(taskIdStr, "task_log", successMsg);

        } catch (Exception e) {
            String errorMsg = "执行任务异常: taskId=" + taskId;
            log.error(errorMsg, e);
            sendMessage(taskIdStr, "task_log", errorMsg + ", 错误: " + e.getMessage());
            sendMessage(taskIdStr, "error", e.getMessage());
        }
    }

    /**
     * 处理厂商返回的结果，并且处理原始图片 提取里面的 label作为json文件 (分类任务)
     */
    private void classificationTaskResult(Integer taskId, FileEntity fileEntity, Object resultData, DataSonEntity dataSonEntity, String taskIdStr, String labelMap) {
        String logMsg = "厂商返回结果: taskId=" + taskId + ", 文件ID=" + fileEntity.getId() + ", 结果=" + resultData;
        log.info(logMsg);
        sendMessage(taskIdStr, "task_log", logMsg);
        try {
            // 人工标注的存放json目录
            String classifyJsonDir = uploadFormalPath + dataSonEntity.getFatherId() + "/v" + dataSonEntity.getVersion() + "/classifyJson/";
            // 第三方存放的json目录
            String thirdClassifyJsonDir = uploadFormalPath + dataSonEntity.getFatherId() + "/v" + dataSonEntity.getVersion() + "/thirdClassifyJson/";
            // 判断目录是否存在，不存在则进行创建
            File classifyJsonDirAddress = new File(classifyJsonDir);
            if (!classifyJsonDirAddress.exists()) {
                classifyJsonDirAddress.mkdirs();
                String mkdirMsg = "创建目录: " + classifyJsonDirAddress;
                log.info(mkdirMsg);
                sendMessage(taskIdStr, "task_log", mkdirMsg);
            }

            File thirdClassifyJsonDirAddress = new File(thirdClassifyJsonDir);
            if (!thirdClassifyJsonDirAddress.exists()) {
                thirdClassifyJsonDirAddress.mkdirs();
                String mkdirMsg = "创建目录: " + thirdClassifyJsonDirAddress;
                log.info(mkdirMsg);
                sendMessage(taskIdStr, "task_log", mkdirMsg);
            }

            // 1. 获取原始文件名（包含后缀）
            String originalFileName = fileEntity.getFdName();
            if (StringUtils.isEmpty(originalFileName)) {
                originalFileName = "result_" + fileEntity.getId();
                String warnMsg = "文件fdName为空，使用默认文件名: " + originalFileName;
                log.warn(warnMsg);
                sendMessage(taskIdStr, "task_log", warnMsg);
            }

            // 2. 提取文件名前缀（去除原有后缀）
            String prefix = removeSuffix(originalFileName);

            // 3. 构建新文件名（文件前缀.json）
            String fileName = prefix + ".json";

            // 4. 替换结果中的img_name为原始文件名
            if (resultData instanceof Map) {
                @SuppressWarnings("unchecked")
                Map<String, Object> resultMap = (Map<String, Object>) resultData;
                resultMap.put("img_name", originalFileName);
            }

            // 5.1 先转换人工标注的
            File sourceMarkPath = new File(uploadFormalPath + dataSonEntity.getFatherId() + "/v" + dataSonEntity.getVersion() + "/json/" + fileName);
            Object classifyJsonResult = processManualAnnotationJson(sourceMarkPath, taskIdStr);
            String resultJson = objectMapper.writeValueAsString(classifyJsonResult);
            // 写入文件
            fileDownloadUtils.writeJsonToFile(resultJson, classifyJsonDir + "/", prefix, ".json");

            // 5.2 提取第三方给的
            Object convertedResult = processThirdPartyJson(resultData, taskIdStr, labelMap);
            String resultJson1 = objectMapper.writeValueAsString(convertedResult);
            // 写入文件
            fileDownloadUtils.writeJsonToFile(resultJson1, thirdClassifyJsonDir + "/", prefix, ".json");


        } catch (Exception e) {
            String fileName = StringUtils.isEmpty(fileEntity.getFdName()) ? "默认文件名" : fileEntity.getFdName();
            String errorMsg = "保存结果失败 [文件ID:" + fileEntity.getId() + " | 原文件名:" + fileName + "]";
            log.error(errorMsg, e);
            sendMessage(taskIdStr, "task_log", errorMsg + ", 错误: " + e.getMessage());
        }
    }


    /**
     * 处理人工标注JSON文件，提取第一个label并返回指定格式
     * 仅包含label字段，格式为 {"label":"提取到的标签"} 或 {"label":null}
     *
     * @param sourceMarkPath 源标注JSON文件路径
     * @param taskIdStr      任务ID字符串
     * @return 仅包含label字段的Map对象
     */
    private Object processManualAnnotationJson(File sourceMarkPath, String taskIdStr) {
        Map<String, Object> resultMap = new HashMap<>(1);
        resultMap.put("label", null);

        if (!sourceMarkPath.exists()) {
            String infoMsg = "创建JSON文件: " + sourceMarkPath.getAbsolutePath() + "，返回label:null";
            log.info(infoMsg);
            sendMessage(taskIdStr, "task_log", infoMsg);
            return resultMap;
        }

        try {
            // 替换原读取代码，指定编码
            String jsonContent = FileUtils.readFileToString(sourceMarkPath, "UTF-8");
            JSONObject jsonObject = JSON.parseObject(jsonContent);
            JSONArray shapesArray = jsonObject.getJSONArray("shapes");

            if (shapesArray != null && shapesArray.size() > 0) {
                JSONObject firstShape = shapesArray.getJSONObject(0);
                String firstLabel = firstShape.getString("label");
                resultMap.put("label", firstLabel);

                String labelMsg = "提取第一个label: " + firstLabel;
                log.info(labelMsg);
                sendMessage(taskIdStr, "task_log", labelMsg);
            } else {
                String warnMsg = "shapes数据为空，返回label:null";
                log.warn(warnMsg);
                sendMessage(taskIdStr, "task_log", warnMsg);
            }

            return resultMap;

        } catch (Exception e) {
            String errorMsg = "处理失败，返回label:null";
            log.error(errorMsg, e);
            sendMessage(taskIdStr, "task_log", errorMsg + ", 错误: " + e.getMessage());
            return resultMap;
        }
    }


    /**
     * 处理第三方返回的JSON数据（适配dec_rsts字段结构，与convertToAnnotationFormat保持一致）
     * 提取并返回仅包含label和score的Map，支持通过labelMap替换label
     *
     * @param resultData  第三方返回的原始数据（含dec_rsts字段）
     * @param taskIdStr   任务ID字符串（用于日志输出）
     * @param labelMapStr 字符串类型的标签映射关系（JSON数组格式）
     * @return Map<String, Object> 仅包含"label"和"score"字段的Map
     */
    private Map<String, Object> processThirdPartyJson(Object resultData, String taskIdStr, String labelMapStr) {
        // 1. 将结果转换为Map处理（适配厂商返回的结构）
        Map<String, Object> resultMap = objectMapper.convertValue(resultData, Map.class);
        log.debug("转换后的resultMap结构: {}", resultMap); // 确认是否包含dec_rsts

        // 2. 解析标签映射关系
        Map<String, String> labelMap = parseLabelMap(labelMapStr);
        
        // 3. 初始化返回结果
        Map<String, Object> result = new HashMap<>(2);
        result.put("label", null);
        result.put("score", null);

        // 4. 提取dec_rsts数组（与convertToAnnotationFormat一致，厂商返回的检测结果字段）
        @SuppressWarnings("unchecked")
        List<Map<String, Object>> detections = (List<Map<String, Object>>) resultMap.get("dec_rsts");
        log.debug("提取到的dec_rsts数组: {}", detections);

        if (detections != null && !detections.isEmpty()) {
            // 处理第一个检测结果（与convertToAnnotationFormat逻辑一致）
            Map<String, Object> firstDetection = detections.get(0);
            log.debug("第一个检测结果内容: {}", firstDetection);

            // 设置label（使用class_name）
            String code = firstDetection.get("class_name").toString();
            if (labelMap != null && !StringUtils.isEmpty(code) && labelMap.containsKey(code)) {
                // key存在！判断value是否为null,如果为空,则不进行替换
                String value = labelMap.get(code);
                if (value != null) {
                    result.put("label", value);
                } else {
                    result.put("label", firstDetection.get("class_name"));
                }
            } else {
                result.put("label", firstDetection.get("class_name"));
            }

            // 7. 提取score（保留原始值）
            Object scoreObj = firstDetection.get("score");
            result.put("score", scoreObj);

            log.debug("第三方数据提取完成，taskId={}, label={}, score={}",
                    taskIdStr, result.get("label"), result.get("score"));
        } else {
            log.warn("第三方数据中dec_rsts数组为空、不存在，taskId={}", taskIdStr);
        }

        return result;
    }

    /**
     * 解析标签映射关系（适配JSON数组格式）
     */
    private Map<String, String> parseLabelMap(String labelMapStr) {
        Map<String, String> labelMap = new HashMap<>();
        if (StringUtils.isEmpty(labelMapStr)) {
            return labelMap;
        }

        try {
            JSONArray jsonArray = JSON.parseArray(labelMapStr);
            for (Object item : jsonArray) {
                @SuppressWarnings("unchecked")
                Map<String, Object> labelItem = (Map<String, Object>) item;
                // 从映射数组中提取code和mapLabel（与第三方label匹配）
                Object codeObj = labelItem.get("code");
                Object englishLabelNameObj = labelItem.get("englishLabelName");
                if (codeObj != null && englishLabelNameObj != null) {
                    labelMap.put(codeObj.toString(), englishLabelNameObj.toString());
                }
            }
        } catch (Exception e) {
            log.error("解析labelMap失败: {}", labelMapStr, e);
        }
        return labelMap;
    }


//------------------------------------------------------------------------------------------------------------------------------------------------------------------

    /**
     * 处理厂商返回的结果并按原始前缀保存为JSON文件  （目标识别）
     */
    private void processResult(Integer taskId, FileEntity fileEntity, Object resultData, DataSonEntity dataSonEntity, String taskIdStr, String labelMap) {
        String logMsg = "厂商返回结果: taskId=" + taskId + ", 文件ID=" + fileEntity.getId() + ", 结果=" + resultData;
        log.info(logMsg);
        sendMessage(taskIdStr, "task_log", logMsg);

        try {
            // 1. 获取原始文件名（包含后缀）
            String originalFileName = fileEntity.getFdName();
            if (StringUtils.isEmpty(originalFileName)) {
                originalFileName = "result_" + fileEntity.getId();
                String warnMsg = "文件fdName为空，使用默认文件名: " + originalFileName;
                log.warn(warnMsg);
                sendMessage(taskIdStr, "task_log", warnMsg);
            }

            // 2. 提取文件名前缀（去除原有后缀）
            String prefix = removeSuffix(originalFileName);

            // 3. 构建新文件名（文件前缀.json）
            String fileName = prefix + ".json";

            // 4. 构建目录路径
            String baseDir = uploadFormalPath + dataSonEntity.getFatherId() + "/v" + dataSonEntity.getVersion() + "/thirdJson/";
            File dir = new File(baseDir);
            if (!dir.exists()) {
                dir.mkdirs();
                String mkdirMsg = "创建目录: " + baseDir;
                log.info(mkdirMsg);
                sendMessage(taskIdStr, "task_log", mkdirMsg);
            }

            // 5. 生成完整文件路径
            String filePath = baseDir + fileName;

            // 6. 替换结果中的img_name为原始文件名
            if (resultData instanceof Map) {
                @SuppressWarnings("unchecked")
                Map<String, Object> resultMap = (Map<String, Object>) resultData;
                resultMap.put("img_name", originalFileName);
            }

            // 7. 转换结果为JSON并写入文件
            Object convertedResult = convertToAnnotationFormat(resultData, originalFileName, labelMap);

            // 获取一下人工标注的和厂商标注的，合并到一起，组成markIndo
            mergeMarkInfo(fileEntity, convertedResult, taskId, dataSonEntity);

            // 8. 写入文件
            String resultJson = objectMapper.writeValueAsString(convertedResult);
            fileDownloadUtils.writeJsonToFile(resultJson, dir.getPath() + "/", prefix, ".json");

            String saveMsg = "结果已保存至: " + filePath;
            log.info(saveMsg);
            sendMessage(taskIdStr, "task_log", saveMsg);
        } catch (Exception e) {
            String fileName = StringUtils.isEmpty(fileEntity.getFdName()) ? "默认文件名" : fileEntity.getFdName();
            String errorMsg = "保存结果失败 [文件ID:" + fileEntity.getId() + " | 原文件名:" + fileName + "]";
            log.error(errorMsg, e);
            sendMessage(taskIdStr, "task_log", errorMsg + ", 错误: " + e.getMessage());
        }
    }


    /**
     * 将人工标注的和厂商生成的合并成一起
     */
    private void mergeMarkInfo(FileEntity fileEntity, Object convertedResult, Integer taskId, DataSonEntity dataSonEntity) {
        List<JsonNode> mergedList = new ArrayList<>();

        // 1. 基础参数初始化（默认值为0，后续从markInfoEntity获取）
        MarkInfoEntity markInfoEntity = markInfoMapper.selectOne(
                new LambdaQueryWrapper<MarkInfoEntity>()
                        .eq(MarkInfoEntity::getFileId, fileEntity.getId())
        );
        Integer originalWidth = 0;       // 原图宽（取自markInfoEntity）
        Integer originalHeight = 0;      // 原图高（取自markInfoEntity）
        Integer targetWidth = 0;         // 等比例缩放宽（优先用operateWidth，为0则用原图宽）
        Integer targetHeight = 0;        // 等比例缩放高（优先用operateHeight，为0则用原图高）


        // 2. 处理人工标注（核心：从markInfoEntity获取原始宽高和缩放宽高）
        if (markInfoEntity != null && !StringUtils.isEmpty(markInfoEntity.getMarkInfo())) {
            // 2.1 获取原图宽高（来自markInfoEntity）
            originalWidth = markInfoEntity.getWidth() != null ? markInfoEntity.getWidth() : 0;
            originalHeight = markInfoEntity.getHeight() != null ? markInfoEntity.getHeight() : 0;

            // 2.2 计算等比例缩放宽高：若operateWidth/Height为null或0，则用原图宽高
            Integer operateWidth = markInfoEntity.getOperateWidth();
            targetWidth = (operateWidth == null || operateWidth == 0) ? originalWidth : operateWidth;

            Integer operateHeight = markInfoEntity.getOperateHeight();
            targetHeight = (operateHeight == null || operateHeight == 0) ? originalHeight : operateHeight;

            System.out.println("原图宽高：" + originalWidth + "x" + originalHeight);
            System.out.println("等比例缩放宽高：" + targetWidth + "x" + targetHeight);

            try {
                JsonNode manualMarkInfo = objectMapper.readTree(markInfoEntity.getMarkInfo());
                if (manualMarkInfo.isArray()) {
                    for (JsonNode element : manualMarkInfo) {
                        // 处理name替换
                        JsonNode propsNode = element.path("props");
                        if (!propsNode.isMissingNode() && propsNode.isObject()) {
                            String currentName = propsNode.path("name").asText();
                            if (!StringUtils.isEmpty(currentName)) {
                                String englishName = returnLabelEnglish(dataSonEntity, currentName);
                                if (englishName != null) {
                                    ((ObjectNode) propsNode).put("name", englishName);
                                }
                            }
                        }
                        // 处理style
                        JsonNode styleNode = element.path("style");
                        if (!styleNode.isMissingNode() && styleNode.isObject()) {
                            ObjectNode styleObject = (ObjectNode) styleNode;
                            styleObject.put("fillStyle", "#1E88E5"); // 深蓝色
                            styleObject.put("strokeStyle", "#0D47A1"); // 深边框蓝
                            styleObject.put("opacity", 1.0);
                            styleObject.put("globalAlpha", 0.7);
                        }
                        mergedList.add(element);
                    }
                }
            } catch (JsonProcessingException e) {
                throw new RuntimeException("人工标注JSON处理异常: " + e.getMessage(), e);
            }
        }


        // 3. 处理厂商JSON（基于markInfoEntity的宽高缩放）
        if (convertedResult != null) {
            try {
                JsonNode vendorJson = objectMapper.readTree(objectMapper.writeValueAsString(convertedResult));
                // 厂商原始宽高：使用markInfoEntity的原图宽高（originalWidth/Height）
                // 强制非0，避免除零错误（若原图宽高为0，默认用1）
                int vendorOriginalWidth = Math.max(originalWidth, 1);
                int vendorOriginalHeight = Math.max(originalHeight, 1);
                JsonNode vendorShapes = vendorJson.path("shapes");

                // 计算缩放比例：基于等比例缩放宽高（targetWidth/Height）和厂商原始宽高
                // 若target为0（可能markInfoEntity为null），默认不缩放（scale=1.0）
                double scaleX = (targetWidth > 0 && vendorOriginalWidth > 0) ? (double) targetWidth / vendorOriginalWidth : 1.0;
                double scaleY = (targetHeight > 0 && vendorOriginalHeight > 0) ? (double) targetHeight / vendorOriginalHeight : 1.0;

                if (vendorShapes.isArray() && !vendorShapes.isEmpty()) {
                    for (int i = 0; i < vendorShapes.size(); i++) {
                        JsonNode vendorShape = vendorShapes.get(i);
                        ObjectNode targetNode = objectMapper.createObjectNode();

                        // 基础字段
                        String uuid = UUID.randomUUID().toString().replaceAll("-", "");
                        targetNode.put("openId", uuid.substring(0, 8));
                        targetNode.put("id", uuid);
                        targetNode.put("type", vendorShape.path("shape_type").asText());
                        targetNode.put("isEye", true);
                        targetNode.put("operateIdx", mergedList.size() + i);

                        // props设置（关联缩放宽高）
                        ObjectNode props = objectMapper.createObjectNode();
                        props.put("name", vendorShape.path("label").asText());
                        props.put("textId", uuid);
                        props.put("deleteMarkerId", uuid);
                        props.put("operateWidth", targetWidth);  // 存储等比例缩放宽
                        props.put("operateHeight", targetHeight); // 存储等比例缩放高
                        targetNode.set("props", props);

                        // 坐标缩放（应用计算好的比例）
                        JsonNode points = vendorShape.path("points");
                        if (points.isArray() && points.size() >= 4) {
                            double originalX1 = points.get(0).get(0).asDouble();
                            double originalY1 = points.get(0).get(1).asDouble();
                            double originalX3 = points.get(2).get(0).asDouble();
                            double originalY3 = points.get(2).get(1).asDouble();

                            // 应用缩放比例
                            double scaledX1 = originalX1 * scaleX;
                            double scaledY1 = originalY1 * scaleY;
                            double scaledX3 = originalX3 * scaleX;
                            double scaledY3 = originalY3 * scaleY;

                            double scaledWidth = scaledX3 - scaledX1;
                            double scaledHeight = scaledY3 - scaledY1;

                            ObjectNode shape = objectMapper.createObjectNode();
                            shape.put("x", scaledX1);
                            shape.put("y", scaledY1);
                            shape.put("width", scaledWidth);
                            shape.put("height", scaledHeight);
                            targetNode.set("shape", shape);
                        }

                        // style
                        ObjectNode style = objectMapper.createObjectNode();
                        style.put("opacity", 1.0);
                        style.put("fillStyle", "#FF9800"); // 橙色
                        style.put("strokeStyle", "#E65100"); // 深橙边框
                        style.put("lineWidth", 2);
                        style.put("fill", true);
                        style.put("globalAlpha", 0.7);
                        targetNode.set("style", style);

                        mergedList.add(targetNode);
                    }
                }
            } catch (JsonProcessingException e) {
                throw new RuntimeException("厂商JSON处理异常: " + e.getMessage(), e);
            }
        }


        // 4. 合并结果存储
        try {
            String mergedJson = objectMapper.writeValueAsString(mergedList);
            ModelMarkInfoEntity existingEntity = modelMarkInfoMapper.selectOne(
                    new LambdaQueryWrapper<ModelMarkInfoEntity>()
                            .eq(ModelMarkInfoEntity::getSonId, dataSonEntity.getSonId())
                            .eq(ModelMarkInfoEntity::getFileId, fileEntity.getId())
                            .eq(ModelMarkInfoEntity::getTaskId, taskId)
            );
            if (existingEntity != null) {
                existingEntity.setMarkInfo(mergedJson);
                modelMarkInfoMapper.updateById(existingEntity);
            } else {
                ModelMarkInfoEntity newEntity = new ModelMarkInfoEntity();
                newEntity.setSonId(dataSonEntity.getSonId());
                newEntity.setFileId(fileEntity.getId());
                newEntity.setTaskId(taskId);
                newEntity.setMarkInfo(mergedJson);
                modelMarkInfoMapper.insert(newEntity);
            }
        } catch (JsonProcessingException e) {
            throw new RuntimeException("合并结果JSON处理异常: " + e.getMessage(), e);
        }
    }

    /**
     * 返回英文标签
     */
    public String returnLabelEnglish(DataSonEntity dataSonEntity, String labelName) {
        // 当前数据集的所有标签
        List<DataSonLabelEntity> list = dataSonLabelMapper.selectList(
                new LambdaQueryWrapper<DataSonLabelEntity>()
                        .eq(DataSonLabelEntity::getSonId, dataSonEntity.getSonId())
        );
        // 提取所有的 labelId
        List<Integer> labelIds = list.stream().map(DataSonLabelEntity::getLabelId).collect(Collectors.toList());
        // 批量查询标签信息
        List<LabelEntity> labelEntities = labelMapper.selectByIds(labelIds);
        // 使用 Map 存储标签名称映射
        Map<String, String> labelMap = new HashMap<>();
        for (LabelEntity labelEntity : labelEntities) {
            labelMap.put(labelEntity.getLabelName().trim(), labelEntity.getEnglishLabelName());
        }
        return labelMap.getOrDefault(labelName, null);
    }


    /**
     * 将厂商返回的结果转换为标注格式
     */
    private Object convertToAnnotationFormat(Object resultData, String originalFileName, String labelMapStr) {
        // 1. 将结果转换为Map以便处理
        Map<String, Object> resultMap = objectMapper.convertValue(resultData, Map.class);

        // 获取当前数据集的所有标签英文，用于替换
        Map<String, String> labelMap = null;
        if (!StringUtils.isEmpty(labelMapStr)) {
            labelMap = JsonParser.parseMapLabelToEnglish(labelMapStr);
        }

        // 2. 创建目标格式的Map
        Map<String, Object> annotationMap = new HashMap<>();

        // 3. 提取图像尺寸信息
        Object widthObj = resultMap.get("img_width");
        Object heightObj = resultMap.get("img_height");

        // 安全转换为Integer
        Integer imageWidth = widthObj instanceof Number ? ((Number) widthObj).intValue() : 0;
        Integer imageHeight = heightObj instanceof Number ? ((Number) heightObj).intValue() : 0;

        // 4. 设置基本信息
        annotationMap.put("imageWidth", imageWidth);
        annotationMap.put("imageHeight", imageHeight);
        annotationMap.put("imagePath", originalFileName);

        // 5. 处理检测结果
        List<Object> shapes = new ArrayList<>();
        List<Map<String, Object>> detections = (List<Map<String, Object>>) resultMap.get("dec_rsts");

        if (detections != null) {
            for (Map<String, Object> detection : detections) {
                // 创建shape对象
                Map<String, Object> shape = new HashMap<>();

                // 设置label（使用class_name）
                String code = detection.get("class_name").toString();
                if (labelMap != null && !StringUtils.isEmpty(code) && labelMap.containsKey(code)) {
                    // key存在！判断value是否为null,如果为空,则不进行替换
                    String value = labelMap.get(code);
                    if (value != null) {
                        shape.put("label", value);
                    } else {
                        shape.put("label", detection.get("class_name"));
                    }
                } else {
                    shape.put("label", detection.get("class_name"));
                }

                // 设置shape_type为RECT（矩形）
                shape.put("shape_type", "RECT");

                // 转换bbox坐标为points格式（矩形的四个顶点）
                Object bboxObj = detection.get("bbox");
                if (bboxObj instanceof List) {
                    @SuppressWarnings("unchecked")
                    List<Object> bboxList = (List<Object>) bboxObj;

                    if (bboxList.size() == 4) {
                        List<List<Double>> points = new ArrayList<>();

                        // 安全地将坐标转换为Double
                        double x1 = toDouble(bboxList.get(0));
                        double y1 = toDouble(bboxList.get(1));
                        double x2 = toDouble(bboxList.get(2));
                        double y2 = toDouble(bboxList.get(3));

                        // 左上角
                        points.add(Arrays.asList(x1, y1));
                        // 右上角
                        points.add(Arrays.asList(x2, y1));
                        // 右下角
                        points.add(Arrays.asList(x2, y2));
                        // 左下角
                        points.add(Arrays.asList(x1, y2));

                        shape.put("points", points);
                    }
                }

                // 新增score字段（从原始数据中提取）
                Object score = detection.get("score");
                if (score != null) {
                    // 安全转换score为Double
                    if (score instanceof Number) {
                        shape.put("score", ((Number) score).doubleValue());
                    } else {
                        try {
                            shape.put("score", Double.parseDouble(score.toString()));
                        } catch (NumberFormatException e) {
                            log.warn("score格式转换失败: {}", score);
                            shape.put("score", score); // 保留原始格式
                        }
                    }
                }

                shapes.add(shape);
            }
        }

        // 6. 设置shapes列表
        annotationMap.put("shapes", shapes);

        return annotationMap;
    }

    /**
     * 安全地将对象转换为Double
     */
    private double toDouble(Object value) {
        if (value instanceof Number) {
            return ((Number) value).doubleValue();
        } else if (value != null) {
            try {
                return Double.parseDouble(value.toString());
            } catch (NumberFormatException e) {
                log.warn("坐标转换失败: {}", value);
                return 0.0;
            }
        }
        return 0.0;
    }

    /**
     * 去除文件名后缀
     */
    private String removeSuffix(String fileName) {
        int dotIndex = fileName.lastIndexOf(".");
        if (dotIndex > 0) {
            return fileName.substring(0, dotIndex);
        }
        return fileName;
    }

    @Override
    public void stopManufacturer(ModelAssessTaskEntity modelAssessTaskEntity) {
        Integer taskId = modelAssessTaskEntity.getId();
        String taskIdStr = taskId.toString();

        AtomicBoolean flag = taskControlFlags.get(taskId);
        if (flag != null) {
            flag.set(false);
            String msg = "任务已停止: taskId=" + taskId;
            log.info(msg);
            sendMessage(taskIdStr, "task_log", msg);
            sendMessage(taskIdStr, "stopped", true);
        }
    }

    @Override
    public void endManufacturer(ModelAssessTaskEntity modelAssessTaskEntity) {
        Integer taskId = modelAssessTaskEntity.getId();
        String taskIdStr = taskId.toString();

        taskControlFlags.remove(taskId);
        String msg = "任务已结束: taskId=" + taskId;
        log.info(msg);
        sendMessage(taskIdStr, "task_log", msg);
        sendMessage(taskIdStr, "ended", true);
    }




    @Override
    public boolean isExecuteTask(ModelAssessTaskEntity modelAssessTaskEntity) {

        // 参数校验
        if (modelAssessTaskEntity == null || modelAssessTaskEntity.getId() == null) {
            log.error("任务实体或任务ID为空，无法执行任务检查");
            throw new RuntimeException("当前任务不存在！");
        }

        // 获取任务配置
        ModelAssessConfigEntity modelAssessConfigEntity = modelAssessConfigMapper.selectOne(
                new LambdaQueryWrapper<ModelAssessConfigEntity>()
                        .eq(ModelAssessConfigEntity::getAssessTaskId, modelAssessTaskEntity.getId())
        );
        if (modelAssessConfigEntity == null) {
            log.error("未找到任务ID为[{}]的配置信息", modelAssessTaskEntity.getId());
            throw new RuntimeException("当前任务配置不存在！");
        }

        // 获取任务详情
        ModelAssessTaskEntity entity = modelAssessTaskMapper.selectById(modelAssessTaskEntity.getId());
        if (entity == null) {
            log.error("未找到ID为[{}]的任务详情", modelAssessTaskEntity.getId());
            throw new RuntimeException("当前任务不存在！");
        }

        // 获取数据集信息
        DataSonEntity dataSonEntity = dataSonMapper.getDataSonBySonId(modelAssessConfigEntity.getSonId());
        if (dataSonEntity == null) {
            log.error("未找到子数据集ID为[{}]的数据集信息", modelAssessConfigEntity.getSonId());
            throw new RuntimeException("数据集不存在！");
        }

        // 解析文件数量（处理fileIds为空的情况）
        String fileIds = dataSonEntity.getFileIds();
        int fileSum = (fileIds == null || fileIds.trim().isEmpty()) ? 0 : fileIds.split(",").length;
        log.debug("数据集[{}]的文件总数为: {}", dataSonEntity.getSonId(), fileSum);

        // 获取任务类型
        Integer taskType = entity.getTaskType();
        log.debug("当前任务[{}]的类型为: {}", modelAssessTaskEntity.getId(), taskType);

        // 构建目录路径
        String fatherId = String.valueOf(dataSonEntity.getFatherId());
        String version = "v" + dataSonEntity.getVersion();

        // 分类任务目录
        Path classifyJsonDirectory = Paths.get(uploadFormalPath, fatherId, version, "classifyJson");
        Path thirdClassifyJsonDirectory = Paths.get(uploadFormalPath, fatherId, version, "thirdClassifyJson");

        // 目标检测目录
        Path detectionJsonDirectory = Paths.get(uploadFormalPath, fatherId, version, "thirdJson");

        // 根据任务类型进行校验
        if (TASK_TYPE_CLASSIFY == taskType) {
            return checkClassifyTask(classifyJsonDirectory, thirdClassifyJsonDirectory, fileSum);
        } else if (TASK_TYPE_DETECTION == taskType) {
            return checkDetectionTask(detectionJsonDirectory, fileSum);
        } else {
            log.warn("未知的任务类型[{}]，任务ID: {}", taskType, modelAssessTaskEntity.getId());
            return false;
        }
    }

    /**
     * 检查分类任务是否可执行
     * @param classifyDir 分类结果目录
     * @param thirdClassifyDir 第三方分类结果目录
     * @param expectedTotal 预期总文件数（原始文件数 * 2）
     * @return 是否可执行
     */
    private boolean checkClassifyTask(Path classifyDir, Path thirdClassifyDir, int expectedTotal) {
        // 检查目录有效性
        if (!isValidDirectory(classifyDir) || !isValidDirectory(thirdClassifyDir)) {
            log.warn("分类任务目录无效，classifyDir: {}, thirdClassifyDir: {}", classifyDir, thirdClassifyDir);
            return false;
        }

        // 计算两个目录的文件总数
        long classifyFileCount = countRegularFiles(classifyDir);
        long thirdClassifyFileCount = countRegularFiles(thirdClassifyDir);

        // 检查文件计数是否成功
        if (classifyFileCount == -1 || thirdClassifyFileCount == -1) {
            log.error("分类任务目录文件计数失败");
            return false;
        }

        long actualTotal = classifyFileCount + thirdClassifyFileCount;
        boolean result = actualTotal == expectedTotal * 2;

        if (!result) {
            log.warn("分类任务文件数量不匹配，预期: {}, 实际: {} (classify: {}, thirdClassify: {})",
                    expectedTotal * 2, actualTotal, classifyFileCount, thirdClassifyFileCount);
        }

        return result;
    }

    /**
     * 检查目标检测任务是否可执行
     * @param detectionDir 检测结果目录
     * @param expectedCount 预期文件数
     * @return 是否可执行
     */
    private boolean checkDetectionTask(Path detectionDir, int expectedCount) {
        // 检查目录有效性
        if (!isValidDirectory(detectionDir)) {
            log.warn("目标检测任务目录无效: {}", detectionDir);
            return false;
        }

        // 计算目录文件数
        long actualFileCount = countRegularFiles(detectionDir);

        // 检查文件计数是否成功
        if (actualFileCount == -1) {
            log.error("目标检测任务目录文件计数失败");
            return false;
        }

        boolean result = actualFileCount == expectedCount;

        if (!result) {
            log.warn("目标检测任务文件数量不匹配，预期: {}, 实际: {}", expectedCount, actualFileCount);
        }

        return result;
    }

    /**
     * 检查目录是否有效（存在且为目录）
     * @param directory 目录路径
     * @return 是否有效
     */
    private boolean isValidDirectory(Path directory) {
        return directory != null && Files.exists(directory) && Files.isDirectory(directory);
    }

    /**
     * 统计目录中的普通文件数量
     * @param directory 目录路径
     * @return 文件数量，-1表示统计失败
     */
    private long countRegularFiles(Path directory) {
        if (!isValidDirectory(directory)) {
            return 0;
        }

        try (DirectoryStream<Path> stream = Files.newDirectoryStream(directory)) {
            long count = 0;
            for (Path path : stream) {
                if (Files.isRegularFile(path)) {
                    count++;
                }
            }
            return count;
        } catch (IOException e) {
            log.error("统计目录[{}]文件数量时发生异常", directory, e);
            return -1;
        }
    }





















    @Override
    public int finishContact(ModelAssessTaskEntity modelAssessTaskEntity) {
        ModelAssessTaskEntity entity = modelAssessTaskMapper.selectById(modelAssessTaskEntity.getId());
        if (entity == null || modelAssessTaskEntity.getId() == null) {
            throw new RuntimeException("当前任务不存在！");
        }
        // 判断是测试还是评估
       /* if (entity.getTaskType() == 1) {
            entity.setTaskStatus(2);
            entity.setTaskProgress("100%");
            ModelAssessConfigEntity modelAssessConfigEntity = modelAssessConfigMapper.selectOne(
                    new LambdaQueryWrapper<ModelAssessConfigEntity>()
                            .eq(ModelAssessConfigEntity::getAssessTaskId, entity.getId())
            );
            if (modelAssessConfigEntity == null || StringUtils.isEmpty(modelAssessConfigEntity.getSonId())) {
                return 1;
            }
            // 恢复数据集状态
            DataSonEntity dataSonEntity = dataSonMapper.selectOne(
                    new LambdaQueryWrapper<DataSonEntity>()
                            .eq(DataSonEntity::getSonId, modelAssessConfigEntity.getSonId())
            );
            if (dataSonEntity != null) {
                dataSonEntity.setIsMany(0);
                dataSonMapper.updateById(dataSonEntity);
            }
        } else {
            entity.setTaskStatus(0);
        }*/
        entity.setTaskStatus(0);
        return modelAssessTaskMapper.updateById(entity);
    }

    /**
     * 发送消息到前端
     */
    private void sendMessage(String taskId, String type, Object message) {
        TaskWebSocketHandler.sendMessage(taskId, type, message);
    }
}