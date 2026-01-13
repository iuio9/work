package com.qczy.distillation.service;

import com.alibaba.fastjson.JSON;
import com.alibaba.fastjson.JSONObject;
import com.qczy.distillation.model.dto.InferenceRequestDTO;
import com.qczy.distillation.model.dto.InferenceResultDTO;
import com.qczy.distillation.model.entity.MdTrainingTaskEntity;
import com.qczy.distillation.mapper.MdTrainingTaskMapper;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.scheduling.annotation.Async;
import org.springframework.stereotype.Service;

import java.io.BufferedReader;
import java.io.File;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

/**
 * 模型推理服务
 *
 * 功能：
 * 1. 使用训练好的模型对图像进行推理
 * 2. 生成自动标注结果
 * 3. 异步执行推理任务
 * 4. 管理推理状态和结果
 *
 * @author AI Assistant
 * @date 2025-01-13
 */
@Service
public class TrainingInferenceService {

    private static final Logger logger = LoggerFactory.getLogger(TrainingInferenceService.class);

    @Autowired
    private MdTrainingTaskMapper trainingTaskMapper;

    @Value("${distillation.python.executable:python3}")
    private String pythonExecutable;

    @Value("${distillation.inference-script.path:/home/user/work/back/datamark-admin/inference_qwen_vl_distilled_models.py}")
    private String inferenceScriptPath;

    // 推理任务状态缓存
    private final Map<String, InferenceResultDTO> inferenceResults = new ConcurrentHashMap<>();

    /**
     * 提交推理任务（异步执行）
     */
    @Async("taskExecutor")
    public void submitInferenceTask(String inferenceId, InferenceRequestDTO request) {
        logger.info("开始推理任务: {}", inferenceId);

        InferenceResultDTO result = new InferenceResultDTO();
        result.setInferenceId(inferenceId);
        result.setTaskId(request.getTaskId());
        result.setStatus("RUNNING");
        result.setStartTime(LocalDateTime.now().format(DateTimeFormatter.ISO_LOCAL_DATE_TIME));
        result.setOutputDir(request.getOutputDir());

        inferenceResults.put(inferenceId, result);

        try {
            // 获取训练任务信息
            MdTrainingTaskEntity task = trainingTaskMapper.selectByTaskId(request.getTaskId());
            if (task == null) {
                throw new RuntimeException("训练任务不存在: " + request.getTaskId());
            }

            if (!"completed".equalsIgnoreCase(task.getStatus())) {
                throw new RuntimeException("训练任务未完成，无法进行推理");
            }

            // 获取模型路径
            String modelPath = task.getModelPath();
            if (modelPath == null || modelPath.isEmpty()) {
                throw new RuntimeException("模型路径为空，请确认训练任务已保存模型");
            }
            if (!new File(modelPath).exists()) {
                throw new RuntimeException("模型文件不存在: " + modelPath);
            }

            // 解析训练配置
            JSONObject trainingConfig = JSON.parseObject(task.getTrainingConfig());
            String studentModelType = trainingConfig.getString("studentModelType");
            Integer numClasses = trainingConfig.getInteger("numClasses");
            String studentModelSize = trainingConfig.getString("studentModelSize");
            Integer imageSize = trainingConfig.getInteger("imageSize");

            result.setModelType(studentModelType);

            // 执行推理
            executeInference(
                    modelPath,
                    studentModelType,
                    request.getInputDir(),
                    request.getOutputDir(),
                    numClasses,
                    studentModelSize,
                    imageSize,
                    request.getBatchSize(),
                    result
            );

            // 统计结果
            File outputDirFile = new File(request.getOutputDir());
            if (outputDirFile.exists() && outputDirFile.isDirectory()) {
                File[] jsonFiles = outputDirFile.listFiles((dir, name) -> name.endsWith(".json"));
                int count = jsonFiles != null ? jsonFiles.length : 0;
                result.setProcessedImages(count);
                result.setSuccessCount(count);
                result.setFailureCount(0);
            }

            result.setStatus("COMPLETED");
            logger.info("推理任务完成: {}", inferenceId);

        } catch (Exception e) {
            logger.error("推理任务失败: " + inferenceId, e);
            result.setStatus("FAILED");
            result.setErrorMessage(e.getMessage());
            result.setFailureCount(1);
        } finally {
            result.setEndTime(LocalDateTime.now().format(DateTimeFormatter.ISO_LOCAL_DATE_TIME));

            // 计算耗时
            if (result.getStartTime() != null && result.getEndTime() != null) {
                try {
                    LocalDateTime start = LocalDateTime.parse(result.getStartTime(), DateTimeFormatter.ISO_LOCAL_DATE_TIME);
                    LocalDateTime end = LocalDateTime.parse(result.getEndTime(), DateTimeFormatter.ISO_LOCAL_DATE_TIME);
                    result.setDuration(java.time.Duration.between(start, end).getSeconds());
                } catch (Exception e) {
                    logger.warn("计算耗时失败", e);
                }
            }

            inferenceResults.put(inferenceId, result);
        }
    }

    /**
     * 执行Python推理脚本
     */
    private void executeInference(
            String modelPath,
            String modelType,
            String inputDir,
            String outputDir,
            Integer numClasses,
            String modelSize,
            Integer imageSize,
            Integer batchSize,
            InferenceResultDTO result
    ) throws Exception {

        // 构建Python命令
        List<String> command = buildInferenceCommand(
                modelPath, modelType, inputDir, outputDir,
                numClasses, modelSize, imageSize, batchSize
        );

        logger.info("执行推理命令: {}", String.join(" ", command));

        ProcessBuilder processBuilder = new ProcessBuilder(command);
        processBuilder.redirectErrorStream(true);

        Process process = processBuilder.start();

        // 读取输出
        StringBuilder output = new StringBuilder();
        try (BufferedReader reader = new BufferedReader(
                new InputStreamReader(process.getInputStream(), StandardCharsets.UTF_8))) {
            String line;
            while ((line = reader.readLine()) != null) {
                logger.info("[推理] {}", line);
                output.append(line).append("\n");
            }
        }

        int exitCode = process.waitFor();
        logger.info("推理脚本执行完成，退出码: {}", exitCode);

        if (exitCode != 0) {
            throw new RuntimeException("推理脚本执行失败，退出码: " + exitCode + "\n输出:\n" + output);
        }
    }

    /**
     * 构建推理命令
     */
    private List<String> buildInferenceCommand(
            String modelPath,
            String modelType,
            String inputDir,
            String outputDir,
            Integer numClasses,
            String modelSize,
            Integer imageSize,
            Integer batchSize
    ) {
        List<String> command = new ArrayList<>();
        command.add(pythonExecutable);
        command.add(inferenceScriptPath);
        command.add("--model_path");
        command.add(modelPath);
        command.add("--model_type");
        command.add(modelType);
        command.add("--input_dir");
        command.add(inputDir);
        command.add("--output_dir");
        command.add(outputDir);
        command.add("--num_classes");
        command.add(String.valueOf(numClasses));

        if (modelSize != null && !modelSize.isEmpty()) {
            command.add("--model_size");
            command.add(modelSize);
        }

        if (imageSize != null) {
            command.add("--image_size");
            command.add(String.valueOf(imageSize));
        }

        if (batchSize != null) {
            command.add("--batch_size");
            command.add(String.valueOf(batchSize));
        }

        return command;
    }

    /**
     * 获取推理结果
     */
    public InferenceResultDTO getInferenceResult(String inferenceId) {
        return inferenceResults.get(inferenceId);
    }

    /**
     * 获取所有推理任务
     */
    public List<InferenceResultDTO> getAllInferenceResults() {
        return new ArrayList<>(inferenceResults.values());
    }

    /**
     * 清除推理结果
     */
    public void clearInferenceResult(String inferenceId) {
        inferenceResults.remove(inferenceId);
    }
}
