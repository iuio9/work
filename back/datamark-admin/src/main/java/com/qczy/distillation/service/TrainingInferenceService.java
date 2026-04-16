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
import org.springframework.http.HttpEntity;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.scheduling.annotation.Async;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;

import java.io.File;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

/**
 * 模型推理服务（远程算法服务器版本）
 *
 * 将推理任务分发到独立的算法服务器（GPU机）上执行。
 * 算法服务器上需要运行 training_agent.py（FastAPI 服务）。
 *
 * 调用链：
 *   Spring Boot → POST http://算法服务器:5000/infer
 *     └─ training_agent.py 在算法服务器启动 subprocess
 *         └─ Python 推理脚本 → 写出结果到 outputDir（NFS 挂载）
 */
@Service
public class TrainingInferenceService {

    private static final Logger logger = LoggerFactory.getLogger(TrainingInferenceService.class);

    @Autowired
    private MdTrainingTaskMapper trainingTaskMapper;

    /** 算法服务器上 training_agent.py 的根 URL */
    @Value("${distillation.agent.url:http://localhost:5000}")
    private String agentUrl;

    /** 算法服务器上的 Python 解释器路径 */
    @Value("${distillation.python.path:python3}")
    private String pythonPath;

    /** 算法服务器上推理脚本的绝对路径 */
    @Value("${distillation.inference-script.path:/opt/datamark/inference_qwen_vl_distilled_models.py}")
    private String inferenceScriptPath;

    /** 轮询推理状态的间隔（毫秒） */
    private static final long POLL_INTERVAL_MS = 5_000;

    /** 推理任务最长等待时间（毫秒，默认 4 小时） */
    private static final long MAX_WAIT_MS = 4 * 60 * 60 * 1_000L;

    private final RestTemplate restTemplate = new RestTemplate();

    // 推理任务状态缓存
    private final Map<String, InferenceResultDTO> inferenceResults = new ConcurrentHashMap<>();

    // ─────────────────────────────────────────────────────────────────
    // 公开接口
    // ─────────────────────────────────────────────────────────────────

    /**
     * 提交推理任务（异步执行）。
     * 向算法服务器发送推理请求，然后轮询直到完成。
     */
    @Async("taskExecutor")
    public void submitInferenceTask(String inferenceId, InferenceRequestDTO request) {
        logger.info("提交推理任务到算法服务器: inferenceId={}", inferenceId);

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

            String modelPath = task.getModelPath();
            if (modelPath == null || modelPath.isEmpty()) {
                throw new RuntimeException("模型路径为空，请确认训练任务已保存模型");
            }

            // 解析训练配置
            String studentModelType = "resnet";
            Integer numClasses = 10;
            String studentModelSize = null;
            Integer imageSize = null;

            if (task.getTrainingConfig() != null && !task.getTrainingConfig().isEmpty()) {
                JSONObject trainingConfig = JSON.parseObject(task.getTrainingConfig());
                if (trainingConfig.getString("studentModelType") != null)
                    studentModelType = trainingConfig.getString("studentModelType");
                if (trainingConfig.getInteger("numClasses") != null)
                    numClasses = trainingConfig.getInteger("numClasses");
                studentModelSize = trainingConfig.getString("studentModelSize");
                imageSize = trainingConfig.getInteger("imageSize");
            }

            result.setModelType(studentModelType);

            // 构建发往 training_agent 的推理请求
            Map<String, String> args = new LinkedHashMap<>();
            args.put("model_path",  modelPath);
            args.put("model_type",  studentModelType);
            args.put("input_dir",   request.getInputDir());
            args.put("output_dir",  request.getOutputDir());
            args.put("num_classes", String.valueOf(numClasses));
            if (studentModelSize != null && !studentModelSize.isEmpty())
                args.put("model_size", studentModelSize);
            if (imageSize != null)
                args.put("image_size", String.valueOf(imageSize));
            if (request.getBatchSize() != null)
                args.put("batch_size", String.valueOf(request.getBatchSize()));

            Map<String, Object> body = new HashMap<>();
            body.put("inferenceId", inferenceId);
            body.put("scriptPath",  inferenceScriptPath);
            body.put("pythonPath",  pythonPath);
            body.put("args",        args);

            // POST 到算法服务器
            String startUrl = agentUrl + "/infer";
            HttpHeaders headers = new HttpHeaders();
            headers.setContentType(MediaType.APPLICATION_JSON);
            HttpEntity<String> entity = new HttpEntity<>(JSON.toJSONString(body), headers);

            logger.info("调用算法服务器推理: POST {}", startUrl);
            restTemplate.postForObject(startUrl, entity, Map.class);

            // 轮询直到推理结束
            String statusUrl = agentUrl + "/infer/status/" + inferenceId;
            long waited = 0;
            boolean success = false;

            while (waited < MAX_WAIT_MS) {
                Thread.sleep(POLL_INTERVAL_MS);
                waited += POLL_INTERVAL_MS;

                Map<?, ?> statusResp = restTemplate.getForObject(statusUrl, Map.class);
                if (statusResp == null) continue;

                boolean running = Boolean.TRUE.equals(statusResp.get("running"));
                if (!running) {
                    Object exitCodeObj = statusResp.get("exitCode");
                    int exitCode = exitCodeObj instanceof Number ? ((Number) exitCodeObj).intValue() : -1;
                    if (exitCode != 0) {
                        throw new RuntimeException("推理脚本执行失败，退出码: " + exitCode);
                    }
                    success = true;
                    break;
                }
            }

            if (!success) {
                throw new RuntimeException("推理任务超时");
            }

            // 统计输出文件数（通过 NFS 挂载路径访问）
            File outputDirFile = new File(request.getOutputDir());
            if (outputDirFile.exists() && outputDirFile.isDirectory()) {
                File[] jsonFiles = outputDirFile.listFiles((dir, name) -> name.endsWith(".json"));
                int count = jsonFiles != null ? jsonFiles.length : 0;
                result.setProcessedImages(count);
                result.setSuccessCount(count);
                result.setFailureCount(0);
            }

            result.setStatus("COMPLETED");
            logger.info("推理任务完成: inferenceId={}", inferenceId);

        } catch (Exception e) {
            logger.error("推理任务失败: inferenceId=" + inferenceId, e);
            result.setStatus("FAILED");
            result.setErrorMessage(e.getMessage());
            result.setFailureCount(1);
        } finally {
            result.setEndTime(LocalDateTime.now().format(DateTimeFormatter.ISO_LOCAL_DATE_TIME));
            if (result.getStartTime() != null && result.getEndTime() != null) {
                try {
                    LocalDateTime start = LocalDateTime.parse(result.getStartTime(), DateTimeFormatter.ISO_LOCAL_DATE_TIME);
                    LocalDateTime end   = LocalDateTime.parse(result.getEndTime(),   DateTimeFormatter.ISO_LOCAL_DATE_TIME);
                    result.setDuration(java.time.Duration.between(start, end).getSeconds());
                } catch (Exception ignored) {}
            }
            inferenceResults.put(inferenceId, result);
        }
    }

    /**
     * 停止推理任务（通知算法服务器终止进程）。
     */
    public boolean stopInference(String inferenceId) {
        try {
            String url = agentUrl + "/infer/stop/" + inferenceId;
            restTemplate.postForObject(url, null, Map.class);
            logger.info("已通知算法服务器停止推理: inferenceId={}", inferenceId);
            InferenceResultDTO result = inferenceResults.get(inferenceId);
            if (result != null) {
                result.setStatus("STOPPED");
                result.setEndTime(LocalDateTime.now().format(DateTimeFormatter.ISO_LOCAL_DATE_TIME));
            }
            return true;
        } catch (Exception e) {
            logger.warn("停止推理任务失败（可能已结束）: inferenceId={}, error={}", inferenceId, e.getMessage());
            return false;
        }
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
