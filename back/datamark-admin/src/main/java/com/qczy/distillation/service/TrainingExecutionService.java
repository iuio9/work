package com.qczy.distillation.service;

import com.alibaba.fastjson.JSON;
import com.qczy.distillation.mapper.MdTrainingTaskMapper;
import com.qczy.distillation.model.dto.TrainingConfigDTO;
import com.qczy.distillation.model.entity.MdTrainingTaskEntity;
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

import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * 训练执行服务（远程算法服务器版本）
 *
 * 将训练任务分发到独立的算法服务器（GPU机）上执行。
 * 算法服务器上需要运行 training_agent.py（FastAPI 服务）。
 *
 * 调用链：
 *   Spring Boot → POST http://算法服务器:5000/train
 *     └─ training_agent.py 在算法服务器启动 subprocess
 *         └─ Python 训练脚本 → PUT http://存储服务器:8080/model-distillation/.../progress
 */
@Service
public class TrainingExecutionService {

    private static final Logger logger = LoggerFactory.getLogger(TrainingExecutionService.class);

    @Autowired
    private MdTrainingTaskMapper trainingTaskMapper;

    @Autowired
    private MdTrainingTaskService trainingTaskService;

    // ── 算法服务器（Training Agent）配置 ──────────────────────────────

    /** 算法服务器上 training_agent.py 的根 URL，例如 http://192.168.1.100:5000 */
    @Value("${distillation.agent.url:http://localhost:5000}")
    private String agentUrl;

    /** 算法服务器上的 Python 解释器路径 */
    @Value("${distillation.python.path:python3}")
    private String pythonPath;

    /** 算法服务器上各训练脚本的绝对路径 */
    @Value("${distillation.script.path:/opt/datamark/train_distillation.py}")
    private String scriptPath;

    @Value("${distillation.qwen-script.path:/opt/datamark/train_qwen_vl_distillation.py}")
    private String qwenScriptPath;

    @Value("${distillation.student-only-script.path:/opt/datamark/train_student_only.py}")
    private String studentOnlyScriptPath;

    // ── 回调地址（存储服务器自身） ────────────────────────────────────

    /** 训练脚本回调进度时使用的地址（存储服务器对外 IP:Port） */
    @Value("${distillation.api.base-url:http://localhost:8080}")
    private String apiBaseUrl;

    // ── 路径（算法服务器视角） ────────────────────────────────────────

    /** 算法服务器上数据集的路径（NFS 挂载点或本地路径） */
    @Value("${distillation.datasets.root}")
    private String datasetsRoot;

    /** 算法服务器上训练输出的路径 */
    @Value("${distillation.output.root}")
    private String outputRoot;

    /** 模型权重根目录 */
    @Value("${distillation.models.root:/data/models}")
    private String modelsRoot;

    private final RestTemplate restTemplate = new RestTemplate();

    // ─────────────────────────────────────────────────────────────────
    // 公开接口
    // ─────────────────────────────────────────────────────────────────

    /**
     * 异步提交训练任务到算法服务器。
     * 立即返回，Python 脚本通过回调接口上报进度。
     */
    @Async("taskExecutor")
    public void startTrainingAsync(String taskId) {
        logger.info("提交训练任务到算法服务器: taskId={}", taskId);

        try {
            MdTrainingTaskEntity task = trainingTaskMapper.selectByTaskId(taskId);
            if (task == null) {
                logger.error("任务不存在: {}", taskId);
                return;
            }

            TrainingConfigDTO config = null;
            if (task.getTrainingConfig() != null && !task.getTrainingConfig().isEmpty()) {
                config = JSON.parseObject(task.getTrainingConfig(), TrainingConfigDTO.class);
            }

            // 构建发往 training_agent 的请求体
            Map<String, Object> requestBody = buildAgentRequest(task, config);

            // 发送到算法服务器
            String url = agentUrl + "/train";
            HttpHeaders headers = new HttpHeaders();
            headers.setContentType(MediaType.APPLICATION_JSON);
            HttpEntity<String> entity = new HttpEntity<>(JSON.toJSONString(requestBody), headers);

            logger.info("调用算法服务器: POST {}", url);
            Map<?, ?> response = restTemplate.postForObject(url, entity, Map.class);
            logger.info("算法服务器响应: {}", response);

        } catch (Exception e) {
            logger.error("提交训练任务失败: taskId={}", taskId, e);
            trainingTaskService.updateError(taskId, "提交到算法服务器失败: " + e.getMessage());
        }
    }

    /**
     * 停止训练任务（通知算法服务器终止进程）。
     */
    public boolean stopTraining(String taskId) {
        try {
            String url = agentUrl + "/stop/" + taskId;
            restTemplate.postForObject(url, null, Map.class);
            logger.info("已通知算法服务器停止任务: {}", taskId);
            return true;
        } catch (Exception e) {
            logger.warn("停止训练任务失败（可能已结束）: taskId={}, error={}", taskId, e.getMessage());
            return false;
        }
    }

    /**
     * 检查任务是否仍在算法服务器上运行。
     */
    public boolean isTrainingRunning(String taskId) {
        try {
            String url = agentUrl + "/status/" + taskId;
            Map<?, ?> resp = restTemplate.getForObject(url, Map.class);
            return resp != null && Boolean.TRUE.equals(resp.get("running"));
        } catch (Exception e) {
            logger.warn("查询训练状态失败: taskId={}, error={}", taskId, e.getMessage());
            return false;
        }
    }

    public int getRunningTaskCount() {
        // 轻量估算，不强依赖 agent 可达性
        try {
            String url = agentUrl + "/health";
            Map<?, ?> resp = restTemplate.getForObject(url, Map.class);
            if (resp != null && resp.get("runningTasks") instanceof Number) {
                return ((Number) resp.get("runningTasks")).intValue();
            }
        } catch (Exception ignored) {}
        return 0;
    }

    public List<String> getRunningTaskIds() {
        return new ArrayList<>(); // agent 暂未暴露列表接口，按需扩展
    }

    // ─────────────────────────────────────────────────────────────────
    // 私有辅助方法
    // ─────────────────────────────────────────────────────────────────

    /**
     * 构建发往 training_agent 的请求体。
     * 结构与 training_agent.py 的 TrainRequest 对应：
     *   { taskId, scriptPath, pythonPath, apiBaseUrl, args{}, flags[] }
     */
    private Map<String, Object> buildAgentRequest(MdTrainingTaskEntity task,
                                                   TrainingConfigDTO config) {
        String trainingMode = task.getTrainingMode() != null ? task.getTrainingMode() : "distillation";
        String script = getTrainingScript(task.getTeacherModel(), trainingMode);

        // args: key 不带 "--"，value 都是字符串
        Map<String, String> args = new LinkedHashMap<>();

        // 模型配置
        if ("distillation".equals(trainingMode)) {
            args.put("teacher_model", task.getTeacherModel());
            args.put("teacher_path", getModelPath(task.getTeacherModel(), config));
        }

        args.put("student_model", task.getStudentModel());

        String studentPath = getStudentModelPath(task.getStudentModel(), config);
        if (studentPath != null && !studentPath.isEmpty()) {
            args.put("student_path", studentPath);
        }

        // 数据集
        args.put("dataset_id", String.valueOf(task.getDatasetId()));
        if (task.getValDatasetId() != null) {
            args.put("val_dataset_id", String.valueOf(task.getValDatasetId()));
        }
        args.put("datasets_root", datasetsRoot);

        // 训练参数
        args.put("epochs",         String.valueOf(task.getTotalEpochs()));
        args.put("batch_size",     String.valueOf(task.getBatchSize()));
        args.put("learning_rate",  task.getLearningRate().toString());

        // LoRA
        args.put("lora_rank",    String.valueOf(task.getLoraRank()));
        args.put("lora_alpha",   String.valueOf(task.getLoraAlpha()));
        args.put("lora_dropout", task.getLoraDropout().toString());

        // 蒸馏参数
        args.put("temperature",       task.getTemperature().toString());
        args.put("hard_label_weight", String.valueOf(1.0 - task.getAlpha().doubleValue()));
        args.put("soft_label_weight", task.getAlpha().toString());

        // 高级配置
        if (config != null) {
            if (config.getOptimizer() != null)     args.put("optimizer",      config.getOptimizer());
            if (config.getLrScheduler() != null)   args.put("lr_scheduler",   config.getLrScheduler());
            if (config.getWeightDecay() != null)   args.put("weight_decay",   config.getWeightDecay().toString());
            if (config.getGradAccumSteps() != null) args.put("grad_accum_steps", String.valueOf(config.getGradAccumSteps()));
            if (config.getMaxGradNorm() != null)   args.put("max_grad_norm",  config.getMaxGradNorm().toString());

            if (config.getGpuDevices() != null && !config.getGpuDevices().isEmpty()) {
                args.put("gpu_devices", config.getGpuDevices().stream()
                        .map(String::valueOf).reduce((a, b) -> a + "," + b).orElse("0"));
            }
            if (config.getCheckpointInterval() != null) {
                args.put("checkpoint_interval", String.valueOf(config.getCheckpointInterval()));
            }
            if (config.getLoraAdvancedConfig() != null) {
                TrainingConfigDTO.LoraAdvancedConfig lora = config.getLoraAdvancedConfig();
                if (lora.getTargetModules() != null && !lora.getTargetModules().isEmpty())
                    args.put("lora_target_modules", String.join(",", lora.getTargetModules()));
                if (lora.getBiasTrain() != null)
                    args.put("lora_bias", lora.getBiasTrain());
            }
            if (config.getDistillationAdvancedConfig() != null) {
                TrainingConfigDTO.DistillationAdvancedConfig d = config.getDistillationAdvancedConfig();
                if (d.getLossType() != null) args.put("distill_loss_type", d.getLossType());
            }

            args.put("student_model_type", config.getStudentModelType()  != null ? config.getStudentModelType()  : "resnet");
            args.put("student_model_size", config.getStudentModelSize()  != null ? config.getStudentModelSize()  : "resnet50");
            args.put("task_type",          config.getTaskType()          != null ? config.getTaskType()          : "classification");
            args.put("num_classes",        String.valueOf(config.getNumClasses()  != null ? config.getNumClasses()  : 10));
            args.put("image_size",         String.valueOf(config.getImageSize()   != null ? config.getImageSize()   : 224));
            if (config.getDistillationType() != null) args.put("distillation_type", config.getDistillationType());
            if (config.getFeatureLossType()  != null) args.put("feature_loss_type",  config.getFeatureLossType());
            if (config.getAlignFeature()     != null) args.put("align_feature",      String.valueOf(config.getAlignFeature()));
        } else {
            args.put("student_model_type", "resnet");
            args.put("student_model_size", "resnet50");
            args.put("task_type",          "classification");
            args.put("num_classes",        "10");
            args.put("image_size",         "224");
        }

        // 输出目录
        args.put("output_dir", outputRoot + "/" + task.getTaskId());

        // flags（bool 开关，如 auto_save_checkpoint）
        List<String> flags = new ArrayList<>();
        if (config != null && Boolean.TRUE.equals(config.getAutoSaveCheckpoint())) {
            flags.add("auto_save_checkpoint");
        }

        Map<String, Object> body = new HashMap<>();
        body.put("taskId",      task.getTaskId());
        body.put("scriptPath",  script);
        body.put("pythonPath",  pythonPath);
        body.put("apiBaseUrl",  apiBaseUrl);
        body.put("args",        args);
        body.put("flags",       flags);
        return body;
    }

    private String getTrainingScript(String teacherModel, String trainingMode) {
        if ("direct".equals(trainingMode)) return studentOnlyScriptPath;
        if (teacherModel != null &&
            (teacherModel.toLowerCase().contains("qwen") ||
             teacherModel.toLowerCase().contains("qwen2"))) {
            return qwenScriptPath;
        }
        return scriptPath;
    }

    private String getModelPath(String modelName, TrainingConfigDTO config) {
        if (config != null && config.getTeacherModelConfig() != null) {
            String p = config.getTeacherModelConfig().getModelPath();
            if (p != null && !p.isEmpty()) return p;
        }
        return modelsRoot + "/" + modelName;
    }

    private String getStudentModelPath(String modelName, TrainingConfigDTO config) {
        if (config != null && config.getStudentModelConfig() != null) {
            String p = config.getStudentModelConfig().getPretrainPath();
            if (p != null && !p.isEmpty()) return p;
            if ("random".equals(config.getStudentModelConfig().getInitMethod())) return "";
        }
        return modelsRoot + "/" + modelName;
    }
}
