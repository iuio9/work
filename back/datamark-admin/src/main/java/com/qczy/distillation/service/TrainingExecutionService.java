package com.qczy.distillation.service;

import com.alibaba.fastjson.JSON;
import com.qczy.distillation.mapper.MdTrainingTaskMapper;
import com.qczy.distillation.model.dto.TrainingConfigDTO;
import com.qczy.distillation.model.entity.MdTrainingTaskEntity;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.scheduling.annotation.Async;
import org.springframework.stereotype.Service;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * 训练执行服务
 *
 * 功能：
 * 1. 异步启动Python训练脚本
 * 2. 管理训练进程
 * 3. 解析training_config JSON并转换为命令行参数
 * 4. 监控训练进程状态
 *
 * @author AI Assistant
 * @date 2025-01-25
 */
@Service
public class TrainingExecutionService {

    private static final Logger logger = LoggerFactory.getLogger(TrainingExecutionService.class);

    @Autowired
    private MdTrainingTaskMapper trainingTaskMapper;

    @Autowired
    private MdTrainingTaskService trainingTaskService;

    /**
     * Python解释器路径
     */
    @Value("${distillation.python.path:python3}")
    private String pythonPath;

    /**
     * 训练脚本路径
     */
    @Value("${distillation.script.path:/home/user/work/back/datamark-admin/train_distillation.py}")
    private String scriptPath;

    /**
     * Qwen2.5-VL训练脚本路径
     */
    @Value("${distillation.qwen-script.path:/home/user/work/back/datamark-admin/train_qwen_vl_distillation.py}")
    private String qwenScriptPath;

    /**
     * 后端API基础URL
     */
    @Value("${distillation.api.base-url:http://localhost:8080}")
    private String apiBaseUrl;

    /**
     * 模型存储根目录
     */
    @Value("${distillation.models.root:/data/models}")
    private String modelsRoot;

    /**
     * 数据集根目录
     */
    @Value("${distillation.datasets.root:/data/datasets}")
    private String datasetsRoot;

    /**
     * 训练输出根目录
     */
    @Value("${distillation.output.root:/data/training_output}")
    private String outputRoot;

    /**
     * 进程管理Map: taskId -> Process
     */
    private final Map<String, Process> runningProcesses = new ConcurrentHashMap<>();

    /**
     * 异步启动训练任务
     *
     * @param taskId 任务ID
     */
    @Async("taskExecutor")
    public void startTrainingAsync(String taskId) {
        logger.info("开始异步训练任务: {}", taskId);

        try {
            // 1. 从数据库读取任务配置
            MdTrainingTaskEntity task = trainingTaskMapper.selectByTaskId(taskId);
            if (task == null) {
                logger.error("任务不存在: {}", taskId);
                return;
            }

            // 2. 解析training_config JSON
            TrainingConfigDTO config = null;
            if (task.getTrainingConfig() != null && !task.getTrainingConfig().isEmpty()) {
                config = JSON.parseObject(task.getTrainingConfig(), TrainingConfigDTO.class);
            }

            // 3. 构建Python命令
            List<String> command = buildPythonCommand(task, config);

            // 4. 打印命令（用于调试）
            logger.info("训练命令: {}", String.join(" ", command));

            // 5. 启动进程
            ProcessBuilder pb = new ProcessBuilder(command);
            pb.redirectErrorStream(true); // 合并标准输出和错误输出

            // 设置工作目录
            pb.directory(new File(System.getProperty("user.dir")));

            // 启动进程
            Process process = pb.start();
            runningProcesses.put(taskId, process);

            logger.info("Python训练进程已启动，任务ID: {}", taskId);

            // 6. 读取并打印输出
            try (BufferedReader reader = new BufferedReader(
                    new InputStreamReader(process.getInputStream()))) {
                String line;
                while ((line = reader.readLine()) != null) {
                    logger.info("[{}] {}", taskId, line);
                }
            }

            // 7. 等待进程结束
            int exitCode = process.waitFor();
            logger.info("训练进程结束: taskId={}, exitCode={}", taskId, exitCode);

            // 8. 清理进程引用
            runningProcesses.remove(taskId);

            // 9. 根据退出码更新任务状态
            if (exitCode != 0) {
                trainingTaskService.updateError(taskId, "训练进程异常退出，退出码: " + exitCode);
            }

        } catch (Exception e) {
            logger.error("训练任务执行失败: taskId={}", taskId, e);
            trainingTaskService.updateError(taskId, "训练执行异常: " + e.getMessage());
            runningProcesses.remove(taskId);
        }
    }

    /**
     * 停止训练任务
     *
     * @param taskId 任务ID
     * @return 是否成功
     */
    public boolean stopTraining(String taskId) {
        Process process = runningProcesses.get(taskId);
        if (process != null && process.isAlive()) {
            logger.info("正在停止训练任务: {}", taskId);
            process.destroy();

            // 等待进程结束（最多5秒）
            try {
                boolean terminated = process.waitFor(5, java.util.concurrent.TimeUnit.SECONDS);
                if (!terminated) {
                    logger.warn("进程未在5秒内结束，强制终止: {}", taskId);
                    process.destroyForcibly();
                }
                runningProcesses.remove(taskId);
                return true;
            } catch (InterruptedException e) {
                logger.error("等待进程结束时被中断: {}", taskId, e);
                Thread.currentThread().interrupt();
                return false;
            }
        } else {
            logger.warn("没有找到运行中的进程: {}", taskId);
            return false;
        }
    }

    /**
     * 检查任务是否正在运行
     *
     * @param taskId 任务ID
     * @return 是否运行中
     */
    public boolean isTrainingRunning(String taskId) {
        Process process = runningProcesses.get(taskId);
        return process != null && process.isAlive();
    }

    /**
     * 构建Python训练命令
     *
     * @param task 训练任务实体
     * @param config 高级配置DTO
     * @return 命令列表
     */
    private List<String> buildPythonCommand(MdTrainingTaskEntity task, TrainingConfigDTO config) {
        List<String> command = new ArrayList<>();

        // Python解释器
        command.add(pythonPath);

        // 训练脚本（根据教师模型类型动态选择）
        command.add(getTrainingScript(task.getTeacherModel()));

        // ========== 基础配置 ==========
        command.add("--task_id");
        command.add(task.getTaskId());

        command.add("--api_base_url");
        command.add(apiBaseUrl);

        // ========== 模型配置 ==========
        command.add("--teacher_model");
        command.add(task.getTeacherModel());

        command.add("--student_model");
        command.add(task.getStudentModel());

        // 教师模型路径
        String teacherPath = getModelPath(task.getTeacherModel(), config);
        command.add("--teacher_path");
        command.add(teacherPath);

        // 学生模型路径（可选）
        String studentPath = getStudentModelPath(task.getStudentModel(), config);
        if (studentPath != null && !studentPath.isEmpty()) {
            command.add("--student_path");
            command.add(studentPath);
        }

        // ========== 数据集配置 ==========
        command.add("--dataset_id");
        command.add(String.valueOf(task.getDatasetId()));

        if (task.getValDatasetId() != null) {
            command.add("--val_dataset_id");
            command.add(String.valueOf(task.getValDatasetId()));
        }

        // 数据集根目录（从配置文件读取）
        logger.info("添加数据集根目录参数: datasetsRoot = {}", datasetsRoot);
        command.add("--datasets_root");
        command.add(datasetsRoot);

        // ========== 训练参数 ==========
        command.add("--epochs");
        command.add(String.valueOf(task.getTotalEpochs()));

        command.add("--batch_size");
        command.add(String.valueOf(task.getBatchSize()));

        command.add("--learning_rate");
        command.add(task.getLearningRate().toString());

        // ========== LoRA配置 ==========
        command.add("--lora_rank");
        command.add(String.valueOf(task.getLoraRank()));

        command.add("--lora_alpha");
        command.add(String.valueOf(task.getLoraAlpha()));

        command.add("--lora_dropout");
        command.add(task.getLoraDropout().toString());

        // ========== 知识蒸馏参数 ==========
        command.add("--temperature");
        command.add(task.getTemperature().toString());

        command.add("--hard_label_weight");
        command.add(String.valueOf(1.0 - task.getAlpha().doubleValue()));

        command.add("--soft_label_weight");
        command.add(task.getAlpha().toString());

        // ========== 高级配置（从JSON解析） ==========
        if (config != null) {
            // 优化器配置
            if (config.getOptimizer() != null) {
                command.add("--optimizer");
                command.add(config.getOptimizer());
            }

            if (config.getLrScheduler() != null) {
                command.add("--lr_scheduler");
                command.add(config.getLrScheduler());
            }

            if (config.getWeightDecay() != null) {
                command.add("--weight_decay");
                command.add(config.getWeightDecay().toString());
            }

            if (config.getGradAccumSteps() != null) {
                command.add("--grad_accum_steps");
                command.add(String.valueOf(config.getGradAccumSteps()));
            }

            if (config.getMaxGradNorm() != null) {
                command.add("--max_grad_norm");
                command.add(config.getMaxGradNorm().toString());
            }

            // GPU配置
            if (config.getGpuDevices() != null && !config.getGpuDevices().isEmpty()) {
                command.add("--gpu_devices");
                command.add(config.getGpuDevices().stream()
                        .map(String::valueOf)
                        .reduce((a, b) -> a + "," + b)
                        .orElse("0"));
            }

            if (config.getAutoSaveCheckpoint() != null) {
                command.add("--auto_save_checkpoint");
                command.add(String.valueOf(config.getAutoSaveCheckpoint()));
            }

            if (config.getCheckpointInterval() != null) {
                command.add("--checkpoint_interval");
                command.add(String.valueOf(config.getCheckpointInterval()));
            }

            // LoRA高级配置
            if (config.getLoraAdvancedConfig() != null) {
                TrainingConfigDTO.LoraAdvancedConfig loraConfig = config.getLoraAdvancedConfig();

                if (loraConfig.getTargetModules() != null && !loraConfig.getTargetModules().isEmpty()) {
                    command.add("--lora_target_modules");
                    command.add(String.join(",", loraConfig.getTargetModules()));
                }

                if (loraConfig.getBiasTrain() != null) {
                    command.add("--lora_bias");
                    command.add(loraConfig.getBiasTrain());
                }
            }

            // 知识蒸馏高级配置
            if (config.getDistillationAdvancedConfig() != null) {
                TrainingConfigDTO.DistillationAdvancedConfig distillConfig =
                        config.getDistillationAdvancedConfig();

                if (distillConfig.getHardLabelWeight() != null) {
                    command.add("--hard_label_weight");
                    command.add(distillConfig.getHardLabelWeight().toString());
                }

                if (distillConfig.getSoftLabelWeight() != null) {
                    command.add("--soft_label_weight");
                    command.add(distillConfig.getSoftLabelWeight().toString());
                }

                if (distillConfig.getLossType() != null) {
                    command.add("--distill_loss_type");
                    command.add(distillConfig.getLossType());
                }
            }

            // ========== Qwen2.5-VL多模型配置 ==========
            // 注意：Qwen脚本要求这些参数必需，所以提供默认值
            String studentModelType = config.getStudentModelType() != null ?
                config.getStudentModelType() : "resnet";
            command.add("--student_model_type");
            command.add(studentModelType);

            String studentModelSize = config.getStudentModelSize() != null ?
                config.getStudentModelSize() : "resnet50";
            command.add("--student_model_size");
            command.add(studentModelSize);

            String taskType = config.getTaskType() != null ?
                config.getTaskType() : "classification";
            command.add("--task_type");
            command.add(taskType);

            Integer numClasses = config.getNumClasses() != null ?
                config.getNumClasses() : 10;
            command.add("--num_classes");
            command.add(String.valueOf(numClasses));

            Integer imageSize = config.getImageSize() != null ?
                config.getImageSize() : 224;
            command.add("--image_size");
            command.add(String.valueOf(imageSize));

            if (config.getDistillationType() != null) {
                command.add("--distillation_type");
                command.add(config.getDistillationType());
            }

            if (config.getFeatureLossType() != null) {
                command.add("--feature_loss_type");
                command.add(config.getFeatureLossType());
            }

            if (config.getAlignFeature() != null) {
                command.add("--align_feature");
                command.add(String.valueOf(config.getAlignFeature()));
            }
        } else {
            // config 为 null时，为 Qwen 脚本提供默认值
            command.add("--student_model_type");
            command.add("resnet");
            command.add("--student_model_size");
            command.add("resnet50");
            command.add("--task_type");
            command.add("classification");
            command.add("--num_classes");
            command.add("10");
            command.add("--image_size");
            command.add("224");
        }

        // ========== 输出配置 ==========
        String outputDir = outputRoot + "/" + task.getTaskId();
        command.add("--output_dir");
        command.add(outputDir);

        return command;
    }

    /**
     * 根据教师模型类型选择训练脚本
     *
     * @param teacherModel 教师模型名称
     * @return 训练脚本路径
     */
    private String getTrainingScript(String teacherModel) {
        if (teacherModel != null &&
            (teacherModel.toLowerCase().contains("qwen") ||
             teacherModel.toLowerCase().contains("qwen2"))) {
            logger.info("检测到Qwen模型，使用Qwen专用训练脚本: {}", qwenScriptPath);
            return qwenScriptPath;
        }
        return scriptPath;
    }

    /**
     * 获取教师模型路径
     *
     * @param modelName 模型名称
     * @param config 配置
     * @return 模型路径
     */
    private String getModelPath(String modelName, TrainingConfigDTO config) {
        // 优先使用JSON配置中的路径
        if (config != null && config.getTeacherModelConfig() != null) {
            String modelPath = config.getTeacherModelConfig().getModelPath();
            if (modelPath != null && !modelPath.isEmpty()) {
                return modelPath;
            }
        }

        // 默认路径：modelsRoot/modelName
        return modelsRoot + "/" + modelName;
    }

    /**
     * 获取学生模型路径
     *
     * @param modelName 模型名称
     * @param config 配置
     * @return 模型路径
     */
    private String getStudentModelPath(String modelName, TrainingConfigDTO config) {
        // 优先使用JSON配置中的路径
        if (config != null && config.getStudentModelConfig() != null) {
            String pretrainPath = config.getStudentModelConfig().getPretrainPath();
            if (pretrainPath != null && !pretrainPath.isEmpty()) {
                return pretrainPath;
            }
        }

        // 如果配置中指定了随机初始化，返回空
        if (config != null && config.getStudentModelConfig() != null) {
            String initMethod = config.getStudentModelConfig().getInitMethod();
            if ("random".equals(initMethod)) {
                return "";
            }
        }

        // 默认路径
        return modelsRoot + "/" + modelName;
    }

    /**
     * 获取正在运行的任务数量
     *
     * @return 运行中的任务数
     */
    public int getRunningTaskCount() {
        return (int) runningProcesses.values().stream()
                .filter(Process::isAlive)
                .count();
    }

    /**
     * 获取所有运行中的任务ID
     *
     * @return 任务ID列表
     */
    public List<String> getRunningTaskIds() {
        return new ArrayList<>(runningProcesses.keySet());
    }
}
