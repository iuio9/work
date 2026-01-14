package com.qczy.distillation.manager;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;

import java.io.BufferedReader;
import java.io.File;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * 模型蒸馏训练进程管理器
 * ========================
 *
 * 功能：
 * 1. 启动模型蒸馏训练 Python 进程
 * 2. 停止训练进程
 * 3. 监控训练进程运行状态
 * 4. 捕获并记录训练日志
 *
 * 工作原理：
 * - 使用 ProcessBuilder 启动 Python 训练脚本
 * - 通过命令行参数传递训练配置（教师模型、学生模型、数据集等）
 * - 异步读取进程输出并记录到日志
 * - 使用 ConcurrentHashMap 管理多个训练任务的进程
 *
 * 配置说明：
 * - training.python.executable: Python 可执行文件路径（默认：python3）
 * - training.python.script-dir: 训练脚本目录
 * - training.python.distillation-script: 蒸馏训练脚本名称
 * - training.python.venv-path: Python虚拟环境路径（可选）
 *
 * @author AI Assistant
 * @date 2026-01-14
 */
@Component
public class DistillationTrainingManager {
    private static final Logger log = LoggerFactory.getLogger(DistillationTrainingManager.class);

    /**
     * Python 可执行文件路径
     * 配置项：training.python.executable（默认：python3）
     * 可以是：
     * - python3
     * - /usr/bin/python3
     * - /home/user/work/venv/bin/python  (虚拟环境中的python)
     */
    @Value("${training.python.executable:python3}")
    private String pythonExecutable;

    /**
     * 训练脚本目录
     * 配置项：training.python.script-dir
     * 默认：项目根目录下的 python_scripts 文件夹
     */
    @Value("${training.python.script-dir:python_scripts}")
    private String scriptDir;

    /**
     * 蒸馏训练脚本名称
     * 配置项：training.python.distillation-script（默认：distillation_train.py）
     */
    @Value("${training.python.distillation-script:distillation_train.py}")
    private String distillationScript;

    /**
     * Python虚拟环境路径（可选）
     * 如果配置了虚拟环境，会优先使用虚拟环境中的python
     */
    @Value("${training.python.venv-path:#{null}}")
    private String venvPath;

    /**
     * 教师模型路径（Qwen2.5-VL等）
     * 配置项：training.models.teacher-model-path
     */
    @Value("${training.models.teacher-model-path:/home/user/models/qwen2.5-vl-3b-instruct}")
    private String teacherModelPath;

    /**
     * 学生模型保存路径
     * 配置项：training.models.student-model-path
     */
    @Value("${training.models.student-model-path:/home/user/work/models/students}")
    private String studentModelPath;

    /**
     * 数据集根目录
     * 配置项：training.datasets.root-path
     */
    @Value("${training.datasets.root-path:/home/user/datasets}")
    private String datasetsRootPath;

    /**
     * 存储每个任务的训练进程
     * Key: 任务ID（taskId）
     * Value: Process 对象
     */
    private final Map<String, Process> trainingProcesses = new ConcurrentHashMap<>();

    /**
     * 启动蒸馏训练进程
     *
     * @param taskId 训练任务ID
     * @param teacherModel 教师模型类型 (qwen2.5-vl-3b, resnet101, vit-base等)
     * @param studentModel 学生模型类型 (resnet18, yolov8n, unet等)
     * @param datasetName 数据集名称 (cifar10, imagenet, coco等)
     * @param batchSize 批次大小
     * @param numEpochs 训练轮数
     * @param learningRate 学习率
     * @param temperature 蒸馏温度
     * @param alpha 蒸馏权重
     * @return 是否启动成功
     */
    public boolean startTraining(
            String taskId,
            String teacherModel,
            String studentModel,
            String datasetName,
            Integer batchSize,
            Integer numEpochs,
            Double learningRate,
            Double temperature,
            Double alpha
    ) {
        try {
            // ====================================================================
            // 步骤1：确定Python可执行文件路径
            // ====================================================================
            String pythonExec = pythonExecutable;

            // 如果配置了虚拟环境，优先使用虚拟环境中的python
            if (venvPath != null && !venvPath.isEmpty()) {
                File venvPython = new File(venvPath, "bin/python");
                if (venvPython.exists()) {
                    pythonExec = venvPython.getAbsolutePath();
                    log.info("使用虚拟环境Python: {}", pythonExec);
                } else {
                    log.warn("虚拟环境Python不存在: {}, 使用默认Python", venvPython.getAbsolutePath());
                }
            }

            // ====================================================================
            // 步骤2：构建Python脚本完整路径
            // ====================================================================
            File scriptFile = new File(scriptDir, distillationScript);
            String scriptPath = scriptFile.getAbsolutePath();

            log.info("训练脚本路径: {}", scriptPath);

            // 检查脚本是否存在
            if (!scriptFile.exists()) {
                log.error("训练脚本不存在: {}", scriptPath);
                return false;
            }

            // ====================================================================
            // 步骤3：构建命令参数
            // ====================================================================
            List<String> command = new ArrayList<>();
            command.add(pythonExec);
            command.add(scriptPath);

            // 必需参数
            command.add("--task-id");
            command.add(taskId);

            command.add("--teacher-model");
            command.add(teacherModel);

            command.add("--student-model");
            command.add(studentModel);

            command.add("--dataset");
            command.add(datasetName);

            // 可选参数
            if (batchSize != null) {
                command.add("--batch-size");
                command.add(String.valueOf(batchSize));
            }

            if (numEpochs != null) {
                command.add("--epochs");
                command.add(String.valueOf(numEpochs));
            }

            if (learningRate != null) {
                command.add("--learning-rate");
                command.add(String.valueOf(learningRate));
            }

            if (temperature != null) {
                command.add("--temperature");
                command.add(String.valueOf(temperature));
            }

            if (alpha != null) {
                command.add("--alpha");
                command.add(String.valueOf(alpha));
            }

            // 路径参数
            command.add("--teacher-model-path");
            command.add(teacherModelPath);

            command.add("--student-model-save-path");
            command.add(studentModelPath);

            command.add("--dataset-root");
            command.add(datasetsRootPath);

            // 打印完整命令（调试用）
            log.info("启动训练命令: {}", String.join(" ", command));

            // ====================================================================
            // 步骤4：配置进程
            // ====================================================================
            ProcessBuilder pb = new ProcessBuilder(command);

            // 设置工作目录
            pb.directory(new File(System.getProperty("user.dir")));

            // 将标准错误重定向到标准输出
            pb.redirectErrorStream(true);

            // 设置环境变量（可选）
            Map<String, String> env = pb.environment();
            // 例如：设置CUDA设备
            // env.put("CUDA_VISIBLE_DEVICES", "0");

            // ====================================================================
            // 步骤5：启动进程
            // ====================================================================
            Process process = pb.start();

            log.info("训练进程已启动，任务ID: {}, PID: {}", taskId, process.pid());

            // ====================================================================
            // 步骤6：异步读取进程输出
            // ====================================================================
            new Thread(() -> {
                try (BufferedReader reader = new BufferedReader(
                        new InputStreamReader(process.getInputStream()))) {
                    String line;
                    while ((line = reader.readLine()) != null) {
                        // 记录训练日志
                        log.info("[训练任务 {}]: {}", taskId, line);

                        // TODO: 可以在这里解析日志，提取训练指标
                        // 例如：Loss, Accuracy, Epoch进度等
                        // 然后更新到数据库或发送给前端
                    }
                } catch (Exception e) {
                    log.error("读取训练进程输出失败", e);
                }
            }).start();

            // ====================================================================
            // 步骤7：监控进程状态
            // ====================================================================
            new Thread(() -> {
                try {
                    // 等待进程结束
                    int exitCode = process.waitFor();

                    if (exitCode == 0) {
                        log.info("训练任务完成，任务ID: {}", taskId);
                        // TODO: 更新任务状态为COMPLETED
                    } else {
                        log.error("训练任务异常退出，任务ID: {}, 退出码: {}", taskId, exitCode);
                        // TODO: 更新任务状态为FAILED
                    }

                    // 从Map中移除
                    trainingProcesses.remove(taskId);

                } catch (InterruptedException e) {
                    log.error("等待训练进程结束被中断", e);
                    Thread.currentThread().interrupt();
                }
            }).start();

            // ====================================================================
            // 步骤8：保存进程引用
            // ====================================================================
            trainingProcesses.put(taskId, process);

            return true;

        } catch (Exception e) {
            log.error("启动训练失败，任务ID: {}", taskId, e);
            return false;
        }
    }

    /**
     * 停止训练进程
     *
     * @param taskId 任务ID
     * @return 是否成功停止
     */
    public boolean stopTraining(String taskId) {
        Process process = trainingProcesses.remove(taskId);

        if (process != null && process.isAlive()) {
            log.info("正在停止训练，任务ID: {}", taskId);

            // 优雅关闭（发送SIGTERM）
            process.destroy();

            try {
                // 等待3秒
                boolean exited = process.waitFor(3, java.util.concurrent.TimeUnit.SECONDS);

                if (!exited) {
                    // 如果3秒后还没停止，强制终止
                    log.warn("训练进程未响应，强制终止，任务ID: {}", taskId);
                    process.destroyForcibly();
                }

                log.info("训练已停止，任务ID: {}", taskId);
                return true;

            } catch (InterruptedException e) {
                log.error("等待训练进程停止被中断", e);
                process.destroyForcibly();
                Thread.currentThread().interrupt();
                return false;
            }
        }

        log.warn("训练进程不存在或已停止，任务ID: {}", taskId);
        return false;
    }

    /**
     * 暂停训练进程（发送SIGSTOP信号）
     * 注意：Java标准库不支持发送SIGSTOP，需要通过系统命令实现
     *
     * @param taskId 任务ID
     * @return 是否成功暂停
     */
    public boolean pauseTraining(String taskId) {
        Process process = trainingProcesses.get(taskId);

        if (process != null && process.isAlive()) {
            try {
                long pid = process.pid();

                // 使用系统命令发送SIGSTOP信号
                ProcessBuilder pb = new ProcessBuilder("kill", "-STOP", String.valueOf(pid));
                Process killProcess = pb.start();
                int exitCode = killProcess.waitFor();

                if (exitCode == 0) {
                    log.info("训练已暂停，任务ID: {}, PID: {}", taskId, pid);
                    return true;
                } else {
                    log.error("暂停训练失败，任务ID: {}, 退出码: {}", taskId, exitCode);
                    return false;
                }

            } catch (Exception e) {
                log.error("暂停训练失败，任务ID: {}", taskId, e);
                return false;
            }
        }

        log.warn("训练进程不存在或已停止，任务ID: {}", taskId);
        return false;
    }

    /**
     * 恢复训练进程（发送SIGCONT信号）
     *
     * @param taskId 任务ID
     * @return 是否成功恢复
     */
    public boolean resumeTraining(String taskId) {
        Process process = trainingProcesses.get(taskId);

        if (process != null && process.isAlive()) {
            try {
                long pid = process.pid();

                // 使用系统命令发送SIGCONT信号
                ProcessBuilder pb = new ProcessBuilder("kill", "-CONT", String.valueOf(pid));
                Process killProcess = pb.start();
                int exitCode = killProcess.waitFor();

                if (exitCode == 0) {
                    log.info("训练已恢复，任务ID: {}, PID: {}", taskId, pid);
                    return true;
                } else {
                    log.error("恢复训练失败，任务ID: {}, 退出码: {}", taskId, exitCode);
                    return false;
                }

            } catch (Exception e) {
                log.error("恢复训练失败，任务ID: {}", taskId, e);
                return false;
            }
        }

        log.warn("训练进程不存在或已停止，任务ID: {}", taskId);
        return false;
    }

    /**
     * 检查训练进程是否正在运行
     *
     * @param taskId 任务ID
     * @return 是否正在运行
     */
    public boolean isTrainingRunning(String taskId) {
        Process process = trainingProcesses.get(taskId);
        return process != null && process.isAlive();
    }

    /**
     * 获取正在运行的训练任务数量
     *
     * @return 运行中的任务数量
     */
    public int getRunningTrainingCount() {
        return (int) trainingProcesses.values().stream()
                .filter(Process::isAlive)
                .count();
    }

    /**
     * 获取所有正在运行的任务ID列表
     *
     * @return 任务ID列表
     */
    public List<String> getRunningTaskIds() {
        return trainingProcesses.entrySet().stream()
                .filter(entry -> entry.getValue().isAlive())
                .map(Map.Entry::getKey)
                .toList();
    }
}
