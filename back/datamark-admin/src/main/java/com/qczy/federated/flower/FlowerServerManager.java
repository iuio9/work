package com.qczy.federated.flower;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;

import java.io.BufferedReader;
import java.io.File;
import java.io.InputStreamReader;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Flower Server 进程管理器
 * ========================
 * 
 * 功能：
 * 1. 启动 Flower Server Python 进程
 * 2. 停止 Flower Server 进程
 * 3. 监控进程运行状态
 * 4. 捕获并记录 Flower Server 的输出日志
 * 
 * 工作原理：
 * - 使用 ProcessBuilder 启动 Python 脚本（flower_server.py）
 * - 通过命令行参数传递配置（模型类型、轮数、端口等）
 * - 异步读取进程输出并记录到日志
 * - 使用 ConcurrentHashMap 管理多个任务的进程
 * 
 * 配置说明：
 * - flower.server.script.path: Flower Server 脚本路径（默认：flower_server.py）
 * - flower.python.executable: Python 可执行文件路径（默认：python3）
 * 
 * @author AI Assistant
 * @date 2025-01-20
 */
@Component
public class FlowerServerManager {
    private static final Logger log = LoggerFactory.getLogger(FlowerServerManager.class);

    /**
     * Flower Server Python 脚本路径
     * 配置项：flower.server.script.path（默认：flower_server.py）
     * 路径相对于项目根目录
     */
    @Value("${flower.server.script.path:flower_server.py}")
    private String flowerServerScriptPath;

    /**
     * Python 可执行文件路径
     * 配置项：flower.python.executable（默认：python3）
     * 可以是 python、python3、或完整路径
     */
    @Value("${flower.python.executable:python3}")
    private String pythonExecutable;

    /**
     * 存储每个任务的 Flower Server 进程
     * Key: 任务ID（jobId）
     * Value: Process 对象
     * 
     * 使用 ConcurrentHashMap 保证线程安全
     */
    private final Map<String, Process> serverProcesses = new ConcurrentHashMap<>();

    /**
     * 启动 Flower Server 进程
     * 
     * @param jobId 任务ID（用于标识和管理进程）
     * @param modelType 模型类型 (YOLO_V8, LSTM, UNET, RESNET, VISION_TRANSFORMER)
     * @param numRounds 训练轮数（Flower Server 将运行这么多轮）
     * @param minClients 最少客户端数（至少需要这么多客户端连接才能开始训练）
     * @param port 服务端口（Flower Server 监听的端口）
     * @param baselineAccuracy 基线精度（可选，如果提供则使用此值，否则使用首次评估结果）
     * @param allowedDropPercent 允许精度下降百分比（默认 5.0，超过此值将标记为 DEGRADED）
     * @return 是否启动成功（true：成功，false：失败）
     * 
     * 流程：
     * 1. 构建 Python 命令（python3 flower_server.py --model-type ...）
     * 2. 设置工作目录为项目根目录
     * 3. 启动进程
     * 4. 异步读取进程输出并记录到日志
     * 5. 将进程保存到 Map 中以便后续管理
     * 
     * 注意事项：
     * - 进程会一直运行直到完成所有训练轮次或手动停止
     * - 标准输出和错误输出都会被重定向到同一个流
     * - 日志会以 [Flower Server {jobId}]: 为前缀，便于区分不同任务
     */
    public boolean startServer(
            String jobId,
            String modelType,
            int numRounds,
            int minClients,
            int port,
            Double baselineAccuracy,
            Double allowedDropPercent
    ) {
        try {
            // ====================================================================
            // 步骤1：构建 Python 命令
            // ====================================================================
            // 基础命令：python3 flower_server.py
            ProcessBuilder pb = new ProcessBuilder(
                    pythonExecutable,           // Python 可执行文件（如：python3）
                    flowerServerScriptPath,      // Flower Server 脚本路径
                    "--model-type", modelType,  // 模型类型
                    "--num-rounds", String.valueOf(numRounds),  // 训练轮数
                    "--min-clients", String.valueOf(minClients), // 最少客户端数
                    "--port", String.valueOf(port)              // 服务端口
            );

            // 可选参数：基线精度（如果提供）
            if (baselineAccuracy != null) {
                pb.command().add("--baseline-accuracy");
                pb.command().add(String.valueOf(baselineAccuracy));
            }

            // 可选参数：允许精度下降百分比（如果提供）
            if (allowedDropPercent != null) {
                pb.command().add("--allowed-drop-percent");
                pb.command().add(String.valueOf(allowedDropPercent));
            }

            // ====================================================================
            // 步骤2：配置进程环境
            // ====================================================================
            // 设置工作目录为项目根目录（确保能找到 flower_server.py）
            pb.directory(new File(System.getProperty("user.dir")));
            // 将标准错误重定向到标准输出（统一处理日志）
            pb.redirectErrorStream(true);

            // ====================================================================
            // 步骤3：启动进程
            // ====================================================================
            Process process = pb.start();

            // ====================================================================
            // 步骤4：异步读取进程输出（避免阻塞）
            // ====================================================================
            // 创建新线程读取进程输出并记录到日志
            new Thread(() -> {
                try (BufferedReader reader = new BufferedReader(
                        new InputStreamReader(process.getInputStream()))) {
                    String line;
                    // 逐行读取输出
                    while ((line = reader.readLine()) != null) {
                        // 记录日志，前缀 [Flower Server {jobId}] 便于区分不同任务
                        log.info("[Flower Server {}]: {}", jobId, line);
                    }
                } catch (Exception e) {
                    log.error("Error reading Flower Server output", e);
                }
            }).start();

            // ====================================================================
            // 步骤5：保存进程到 Map 中以便后续管理
            // ====================================================================
            serverProcesses.put(jobId, process);
            log.info("Flower Server started for job: {} on port: {}", jobId, port);

            return true;
        } catch (Exception e) {
            log.error("Failed to start Flower Server for job: {}", jobId, e);
            return false;
        }
    }

    /**
     * 停止 Flower Server 进程
     * 
     * @param jobId 任务ID
     * @return 是否成功停止（true：成功停止，false：进程不存在或已停止）
     * 
     * 流程：
     * 1. 从 Map 中移除并获取进程对象
     * 2. 检查进程是否存在且仍在运行
     * 3. 强制终止进程（destroyForcibly）
     * 
     * 说明：
     * - destroyForcibly() 会强制终止进程，即使进程正在运行
     * - 进程终止后，Flower Server 会停止接受新的客户端连接
     * - 已连接的客户端会自动断开连接
     */
    public boolean stopServer(String jobId) {
        // 从 Map 中移除并获取进程对象
        Process process = serverProcesses.remove(jobId);
        // 检查进程是否存在且仍在运行
        if (process != null && process.isAlive()) {
            // 强制终止进程
            process.destroyForcibly();
            log.info("Flower Server stopped for job: {}", jobId);
            return true;
        }
        // 进程不存在或已停止
        return false;
    }

    /**
     * 检查 Flower Server 是否正在运行
     * 
     * @param jobId 任务ID
     * @return 是否正在运行（true：运行中，false：未运行或不存在）
     * 
     * 说明：
     * - 检查进程是否存在于 Map 中
     * - 检查进程是否仍在运行（isAlive()）
     * - 如果进程已结束但仍在 Map 中，会返回 false
     */
    public boolean isServerRunning(String jobId) {
        Process process = serverProcesses.get(jobId);
        return process != null && process.isAlive();
    }
}

