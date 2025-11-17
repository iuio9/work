package com.qczy.federated.service;

import com.qczy.federated.model.FederatedNode;
import com.qczy.federated.model.ModelType;
import com.qczy.federated.model.TrainingJob;
import com.qczy.federated.optimizer.FedAvgOptimizer;
import com.qczy.federated.adapters.DefaultAdapterFactory;
import com.qczy.federated.spi.ModelAdapter;
import com.qczy.federated.flower.FlowerServerManager;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Service;

import java.time.Instant;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

@Service
public class FederatedCoordinatorService {

    private final Map<String, FederatedNode> nodes = new ConcurrentHashMap<>();

    private final Map<String, TrainingJob> jobs = new ConcurrentHashMap<>();

    private final FedAvgOptimizer optimizer = new FedAvgOptimizer();

    @Autowired
    private FlowerServerManager flowerServerManager;

    @Value("${flower.server.port:8080}")
    private int defaultFlowerPort;

    public FederatedNode registerNode(FederatedNode node) {
        node.setActive(true);
        node.setLastHeartbeatAt(Instant.now());
        nodes.put(node.getNodeId(), node);
        return node;
    }

    public void heartbeat(String nodeId, Map<String, Object> metadata) {
        FederatedNode n = nodes.get(nodeId);
        if (n != null) {
            n.setLastHeartbeatAt(Instant.now());
            n.setActive(true);
            n.setMetadata(metadata);
        }
    }

    public TrainingJob createJob(ModelType type, Map<String, Object> hyperParams, List<String> participantNodeIds) {
        TrainingJob job = new TrainingJob();
        job.setJobId(UUID.randomUUID().toString());
        job.setModelType(type);
        job.setHyperParameters(hyperParams);
        job.setParticipantNodeIds(participantNodeIds);
        job.setStatus("CREATED");
        job.setCreatedAt(Instant.now());
        job.setUpdatedAt(Instant.now());
        jobs.put(job.getJobId(), job);
        return job;
    }

    // 周期性参数同步：每 10 秒一次（占位）。实际可按轮次或步数同步
    @Scheduled(fixedDelay = 10000)
    public void synchronizeParameters() {
        for (TrainingJob job : jobs.values()) {
            if (!"RUNNING".equals(job.getStatus())) {
                continue;
            }
            ModelAdapter adapter = DefaultAdapterFactory.forType(job.getModelType());
            List<Map<String, double[]>> updates = new ArrayList<>();
            for (String nodeId : job.getParticipantNodeIds()) {
                FederatedNode node = nodes.get(nodeId);
                if (node != null && node.isActive()) {
                    Map<String, double[]> delta = adapter.trainOneRound(job, nodeId);
                    if (delta != null) {
                        updates.add(delta);
                    }
                }
            }
            if (updates.isEmpty()) {
                continue;
            }
            Map<String, double[]> global = optimizer.aggregate(updates);
            for (String nodeId : job.getParticipantNodeIds()) {
                FederatedNode node = nodes.get(nodeId);
                if (node != null && node.isActive()) {
                    adapter.applyGlobalParameters(global, job, nodeId);
                }
            }
            job.setGlobalParameters(global);
            job.setCurrentRound(job.getCurrentRound() + 1);

            // 聚合后评估一次，建立或对比基线精度
            double acc = 0.0;
            int evalCnt = 0;
            for (String nodeId : job.getParticipantNodeIds()) {
                FederatedNode node = nodes.get(nodeId);
                if (node != null && node.isActive()) {
                    double a = adapter.evaluate(job, nodeId);
                    if (!Double.isNaN(a) && a > 0) {
                        acc += a;
                        evalCnt++;
                    }
                }
            }
            if (evalCnt > 0) {
                acc = acc / evalCnt; // 简化：平均精度
                if (job.getBaselineAccuracy() == null) {
                    job.setBaselineAccuracy(acc);
                }
                job.setLastGlobalAccuracy(acc);

                // 若精度下降超过阈值，则进行保护：标记降级或暂停
                if (job.getBaselineAccuracy() != null && job.getAllowedDropPercent() != null) {
                    double drop = (job.getBaselineAccuracy() - acc) / job.getBaselineAccuracy() * 100.0;
                    if (drop > job.getAllowedDropPercent()) {
                        job.setStatus("DEGRADED");
                    }
                }
            }
            job.setUpdatedAt(Instant.now());
        }
    }

    // 节点存活监控：若超过 30 秒无心跳则标记为不活跃
    @Scheduled(fixedDelay = 5000)
    public void monitorNodes() {
        Instant now = Instant.now();
        for (FederatedNode node : nodes.values()) {
            if (node.getLastHeartbeatAt() == null) {
                node.setActive(false);
                continue;
            }
            if (now.minusSeconds(30).isAfter(node.getLastHeartbeatAt())) {
                node.setActive(false);
            }
        }
    }

    /**
     * 启动训练任务
     * 
     * 功能：
     * 1. 从任务配置中提取训练参数
     * 2. 自动启动 Flower Server 进程
     * 3. 更新任务状态为 RUNNING 或 FAILED
     * 
     * @param jobId 任务ID
     * 
     * 流程：
     * 1. 获取任务对象
     * 2. 从超参数中提取训练轮数（默认 10）
     * 3. 计算最少客户端数（参与节点数）
     * 4. 分配端口（避免冲突：defaultFlowerPort + 任务数）
     * 5. 启动 Flower Server 进程
     * 6. 根据启动结果更新任务状态
     * 
     * 说明：
     * - 启动任务时会自动启动 Flower Server，无需手动启动
     * - 端口自动分配，多个任务不会冲突
     * - 如果启动失败，任务状态会变为 FAILED
     */
    public void startJob(String jobId) {
        TrainingJob job = jobs.get(jobId);
        if (job != null) {
            // ====================================================================
            // 步骤1：提取训练参数
            // ====================================================================
            // 从超参数中获取训练轮数，如果不存在则使用默认值 10
            int numRounds = job.getHyperParameters() != null && job.getHyperParameters().containsKey("numRounds") 
                    ? ((Number) job.getHyperParameters().get("numRounds")).intValue() : 10;
            // 计算最少客户端数：参与节点数（如果为空则使用默认值 2）
            int minClients = job.getParticipantNodeIds() != null ? job.getParticipantNodeIds().size() : 2;
            // 端口分配：默认端口 + 任务数（简单策略，避免端口冲突）
            // 例如：任务1使用8080，任务2使用8081，任务3使用8082...
            int port = defaultFlowerPort + jobs.size();
            
            // ====================================================================
            // 步骤2：启动 Flower Server 进程
            // ====================================================================
            boolean started = flowerServerManager.startServer(
                    jobId,                              // 任务ID
                    job.getModelType().name(),          // 模型类型（YOLO_V8, LSTM等）
                    numRounds,                          // 训练轮数
                    minClients,                         // 最少客户端数
                    port,                               // 服务端口
                    job.getBaselineAccuracy(),          // 基线精度（可选）
                    job.getAllowedDropPercent()        // 允许精度下降百分比
            );
            
            // ====================================================================
            // 步骤3：更新任务状态
            // ====================================================================
            if (started) {
                // 启动成功：状态改为 RUNNING
                job.setStatus("RUNNING");
                job.setUpdatedAt(Instant.now());
            } else {
                // 启动失败：状态改为 FAILED
                job.setStatus("FAILED");
            }
        }
    }

    /**
     * 停止训练任务
     * 
     * 功能：
     * 1. 停止 Flower Server 进程
     * 2. 更新任务状态为 STOPPED
     * 
     * @param jobId 任务ID
     * 
     * 流程：
     * 1. 获取任务对象
     * 2. 调用 FlowerServerManager 停止进程
     * 3. 更新任务状态和更新时间
     * 
     * 说明：
     * - 停止任务时会自动停止对应的 Flower Server 进程
     * - 已连接的客户端会自动断开连接
     * - 任务状态会更新为 STOPPED
     */
    public void stopJob(String jobId) {
        TrainingJob job = jobs.get(jobId);
        if (job != null) {
            // 停止 Flower Server 进程（如果正在运行）
            flowerServerManager.stopServer(jobId);
            // 更新任务状态
            job.setStatus("STOPPED");
            job.setUpdatedAt(Instant.now());
        }
    }

    public Collection<FederatedNode> listNodes() {
        return nodes.values();
    }

    public Collection<TrainingJob> listJobs() {
        return jobs.values();
    }
}





