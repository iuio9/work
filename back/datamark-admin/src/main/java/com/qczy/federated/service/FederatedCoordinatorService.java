package com.qczy.federated.service;

import com.alibaba.fastjson.JSON;
import com.qczy.federated.mapper.FederatedNodeMapper;
import com.qczy.federated.mapper.TrainingJobMapper;
import com.qczy.federated.model.FederatedNode;
import com.qczy.federated.model.ModelType;
import com.qczy.federated.model.TrainingJob;
import com.qczy.federated.model.entity.FederatedNodeEntity;
import com.qczy.federated.model.entity.TrainingJobEntity;
import com.qczy.federated.flower.FlowerServerManager;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.BeanUtils;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import javax.annotation.PostConstruct;
import java.math.BigDecimal;
import java.time.Instant;
import java.time.LocalDateTime;
import java.time.ZoneId;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Collectors;

/**
 * 联邦学习协调服务
 *
 * 功能：
 * 1. 节点管理：注册、心跳、状态监控
 * 2. 任务管理：创建、启动、停止、查询
 * 3. 训练协调：参数同步、精度监控、容错处理
 * 4. 数据持久化：与数据库同步
 *
 * 架构：
 * - 内存缓存 + 数据库持久化双层架构
 * - 内存缓存用于快速访问和实时更新
 * - 数据库用于持久化和历史记录
 *
 * @author AI Assistant
 * @date 2025-01-20
 */
@Service
public class FederatedCoordinatorService {

    private static final Logger logger = LoggerFactory.getLogger(FederatedCoordinatorService.class);

    // ========== 内存缓存：用于快速访问 ==========
    private final Map<String, FederatedNode> nodeCache = new ConcurrentHashMap<>();
    private final Map<String, TrainingJob> jobCache = new ConcurrentHashMap<>();

    // ========== 依赖注入 ==========
    @Autowired
    private FederatedNodeMapper nodeMapper;

    @Autowired
    private TrainingJobMapper jobMapper;

    @Autowired
    private FlowerServerManager flowerServerManager;

    // ========== 配置参数 ==========
    @Value("${flower.server.port:8080}")
    private int defaultFlowerPort;

    @Value("${federated.node.heartbeat.timeout:30}")
    private int heartbeatTimeout;

    // ========== 初始化：从数据库加载数据到缓存 ==========
    @PostConstruct
    public void init() {
        try {
            // 加载所有节点到缓存
            List<FederatedNodeEntity> nodeEntities = nodeMapper.selectList(null);
            for (FederatedNodeEntity entity : nodeEntities) {
                FederatedNode node = entityToNode(entity);
                nodeCache.put(node.getNodeId(), node);
            }
            logger.info("Loaded {} nodes from database", nodeCache.size());

            // 加载所有任务到缓存
            List<TrainingJobEntity> jobEntities = jobMapper.selectList(null);
            for (TrainingJobEntity entity : jobEntities) {
                TrainingJob job = entityToJob(entity);
                jobCache.put(job.getJobId(), job);
            }
            logger.info("Loaded {} jobs from database", jobCache.size());

        } catch (Exception e) {
            logger.warn("Failed to load data from database, using empty cache: {}", e.getMessage());
        }
    }

    // ========== 节点管理 ==========

    /**
     * 注册节点
     * @param node 节点信息
     * @return 注册后的节点
     */
    @Transactional
    public FederatedNode registerNode(FederatedNode node) {
        node.setActive(true);
        node.setLastHeartbeatAt(Instant.now());

        // 如果没有提供nodeId，则生成一个
        if (node.getNodeId() == null || node.getNodeId().isEmpty()) {
            node.setNodeId(UUID.randomUUID().toString());
        }

        // 保存到数据库
        try {
            FederatedNodeEntity entity = nodeToEntity(node);
            entity.setStatus("ACTIVE");
            entity.setRegisteredAt(LocalDateTime.now());
            entity.setUpdatedAt(LocalDateTime.now());

            // 检查是否已存在
            FederatedNodeEntity existing = nodeMapper.selectByNodeId(node.getNodeId());
            if (existing != null) {
                // 更新现有节点
                entity.setId(existing.getId());
                nodeMapper.updateById(entity);
                logger.info("Updated existing node: {}", node.getNodeId());
            } else {
                // 插入新节点
                nodeMapper.insert(entity);
                logger.info("Registered new node: {}", node.getNodeId());
            }
        } catch (Exception e) {
            logger.error("Failed to save node to database: {}", e.getMessage());
        }

        // 更新缓存
        nodeCache.put(node.getNodeId(), node);

        return node;
    }

    /**
     * 节点心跳
     * @param nodeId 节点ID
     * @param metadata 元数据
     */
    @Transactional
    public void heartbeat(String nodeId, Map<String, Object> metadata) {
        FederatedNode node = nodeCache.get(nodeId);
        if (node != null) {
            node.setLastHeartbeatAt(Instant.now());
            node.setActive(true);
            node.setMetadata(metadata);

            // 更新数据库
            try {
                LocalDateTime heartbeatTime = LocalDateTime.ofInstant(
                    node.getLastHeartbeatAt(),
                    ZoneId.systemDefault()
                );
                nodeMapper.updateHeartbeat(nodeId, heartbeatTime);

                // 更新元数据（如果提供）
                if (metadata != null && !metadata.isEmpty()) {
                    String metadataJson = JSON.toJSONString(metadata);
                    nodeMapper.updateMetadata(nodeId, metadataJson);
                }
            } catch (Exception e) {
                logger.error("Failed to update heartbeat in database: {}", e.getMessage());
            }
        } else {
            logger.warn("Heartbeat received from unknown node: {}", nodeId);
        }
    }

    /**
     * 节点状态监控（定时任务）
     * 每5秒检查一次，标记超时节点为不活跃
     */
    @Scheduled(fixedDelay = 5000)
    @Transactional
    public void monitorNodes() {
        Instant now = Instant.now();
        List<String> inactiveNodeIds = new ArrayList<>();

        for (FederatedNode node : nodeCache.values()) {
            if (node.getLastHeartbeatAt() == null) {
                node.setActive(false);
                inactiveNodeIds.add(node.getNodeId());
                continue;
            }

            // 检查心跳超时
            if (now.minusSeconds(heartbeatTimeout).isAfter(node.getLastHeartbeatAt())) {
                node.setActive(false);
                inactiveNodeIds.add(node.getNodeId());
                logger.warn("Node {} marked as inactive due to heartbeat timeout", node.getNodeId());
            }
        }

        // 批量更新数据库中的不活跃节点
        if (!inactiveNodeIds.isEmpty()) {
            try {
                nodeMapper.batchUpdateInactive(inactiveNodeIds);
                logger.info("Marked {} nodes as inactive", inactiveNodeIds.size());
            } catch (Exception e) {
                logger.error("Failed to update inactive nodes in database: {}", e.getMessage());
            }
        }
    }

    // ========== 任务管理 ==========

    /**
     * 创建训练任务
     * @param type 模型类型
     * @param hyperParams 超参数
     * @param participantNodeIds 参与节点ID列表
     * @return 创建的任务
     */
    @Transactional
    public TrainingJob createJob(ModelType type, Map<String, Object> hyperParams, List<String> participantNodeIds) {
        TrainingJob job = new TrainingJob();
        job.setJobId(UUID.randomUUID().toString());
        job.setModelType(type);
        job.setHyperParameters(hyperParams);
        job.setParticipantNodeIds(participantNodeIds);
        job.setStatus("CREATED");
        job.setCreatedAt(Instant.now());
        job.setUpdatedAt(Instant.now());
        job.setCurrentRound(0);

        // 从超参数中提取训练轮数
        if (hyperParams != null && hyperParams.containsKey("numRounds")) {
            // numRounds 已经在超参数中，不需要额外存储
        }

        // 保存到数据库
        try {
            TrainingJobEntity entity = jobToEntity(job);
            entity.setStrategy("FedAvg"); // 默认策略
            entity.setNumRounds(10); // 默认轮数
            entity.setMinClients(participantNodeIds != null ? participantNodeIds.size() : 2);
            entity.setParticipantCount(participantNodeIds != null ? participantNodeIds.size() : 0);
            entity.setCreatedAt(LocalDateTime.now());
            entity.setUpdatedAt(LocalDateTime.now());

            jobMapper.insert(entity);
            logger.info("Created new training job: {} with model type: {}", job.getJobId(), type);
        } catch (Exception e) {
            logger.error("Failed to save job to database: {}", e.getMessage());
        }

        // 更新缓存
        jobCache.put(job.getJobId(), job);

        return job;
    }

    /**
     * 启动训练任务
     * @param jobId 任务ID
     */
    @Transactional
    public void startJob(String jobId) {
        TrainingJob job = jobCache.get(jobId);
        if (job == null) {
            logger.error("Job not found: {}", jobId);
            return;
        }

        // 提取训练参数
        int numRounds = 10; // 默认值
        if (job.getHyperParameters() != null && job.getHyperParameters().containsKey("numRounds")) {
            numRounds = ((Number) job.getHyperParameters().get("numRounds")).intValue();
        }

        int minClients = job.getParticipantNodeIds() != null ? job.getParticipantNodeIds().size() : 2;

        // 动态分配端口（避免冲突）
        int port = defaultFlowerPort + jobCache.size();

        // 启动 Flower Server
        boolean started = flowerServerManager.startServer(
                jobId,
                job.getModelType().name(),
                numRounds,
                minClients,
                port,
                job.getBaselineAccuracy(),
                job.getAllowedDropPercent()
        );

        // 更新任务状态
        if (started) {
            job.setStatus("RUNNING");
            job.setUpdatedAt(Instant.now());

            // 更新数据库
            try {
                jobMapper.startJob(jobId, LocalDateTime.now());
                jobMapper.updateStatus(jobId, "RUNNING");
                logger.info("Started training job: {} on port: {}", jobId, port);
            } catch (Exception e) {
                logger.error("Failed to update job status in database: {}", e.getMessage());
            }
        } else {
            job.setStatus("FAILED");
            job.setUpdatedAt(Instant.now());

            // 更新数据库
            try {
                jobMapper.updateError(jobId, "Failed to start Flower Server");
                logger.error("Failed to start training job: {}", jobId);
            } catch (Exception e) {
                logger.error("Failed to update job error in database: {}", e.getMessage());
            }
        }
    }

    /**
     * 停止训练任务
     * @param jobId 任务ID
     */
    @Transactional
    public void stopJob(String jobId) {
        TrainingJob job = jobCache.get(jobId);
        if (job == null) {
            logger.error("Job not found: {}", jobId);
            return;
        }

        // 停止 Flower Server
        flowerServerManager.stopServer(jobId);

        // 更新任务状态
        job.setStatus("STOPPED");
        job.setUpdatedAt(Instant.now());

        // 更新数据库
        try {
            jobMapper.updateStatus(jobId, "STOPPED");
            logger.info("Stopped training job: {}", jobId);
        } catch (Exception e) {
            logger.error("Failed to update job status in database: {}", e.getMessage());
        }
    }

    /**
     * 更新任务进度（由Flower Server回调）
     * @param jobId 任务ID
     * @param currentRound 当前轮次
     * @param currentAccuracy 当前精度
     */
    @Transactional
    public void updateJobProgress(String jobId, int currentRound, Double currentAccuracy) {
        TrainingJob job = jobCache.get(jobId);
        if (job == null) {
            return;
        }

        job.setCurrentRound(currentRound);
        job.setLastGlobalAccuracy(currentAccuracy);
        job.setUpdatedAt(Instant.now());

        // 更新基线精度（首次评估）
        if (job.getBaselineAccuracy() == null && currentAccuracy != null) {
            job.setBaselineAccuracy(currentAccuracy);
            try {
                jobMapper.updateBaselineAccuracy(jobId, currentAccuracy);
            } catch (Exception e) {
                logger.error("Failed to update baseline accuracy: {}", e.getMessage());
            }
        }

        // 更新最佳精度
        if (currentAccuracy != null) {
            if (job.getBestAccuracy() == null || currentAccuracy > job.getBestAccuracy()) {
                job.setBestAccuracy(currentAccuracy);
                try {
                    jobMapper.updateBestAccuracy(jobId, currentAccuracy);
                } catch (Exception e) {
                    logger.error("Failed to update best accuracy: {}", e.getMessage());
                }
            }
        }

        // 精度下降检查
        if (job.getBaselineAccuracy() != null && currentAccuracy != null && job.getAllowedDropPercent() != null) {
            double drop = (job.getBaselineAccuracy() - currentAccuracy) / job.getBaselineAccuracy() * 100.0;
            if (drop > job.getAllowedDropPercent()) {
                job.setStatus("DEGRADED");
                logger.warn("Job {} accuracy degraded: baseline={}, current={}, drop={}%",
                    jobId, job.getBaselineAccuracy(), currentAccuracy, drop);
            }
        }

        // 更新数据库
        try {
            jobMapper.updateProgress(jobId, currentRound, currentAccuracy);
        } catch (Exception e) {
            logger.error("Failed to update job progress in database: {}", e.getMessage());
        }
    }

    // ========== 查询方法 ==========

    /**
     * 获取所有节点
     * @return 节点列表
     */
    public Collection<FederatedNode> listNodes() {
        return nodeCache.values();
    }

    /**
     * 获取活跃节点
     * @return 活跃节点列表
     */
    public List<FederatedNode> listActiveNodes() {
        return nodeCache.values().stream()
                .filter(FederatedNode::isActive)
                .collect(Collectors.toList());
    }

    /**
     * 获取所有任务
     * @return 任务列表
     */
    public Collection<TrainingJob> listJobs() {
        return jobCache.values();
    }

    /**
     * 根据ID获取任务
     * @param jobId 任务ID
     * @return 任务对象
     */
    public TrainingJob getJob(String jobId) {
        return jobCache.get(jobId);
    }

    /**
     * 根据ID获取节点
     * @param nodeId 节点ID
     * @return 节点对象
     */
    public FederatedNode getNode(String nodeId) {
        return nodeCache.get(nodeId);
    }

    // ========== 实体转换方法 ==========

    /**
     * FederatedNodeEntity -> FederatedNode
     */
    private FederatedNode entityToNode(FederatedNodeEntity entity) {
        FederatedNode node = new FederatedNode();
        node.setNodeId(entity.getNodeId());
        node.setHost(entity.getHost());
        node.setPort(entity.getPort());
        node.setActive(entity.getIsActive() != null && entity.getIsActive());

        if (entity.getLastHeartbeatAt() != null) {
            node.setLastHeartbeatAt(
                entity.getLastHeartbeatAt().atZone(ZoneId.systemDefault()).toInstant()
            );
        }

        // 解析元数据JSON
        if (entity.getMetadata() != null && !entity.getMetadata().isEmpty()) {
            try {
                @SuppressWarnings("unchecked")
                Map<String, Object> metadata = JSON.parseObject(entity.getMetadata(), Map.class);
                node.setMetadata(metadata);
            } catch (Exception e) {
                logger.warn("Failed to parse node metadata: {}", e.getMessage());
            }
        }

        return node;
    }

    /**
     * FederatedNode -> FederatedNodeEntity
     */
    private FederatedNodeEntity nodeToEntity(FederatedNode node) {
        FederatedNodeEntity entity = new FederatedNodeEntity();
        entity.setNodeId(node.getNodeId());
        entity.setHost(node.getHost());
        entity.setPort(node.getPort());
        entity.setIsActive(node.isActive());
        entity.setStatus(node.isActive() ? "ACTIVE" : "INACTIVE");

        if (node.getLastHeartbeatAt() != null) {
            entity.setLastHeartbeatAt(
                LocalDateTime.ofInstant(node.getLastHeartbeatAt(), ZoneId.systemDefault())
            );
        }

        // 序列化元数据为JSON
        if (node.getMetadata() != null && !node.getMetadata().isEmpty()) {
            try {
                entity.setMetadata(JSON.toJSONString(node.getMetadata()));
            } catch (Exception e) {
                logger.warn("Failed to serialize node metadata: {}", e.getMessage());
            }
        }

        return entity;
    }

    /**
     * TrainingJobEntity -> TrainingJob
     */
    private TrainingJob entityToJob(TrainingJobEntity entity) {
        TrainingJob job = new TrainingJob();
        job.setJobId(entity.getJobId());
        job.setModelType(ModelType.valueOf(entity.getModelType()));
        job.setStatus(entity.getStatus());

        if (entity.getCreatedAt() != null) {
            job.setCreatedAt(entity.getCreatedAt().atZone(ZoneId.systemDefault()).toInstant());
        }
        if (entity.getUpdatedAt() != null) {
            job.setUpdatedAt(entity.getUpdatedAt().atZone(ZoneId.systemDefault()).toInstant());
        }

        // 解析超参数JSON
        if (entity.getHyperparameters() != null && !entity.getHyperparameters().isEmpty()) {
            try {
                @SuppressWarnings("unchecked")
                Map<String, Object> hyperParams = JSON.parseObject(entity.getHyperparameters(), Map.class);
                job.setHyperParameters(hyperParams);
            } catch (Exception e) {
                logger.warn("Failed to parse job hyperparameters: {}", e.getMessage());
            }
        }

        // 解析参与节点ID列表
        if (entity.getParticipantNodeIds() != null && !entity.getParticipantNodeIds().isEmpty()) {
            try {
                List<String> nodeIds = JSON.parseArray(entity.getParticipantNodeIds(), String.class);
                job.setParticipantNodeIds(nodeIds);
            } catch (Exception e) {
                logger.warn("Failed to parse participant node IDs: {}", e.getMessage());
            }
        }

        // 精度相关
        if (entity.getBaselineAccuracy() != null) {
            job.setBaselineAccuracy(entity.getBaselineAccuracy().doubleValue());
        }
        if (entity.getCurrentAccuracy() != null) {
            job.setLastGlobalAccuracy(entity.getCurrentAccuracy().doubleValue());
        }
        if (entity.getAllowedDropPercent() != null) {
            job.setAllowedDropPercent(entity.getAllowedDropPercent().doubleValue());
        }

        job.setCurrentRound(entity.getCurrentRound() != null ? entity.getCurrentRound() : 0);

        return job;
    }

    /**
     * TrainingJob -> TrainingJobEntity
     */
    private TrainingJobEntity jobToEntity(TrainingJob job) {
        TrainingJobEntity entity = new TrainingJobEntity();
        entity.setJobId(job.getJobId());
        entity.setModelType(job.getModelType().name());
        entity.setStatus(job.getStatus());

        if (job.getCreatedAt() != null) {
            entity.setCreatedAt(LocalDateTime.ofInstant(job.getCreatedAt(), ZoneId.systemDefault()));
        }
        if (job.getUpdatedAt() != null) {
            entity.setUpdatedAt(LocalDateTime.ofInstant(job.getUpdatedAt(), ZoneId.systemDefault()));
        }

        // 序列化超参数为JSON
        if (job.getHyperParameters() != null && !job.getHyperParameters().isEmpty()) {
            try {
                entity.setHyperparameters(JSON.toJSONString(job.getHyperParameters()));
            } catch (Exception e) {
                logger.warn("Failed to serialize job hyperparameters: {}", e.getMessage());
            }
        }

        // 序列化参与节点ID列表
        if (job.getParticipantNodeIds() != null && !job.getParticipantNodeIds().isEmpty()) {
            try {
                entity.setParticipantNodeIds(JSON.toJSONString(job.getParticipantNodeIds()));
                entity.setParticipantCount(job.getParticipantNodeIds().size());
            } catch (Exception e) {
                logger.warn("Failed to serialize participant node IDs: {}", e.getMessage());
            }
        }

        // 精度相关
        if (job.getBaselineAccuracy() != null) {
            entity.setBaselineAccuracy(BigDecimal.valueOf(job.getBaselineAccuracy()));
        }
        if (job.getLastGlobalAccuracy() != null) {
            entity.setCurrentAccuracy(BigDecimal.valueOf(job.getLastGlobalAccuracy()));
        }
        if (job.getAllowedDropPercent() != null) {
            entity.setAllowedDropPercent(BigDecimal.valueOf(job.getAllowedDropPercent()));
        }

        entity.setCurrentRound(job.getCurrentRound());

        return entity;
    }
}
