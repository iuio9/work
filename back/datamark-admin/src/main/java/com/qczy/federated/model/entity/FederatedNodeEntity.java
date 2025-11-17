package com.qczy.federated.model.entity;

import com.baomidou.mybatisplus.annotation.*;
import lombok.Data;

import java.math.BigDecimal;
import java.time.LocalDateTime;

/**
 * 联邦学习节点实体类
 *
 * 对应数据库表：fl_federated_node
 * 功能：存储联邦学习节点的完整信息
 *
 * @author AI Assistant
 * @date 2025-01-20
 */
@Data
@TableName("fl_federated_node")
public class FederatedNodeEntity {

    /**
     * 主键ID（自增）
     */
    @TableId(value = "id", type = IdType.AUTO)
    private Long id;

    /**
     * 节点唯一标识（UUID）
     */
    @TableField("node_id")
    private String nodeId;

    /**
     * 节点名称
     */
    @TableField("node_name")
    private String nodeName;

    /**
     * 节点主机地址
     */
    @TableField("host")
    private String host;

    /**
     * 节点端口号
     */
    @TableField("port")
    private Integer port;

    /**
     * 节点状态：ACTIVE/INACTIVE/DISCONNECTED/ERROR
     */
    @TableField("status")
    private String status;

    /**
     * 是否活跃：0-否，1-是
     */
    @TableField("is_active")
    private Boolean isActive;

    /**
     * CPU核心数
     */
    @TableField("cpu_cores")
    private Integer cpuCores;

    /**
     * 内存大小（GB）
     */
    @TableField("memory_gb")
    private BigDecimal memoryGb;

    /**
     * GPU数量
     */
    @TableField("gpu_count")
    private Integer gpuCount;

    /**
     * GPU型号
     */
    @TableField("gpu_model")
    private String gpuModel;

    /**
     * 数据集样本数量
     */
    @TableField("dataset_size")
    private Long datasetSize;

    /**
     * 节点元数据（JSON格式）
     */
    @TableField("metadata")
    private String metadata;

    /**
     * 最后心跳时间
     */
    @TableField("last_heartbeat_at")
    private LocalDateTime lastHeartbeatAt;

    /**
     * 注册时间
     */
    @TableField(value = "registered_at", fill = FieldFill.INSERT)
    private LocalDateTime registeredAt;

    /**
     * 更新时间
     */
    @TableField(value = "updated_at", fill = FieldFill.INSERT_UPDATE)
    private LocalDateTime updatedAt;
}
