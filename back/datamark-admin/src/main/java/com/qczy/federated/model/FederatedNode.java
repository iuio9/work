package com.qczy.federated.model;

import lombok.Data;

import java.time.Instant;
import java.util.Map;

@Data
public class FederatedNode {

    private String nodeId;

    private String host;

    private int port;

    private boolean active;

    private Instant lastHeartbeatAt;

    // 可扩展的节点元数据，例如GPU、内存等
    private Map<String, Object> metadata;
}



