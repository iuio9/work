package com.qczy.federated.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.qczy.federated.model.entity.FederatedNodeEntity;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;
import org.apache.ibatis.annotations.Select;
import org.apache.ibatis.annotations.Update;

import java.time.LocalDateTime;
import java.util.List;

/**
 * 联邦学习节点 Mapper 接口
 *
 * 功能：
 * 1. 节点的增删改查
 * 2. 节点状态管理
 * 3. 节点心跳更新
 * 4. 活跃节点查询
 *
 * @author AI Assistant
 * @date 2025-01-20
 */
@Mapper
public interface FederatedNodeMapper extends BaseMapper<FederatedNodeEntity> {

    /**
     * 根据节点ID查询节点
     * @param nodeId 节点ID
     * @return 节点实体
     */
    @Select("SELECT * FROM fl_federated_node WHERE node_id = #{nodeId}")
    FederatedNodeEntity selectByNodeId(@Param("nodeId") String nodeId);

    /**
     * 更新节点心跳时间
     * @param nodeId 节点ID
     * @param lastHeartbeatAt 心跳时间
     * @return 更新行数
     */
    @Update("UPDATE fl_federated_node SET last_heartbeat_at = #{lastHeartbeatAt}, " +
            "is_active = 1, status = 'ACTIVE', updated_at = NOW() WHERE node_id = #{nodeId}")
    int updateHeartbeat(@Param("nodeId") String nodeId,
                       @Param("lastHeartbeatAt") LocalDateTime lastHeartbeatAt);

    /**
     * 查询所有活跃节点
     * @return 活跃节点列表
     */
    @Select("SELECT * FROM fl_federated_node WHERE is_active = 1 AND status = 'ACTIVE' " +
            "ORDER BY last_heartbeat_at DESC")
    List<FederatedNodeEntity> selectActiveNodes();

    /**
     * 查询心跳超时的节点（超过指定秒数未心跳）
     * @param timeoutSeconds 超时秒数
     * @return 超时节点列表
     */
    @Select("SELECT * FROM fl_federated_node WHERE is_active = 1 " +
            "AND last_heartbeat_at < DATE_SUB(NOW(), INTERVAL #{timeoutSeconds} SECOND)")
    List<FederatedNodeEntity> selectTimeoutNodes(@Param("timeoutSeconds") int timeoutSeconds);

    /**
     * 批量更新节点状态为不活跃
     * @param nodeIds 节点ID列表
     * @return 更新行数
     */
    @Update("<script>" +
            "UPDATE fl_federated_node SET is_active = 0, status = 'INACTIVE', updated_at = NOW() " +
            "WHERE node_id IN " +
            "<foreach collection='nodeIds' item='id' open='(' separator=',' close=')'>" +
            "#{id}" +
            "</foreach>" +
            "</script>")
    int batchUpdateInactive(@Param("nodeIds") List<String> nodeIds);

    /**
     * 更新节点元数据
     * @param nodeId 节点ID
     * @param metadata 元数据（JSON字符串）
     * @return 更新行数
     */
    @Update("UPDATE fl_federated_node SET metadata = #{metadata}, updated_at = NOW() " +
            "WHERE node_id = #{nodeId}")
    int updateMetadata(@Param("nodeId") String nodeId, @Param("metadata") String metadata);
}
