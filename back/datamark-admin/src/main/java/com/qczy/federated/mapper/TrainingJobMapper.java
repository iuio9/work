package com.qczy.federated.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.qczy.federated.model.entity.TrainingJobEntity;
import org.apache.ibatis.annotations.*;

import java.time.LocalDateTime;
import java.util.List;

/**
 * 联邦学习训练任务 Mapper 接口
 *
 * 功能：
 * 1. 训练任务的增删改查
 * 2. 任务状态管理
 * 3. 任务进度更新
 * 4. 精度监控数据更新
 *
 * @author AI Assistant
 * @date 2025-01-20
 */
@Mapper
public interface TrainingJobMapper extends BaseMapper<TrainingJobEntity> {

    /**
     * 根据任务ID查询任务
     * @param jobId 任务ID
     * @return 任务实体
     */
    @Select("SELECT * FROM fl_training_job WHERE job_id = #{jobId}")
    TrainingJobEntity selectByJobId(@Param("jobId") String jobId);

    /**
     * 更新任务状态
     * @param jobId 任务ID
     * @param status 新状态
     * @return 更新行数
     */
    @Update("UPDATE fl_training_job SET status = #{status}, updated_at = NOW() " +
            "WHERE job_id = #{jobId}")
    int updateStatus(@Param("jobId") String jobId, @Param("status") String status);

    /**
     * 更新任务进度
     * @param jobId 任务ID
     * @param currentRound 当前轮次
     * @param currentAccuracy 当前精度
     * @return 更新行数
     */
    @Update("UPDATE fl_training_job SET current_round = #{currentRound}, " +
            "current_accuracy = #{currentAccuracy}, updated_at = NOW() " +
            "WHERE job_id = #{jobId}")
    int updateProgress(@Param("jobId") String jobId,
                      @Param("currentRound") int currentRound,
                      @Param("currentAccuracy") Double currentAccuracy);

    /**
     * 更新基线精度
     * @param jobId 任务ID
     * @param baselineAccuracy 基线精度
     * @return 更新行数
     */
    @Update("UPDATE fl_training_job SET baseline_accuracy = #{baselineAccuracy}, " +
            "updated_at = NOW() WHERE job_id = #{jobId}")
    int updateBaselineAccuracy(@Param("jobId") String jobId,
                              @Param("baselineAccuracy") Double baselineAccuracy);

    /**
     * 更新最佳精度
     * @param jobId 任务ID
     * @param bestAccuracy 最佳精度
     * @return 更新行数
     */
    @Update("UPDATE fl_training_job SET best_accuracy = #{bestAccuracy}, " +
            "updated_at = NOW() WHERE job_id = #{jobId}")
    int updateBestAccuracy(@Param("jobId") String jobId,
                          @Param("bestAccuracy") Double bestAccuracy);

    /**
     * 启动任务（更新状态和开始时间）
     * @param jobId 任务ID
     * @param startedAt 开始时间
     * @return 更新行数
     */
    @Update("UPDATE fl_training_job SET status = 'RUNNING', started_at = #{startedAt}, " +
            "updated_at = NOW() WHERE job_id = #{jobId}")
    int startJob(@Param("jobId") String jobId, @Param("startedAt") LocalDateTime startedAt);

    /**
     * 完成任务（更新状态和完成时间）
     * @param jobId 任务ID
     * @param completedAt 完成时间
     * @return 更新行数
     */
    @Update("UPDATE fl_training_job SET status = 'COMPLETED', completed_at = #{completedAt}, " +
            "updated_at = NOW() WHERE job_id = #{jobId}")
    int completeJob(@Param("jobId") String jobId, @Param("completedAt") LocalDateTime completedAt);

    /**
     * 查询正在运行的任务
     * @return 运行中的任务列表
     */
    @Select("SELECT * FROM fl_training_job WHERE status = 'RUNNING' ORDER BY created_at DESC")
    List<TrainingJobEntity> selectRunningJobs();

    /**
     * 按模型类型查询任务
     * @param modelType 模型类型
     * @return 任务列表
     */
    @Select("SELECT * FROM fl_training_job WHERE model_type = #{modelType} " +
            "ORDER BY created_at DESC")
    List<TrainingJobEntity> selectByModelType(@Param("modelType") String modelType);

    /**
     * 按状态查询任务
     * @param status 任务状态
     * @return 任务列表
     */
    @Select("SELECT * FROM fl_training_job WHERE status = #{status} ORDER BY created_at DESC")
    List<TrainingJobEntity> selectByStatus(@Param("status") String status);

    /**
     * 查询最近的任务（分页）
     * @param limit 数量限制
     * @return 任务列表
     */
    @Select("SELECT * FROM fl_training_job ORDER BY created_at DESC LIMIT #{limit}")
    List<TrainingJobEntity> selectRecentJobs(@Param("limit") int limit);

    /**
     * 更新任务错误信息
     * @param jobId 任务ID
     * @param errorMessage 错误信息
     * @return 更新行数
     */
    @Update("UPDATE fl_training_job SET status = 'FAILED', error_message = #{errorMessage}, " +
            "updated_at = NOW() WHERE job_id = #{jobId}")
    int updateError(@Param("jobId") String jobId, @Param("errorMessage") String errorMessage);

    /**
     * 更新全局模型路径
     * @param jobId 任务ID
     * @param modelPath 模型路径
     * @return 更新行数
     */
    @Update("UPDATE fl_training_job SET global_model_path = #{modelPath}, updated_at = NOW() " +
            "WHERE job_id = #{jobId}")
    int updateModelPath(@Param("jobId") String jobId, @Param("modelPath") String modelPath);
}
