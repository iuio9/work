package com.qczy.distillation.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.qczy.distillation.model.entity.MdTrainingHistoryEntity;
import org.apache.ibatis.annotations.*;

import java.util.List;

/**
 * 训练历史记录 Mapper 接口
 *
 * 功能：
 * 1. 训练历史记录的增删改查
 * 2. 按任务ID查询历史记录
 * 3. 按epoch查询记录
 *
 * @author AI Assistant
 * @date 2025-01-25
 */
@Mapper
public interface MdTrainingHistoryMapper extends BaseMapper<MdTrainingHistoryEntity> {

    /**
     * 根据任务ID查询所有训练历史记录
     * @param taskId 任务ID
     * @return 训练历史记录列表
     */
    @Select("SELECT * FROM md_training_history WHERE task_id = #{taskId} " +
            "ORDER BY epoch ASC")
    List<MdTrainingHistoryEntity> selectByTaskId(@Param("taskId") String taskId);

    /**
     * 根据任务ID和epoch查询训练历史
     * @param taskId 任务ID
     * @param epoch 训练轮次
     * @return 训练历史记录
     */
    @Select("SELECT * FROM md_training_history WHERE task_id = #{taskId} AND epoch = #{epoch}")
    MdTrainingHistoryEntity selectByTaskIdAndEpoch(@Param("taskId") String taskId,
                                                   @Param("epoch") int epoch);

    /**
     * 查询任务的最新训练记录
     * @param taskId 任务ID
     * @param limit 记录数量
     * @return 最新的训练历史记录列表
     */
    @Select("SELECT * FROM md_training_history WHERE task_id = #{taskId} " +
            "ORDER BY epoch DESC LIMIT #{limit}")
    List<MdTrainingHistoryEntity> selectLatestByTaskId(@Param("taskId") String taskId,
                                                       @Param("limit") int limit);

    /**
     * 查询任务的最佳训练记录（按验证准确率）
     * @param taskId 任务ID
     * @return 最佳训练记录
     */
    @Select("SELECT * FROM md_training_history WHERE task_id = #{taskId} " +
            "ORDER BY val_accuracy DESC LIMIT 1")
    MdTrainingHistoryEntity selectBestByTaskId(@Param("taskId") String taskId);

    /**
     * 删除任务的所有训练历史记录
     * @param taskId 任务ID
     * @return 删除行数
     */
    @Delete("DELETE FROM md_training_history WHERE task_id = #{taskId}")
    int deleteByTaskId(@Param("taskId") String taskId);

    /**
     * 获取任务的训练历史统计信息
     * @param taskId 任务ID
     * @return 历史记录总数
     */
    @Select("SELECT COUNT(*) FROM md_training_history WHERE task_id = #{taskId}")
    int countByTaskId(@Param("taskId") String taskId);
}
