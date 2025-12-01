package com.qczy.distillation.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.qczy.distillation.model.entity.MdModelEvaluationEntity;
import org.apache.ibatis.annotations.*;

import java.util.List;

/**
 * 模型评估结果 Mapper 接口
 *
 * 功能：
 * 1. 模型评估结果的增删改查
 * 2. 按任务ID查询评估结果
 * 3. 查询最新评估结果
 *
 * @author AI Assistant
 * @date 2025-01-25
 */
@Mapper
public interface MdModelEvaluationMapper extends BaseMapper<MdModelEvaluationEntity> {

    /**
     * 根据任务ID查询所有评估结果
     * @param taskId 任务ID
     * @return 评估结果列表
     */
    @Select("SELECT * FROM md_model_evaluation WHERE task_id = #{taskId} " +
            "ORDER BY eval_time DESC")
    List<MdModelEvaluationEntity> selectByTaskId(@Param("taskId") String taskId);

    /**
     * 查询任务的最新评估结果
     * @param taskId 任务ID
     * @return 最新的评估结果
     */
    @Select("SELECT * FROM md_model_evaluation WHERE task_id = #{taskId} " +
            "ORDER BY eval_time DESC LIMIT 1")
    MdModelEvaluationEntity selectLatestByTaskId(@Param("taskId") String taskId);

    /**
     * 查询任务的最佳评估结果（按F1分数）
     * @param taskId 任务ID
     * @return 最佳评估结果
     */
    @Select("SELECT * FROM md_model_evaluation WHERE task_id = #{taskId} " +
            "ORDER BY f1_score DESC LIMIT 1")
    MdModelEvaluationEntity selectBestByTaskId(@Param("taskId") String taskId);

    /**
     * 删除任务的所有评估结果
     * @param taskId 任务ID
     * @return 删除行数
     */
    @Delete("DELETE FROM md_model_evaluation WHERE task_id = #{taskId}")
    int deleteByTaskId(@Param("taskId") String taskId);

    /**
     * 获取任务的评估结果数量
     * @param taskId 任务ID
     * @return 评估结果总数
     */
    @Select("SELECT COUNT(*) FROM md_model_evaluation WHERE task_id = #{taskId}")
    int countByTaskId(@Param("taskId") String taskId);
}
