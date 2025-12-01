package com.qczy.distillation.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.qczy.distillation.model.entity.MdTrainingTaskEntity;
import org.apache.ibatis.annotations.*;

import java.math.BigDecimal;
import java.time.LocalDateTime;
import java.util.List;

/**
 * 大小模型协同训练任务 Mapper 接口
 *
 * 功能：
 * 1. 训练任务的增删改查
 * 2. 任务状态管理
 * 3. 任务进度更新
 * 4. 模型查询和筛选
 *
 * @author AI Assistant
 * @date 2025-01-25
 */
@Mapper
public interface MdTrainingTaskMapper extends BaseMapper<MdTrainingTaskEntity> {

    /**
     * 根据任务ID查询任务
     * @param taskId 任务ID
     * @return 任务实体
     */
    @Select("SELECT * FROM md_training_task WHERE task_id = #{taskId} AND del_flag = 0")
    MdTrainingTaskEntity selectByTaskId(@Param("taskId") String taskId);

    /**
     * 查询已完成的训练任务（用于自动标注）
     * @param minAccuracy 最小准确率
     * @return 已完成的任务列表
     */
    @Select("<script>" +
            "SELECT * FROM md_training_task " +
            "WHERE status = 'COMPLETED' AND del_flag = 0 " +
            "<if test='minAccuracy != null'> AND accuracy &gt;= #{minAccuracy} </if>" +
            "ORDER BY accuracy DESC, create_time DESC" +
            "</script>")
    List<MdTrainingTaskEntity> selectCompletedTasks(@Param("minAccuracy") BigDecimal minAccuracy);

    /**
     * 按教师模型和学生模型查询已完成任务
     * @param teacherModel 教师模型
     * @param studentModel 学生模型
     * @param minAccuracy 最小准确率
     * @return 任务列表
     */
    @Select("<script>" +
            "SELECT * FROM md_training_task " +
            "WHERE status = 'COMPLETED' AND del_flag = 0 " +
            "<if test='teacherModel != null and teacherModel != \"\"'> AND teacher_model = #{teacherModel} </if>" +
            "<if test='studentModel != null and studentModel != \"\"'> AND student_model = #{studentModel} </if>" +
            "<if test='minAccuracy != null'> AND accuracy &gt;= #{minAccuracy} </if>" +
            "ORDER BY accuracy DESC, create_time DESC" +
            "</script>")
    List<MdTrainingTaskEntity> selectCompletedTasksByModel(
            @Param("teacherModel") String teacherModel,
            @Param("studentModel") String studentModel,
            @Param("minAccuracy") BigDecimal minAccuracy);

    /**
     * 更新任务状态
     * @param taskId 任务ID
     * @param status 新状态
     * @return 更新行数
     */
    @Update("UPDATE md_training_task SET status = #{status}, update_time = NOW() " +
            "WHERE task_id = #{taskId} AND del_flag = 0")
    int updateStatus(@Param("taskId") String taskId, @Param("status") String status);

    /**
     * 更新任务进度
     * @param taskId 任务ID
     * @param currentEpoch 当前轮次
     * @param progress 进度百分比
     * @return 更新行数
     */
    @Update("UPDATE md_training_task SET current_epoch = #{currentEpoch}, " +
            "progress = #{progress}, update_time = NOW() " +
            "WHERE task_id = #{taskId} AND del_flag = 0")
    int updateProgress(@Param("taskId") String taskId,
                      @Param("currentEpoch") int currentEpoch,
                      @Param("progress") int progress);

    /**
     * 更新训练结果
     * @param taskId 任务ID
     * @param accuracy 准确率
     * @param loss 损失值
     * @return 更新行数
     */
    @Update("UPDATE md_training_task SET accuracy = #{accuracy}, loss = #{loss}, " +
            "update_time = NOW() WHERE task_id = #{taskId} AND del_flag = 0")
    int updateResults(@Param("taskId") String taskId,
                     @Param("accuracy") BigDecimal accuracy,
                     @Param("loss") BigDecimal loss);

    /**
     * 更新最佳准确率
     * @param taskId 任务ID
     * @param bestAccuracy 最佳准确率
     * @return 更新行数
     */
    @Update("UPDATE md_training_task SET best_accuracy = #{bestAccuracy}, " +
            "update_time = NOW() WHERE task_id = #{taskId} AND del_flag = 0")
    int updateBestAccuracy(@Param("taskId") String taskId,
                          @Param("bestAccuracy") BigDecimal bestAccuracy);

    /**
     * 启动任务（更新状态和开始时间）
     * @param taskId 任务ID
     * @param startTime 开始时间
     * @return 更新行数
     */
    @Update("UPDATE md_training_task SET status = 'RUNNING', start_time = #{startTime}, " +
            "update_time = NOW() WHERE task_id = #{taskId} AND del_flag = 0")
    int startTask(@Param("taskId") String taskId, @Param("startTime") LocalDateTime startTime);

    /**
     * 完成任务（更新状态和完成时间）
     * @param taskId 任务ID
     * @param endTime 结束时间
     * @param duration 训练时长（秒）
     * @return 更新行数
     */
    @Update("UPDATE md_training_task SET status = 'COMPLETED', end_time = #{endTime}, " +
            "duration = #{duration}, update_time = NOW() " +
            "WHERE task_id = #{taskId} AND del_flag = 0")
    int completeTask(@Param("taskId") String taskId,
                    @Param("endTime") LocalDateTime endTime,
                    @Param("duration") Long duration);

    /**
     * 停止任务
     * @param taskId 任务ID
     * @return 更新行数
     */
    @Update("UPDATE md_training_task SET status = 'STOPPED', update_time = NOW() " +
            "WHERE task_id = #{taskId} AND del_flag = 0")
    int stopTask(@Param("taskId") String taskId);

    /**
     * 查询正在运行的任务
     * @return 运行中的任务列表
     */
    @Select("SELECT * FROM md_training_task WHERE status = 'RUNNING' AND del_flag = 0 " +
            "ORDER BY create_time DESC")
    List<MdTrainingTaskEntity> selectRunningTasks();

    /**
     * 按状态查询任务
     * @param status 任务状态
     * @return 任务列表
     */
    @Select("SELECT * FROM md_training_task WHERE status = #{status} AND del_flag = 0 " +
            "ORDER BY create_time DESC")
    List<MdTrainingTaskEntity> selectByStatus(@Param("status") String status);

    /**
     * 查询最近的任务（分页）
     * @param limit 数量限制
     * @return 任务列表
     */
    @Select("SELECT * FROM md_training_task WHERE del_flag = 0 " +
            "ORDER BY create_time DESC LIMIT #{limit}")
    List<MdTrainingTaskEntity> selectRecentTasks(@Param("limit") int limit);

    /**
     * 更新任务错误信息
     * @param taskId 任务ID
     * @param errorMessage 错误信息
     * @return 更新行数
     */
    @Update("UPDATE md_training_task SET status = 'FAILED', error_message = #{errorMessage}, " +
            "update_time = NOW() WHERE task_id = #{taskId} AND del_flag = 0")
    int updateError(@Param("taskId") String taskId, @Param("errorMessage") String errorMessage);

    /**
     * 更新模型文件路径
     * @param taskId 任务ID
     * @param modelPath 模型文件路径
     * @param modelUrl 模型访问URL
     * @return 更新行数
     */
    @Update("UPDATE md_training_task SET model_path = #{modelPath}, model_url = #{modelUrl}, " +
            "update_time = NOW() WHERE task_id = #{taskId} AND del_flag = 0")
    int updateModelPath(@Param("taskId") String taskId,
                       @Param("modelPath") String modelPath,
                       @Param("modelUrl") String modelUrl);

    /**
     * 逻辑删除任务
     * @param taskId 任务ID
     * @return 更新行数
     */
    @Update("UPDATE md_training_task SET del_flag = 1, update_time = NOW() " +
            "WHERE task_id = #{taskId}")
    int deleteByTaskId(@Param("taskId") String taskId);
}
