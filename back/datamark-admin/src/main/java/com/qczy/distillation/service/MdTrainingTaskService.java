package com.qczy.distillation.service;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.core.conditions.query.QueryWrapper;
import com.qczy.distillation.mapper.*;
import com.qczy.distillation.model.entity.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.math.BigDecimal;
import java.time.Duration;
import java.time.LocalDateTime;
import java.util.List;
import java.util.UUID;

/**
 * 大小模型协同训练服务类
 *
 * 功能：
 * 1. 训练任务管理：创建、启动、停止、查询、删除
 * 2. 训练进度更新：epoch、accuracy、loss等
 * 3. 训练历史记录：记录每个epoch的详细信息
 * 4. 模型评估：保存评估结果
 * 5. LoRA预设管理：查询和使用预设配置
 * 6. 自动标注集成：查询已完成的模型供标注使用
 *
 * @author AI Assistant
 * @date 2025-01-25
 */
@Service
public class MdTrainingTaskService {

    private static final Logger logger = LoggerFactory.getLogger(MdTrainingTaskService.class);

    @Autowired
    private MdTrainingTaskMapper trainingTaskMapper;

    @Autowired
    private MdTrainingHistoryMapper trainingHistoryMapper;

    @Autowired
    private MdLoraPresetMapper loraPresetMapper;

    @Autowired
    private MdModelEvaluationMapper modelEvaluationMapper;

    // ========== 训练任务管理 ==========

    /**
     * 创建训练任务
     * @param task 训练任务实体
     * @return 创建的任务
     */
    @Transactional
    public MdTrainingTaskEntity createTask(MdTrainingTaskEntity task) {
        // 生成唯一任务ID
        if (task.getTaskId() == null || task.getTaskId().isEmpty()) {
            task.setTaskId("TASK_" + UUID.randomUUID().toString().substring(0, 8).toUpperCase());
        }

        // 设置默认值
        if (task.getStatus() == null) {
            task.setStatus("PENDING");
        }
        if (task.getCurrentEpoch() == null) {
            task.setCurrentEpoch(0);
        }
        if (task.getProgress() == null) {
            task.setProgress(0);
        }
        if (task.getDelFlag() == null) {
            task.setDelFlag(0);
        }

        // 设置时间字段（防止 create_time cannot be null 错误）
        LocalDateTime now = LocalDateTime.now();
        if (task.getCreateTime() == null) {
            task.setCreateTime(now);
        }
        if (task.getUpdateTime() == null) {
            task.setUpdateTime(now);
        }

        // 插入数据库
        trainingTaskMapper.insert(task);
        logger.info("Created training task: {}", task.getTaskId());

        return task;
    }

    /**
     * 根据任务ID查询任务
     * @param taskId 任务ID
     * @return 训练任务
     */
    public MdTrainingTaskEntity getTaskByTaskId(String taskId) {
        return trainingTaskMapper.selectByTaskId(taskId);
    }

    /**
     * 查询所有训练任务
     * @return 任务列表
     */
    public List<MdTrainingTaskEntity> getAllTasks() {
        LambdaQueryWrapper<MdTrainingTaskEntity> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(MdTrainingTaskEntity::getDelFlag, 0)
               .orderByDesc(MdTrainingTaskEntity::getCreateTime);
        return trainingTaskMapper.selectList(wrapper);
    }

    /**
     * 查询已完成的训练任务（用于自动标注）
     * @param minAccuracy 最小准确率
     * @param teacherModel 教师模型（可选）
     * @param studentModel 学生模型（可选）
     * @return 已完成的任务列表
     */
    public List<MdTrainingTaskEntity> getCompletedTasks(BigDecimal minAccuracy,
                                                        String teacherModel,
                                                        String studentModel) {
        if (teacherModel != null || studentModel != null) {
            return trainingTaskMapper.selectCompletedTasksByModel(teacherModel, studentModel, minAccuracy);
        } else {
            return trainingTaskMapper.selectCompletedTasks(minAccuracy);
        }
    }

    /**
     * 启动训练任务
     * @param taskId 任务ID
     * @return 是否成功
     */
    @Transactional
    public boolean startTask(String taskId) {
        MdTrainingTaskEntity task = trainingTaskMapper.selectByTaskId(taskId);
        if (task == null) {
            logger.error("Task not found: {}", taskId);
            return false;
        }

        // 更新任务状态和开始时间
        int result = trainingTaskMapper.startTask(taskId, LocalDateTime.now());
        if (result > 0) {
            logger.info("Started training task: {}", taskId);
            return true;
        }
        return false;
    }

    /**
     * 停止训练任务
     * @param taskId 任务ID
     * @return 是否成功
     */
    @Transactional
    public boolean stopTask(String taskId) {
        int result = trainingTaskMapper.stopTask(taskId);
        if (result > 0) {
            logger.info("Stopped training task: {}", taskId);
            return true;
        }
        return false;
    }

    /**
     * 完成训练任务
     * @param taskId 任务ID
     * @return 是否成功
     */
    @Transactional
    public boolean completeTask(String taskId) {
        MdTrainingTaskEntity task = trainingTaskMapper.selectByTaskId(taskId);
        if (task == null) {
            logger.error("Task not found: {}", taskId);
            return false;
        }

        LocalDateTime endTime = LocalDateTime.now();
        Long duration = null;

        // 计算训练时长
        if (task.getStartTime() != null) {
            duration = Duration.between(task.getStartTime(), endTime).getSeconds();
        }

        int result = trainingTaskMapper.completeTask(taskId, endTime, duration);
        if (result > 0) {
            logger.info("Completed training task: {}", taskId);
            return true;
        }
        return false;
    }

    /**
     * 更新训练进度
     * @param taskId 任务ID
     * @param currentEpoch 当前轮次
     * @param accuracy 当前准确率
     * @param loss 当前损失
     * @return 是否成功
     */
    @Transactional
    public boolean updateProgress(String taskId, int currentEpoch,
                                  BigDecimal accuracy, BigDecimal loss) {
        MdTrainingTaskEntity task = trainingTaskMapper.selectByTaskId(taskId);
        if (task == null) {
            return false;
        }

        // 计算进度百分比
        int progress = (int) ((double) currentEpoch / task.getTotalEpochs() * 100);

        // 更新进度
        trainingTaskMapper.updateProgress(taskId, currentEpoch, progress);

        // 更新准确率和损失
        trainingTaskMapper.updateResults(taskId, accuracy, loss);

        // 更新最佳准确率
        if (task.getBestAccuracy() == null ||
            (accuracy != null && accuracy.compareTo(task.getBestAccuracy()) > 0)) {
            trainingTaskMapper.updateBestAccuracy(taskId, accuracy);
        }

        logger.info("Updated progress for task {}: epoch={}, accuracy={}, loss={}",
                   taskId, currentEpoch, accuracy, loss);
        return true;
    }

    /**
     * 更新任务错误信息
     * @param taskId 任务ID
     * @param errorMessage 错误信息
     * @return 是否成功
     */
    @Transactional
    public boolean updateError(String taskId, String errorMessage) {
        int result = trainingTaskMapper.updateError(taskId, errorMessage);
        if (result > 0) {
            logger.error("Task {} failed: {}", taskId, errorMessage);
            return true;
        }
        return false;
    }

    /**
     * 更新模型文件路径
     * @param taskId 任务ID
     * @param modelPath 模型文件路径
     * @param modelUrl 模型访问URL
     * @return 是否成功
     */
    @Transactional
    public boolean updateModelPath(String taskId, String modelPath, String modelUrl) {
        int result = trainingTaskMapper.updateModelPath(taskId, modelPath, modelUrl);
        if (result > 0) {
            logger.info("Updated model path for task {}: path={}, url={}",
                       taskId, modelPath, modelUrl);
            return true;
        }
        return false;
    }

    /**
     * 删除训练任务
     * @param taskId 任务ID
     * @return 是否成功
     */
    @Transactional
    public boolean deleteTask(String taskId) {
        // 逻辑删除任务
        int result = trainingTaskMapper.deleteByTaskId(taskId);

        // 同时删除相关的训练历史和评估结果
        if (result > 0) {
            trainingHistoryMapper.deleteByTaskId(taskId);
            modelEvaluationMapper.deleteByTaskId(taskId);
            logger.info("Deleted training task: {}", taskId);
            return true;
        }
        return false;
    }

    // ========== 训练历史记录管理 ==========

    /**
     * 记录训练历史
     * @param history 训练历史记录
     * @return 是否成功
     */
    @Transactional
    public boolean recordHistory(MdTrainingHistoryEntity history) {
        int result = trainingHistoryMapper.insert(history);
        if (result > 0) {
            logger.debug("Recorded training history for task {}, epoch {}",
                        history.getTaskId(), history.getEpoch());
            return true;
        }
        return false;
    }

    /**
     * 查询任务的训练历史
     * @param taskId 任务ID
     * @return 训练历史列表
     */
    public List<MdTrainingHistoryEntity> getTaskHistory(String taskId) {
        return trainingHistoryMapper.selectByTaskId(taskId);
    }

    /**
     * 查询任务的最新训练记录
     * @param taskId 任务ID
     * @param limit 记录数量
     * @return 最新的训练历史记录
     */
    public List<MdTrainingHistoryEntity> getLatestHistory(String taskId, int limit) {
        return trainingHistoryMapper.selectLatestByTaskId(taskId, limit);
    }

    // ========== LoRA预设管理 ==========

    /**
     * 查询所有LoRA预设
     * @return LoRA预设列表
     */
    public List<MdLoraPresetEntity> getAllLoraPresets() {
        return loraPresetMapper.selectAllPresets();
    }

    /**
     * 根据名称查询LoRA预设
     * @param presetName 预设名称
     * @return LoRA预设
     */
    public MdLoraPresetEntity getLoraPresetByName(String presetName) {
        return loraPresetMapper.selectByPresetName(presetName);
    }

    /**
     * 创建LoRA预设
     * @param preset LoRA预设实体
     * @return 是否成功
     */
    @Transactional
    public boolean createLoraPreset(MdLoraPresetEntity preset) {
        // 检查预设名称是否已存在
        int count = loraPresetMapper.countByPresetName(preset.getPresetName());
        if (count > 0) {
            logger.warn("LoRA preset name already exists: {}", preset.getPresetName());
            return false;
        }

        if (preset.getDelFlag() == null) {
            preset.setDelFlag(0);
        }

        int result = loraPresetMapper.insert(preset);
        if (result > 0) {
            logger.info("Created LoRA preset: {}", preset.getPresetName());
            return true;
        }
        return false;
    }

    // ========== 模型评估管理 ==========

    /**
     * 保存模型评估结果
     * @param evaluation 评估结果实体
     * @return 是否成功
     */
    @Transactional
    public boolean saveEvaluation(MdModelEvaluationEntity evaluation) {
        int result = modelEvaluationMapper.insert(evaluation);
        if (result > 0) {
            logger.info("Saved evaluation for task {}", evaluation.getTaskId());
            return true;
        }
        return false;
    }

    /**
     * 查询任务的所有评估结果
     * @param taskId 任务ID
     * @return 评估结果列表
     */
    public List<MdModelEvaluationEntity> getTaskEvaluations(String taskId) {
        return modelEvaluationMapper.selectByTaskId(taskId);
    }

    /**
     * 查询任务的最新评估结果
     * @param taskId 任务ID
     * @return 最新评估结果
     */
    public MdModelEvaluationEntity getLatestEvaluation(String taskId) {
        return modelEvaluationMapper.selectLatestByTaskId(taskId);
    }

    /**
     * 查询任务的最佳评估结果
     * @param taskId 任务ID
     * @return 最佳评估结果
     */
    public MdModelEvaluationEntity getBestEvaluation(String taskId) {
        return modelEvaluationMapper.selectBestByTaskId(taskId);
    }

    // ========== 统计信息 ==========

    /**
     * 查询正在运行的任务
     * @return 运行中的任务列表
     */
    public List<MdTrainingTaskEntity> getRunningTasks() {
        return trainingTaskMapper.selectRunningTasks();
    }

    /**
     * 按状态查询任务
     * @param status 任务状态
     * @return 任务列表
     */
    public List<MdTrainingTaskEntity> getTasksByStatus(String status) {
        return trainingTaskMapper.selectByStatus(status);
    }

    /**
     * 查询最近的任务
     * @param limit 数量限制
     * @return 最近的任务列表
     */
    public List<MdTrainingTaskEntity> getRecentTasks(int limit) {
        return trainingTaskMapper.selectRecentTasks(limit);
    }
}
