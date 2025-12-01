package com.qczy.controller.distillation;

import com.qczy.common.result.Result;
import com.qczy.distillation.model.entity.MdTrainingTaskEntity;
import com.qczy.distillation.model.entity.MdTrainingHistoryEntity;
import com.qczy.distillation.model.entity.MdLoraPresetEntity;
import com.qczy.distillation.model.entity.MdModelEvaluationEntity;
import com.qczy.distillation.service.MdTrainingTaskService;
import io.swagger.annotations.Api;
import io.swagger.annotations.ApiOperation;
import io.swagger.annotations.ApiParam;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;

import java.math.BigDecimal;
import java.util.List;

/**
 * 大小模型协同训练（Model Distillation）控制器
 *
 * @Author: AI Assistant
 * @Date: 2025-01-25
 * @Description: 处理大小模型协同训练相关的API请求
 */
@RestController
@RequestMapping("/model-distillation")
@Api(tags = "大小模型协同训练")
@CrossOrigin
public class ModelDistillationController {

    @Autowired
    private MdTrainingTaskService trainingTaskService;

    // ========== 训练任务管理 ==========

    @GetMapping("/tasks")
    @ApiOperation("获取所有训练任务")
    public Result<?> getAllTasks() {
        try {
            List<MdTrainingTaskEntity> tasks = trainingTaskService.getAllTasks();
            return Result.ok(tasks);
        } catch (Exception e) {
            return Result.fail(null).message("获取任务列表失败: " + e.getMessage());
        }
    }

    @GetMapping("/completed-models")
    @ApiOperation("获取已完成的训练任务（用于自动标注）")
    public Result<?> getCompletedModels(
        @ApiParam("最小准确率") @RequestParam(required = false) Double minAccuracy,
        @ApiParam("教师模型") @RequestParam(required = false) String teacherModel,
        @ApiParam("学生模型") @RequestParam(required = false) String studentModel
    ) {
        try {
            BigDecimal minAccuracyDecimal = minAccuracy != null ?
                    BigDecimal.valueOf(minAccuracy) : null;

            List<MdTrainingTaskEntity> completedTasks = trainingTaskService.getCompletedTasks(
                    minAccuracyDecimal, teacherModel, studentModel);

            return Result.ok(completedTasks);
        } catch (Exception e) {
            return Result.fail(null).message("获取已完成任务失败: " + e.getMessage());
        }
    }

    @GetMapping("/tasks/{taskId}")
    @ApiOperation("获取任务详情")
    public Result<?> getTaskDetail(@PathVariable String taskId) {
        try {
            MdTrainingTaskEntity task = trainingTaskService.getTaskByTaskId(taskId);
            if (task != null) {
                return Result.ok(task);
            } else {
                return Result.fail(null).message("任务不存在");
            }
        } catch (Exception e) {
            return Result.fail(null).message("获取任务详情失败: " + e.getMessage());
        }
    }

    @PostMapping("/tasks")
    @ApiOperation("创建训练任务")
    public Result<?> createTask(@RequestBody MdTrainingTaskEntity taskData) {
        try {
            // 设置默认值
            if (taskData.getTotalEpochs() == null) {
                taskData.setTotalEpochs(50);
            }
            if (taskData.getBatchSize() == null) {
                taskData.setBatchSize(32);
            }
            if (taskData.getLearningRate() == null) {
                taskData.setLearningRate(BigDecimal.valueOf(0.001));
            }
            if (taskData.getTemperature() == null) {
                taskData.setTemperature(BigDecimal.valueOf(3.0));
            }
            if (taskData.getAlpha() == null) {
                taskData.setAlpha(BigDecimal.valueOf(0.7));
            }
            if (taskData.getLoraRank() == null) {
                taskData.setLoraRank(16);
            }

            MdTrainingTaskEntity newTask = trainingTaskService.createTask(taskData);
            return Result.ok(newTask).message("任务创建成功");
        } catch (Exception e) {
            return Result.fail(null).message("创建任务失败: " + e.getMessage());
        }
    }

    @PostMapping("/tasks/{taskId}/start")
    @ApiOperation("启动训练任务")
    public Result<?> startTask(@PathVariable String taskId) {
        try {
            boolean success = trainingTaskService.startTask(taskId);
            if (success) {
                MdTrainingTaskEntity task = trainingTaskService.getTaskByTaskId(taskId);
                return Result.ok(task).message("任务已启动");
            } else {
                return Result.fail(null).message("启动任务失败");
            }
        } catch (Exception e) {
            return Result.fail(null).message("启动任务失败: " + e.getMessage());
        }
    }

    @PostMapping("/tasks/{taskId}/stop")
    @ApiOperation("停止训练任务")
    public Result<?> stopTask(@PathVariable String taskId) {
        try {
            boolean success = trainingTaskService.stopTask(taskId);
            if (success) {
                MdTrainingTaskEntity task = trainingTaskService.getTaskByTaskId(taskId);
                return Result.ok(task).message("任务已停止");
            } else {
                return Result.fail(null).message("停止任务失败");
            }
        } catch (Exception e) {
            return Result.fail(null).message("停止任务失败: " + e.getMessage());
        }
    }

    @PostMapping("/tasks/{taskId}/complete")
    @ApiOperation("完成训练任务")
    public Result<?> completeTask(@PathVariable String taskId) {
        try {
            boolean success = trainingTaskService.completeTask(taskId);
            if (success) {
                MdTrainingTaskEntity task = trainingTaskService.getTaskByTaskId(taskId);
                return Result.ok(task).message("任务已完成");
            } else {
                return Result.fail(null).message("完成任务失败");
            }
        } catch (Exception e) {
            return Result.fail(null).message("完成任务失败: " + e.getMessage());
        }
    }

    @DeleteMapping("/tasks/{taskId}")
    @ApiOperation("删除训练任务")
    public Result<?> deleteTask(@PathVariable String taskId) {
        try {
            boolean success = trainingTaskService.deleteTask(taskId);
            if (success) {
                return Result.ok(null).message("任务已删除");
            } else {
                return Result.fail(null).message("删除任务失败");
            }
        } catch (Exception e) {
            return Result.fail(null).message("删除任务失败: " + e.getMessage());
        }
    }

    @PutMapping("/tasks/{taskId}/progress")
    @ApiOperation("更新训练进度")
    public Result<?> updateProgress(
            @PathVariable String taskId,
            @RequestParam int currentEpoch,
            @RequestParam(required = false) Double accuracy,
            @RequestParam(required = false) Double loss
    ) {
        try {
            BigDecimal accuracyDecimal = accuracy != null ? BigDecimal.valueOf(accuracy) : null;
            BigDecimal lossDecimal = loss != null ? BigDecimal.valueOf(loss) : null;

            boolean success = trainingTaskService.updateProgress(
                    taskId, currentEpoch, accuracyDecimal, lossDecimal);

            if (success) {
                MdTrainingTaskEntity task = trainingTaskService.getTaskByTaskId(taskId);
                return Result.ok(task).message("进度更新成功");
            } else {
                return Result.fail(null).message("更新进度失败");
            }
        } catch (Exception e) {
            return Result.fail(null).message("更新进度失败: " + e.getMessage());
        }
    }

    @PutMapping("/tasks/{taskId}/model-path")
    @ApiOperation("更新模型文件路径")
    public Result<?> updateModelPath(
            @PathVariable String taskId,
            @RequestParam String modelPath,
            @RequestParam String modelUrl
    ) {
        try {
            boolean success = trainingTaskService.updateModelPath(taskId, modelPath, modelUrl);
            if (success) {
                return Result.ok(null).message("模型路径更新成功");
            } else {
                return Result.fail(null).message("更新模型路径失败");
            }
        } catch (Exception e) {
            return Result.fail(null).message("更新模型路径失败: " + e.getMessage());
        }
    }

    @PutMapping("/tasks/{taskId}/error")
    @ApiOperation("更新任务错误信息")
    public Result<?> updateError(
            @PathVariable String taskId,
            @RequestParam String errorMessage
    ) {
        try {
            boolean success = trainingTaskService.updateError(taskId, errorMessage);
            if (success) {
                return Result.ok(null).message("错误信息已记录");
            } else {
                return Result.fail(null).message("记录错误信息失败");
            }
        } catch (Exception e) {
            return Result.fail(null).message("记录错误信息失败: " + e.getMessage());
        }
    }

    // ========== 训练历史记录 ==========

    @GetMapping("/tasks/{taskId}/history")
    @ApiOperation("获取任务训练历史")
    public Result<?> getTaskHistory(@PathVariable String taskId) {
        try {
            List<MdTrainingHistoryEntity> history = trainingTaskService.getTaskHistory(taskId);
            return Result.ok(history);
        } catch (Exception e) {
            return Result.fail(null).message("获取训练历史失败: " + e.getMessage());
        }
    }

    @GetMapping("/tasks/{taskId}/history/latest")
    @ApiOperation("获取任务最新训练记录")
    public Result<?> getLatestHistory(
            @PathVariable String taskId,
            @RequestParam(defaultValue = "10") int limit
    ) {
        try {
            List<MdTrainingHistoryEntity> history =
                    trainingTaskService.getLatestHistory(taskId, limit);
            return Result.ok(history);
        } catch (Exception e) {
            return Result.fail(null).message("获取最新训练记录失败: " + e.getMessage());
        }
    }

    @PostMapping("/tasks/{taskId}/history")
    @ApiOperation("记录训练历史")
    public Result<?> recordHistory(@RequestBody MdTrainingHistoryEntity history) {
        try {
            boolean success = trainingTaskService.recordHistory(history);
            if (success) {
                return Result.ok(null).message("训练历史已记录");
            } else {
                return Result.fail(null).message("记录训练历史失败");
            }
        } catch (Exception e) {
            return Result.fail(null).message("记录训练历史失败: " + e.getMessage());
        }
    }

    // ========== LoRA预设管理 ==========

    @GetMapping("/lora-presets")
    @ApiOperation("获取所有LoRA预设")
    public Result<?> getAllLoraPresets() {
        try {
            List<MdLoraPresetEntity> presets = trainingTaskService.getAllLoraPresets();
            return Result.ok(presets);
        } catch (Exception e) {
            return Result.fail(null).message("获取LoRA预设失败: " + e.getMessage());
        }
    }

    @GetMapping("/lora-presets/{presetName}")
    @ApiOperation("根据名称获取LoRA预设")
    public Result<?> getLoraPresetByName(@PathVariable String presetName) {
        try {
            MdLoraPresetEntity preset = trainingTaskService.getLoraPresetByName(presetName);
            if (preset != null) {
                return Result.ok(preset);
            } else {
                return Result.fail(null).message("预设不存在");
            }
        } catch (Exception e) {
            return Result.fail(null).message("获取LoRA预设失败: " + e.getMessage());
        }
    }

    @PostMapping("/lora-presets")
    @ApiOperation("创建LoRA预设")
    public Result<?> createLoraPreset(@RequestBody MdLoraPresetEntity preset) {
        try {
            boolean success = trainingTaskService.createLoraPreset(preset);
            if (success) {
                return Result.ok(preset).message("LoRA预设创建成功");
            } else {
                return Result.fail(null).message("LoRA预设名称已存在");
            }
        } catch (Exception e) {
            return Result.fail(null).message("创建LoRA预设失败: " + e.getMessage());
        }
    }

    // ========== 模型评估 ==========

    @GetMapping("/tasks/{taskId}/evaluations")
    @ApiOperation("获取任务评估结果")
    public Result<?> getTaskEvaluations(@PathVariable String taskId) {
        try {
            List<MdModelEvaluationEntity> evaluations =
                    trainingTaskService.getTaskEvaluations(taskId);
            return Result.ok(evaluations);
        } catch (Exception e) {
            return Result.fail(null).message("获取评估结果失败: " + e.getMessage());
        }
    }

    @GetMapping("/tasks/{taskId}/evaluations/latest")
    @ApiOperation("获取任务最新评估结果")
    public Result<?> getLatestEvaluation(@PathVariable String taskId) {
        try {
            MdModelEvaluationEntity evaluation =
                    trainingTaskService.getLatestEvaluation(taskId);
            return Result.ok(evaluation);
        } catch (Exception e) {
            return Result.fail(null).message("获取最新评估结果失败: " + e.getMessage());
        }
    }

    @GetMapping("/tasks/{taskId}/evaluations/best")
    @ApiOperation("获取任务最佳评估结果")
    public Result<?> getBestEvaluation(@PathVariable String taskId) {
        try {
            MdModelEvaluationEntity evaluation =
                    trainingTaskService.getBestEvaluation(taskId);
            return Result.ok(evaluation);
        } catch (Exception e) {
            return Result.fail(null).message("获取最佳评估结果失败: " + e.getMessage());
        }
    }

    @PostMapping("/tasks/{taskId}/evaluations")
    @ApiOperation("保存模型评估结果")
    public Result<?> saveEvaluation(@RequestBody MdModelEvaluationEntity evaluation) {
        try {
            boolean success = trainingTaskService.saveEvaluation(evaluation);
            if (success) {
                return Result.ok(null).message("评估结果已保存");
            } else {
                return Result.fail(null).message("保存评估结果失败");
            }
        } catch (Exception e) {
            return Result.fail(null).message("保存评估结果失败: " + e.getMessage());
        }
    }

    // ========== 统计信息 ==========

    @GetMapping("/tasks/running")
    @ApiOperation("获取正在运行的任务")
    public Result<?> getRunningTasks() {
        try {
            List<MdTrainingTaskEntity> tasks = trainingTaskService.getRunningTasks();
            return Result.ok(tasks);
        } catch (Exception e) {
            return Result.fail(null).message("获取运行中任务失败: " + e.getMessage());
        }
    }

    @GetMapping("/tasks/status/{status}")
    @ApiOperation("按状态查询任务")
    public Result<?> getTasksByStatus(@PathVariable String status) {
        try {
            List<MdTrainingTaskEntity> tasks = trainingTaskService.getTasksByStatus(status);
            return Result.ok(tasks);
        } catch (Exception e) {
            return Result.fail(null).message("查询任务失败: " + e.getMessage());
        }
    }

    @GetMapping("/tasks/recent")
    @ApiOperation("获取最近的任务")
    public Result<?> getRecentTasks(@RequestParam(defaultValue = "10") int limit) {
        try {
            List<MdTrainingTaskEntity> tasks = trainingTaskService.getRecentTasks(limit);
            return Result.ok(tasks);
        } catch (Exception e) {
            return Result.fail(null).message("获取最近任务失败: " + e.getMessage());
        }
    }
}
