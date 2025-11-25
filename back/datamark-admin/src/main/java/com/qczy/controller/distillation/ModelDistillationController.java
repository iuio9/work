package com.qczy.controller.distillation;

import com.qczy.common.result.Result;
import io.swagger.annotations.Api;
import io.swagger.annotations.ApiOperation;
import io.swagger.annotations.ApiParam;
import org.springframework.web.bind.annotation.*;

import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.*;
import java.util.stream.Collectors;

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

    // 模拟数据存储（实际项目应使用数据库）
    private static final List<Map<String, Object>> MOCK_TASKS = new ArrayList<>();

    static {
        // 初始化一些模拟数据
        DateTimeFormatter formatter = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss");

        MOCK_TASKS.add(createMockTask(
            "TASK_001",
            "目标检测协同训练-YOLOv5",
            "llama2-7b",
            "yolov5s",
            "COMPLETED",
            92.5,
            50,
            32,
            0.001,
            3.0,
            0.7,
            16,
            LocalDateTime.now().minusDays(2).format(formatter)
        ));

        MOCK_TASKS.add(createMockTask(
            "TASK_002",
            "图像分类协同训练-ResNet",
            "qwen-7b",
            "resnet50",
            "COMPLETED",
            88.3,
            40,
            64,
            0.0005,
            2.5,
            0.6,
            8,
            LocalDateTime.now().minusDays(5).format(formatter)
        ));

        MOCK_TASKS.add(createMockTask(
            "TASK_003",
            "语义分割协同训练-UNet",
            "llama2-13b",
            "unet",
            "COMPLETED",
            85.7,
            60,
            16,
            0.002,
            4.0,
            0.8,
            16,
            LocalDateTime.now().minusDays(7).format(formatter)
        ));

        MOCK_TASKS.add(createMockTask(
            "TASK_004",
            "序列预测协同训练-LSTM",
            "qwen-14b",
            "lstm",
            "RUNNING",
            null,
            30,
            32,
            0.001,
            3.5,
            0.7,
            8,
            LocalDateTime.now().minusHours(3).format(formatter)
        ));

        MOCK_TASKS.add(createMockTask(
            "TASK_005",
            "视觉Transformer协同训练",
            "llama2-7b",
            "vit",
            "COMPLETED",
            90.2,
            45,
            32,
            0.0008,
            3.0,
            0.65,
            16,
            LocalDateTime.now().minusDays(10).format(formatter)
        ));
    }

    private static Map<String, Object> createMockTask(
        String taskId,
        String taskName,
        String teacherModel,
        String studentModel,
        String status,
        Double accuracy,
        int totalEpochs,
        int batchSize,
        double learningRate,
        double temperature,
        double alpha,
        int loraRank,
        String createTime
    ) {
        Map<String, Object> task = new HashMap<>();
        task.put("taskId", taskId);
        task.put("taskName", taskName);
        task.put("teacherModel", teacherModel);
        task.put("studentModel", studentModel);
        task.put("status", status);
        task.put("accuracy", accuracy);
        task.put("totalEpochs", totalEpochs);
        task.put("currentEpoch", status.equals("COMPLETED") ? totalEpochs : (int)(Math.random() * totalEpochs));
        task.put("batchSize", batchSize);
        task.put("learningRate", learningRate);
        task.put("temperature", temperature);
        task.put("alpha", alpha);
        task.put("loraRank", loraRank);
        task.put("createTime", createTime);
        task.put("updateTime", createTime);
        return task;
    }

    @GetMapping("/tasks")
    @ApiOperation("获取所有训练任务")
    public Result<?> getAllTasks() {
        return Result.success(MOCK_TASKS);
    }

    @GetMapping("/completed-models")
    @ApiOperation("获取已完成的训练任务（用于自动标注）")
    public Result<?> getCompletedModels(
        @ApiParam("最小准确率") @RequestParam(required = false) Double minAccuracy,
        @ApiParam("教师模型") @RequestParam(required = false) String teacherModel,
        @ApiParam("学生模型") @RequestParam(required = false) String studentModel
    ) {
        // 筛选已完成的任务
        List<Map<String, Object>> completedTasks = MOCK_TASKS.stream()
            .filter(task -> "COMPLETED".equals(task.get("status")))
            .filter(task -> {
                if (minAccuracy != null && task.get("accuracy") != null) {
                    return (Double) task.get("accuracy") >= minAccuracy;
                }
                return true;
            })
            .filter(task -> {
                if (teacherModel != null && !teacherModel.isEmpty()) {
                    return teacherModel.equals(task.get("teacherModel"));
                }
                return true;
            })
            .filter(task -> {
                if (studentModel != null && !studentModel.isEmpty()) {
                    return studentModel.equals(task.get("studentModel"));
                }
                return true;
            })
            .collect(Collectors.toList());

        return Result.success(completedTasks);
    }

    @GetMapping("/tasks/{taskId}")
    @ApiOperation("获取任务详情")
    public Result<?> getTaskDetail(@PathVariable String taskId) {
        Optional<Map<String, Object>> task = MOCK_TASKS.stream()
            .filter(t -> taskId.equals(t.get("taskId")))
            .findFirst();

        if (task.isPresent()) {
            return Result.success(task.get());
        } else {
            return Result.failed("任务不存在");
        }
    }

    @PostMapping("/tasks")
    @ApiOperation("创建训练任务")
    public Result<?> createTask(@RequestBody Map<String, Object> taskData) {
        String taskId = "TASK_" + String.format("%03d", MOCK_TASKS.size() + 1);
        DateTimeFormatter formatter = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss");
        String now = LocalDateTime.now().format(formatter);

        Map<String, Object> newTask = new HashMap<>();
        newTask.put("taskId", taskId);
        newTask.put("taskName", taskData.get("taskName"));
        newTask.put("teacherModel", taskData.get("teacherModel"));
        newTask.put("studentModel", taskData.get("studentModel"));
        newTask.put("status", "PENDING");
        newTask.put("accuracy", null);
        newTask.put("totalEpochs", taskData.get("totalEpochs"));
        newTask.put("currentEpoch", 0);
        newTask.put("batchSize", taskData.get("batchSize"));
        newTask.put("learningRate", taskData.get("learningRate"));
        newTask.put("temperature", taskData.get("temperature"));
        newTask.put("alpha", taskData.get("alpha"));
        newTask.put("loraRank", taskData.get("loraRank"));
        newTask.put("createTime", now);
        newTask.put("updateTime", now);

        MOCK_TASKS.add(newTask);

        return Result.success(newTask, "任务创建成功");
    }

    @PostMapping("/tasks/{taskId}/start")
    @ApiOperation("启动训练任务")
    public Result<?> startTask(@PathVariable String taskId) {
        Optional<Map<String, Object>> taskOpt = MOCK_TASKS.stream()
            .filter(t -> taskId.equals(t.get("taskId")))
            .findFirst();

        if (taskOpt.isPresent()) {
            Map<String, Object> task = taskOpt.get();
            task.put("status", "RUNNING");
            DateTimeFormatter formatter = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss");
            task.put("updateTime", LocalDateTime.now().format(formatter));
            return Result.success(task, "任务已启动");
        } else {
            return Result.failed("任务不存在");
        }
    }

    @PostMapping("/tasks/{taskId}/stop")
    @ApiOperation("停止训练任务")
    public Result<?> stopTask(@PathVariable String taskId) {
        Optional<Map<String, Object>> taskOpt = MOCK_TASKS.stream()
            .filter(t -> taskId.equals(t.get("taskId")))
            .findFirst();

        if (taskOpt.isPresent()) {
            Map<String, Object> task = taskOpt.get();
            task.put("status", "STOPPED");
            DateTimeFormatter formatter = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss");
            task.put("updateTime", LocalDateTime.now().format(formatter));
            return Result.success(task, "任务已停止");
        } else {
            return Result.failed("任务不存在");
        }
    }

    @DeleteMapping("/tasks/{taskId}")
    @ApiOperation("删除训练任务")
    public Result<?> deleteTask(@PathVariable String taskId) {
        boolean removed = MOCK_TASKS.removeIf(t -> taskId.equals(t.get("taskId")));

        if (removed) {
            return Result.success(null, "任务已删除");
        } else {
            return Result.failed("任务不存在");
        }
    }
}
