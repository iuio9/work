package com.qczy.common.generate;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.qczy.mapper.ModelAssessTaskMapper;
import com.qczy.mapper.ModelBaseMapper;
import com.qczy.model.entity.ModelAssessTaskEntity;
import com.qczy.model.entity.ModelBaseEntity;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;
import org.springframework.util.CollectionUtils;
import org.springframework.util.StringUtils;

import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.net.URLEncoder;
import java.nio.charset.StandardCharsets;
import java.util.*;
import java.util.stream.Collectors;
import java.util.zip.ZipEntry;
import java.util.zip.ZipOutputStream;

@Component
public class GenerateWordByApplyNoForm {

    @Autowired
    private ModelBaseMapper modelBaseMapper;
    @Autowired
    private ModelAssessTaskMapper modelAssessTaskMapper;
    @Autowired
    private GenerateWordForm generateWordForm;

    private static final ObjectMapper objectMapper = new ObjectMapper();

    /**
     * 下载包含任务和测试指标的压缩包
     */
    public void downloadWordZip(Integer modelId, HttpServletRequest request, HttpServletResponse response) {
        if (modelId == null) return;

        ModelBaseEntity model = modelBaseMapper.selectById(modelId);
        if (model == null) return;

        // 查询已完成的任务
        List<ModelAssessTaskEntity> taskList = modelAssessTaskMapper.selectList(
                new LambdaQueryWrapper<ModelAssessTaskEntity>()
                        .eq(ModelAssessTaskEntity::getModelBaseId, model.getId())
                        .eq(ModelAssessTaskEntity::getTaskStatus, 2)  //已完成
        );
        if (CollectionUtils.isEmpty(taskList)) return;

        // 创建临时目录
        File tempDir = new File(System.getProperty("java.io.tmpdir") + "/word_zip_" + System.currentTimeMillis());
        if (!tempDir.exists()) tempDir.mkdirs();

        try {
            // 生成任务 Word 文件
            for (ModelAssessTaskEntity task : taskList) {
                String fileName = "任务_" + task.getTaskName() + ".docx";
                generateWordForm.generateToFile(task.getId(), new File(tempDir, fileName));
            }

            // 生成测试指标 Word 文件
            Map<String, String> indicatorData = getTestIndicatorData(taskList);
            generateWordForm.generateTestIndicatorWord(
                    new File(tempDir, "测试指标（平均指标）.docx"),
                    indicatorData
            );

            // 打包下载
            response.setContentType("application/zip");
            String zipName = URLEncoder.encode(model.getModelName() + ".zip", StandardCharsets.UTF_8.name());
            response.setHeader("Content-Disposition", "attachment; filename=\"" + zipName + "\"");

            try (ZipOutputStream zos = new ZipOutputStream(response.getOutputStream())) {
                for (File file : tempDir.listFiles()) {
                    if (file.isFile()) {
                        zos.putNextEntry(new ZipEntry(file.getName()));
                        try (FileInputStream fis = new FileInputStream(file)) {
                            byte[] buffer = new byte[8092];
                            int len;
                            while ((len = fis.read(buffer)) > 0) {
                                zos.write(buffer, 0, len);
                            }
                        }
                        zos.closeEntry();
                    }
                }
            }

        } catch (Exception e) {
            e.printStackTrace();
            try {
                response.sendError(500, "生成压缩包失败");
            } catch (IOException ex) {
                ex.printStackTrace();
            }
        } finally {
            deleteDir(tempDir); // 清理临时文件
        }
    }

    // 以下方法与之前一致（获取指标数据、计算平均值等）
    public Map<String, String> getTestIndicatorData(List<ModelAssessTaskEntity> taskList) {
        Map<String, String> dataMap = new HashMap<>();
        if (CollectionUtils.isEmpty(taskList)) return dataMap;

        Map<String, List<Double>> metricsMap = new LinkedHashMap<>();
        metricsMap.put("mPrecisionAvg", new ArrayList<>());
        metricsMap.put("mRecallAvg", new ArrayList<>());
        metricsMap.put("mAP@0.5Avg", new ArrayList<>());
        metricsMap.put("mMissRateAvg", new ArrayList<>());
        metricsMap.put("mFalseAlarmRateAvg", new ArrayList<>());
        metricsMap.put("mAccuracyAvg", new ArrayList<>());

        for (ModelAssessTaskEntity task : taskList) {
            if (StringUtils.isEmpty(task.getTaskResult())) continue;

            try {
                Map<String, Object> result = objectMapper.readValue(task.getTaskResult(), Map.class);
                metricsMap.get("mPrecisionAvg").add(parseDouble(result.get("mPrecision")));
                metricsMap.get("mRecallAvg").add(parseDouble(result.get("mRecall")));
                metricsMap.get("mAP@0.5Avg").add(parseDouble(result.get("mAP@0.5")));
                metricsMap.get("mAccuracyAvg").add(parseDouble(result.get("mAccuracy")));
                metricsMap.get("mMissRateAvg").add(
                        result.get("mMissRate") != null ? parseDouble(result.get("mMissRate")) : null
                );
                metricsMap.get("mFalseAlarmRateAvg").add(
                        result.get("mFalseAlarmRate") != null ? parseDouble(result.get("mFalseAlarmRate")) : null
                );
            } catch (JsonProcessingException e) {
                System.err.println("解析任务结果失败，任务ID：" + task.getId() + "，错误：" + e.getMessage());
            }
        }

        calculateAverageMetrics(metricsMap, dataMap);
        return dataMap;
    }

    private Double parseDouble(Object value) {
        if (value == null) return 0.0;
        try {
            return Double.parseDouble(value.toString());
        } catch (NumberFormatException e) {
            System.err.println("指标格式错误，使用0替代：" + value);
            return 0.0;
        }
    }

    private void calculateAverageMetrics(Map<String, List<Double>> metricsMap, Map<String, String> dataMap) {
        for (Map.Entry<String, List<Double>> entry : metricsMap.entrySet()) {
            List<Double> validValues = entry.getValue().stream()
                    .filter(Objects::nonNull)
                    .collect(Collectors.toList());

            if (validValues.isEmpty()) {
                dataMap.put(entry.getKey(), "--");
                continue;
            }

            double avg = validValues.stream().mapToDouble(Double::doubleValue).average().orElse(0.0);
            dataMap.put(entry.getKey(), String.valueOf(avg));
        }
    }

    private boolean deleteDir(File dir) {
        if (dir.isDirectory()) {
            File[] children = dir.listFiles();
            if (children != null) for (File child : children) deleteDir(child);
        }
        return dir.delete();
    }
}