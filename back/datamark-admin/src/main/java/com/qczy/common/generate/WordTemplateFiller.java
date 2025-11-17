package com.qczy.common.generate;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.qczy.mapper.ModelAssessConfigMapper;
import com.qczy.mapper.ModelAssessTaskMapper;
import com.qczy.mapper.ModelBaseMapper;
import com.qczy.mapper.ModelConfigureMapper;
import com.qczy.model.entity.ModelAssessTaskEntity;
import com.qczy.model.entity.ModelBaseEntity;
import com.qczy.model.entity.ModelConfigureEntity;
import com.qczy.utils.EntityUtils;
import org.apache.poi.xwpf.usermodel.*;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;
import org.apache.commons.beanutils.BeanUtils;

import java.io.*;
import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

@Component
public class WordTemplateFiller {

    @Autowired
    private ModelAssessTaskMapper modelAssessTaskMapper;
    @Autowired
    private ModelAssessConfigMapper modelAssessConfigMapper;
    @Autowired
    private ModelBaseMapper modelBaseMapper;
    @Autowired
    private ModelConfigureMapper modelConfigureMapper;

    public void genWord() {
        try {
            // 1. 加载模板文件（确保模板文件在 resources 目录下）
            String templateName = "111222.docx";
            InputStream is = WordTemplateFiller.class.getClassLoader().getResourceAsStream(templateName);
            if (is == null) {
                throw new RuntimeException("模板文件未找到，请检查 resources 目录下是否存在：" + templateName);
            }

            XWPFDocument document = new XWPFDocument(is);

            // 2. 准备数据（键值对）
            // 获取动态数据
            Map<String, String> dataMap = getDynamicData();
          /*  for (Map.Entry<String, String> entry : dataMap.entrySet()) {
                System.out.println(entry.getKey() + ":" + entry.getValue());
            }*/
           /* Map<String, String> dataMap = new HashMap<>();
            dataMap.put("model_name", "电力系统稳定性预测模型");
            dataMap.put("project_id", "SG2025-0001");
            dataMap.put("construction_unit", "国家电网XX分公司");
            dataMap.put("construction_address", "北京市西城区");
            dataMap.put("construction_contact", "张三");
            dataMap.put("building_unit", "中国电力科学研究院");
            dataMap.put("building_address", "北京市海淀区");
            dataMap.put("building_contact", "李四");
*/
            // 3. 替换段落中的占位符
            for (XWPFParagraph paragraph : document.getParagraphs()) {
                replacePlaceholdersInParagraph(paragraph, dataMap);
            }

            // 4. 替换表格中的占位符（包括嵌套表格）
            for (XWPFTable table : document.getTables()) {
                replacePlaceholdersInTable(table, dataMap);
            }

            // 5. 保存输出文件
            String outputFileName = "填写后的模型测评报告.docx";
            try (FileOutputStream fos = new FileOutputStream(outputFileName)) {
                document.write(fos);
            }

            System.out.println("✅ 文档已成功生成：" + outputFileName);

        } catch (IOException e) {
            System.err.println("❌ 文档处理过程中发生错误：");
            e.printStackTrace();
        }
    }

    /**
     * 获取动态数据
     */
    private Map<String, String> getDynamicData() {
        // 根据任务id获取数据
        Map<String, String> dataMap = new HashMap<>();
        ModelAssessTaskEntity modelAssessTaskEntity = modelAssessTaskMapper.selectById(20);
        if (modelAssessTaskEntity == null || modelAssessTaskEntity.getModelBaseId() == null) {
            return dataMap;
        }
        dataMap.putAll(EntityUtils.convertToMap(modelAssessTaskEntity));
        ModelBaseEntity modelBaseEntity = modelBaseMapper.selectOne(
                new LambdaQueryWrapper<ModelBaseEntity>()
                        .eq(ModelBaseEntity::getId, modelAssessTaskEntity.getModelBaseId())
        );
        dataMap.putAll(EntityUtils.convertToMap(modelBaseEntity));
        if (modelBaseEntity == null) {
            return dataMap;
        }
        ModelConfigureEntity modelConfigureEntity = modelConfigureMapper.selectOne(
                new LambdaQueryWrapper<ModelConfigureEntity>()
                        .eq(ModelConfigureEntity::getModelBaseId, modelBaseEntity.getId())
        );

        dataMap.putAll(EntityUtils.convertToMap(modelConfigureEntity));
        return dataMap;
    }

    /**
     * 替换段落中的占位符（支持跨 XWPFRun 的占位符）
     * 占位符格式：{{key}}
     */
    private static void replacePlaceholdersInParagraph(XWPFParagraph paragraph, Map<String, String> dataMap) {
        List<XWPFRun> runs = paragraph.getRuns();
        if (runs == null || runs.isEmpty()) return;

        // 1. 合并所有 Run 的文本
        StringBuilder fullText = new StringBuilder();
        for (XWPFRun run : runs) {
            String text = run.getText(0);
            if (text != null) fullText.append(text);
        }

        // 2. 替换占位符
        String replacedText = replacePlaceholders(fullText.toString(), dataMap);

        // 3. 清空原有 Run，重新设置文本
        while (runs.size() > 1) {
            paragraph.removeRun(1); // 保留第一个 Run
        }
        XWPFRun firstRun = runs.get(0);
        firstRun.setText(replacedText, 0);
    }

    /**
     * 替换表格中的占位符（递归处理每个单元格的段落和嵌套表格）
     */
    private static void replacePlaceholdersInTable(XWPFTable table, Map<String, String> dataMap) {
        for (XWPFTableRow row : table.getRows()) {
            for (XWPFTableCell cell : row.getTableCells()) {
                // 处理单元格中的段落
                for (XWPFParagraph paragraph : cell.getParagraphs()) {
                    replacePlaceholdersInParagraph(paragraph, dataMap);
                }

                // 递归处理嵌套表格
                for (XWPFTable nestedTable : cell.getTables()) {
                    replacePlaceholdersInTable(nestedTable, dataMap);
                }
            }
        }
    }

    /**
     * 替换占位符核心逻辑
     */
    private static String replacePlaceholders(String text, Map<String, String> dataMap) {
        if (text == null || text.isEmpty()) return text;

        StringBuilder result = new StringBuilder();
        int lastEnd = 0;
        Matcher matcher = Pattern.compile("\\{\\{([^{}]+)\\}\\}").matcher(text);

        while (matcher.find()) {
            // 添加非占位符部分
            result.append(text, lastEnd, matcher.start());

            // 获取键名并替换
            String key = matcher.group(1).trim();
            String replacement = dataMap.getOrDefault(key, matcher.group(0)); // 如果不存在键，保留原占位符
            result.append(replacement);

            // 更新最后位置
            lastEnd = matcher.end();
        }

        // 添加剩余部分
        result.append(text.substring(lastEnd));
        return result.toString();
    }
}