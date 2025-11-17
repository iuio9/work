package com.qczy.common.generate;

import org.apache.poi.xwpf.usermodel.*;

import java.io.InputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class ModelEvaluationReportFiller {

    public static void main(String[] args) {
        // 加载模板文件
        String templateName = "111222.docx";
        try (InputStream is = ModelEvaluationReportFiller.class.getClassLoader().getResourceAsStream(templateName)) {
            if (is == null) {
                throw new RuntimeException("模板文件未找到，请检查 resources 目录下是否存在：" + templateName);
            }

            XWPFDocument document = new XWPFDocument(is);

            // 准备要填入的数据
            Map<String, String> dataMap = new HashMap<>();
            dataMap.put("model_name", "电力系统稳定性预测模型");
            dataMap.put("project_id", "SG2025-0001");
            dataMap.put("construction_unit", "国家电网XX分公司");
            dataMap.put("construction_address", "北京市西城区");
            dataMap.put("construction_contact", "张三");
            dataMap.put("building_unit", "中国电力科学研究院");
            dataMap.put("building_address", "北京市海淀区");
            dataMap.put("building_contact", "李四");

            // 替换段落中的变量
            replacePlaceholdersInParagraphs(document.getParagraphs(), dataMap);

            // 替换表格中的变量
            for (XWPFTable table : document.getTables()) {
                replacePlaceholdersInTable(table, dataMap);
            }

            // 保存修改后的文档
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
     * 替换段落中的占位符，支持 (key) 和 ${key} 两种格式
     */
    private static void replacePlaceholdersInParagraphs(List<XWPFParagraph> paragraphs, Map<String, String> dataMap) {
        for (XWPFParagraph paragraph : paragraphs) {
            List<XWPFRun> runs = paragraph.getRuns();
            if (runs != null && !runs.isEmpty()) {
                StringBuilder textBuilder = new StringBuilder();
                for (XWPFRun run : runs) {
                    String text = run.getText(0);
                    if (text != null) {
                        textBuilder.append(text);
                    }
                }
                String fullText = textBuilder.toString();

                // 匹配所有占位符
                Pattern pattern = Pattern.compile("\\$\\w+\\$|\\$\\{\\w+\\}|\\$$(\\w+)\\$$");
                Matcher matcher = pattern.matcher(fullText);

                while (matcher.find()) {
                    String placeholder = matcher.group();
                    String key = placeholder.replaceAll("[(){}$]", "");
                    if (dataMap.containsKey(key)) {
                        String value = dataMap.get(key);
                        int start = matcher.start();
                        int end = matcher.end();
                        int index = 0;
                        for (XWPFRun run : runs) {
                            String runText = run.getText(0);
                            if (runText != null) {
                                int length = runText.length();
                                if (index + length >= start && index < end) {
                                    String newText = runText.replace(placeholder, value);
                                    run.setText(newText, 0);
                                }
                                index += length;
                            }
                        }
                    }
                }
            }
        }
    }

    /**
     * 替换表格中的占位符，支持 (key) 和 ${key} 两种格式
     */
    private static void replacePlaceholdersInTable(XWPFTable table, Map<String, String> dataMap) {
        for (XWPFTableRow row : table.getRows()) {
            for (XWPFTableCell cell : row.getTableCells()) {
                replacePlaceholdersInParagraphs(cell.getParagraphs(), dataMap);
            }
        }
    }
}