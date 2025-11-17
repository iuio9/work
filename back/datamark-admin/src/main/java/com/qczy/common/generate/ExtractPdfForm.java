/*
package com.qczy.common.generate;

import com.qczy.model.entity.ModelBaseEntity;
import com.qczy.model.request.ModelApplyForRequest;
import org.apache.pdfbox.Loader;
import org.apache.pdfbox.pdmodel.PDDocument;
import org.apache.pdfbox.text.PDFTextStripper;

import java.io.File;
import java.io.IOException;
import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.*;

import org.apache.pdfbox.pdmodel.PDDocument;
import org.apache.pdfbox.text.PDFTextStripper;
import org.apache.pdfbox.text.TextPosition;

*/
/**
 * @Author: hh
 * @Version: 1.0
 * @Date: 2025/5/20 10:29
 * @Description:
 *//*


public class ExtractPdfForm {


    public static Map<String, String> extractTwoColumnTable(String filePath) throws IOException {
        try (PDDocument document = Loader.loadPDF(new File(filePath))) {
            List<TextChunk> textChunks = new ArrayList<>();

            // 第一步：收集所有文本块及其坐标信息
            new PDFTextStripper() {
                @Override
                protected void writeString(String text, List<TextPosition> textPositions) {
                    float startX = textPositions.get(0).getX();
                    float endX = textPositions.get(textPositions.size() - 1).getEndX();
                    float y = textPositions.get(0).getY();
                    textChunks.add(new TextChunk(text.trim(), startX, endX, y));
                }
            }.getText(document);

            // 第二步：智能列边界检测
            ColumnBoundary columnBoundary = detectColumnBoundary(textChunks);

            // 第三步：行合并与数据处理
            return processTextChunks(textChunks, columnBoundary);
        }
    }

    // 列边界检测（动态容差）
    private static ColumnBoundary detectColumnBoundary(List<TextChunk> chunks) {
        Map<Integer, Integer> xFrequency = new HashMap<>();

        // 统计X坐标出现频率
        chunks.forEach(chunk -> {
            int xKey = Math.round(chunk.startX);
            xFrequency.put(xKey, xFrequency.getOrDefault(xKey, 0) + 1);
        });

        // 寻找第一列结束位置（最高频右边界）
        int firstColEnd = xFrequency.entrySet().stream()
                .max(Map.Entry.comparingByValue())
                .map(Map.Entry::getKey)
                .orElse(200); // 默认值

        // 动态计算列间距容差
        int tolerance = calculateDynamicTolerance(chunks);

        return new ColumnBoundary(firstColEnd, tolerance);
    }

    // 动态容差计算（基于字符平均宽度）
    private static int calculateDynamicTolerance(List<TextChunk> chunks) {
        if (chunks.isEmpty()) return 10;

        // 使用mapToDouble处理浮点运算
        double totalWidth = chunks.stream()
                .mapToDouble(c -> c.endX - c.startX) // 自动将float转为double
                .sum();

        int totalChars = chunks.stream()
                .mapToInt(c -> c.text.length())
                .sum();

        // 计算平均字符宽度时进行类型转换
        float avgCharWidth = (float) (totalWidth / totalChars);

        return Math.round(avgCharWidth * 1.5f);
    }

    // 文本处理核心逻辑
    private static Map<String, String> processTextChunks(List<TextChunk> chunks, ColumnBoundary boundary) {
        Map<Float, TableRow> rowMap = new TreeMap<>();

        // 按Y坐标分组（行合并）
        chunks.forEach(chunk -> {
            TableRow row = rowMap.computeIfAbsent(chunk.y, k -> new TableRow());

            if (chunk.endX < boundary.firstColEnd + boundary.tolerance) {
                row.key.append(chunk.text).append(" ");
            } else {
                row.value.append(chunk.text).append(" ");
            }
        });

        // 构建结果Map
        Map<String, String> result = new LinkedHashMap<>();
        rowMap.values().forEach(row -> {
            String key = row.key.toString().replaceAll("\\s+", "").trim();
            String value = row.value.toString().replaceAll("\\s+", "").trim();
            if (!key.isEmpty() && !value.isEmpty()) {
                result.put(key, value);
            }
        });

        return result;
    }

    // 辅助类定义
    static class TextChunk {
        String text;
        float startX;
        float endX;
        float y;

        TextChunk(String text, float startX, float endX, float y) {
            this.text = text;
            this.startX = startX;
            this.endX = endX;
            this.y = y;
        }
    }

    static class ColumnBoundary {
        int firstColEnd;
        int tolerance;

        ColumnBoundary(int firstColEnd, int tolerance) {
            this.firstColEnd = firstColEnd;
            this.tolerance = tolerance;
        }
    }

    static class TableRow {
        StringBuilder key = new StringBuilder();
        StringBuilder value = new StringBuilder();
    }


    public static int checkPfd(ModelApplyForRequest request, String filePath) {
        try {
            Map<String, String> result = extractTwoColumnTable("C:\\Users\\c\\Desktop\\model_20250516697.pdf");
            ModelBaseEntity modelBaseEntity = request.getModelBaseEntity();
            ModelBaseEntity checkModelBaseEntity = new ModelBaseEntity();

            checkModelBaseEntity.setApplyForNum(result.getOrDefault("申请单号", null));
            checkModelBaseEntity.setBusineUnit(result.getOrDefault("业务单位", null));
            checkModelBaseEntity.setBusinePeople(result.getOrDefault("业务单位负责⼈", null));
            checkModelBaseEntity.setBusinePhone(result.getOrDefault("业务单位联系⽅式", null));
            checkModelBaseEntity.setDevUnit(result.getOrDefault("开发单位", null));
            checkModelBaseEntity.setDevPeople(result.getOrDefault("开发单位负责⼈", null));
            checkModelBaseEntity.setDevPhone(result.getOrDefault("开发单位联系⽅式", null));
            checkModelBaseEntity.setApplyForDate(result.getOrDefault("申请⽇期", null) == null ? null :
                    new SimpleDateFormat("yyyy-MM-dd").parse(result.getOrDefault("申请⽇期", null)));


            System.out.println("旧的：" + modelBaseEntity);
            System.out.println("新的：" + checkModelBaseEntity);
            boolean b = compareModel(modelBaseEntity, checkModelBaseEntity);
            System.out.println("结果：" + b);

        } catch (RuntimeException e) {
            e.printStackTrace();
        } catch (IOException e) {
            throw new RuntimeException(e);
        } catch (ParseException e) {
            throw new RuntimeException(e);
        }

        return 0;

    }


    */
/**
     * 比较两个ModelBaseEntity对象的属性值是否相同
     *//*

    private static boolean compareModel(ModelBaseEntity entity1, ModelBaseEntity entity2) {
        // 比较各个属性值
        return Objects.equals(entity1.getApplyForNum(), entity2.getApplyForNum()) &&
                (entity1.getDevUnit().equals(entity2.getDevUnit())) &&
                (Objects.equals(entity1.getBusinePeople(), entity2.getBusinePeople()));

    }


    public static void main(String[] args) {
        try {
            Map<String, String> result = extractTwoColumnTable("C:\\Users\\c\\Desktop\\model_20250516697.pdf");
            result.forEach((k, v) -> System.out.println("[" + k + "] => [" + v + "]"));
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

}


*/
