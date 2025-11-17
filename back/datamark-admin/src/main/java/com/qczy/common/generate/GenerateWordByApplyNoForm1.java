package com.qczy.common.generate;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.core.toolkit.CollectionUtils;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.qczy.mapper.ModelAssessConfigMapper;
import com.qczy.mapper.ModelAssessTaskMapper;
import com.qczy.mapper.ModelBaseMapper;
import com.qczy.mapper.ModelConfigureMapper;
import com.qczy.model.entity.ModelAssessTaskEntity;
import com.qczy.model.entity.ModelBaseEntity;
import com.qczy.model.entity.ModelConfigureEntity;
import com.qczy.utils.EntityUtils;
import com.qczy.utils.StringUtils;
import org.apache.commons.io.IOUtils;
import org.apache.pdfbox.Loader;
import org.apache.pdfbox.pdmodel.PDDocument;
import org.apache.pdfbox.rendering.ImageType;
import org.apache.pdfbox.rendering.PDFRenderer;
import org.apache.poi.util.Units;
import org.apache.poi.xwpf.usermodel.*;
import org.openxmlformats.schemas.wordprocessingml.x2006.main.CTTblPr;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

import javax.imageio.ImageIO;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import java.awt.image.BufferedImage;
import java.io.*;
import java.math.BigInteger;
import java.net.HttpURLConnection;
import java.net.URL;
import java.net.URLEncoder;
import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

@Component
public class GenerateWordByApplyNoForm1 {

    @Autowired
    private ModelAssessTaskMapper modelAssessTaskMapper;

    @Autowired
    private ModelBaseMapper modelBaseMapper;
    @Autowired
    private ModelConfigureMapper modelConfigureMapper;
    @Autowired
    private GenerateWordByApplyNoForm generateWordByApplyNoForm;

    // JSON解析器
    private static final ObjectMapper objectMapper = new ObjectMapper();
    // 临时文件存储目录
    private static final String TEMP_IMAGE_DIR = "temp_images/";
    // 图片格式
    private static final String IMAGE_FORMAT = "png";
    // 常量定义
    private static final int MAX_PAGES_PER_CELL = 3; // 单元格最多容纳的PDF页数
    private static final double MAX_HEIGHT_RATIO = 2.0; // 图片总高度相对于单元格高度的最大比例

    // PDF临时图片占位符前缀
    private static final String PDF_IMAGE_PLACEHOLDER_PREFIX = "PDF_IMAGE_PLACEHOLDER_";
    // 网络图片临时文件前缀
    private static final String NETWORK_IMAGE_PREFIX = "network_";

    // 任务 Word 模板文件名（放在 resources 目录）
    private static final String TASK_TEMPLATE = "222333.docx";

    // 第一个任务名称
    private String taskName;

    // 持有当前处理的文档引用
    private XWPFDocument currentDocument;

    public void downloadWord(Integer applyNo, HttpServletRequest request, HttpServletResponse response) {
        // List<Integer> taskIds = getTaskIdsByApplyNo().stream(applyNo);
        List<ModelAssessTaskEntity> modelAssessTaskEntityList = getTaskIdsByApplyNo(applyNo);
        List<Integer> taskIds = modelAssessTaskEntityList.stream().map(ModelAssessTaskEntity::getId).collect(Collectors.toList());
        if (taskIds.isEmpty()) {
            try {
                response.sendError(HttpServletResponse.SC_NOT_FOUND, "未找到对应申请单号的任务");
            } catch (IOException e) {
                e.printStackTrace();
            }
            return;
        }

        try {
            InputStream is = GenerateWordByApplyNoForm1.class.getClassLoader().getResourceAsStream(TASK_TEMPLATE);
            if (is == null) {
                response.sendError(HttpServletResponse.SC_NOT_FOUND, "模板文件未找到");
                return;
            }

            currentDocument = new XWPFDocument(is);
            List<XWPFTable> tables = currentDocument.getTables();
            if (tables.isEmpty()) {
                response.sendError(HttpServletResponse.SC_INTERNAL_SERVER_ERROR, "模板文件中未找到表格");
                return;
            }

            XWPFTable lastTable = tables.get(tables.size() - 1);

            // 为每个额外的任务复制表格（从索引1开始，对应taskIndex=1）
            for (int i = 1; i < taskIds.size(); i++) {
                // 关键：在新表格前插入空段落，设置段后间距（控制表格间空隙）
                XWPFParagraph separatorPara = currentDocument.createParagraph();
                // POI 4.2.1 支持的方法：设置段后间距（单位：行，1.0 = 1行，0.5 = 半行）
                separatorPara.setSpacingAfter(1); // 段后间距1行（可根据需要调整）
                // 去除段落默认缩进（可选，避免空隙过大）
                separatorPara.setIndentationLeft(0);
                separatorPara.setIndentationRight(0);

                // 创建并复制表格内容
                currentDocument.createTable();
                XWPFTable newTable = currentDocument.getTables().get(currentDocument.getTables().size() - 1);
                cloneTableContent(lastTable, newTable);
                modifyTablePlaceholders(newTable, String.valueOf(i));

                System.out.println("克隆并修改表格占位符成功，当前总表格数: " + currentDocument.getTables().size());
            }

            // 填充数据（taskIndex与表格索引一致）
            Map<String, String> dataMap1 = new HashMap<>();
            for (int i = 0; i < taskIds.size(); i++) {
                Integer taskId = taskIds.get(i);
                int taskIndex = i; // 任务索引从0开始，与表格索引对应（0:原表格，1:复制表格1）
                Map<String, String> dataMap = getDynamicData(taskId, taskIndex);
                dataMap1.putAll(dataMap);
                dataMap.put("taskIndex", String.valueOf(i + 1));
                System.out.println("任务ID: " + taskId + " 数据: " + dataMap);

            }

            dataMap1.put("taskName", taskName);

            // 计算平均值
            Map<String, String> testIndicatorData = generateWordByApplyNoForm.getTestIndicatorData(modelAssessTaskEntityList);
            if (!CollectionUtils.isEmpty(testIndicatorData)) {
                dataMap1.putAll(testIndicatorData);
            }


            // 最后在进行替换
            replaceAllPlaceholders(currentDocument, dataMap1);

           /* // 后续响应处理（保持不变）
            System.out.println("========================完整的数据========================");
            for (Map.Entry<String, String> entry : dataMap1.entrySet()) {
                System.out.println(entry.getKey() + ": " + entry.getValue());
            }
            System.out.println("=========================================================");
*/
            // 设置响应头并输出
            response.setContentType("application/vnd.openxmlformats-officedocument.wordprocessingml.document");
            response.setCharacterEncoding("UTF-8");
            String fileName = "测评报告_" + applyNo + ".docx";
            String userAgent = request.getHeader("User-Agent");
            String encodedFileName = userAgent != null && userAgent.matches("(?i).*(MSIE|Trident|Edge).*")
                    ? URLEncoder.encode(fileName, "UTF-8").replace("+", "%20")
                    : new String(fileName.getBytes("UTF-8"), "ISO-8859-1");
            response.setHeader("Content-Disposition", "attachment; filename=\"" + encodedFileName + "\"");
            response.setHeader("Cache-Control", "no-cache, no-store, must-revalidate");
            response.setHeader("Pragma", "no-cache");
            response.setDateHeader("Expires", 0);

            try (OutputStream os = response.getOutputStream()) {
                currentDocument.write(os);
                os.flush();
            }

        } catch (Exception e) {
            System.err.println("❌ 文档处理过程中发生错误：");
            e.printStackTrace();
            try {
                response.sendError(HttpServletResponse.SC_INTERNAL_SERVER_ERROR, "生成文档失败");
            } catch (IOException ioException) {
                ioException.printStackTrace();
            }
        } finally {
            cleanTempImages();
        }
    }




    /**
     * 修改表格中的占位符，添加唯一标识后缀，解决占位符被拆分到多个XWPFRun的问题
     *
     * @param table        要修改的表格
     * @param uniqueSuffix 唯一标识后缀（如任务ID、索引等）
     */
    private void modifyTablePlaceholders(XWPFTable table, String uniqueSuffix) {
        if (table == null || StringUtils.isEmpty(uniqueSuffix)) {
            System.out.println("⚠️ 表格为空或唯一后缀为空，不执行占位符修改");
            return;
        }
        System.out.println("📌 开始修改表格占位符，唯一后缀：" + uniqueSuffix);

        // 遍历表格所有行
        for (int rowIdx = 0; rowIdx < table.getRows().size(); rowIdx++) {
            XWPFTableRow row = table.getRow(rowIdx);
            if (row == null) continue;

            // 遍历行中所有单元格
            for (int cellIdx = 0; cellIdx < row.getTableCells().size(); cellIdx++) {
                XWPFTableCell cell = row.getCell(cellIdx);
                if (cell == null) continue;

                // 处理单元格内的段落（修正：通过索引从段落列表中获取段落）
                List<XWPFParagraph> paragraphs = cell.getParagraphs();
                for (int paraIdx = 0; paraIdx < paragraphs.size(); paraIdx++) {
                    XWPFParagraph paragraph = paragraphs.get(paraIdx);
                    if (paragraph == null) continue;

                    // 1. 合并段落中所有run的文本，解决占位符被拆分的问题
                    StringBuilder originalText = new StringBuilder();
                    for (XWPFRun run : paragraph.getRuns()) {
                        String text = run.getText(0);
                        if (text != null) {
                            originalText.append(text);
                        }
                    }
                    String paraOriginalText = originalText.toString();

                    // 2. 匹配并修改占位符（支持 {{key}}、{{ key }} 等带空格的格式）
                    Pattern pattern = Pattern.compile("\\{\\{\\s*([^{}]+?)\\s*\\}\\}");
                    Matcher matcher = pattern.matcher(paraOriginalText);
                    StringBuffer modifiedText = new StringBuffer();
                    boolean hasMatch = false;

                    while (matcher.find()) {
                        hasMatch = true;
                        String originalKey = matcher.group(1).trim();
                        String newPlaceholder = String.format("{{%s_%s}}", originalKey, uniqueSuffix);
                        matcher.appendReplacement(modifiedText, newPlaceholder);
                    }
                    matcher.appendTail(modifiedText);
                    String paraModifiedText = modifiedText.toString();

                    // 3. 清空原段落的run，写入修改后的文本（解决拆分问题）
                    if (hasMatch) {
                        // 清除原有run
                        while (paragraph.getRuns().size() > 0) {
                            paragraph.removeRun(0);
                        }
                        // 创建新run写入修改后的文本
                        XWPFRun newRun = paragraph.createRun();
                        newRun.setText(paraModifiedText);
                    }
                }

                // 处理嵌套表格（递归）
                for (XWPFTable nestedTable : cell.getTables()) {
                    modifyTablePlaceholders(nestedTable, uniqueSuffix);
                }
            }
        }
    }


    /**
     * 复制表格内容（不创建新表格）
     */
    private void cloneTableContent(XWPFTable source, XWPFTable target) {
        // 复制表格属性
        target.getCTTbl().setTblPr(source.getCTTbl().getTblPr());

        // 确保目标表格有足够的行数
        while (target.getRows().size() < source.getRows().size()) {
            target.createRow();
        }

        // 复制每个单元格的内容
        for (int i = 0; i < source.getRows().size(); i++) {
            XWPFTableRow sourceRow = source.getRow(i);
            XWPFTableRow targetRow = target.getRow(i);

            // 复制行属性
            if (sourceRow.getCtRow().isSetTrPr()) {
                targetRow.getCtRow().setTrPr(sourceRow.getCtRow().getTrPr());
            }

            // 确保目标行有足够的单元格
            while (targetRow.getTableCells().size() < sourceRow.getTableCells().size()) {
                targetRow.addNewTableCell();
            }

            for (int j = 0; j < sourceRow.getTableCells().size(); j++) {
                XWPFTableCell sourceCell = sourceRow.getCell(j);
                XWPFTableCell targetCell = targetRow.getCell(j);

                // 复制单元格属性
                if (sourceCell.getCTTc().isSetTcPr()) {
                    targetCell.getCTTc().setTcPr(sourceCell.getCTTc().getTcPr());
                }

                // 清空目标单元格的默认段落
                while (targetCell.getParagraphs().size() > 0) {
                    targetCell.removeParagraph(0);
                }

                // 复制源单元格的所有段落
                for (XWPFParagraph sourcePara : sourceCell.getParagraphs()) {
                    XWPFParagraph targetPara = targetCell.addParagraph();

                    // 复制段落属性
                    targetPara.getCTP().setPPr(sourcePara.getCTP().getPPr());

                    // 复制段落中的所有运行
                    for (XWPFRun sourceRun : sourcePara.getRuns()) {
                        XWPFRun targetRun = targetPara.createRun();

                        // 复制运行属性
                        if (sourceRun.getCTR().isSetRPr()) {
                            targetRun.getCTR().setRPr(sourceRun.getCTR().getRPr());
                        }

                        // 复制文本和其他属性
                        targetRun.setText(sourceRun.getText(0));
                        targetRun.setBold(sourceRun.isBold());
                        targetRun.setItalic(sourceRun.isItalic());
                        // 可以根据需要复制更多属性
                    }
                }
            }
        }
    }


    /**
     * 根据申请单号获取任务 ID 列表
     *
     * @param applyNo 申请单号
     * @return 任务 ID 列表
     */
    private List<ModelAssessTaskEntity> getTaskIdsByApplyNo(Integer applyNo) {
        LambdaQueryWrapper<ModelAssessTaskEntity> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(ModelAssessTaskEntity::getModelBaseId, applyNo);
        wrapper.eq(ModelAssessTaskEntity::getTaskStatus, 2);
        return modelAssessTaskMapper.selectList(wrapper);

    }


    /**
     * 获取动态数据，确保图片键带后缀且不被覆盖
     */
    private Map<String, String> getDynamicData(Integer taskId, Integer taskIndex) {
        Map<String, String> dataMap = new HashMap<>();
        try {
            System.out.println("📌 开始获取任务ID=" + taskId + " 的动态数据，taskIndex=" + taskIndex);

            ModelAssessTaskEntity modelAssessTaskEntity = modelAssessTaskMapper.selectById(taskId);
            if (modelAssessTaskEntity == null || modelAssessTaskEntity.getModelBaseId() == null) {
                System.err.println("⚠️ 任务ID=" + taskId + " 数据不存在或不完整");
                return dataMap;
            }

            // 1. 先加载模型基础数据和配置数据（低优先级，避免覆盖任务结果）
            ModelBaseEntity modelBaseEntity = modelBaseMapper.selectOne(
                    new LambdaQueryWrapper<ModelBaseEntity>()
                            .eq(ModelBaseEntity::getId, modelAssessTaskEntity.getModelBaseId())
            );
            if (modelBaseEntity != null) {
                // 过滤模型基础数据中可能覆盖图片键的字段
                Map<String, String> baseMap = EntityUtils.convertToMap(modelBaseEntity);
                baseMap.keySet().removeAll(Arrays.asList("PR_curve", "confusion_matrix", "PR_curve_1", "confusion_matrix_1"));
                dataMap.putAll(baseMap);
                System.out.println("✅ 加载模型基础数据(过滤后): " + baseMap.keySet());

                // 处理模型名称
                if (!StringUtils.isEmpty(modelBaseEntity.getModelName())) {
                    boolean endsWith = modelBaseEntity.getModelName().trim().endsWith("模型");
                    dataMap.put("modelName", endsWith ? modelBaseEntity.getModelName() : modelBaseEntity.getModelName() + "模型");
                }

                // 加载模型配置数据
                ModelConfigureEntity modelConfigureEntity = modelConfigureMapper.selectOne(
                        new LambdaQueryWrapper<ModelConfigureEntity>()
                                .eq(ModelConfigureEntity::getModelBaseId, modelBaseEntity.getId())
                );
                if (modelConfigureEntity != null) {
                    // 过滤配置数据中可能覆盖图片键的字段
                    Map<String, String> configMap = EntityUtils.convertToMap(modelConfigureEntity);
                    configMap.keySet().removeAll(Arrays.asList("PR_curve", "confusion_matrix", "PR_curve_1", "confusion_matrix_1"));
                    dataMap.putAll(configMap);
                    System.out.println("✅ 加载模型配置数据(过滤后): " + configMap.keySet());

                    if (!StringUtils.isEmpty(modelConfigureEntity.getModelInterfaceDesc())) {
                        dataMap.put("v1", "详见附表1");
                    }
                    if (!StringUtils.isEmpty(modelConfigureEntity.getModelCase())) {
                        dataMap.put("v2", "详见附表2");
                    }
                }
            }

            // 2. 加载任务基本数据（中优先级）
            Map<String, String> taskBaseMap = EntityUtils.convertToMap(modelAssessTaskEntity);
            // 过滤任务数据中可能覆盖图片键的字段
            taskBaseMap.keySet().removeAll(Arrays.asList("PR_curve", "confusion_matrix", "PR_curve_1", "confusion_matrix_1"));
            dataMap.putAll(taskBaseMap);
            System.out.println("✅ 加载任务基本数据(过滤后): " + taskBaseMap.keySet());

            // 3. 最后处理任务结果（高优先级，确保带后缀的图片键不被覆盖）
            if (!StringUtils.isEmpty(modelAssessTaskEntity.getTaskResult())) {
                try {
                    Map<String, Object> resultMap = objectMapper.readValue(
                            modelAssessTaskEntity.getTaskResult(),
                            Map.class
                    );

                    // 单独处理图片相关键，强制添加后缀
                    List<String> imageKeys = Arrays.asList("PR_curve", "confusion_matrix");
                    for (String key : imageKeys) {
                        Object value = resultMap.get(key);
                        if (value != null) {
                            String valueStr = value.toString();
                            // 生成带后缀的键（如PR_curve_1）
                            if (taskIndex != null && taskIndex >= 0) {
                                String suffixedKey = taskIndex == 0 ? key : key + "_" + taskIndex;
                                dataMap.put(suffixedKey, valueStr);
                                System.out.println("✅ 生成图片键: " + suffixedKey + " = " + valueStr);
                            } else {
                                dataMap.put(key, valueStr);
                            }
                        }
                    }

                    // 处理其他任务结果键
                    for (Map.Entry<String, Object> entry : resultMap.entrySet()) {
                        if (!imageKeys.contains(entry.getKey()) && entry.getValue() != null) {
                            String key = entry.getKey();
                            String valueStr = entry.getValue().toString();
                            if (taskIndex != null && taskIndex > 0) {
                                dataMap.put(key + "_" + taskIndex, valueStr);
                                dataMap.put(("taskName_" + taskIndex), modelAssessTaskEntity.getTaskName());
                            } else {
                                dataMap.put(key, valueStr);
                                taskName = modelAssessTaskEntity.getTaskName();
                            }
                        }
                    }
                    System.out.println("✅ 加载任务结果数据: " + resultMap.keySet());
                } catch (JsonProcessingException e) {
                    System.err.println("⚠️ 任务结果JSON解析失败：" + e.getMessage());
                }
            }

            // 处理PDF文件（保持不变）
            String pdfPath = modelBaseEntity != null ? modelBaseEntity.getApplyForPdf() : "";
            if (StringUtils.isNotEmpty(pdfPath) && new File(pdfPath).exists()) {
                System.out.println("📌 PDF文件存在，路径：" + pdfPath + "，开始预处理");
                Map<String, List<String>> pdfPlaceholders = processPdfToImages(pdfPath);
                for (Map.Entry<String, List<String>> entry : pdfPlaceholders.entrySet()) {
                    try {
                        dataMap.put(entry.getKey(), objectMapper.writeValueAsString(entry.getValue()));
                        System.out.println("✅ PDF占位符映射: " + entry.getKey() + " => " + entry.getValue());
                    } catch (JsonProcessingException e) {
                        System.err.println("⚠️ PDF占位符JSON序列化失败：" + e.getMessage());
                    }
                }
                dataMap.put("pdfPath", pdfPath);
            } else {
                dataMap.put("pdfPath", "");
                System.out.println("⚠️ PDF文件不存在或路径为空");
            }

        } catch (Exception e) {
            System.err.println("⚠️ 获取任务ID=" + taskId + " 动态数据失败：" + e.getMessage());
            e.printStackTrace();
        }

        System.out.println("📌 任务ID=" + taskId + " 最终数据Map包含图片键: " +
                dataMap.containsKey("PR_curve" + (taskIndex != null && taskIndex > 0 ? "_" + taskIndex : "")));
        return dataMap;
    }


    /**
     * 处理PDF文件并返回占位符与图片路径的映射
     */
    private Map<String, List<String>> processPdfToImages(String pdfPath) {
        Map<String, List<String>> result = new HashMap<>();
        try {
            File tempDir = new File(TEMP_IMAGE_DIR);
            if (!tempDir.exists()) {
                tempDir.mkdirs();
            }

            String baseFileName = System.currentTimeMillis() + "_" + new Random().nextInt(1000);
            List<String> tempImagePaths = new ArrayList<>();

            try (PDDocument document = loadPdfDocument(pdfPath)) {
                PDFRenderer renderer = new PDFRenderer(document);
                int pageCount = document.getNumberOfPages();

                for (int page = 0; page < pageCount; page++) {
                    String tempImagePath = TEMP_IMAGE_DIR + baseFileName + "_page" + (page + 1) + "." + IMAGE_FORMAT;
                    tempImagePaths.add(tempImagePath);

                    BufferedImage image = renderer.renderImageWithDPI(page, 300, ImageType.RGB);
                    ImageIO.write(image, IMAGE_FORMAT, new File(tempImagePath));
                }
            }

            // 创建唯一占位符
            String placeholderKey = PDF_IMAGE_PLACEHOLDER_PREFIX + UUID.randomUUID().toString();
            result.put(placeholderKey, tempImagePaths);

            // 同时保存占位符与原始PDF路径的映射，用于后续处理
            result.put("pdfPath_placeholder", Collections.singletonList(placeholderKey));

            System.out.println("【文件相关】PDF预处理完成，生成占位符：" + placeholderKey);
            return result;
        } catch (Exception e) {
            System.err.println("【文件相关】PDF处理异常：" + e.getMessage());
            result.put("error", Collections.singletonList("PDF处理失败"));
            return result;
        }
    }


    /**
     * 替换所有占位符（支持传入文档对象，不依赖成员变量）
     *
     * @param document 要处理的Word文档对象
     * @param dataMap  替换数据（key:占位符，value:替换值）
     */
    private void replaceAllPlaceholders(XWPFDocument document, Map<String, String> dataMap) {
        try {
            System.out.println("开始替换占位符，当前文档表格数: " + document.getTables().size());

            // 处理段落中的占位符
            for (XWPFParagraph paragraph : document.getParagraphs()) {
                replacePlaceholdersInParagraphWithImage(paragraph, dataMap, null);
            }

            // 处理表格中的占位符（修复：使用表格副本避免并发修改）
            List<XWPFTable> tablesCopy = new ArrayList<>(document.getTables());
            for (XWPFTable table : tablesCopy) {
                replacePlaceholdersInTable(table, dataMap);
            }
        } catch (Exception e) {
            System.err.println("替换占位符时发生错误：" + e.getMessage());
            e.printStackTrace();
        }
    }

    /**
     * 替换段落中的占位符，并处理图片/PDF插入
     */
    private void replacePlaceholdersInParagraphWithImage(XWPFParagraph paragraph, Map<String, String> dataMap, XWPFTableCell cell) {
        try {
            List<XWPFRun> runs = new ArrayList<>(paragraph.getRuns());
            if (runs.isEmpty()) return;

            StringBuilder fullText = new StringBuilder();
            for (XWPFRun run : runs) {
                String text = run.getText(0);
                if (text != null) fullText.append(text);
            }

            String originalText = fullText.toString();
            // 清空原有文本
            while (paragraph.getRuns().size() > 1) {
                paragraph.removeRun(1);
            }
            if (!paragraph.getRuns().isEmpty()) {
                paragraph.getRuns().get(0).setText("", 0);
            }

            // 处理图片/PDF占位符
            Matcher imgMatcher = Pattern.compile("\\{\\{@([a-zA-Z0-9_@.]+)\\}\\}").matcher(originalText);
            boolean hasImage = false;

            while (imgMatcher.find()) {
                hasImage = true;
                String imgKey = imgMatcher.group(1).trim();
                System.out.println("检查键是否存在：" + imgKey + " -> " + dataMap.containsKey(imgKey));
                String resourcePath = dataMap.get(imgKey);
                System.out.println("资源路径：" + resourcePath);

                System.out.println("匹配到图片占位符：" + imgKey);
                System.out.println("对应资源路径：" + resourcePath);

                if (resourcePath != null) {
                    // 处理PDF文件
                    if (imgKey.startsWith("pdfPath") && !resourcePath.isEmpty()) {
                        String pdfPlaceholderKeyJson = dataMap.get(imgKey + "_placeholder");
                        System.out.println("pdfPath_placeholder值：" + pdfPlaceholderKeyJson);

                        if (pdfPlaceholderKeyJson != null) {
                            try {
                                List<String> pdfPlaceholderKeys = objectMapper.readValue(pdfPlaceholderKeyJson, List.class);
                                if (!pdfPlaceholderKeys.isEmpty()) {
                                    String placeholderKey = pdfPlaceholderKeys.get(0);
                                    String imagePathsJson = dataMap.get(placeholderKey);
                                    System.out.println("PDF图片路径列表：" + imagePathsJson);

                                    if (imagePathsJson != null) {
                                        List<String> imagePaths = objectMapper.readValue(imagePathsJson, List.class);
                                        insertPdfImagesAtCurrentPosition(paragraph, imagePaths);
                                        continue;
                                    }
                                }
                            } catch (JsonProcessingException e) {
                                System.err.println("PDF占位符解析失败：" + e.getMessage());
                            }
                        }

                        // 备用逻辑
                        if (cell != null) {
                            boolean insertedSuccessfully = insertPdfAsImagesInCell(cell, resourcePath);
                            if (!insertedSuccessfully && cell.getTableRow() != null) {
                                XWPFTable table = cell.getTableRow().getTable();
                                int tablePos = currentDocument.getPosOfTable(table);
                                currentDocument.removeBodyElement(tablePos);
                                XWPFParagraph newPara = currentDocument.createParagraph();
                                insertPdfAsImagesInParagraph(newPara, resourcePath);
                            }
                        } else {
                            insertPdfAsImagesInParagraph(paragraph, resourcePath);
                        }
                    }
                    // 处理常规图片
                    else {
                        if (cell != null) {
                            System.out.println("尝试在单元格中插入图片：" + resourcePath);
                            insertImageInCell(cell, resourcePath);
                        } else {
                            System.out.println("尝试在段落中插入图片：" + resourcePath);
                            insertImageInParagraph(paragraph, resourcePath);
                        }
                    }
                } else {
                    System.out.println("未找到图片资源路径，使用空文本");
                    insertTextInElement(cell, paragraph, "");
                }
            }

            // 处理普通文本占位符
            if (!hasImage) {
                String replacedText = replacePlaceholders(originalText, dataMap);
                if (!paragraph.getRuns().isEmpty()) {
                    paragraph.getRuns().get(0).setText(replacedText, 0);
                }
            }
        } catch (Exception e) {
            System.err.println("处理段落占位符时发生错误：" + e.getMessage());
            e.printStackTrace();
            insertTextInElement(cell, paragraph, "-");
        }
    }

    /**
     * 在当前位置插入PDF图片
     */
    private void insertPdfImagesAtCurrentPosition(XWPFParagraph paragraph, List<String> imagePaths) {
        try {
            for (String imagePath : imagePaths) {
                insertImageInParagraph(paragraph, imagePath);
            }
        } catch (Exception e) {
            System.err.println("插入PDF图片时发生错误：" + e.getMessage());
            insertTextInElement(null, paragraph, "-");
        }
    }

    /**
     * 在段落（非表格段落）中插入 PDF 转换的图片
     */
    private void insertPdfAsImagesInParagraph(XWPFParagraph paragraph, String pdfPath) {
        try {
            if (pdfPath == null || !new File(pdfPath).exists()) {
                insertTextInElement(null, paragraph, "PDF文件不存在");
                return;
            }

            File tempDir = new File(TEMP_IMAGE_DIR);
            if (!tempDir.exists()) {
                tempDir.mkdirs();
            }

            String baseFileName = System.currentTimeMillis() + "_" + new Random().nextInt(1000);
            List<String> tempImagePaths = new ArrayList<>();

            try (PDDocument document = loadPdfDocument(pdfPath)) {
                PDFRenderer renderer = new PDFRenderer(document);
                int pageCount = document.getNumberOfPages();

                for (int page = 0; page < pageCount; page++) {
                    String tempImagePath = TEMP_IMAGE_DIR + baseFileName + "_page" + (page + 1) + "." + IMAGE_FORMAT;
                    tempImagePaths.add(tempImagePath);

                    BufferedImage image = renderer.renderImageWithDPI(page, 300, ImageType.RGB);
                    ImageIO.write(image, IMAGE_FORMAT, new File(tempImagePath));
                }
            }

            for (String imagePath : tempImagePaths) {
                insertImageInParagraph(paragraph, imagePath);
            }
        } catch (Exception e) {
            insertTextInElement(null, paragraph, "PDF处理失败: " + e.getMessage());
        }
    }

    /**
     * 在段落（非表格段落）中插入常规图片（增大尺寸并添加边框）
     */
    private void insertImageInParagraph(XWPFParagraph paragraph, String imagePath) {
        if (imagePath == null || imagePath.trim().isEmpty()) {
            insertTextInElement(null, paragraph, "-");
            return;
        }

        try {
            // 处理网络图片地址
            if (isUrl(imagePath)) {
                imagePath = downloadImageFromUrl(imagePath);
            }

            File imageFile = new File(imagePath);
            if (!imageFile.exists() || !imageFile.canRead()) {
                insertTextInElement(null, paragraph, "文件不可读: " + imagePath);
                return;
            }

            int imageWidthPoints = 350;
            int imageHeightPoints = 350;

            XWPFRun run = paragraph.createRun();

            try (InputStream imageStream = new FileInputStream(imageFile)) {
                byte[] bytes = IOUtils.toByteArray(imageStream);
                String ext = imagePath.substring(imagePath.lastIndexOf(".") + 1).toLowerCase();
                int pictureType;

                switch (ext) {
                    case "png":
                        pictureType = XWPFDocument.PICTURE_TYPE_PNG;
                        break;
                    case "jpg":
                    case "jpeg":
                        pictureType = XWPFDocument.PICTURE_TYPE_JPEG;
                        break;
                    case "gif":
                        pictureType = XWPFDocument.PICTURE_TYPE_GIF;
                        break;
                    case "bmp":
                        pictureType = XWPFDocument.PICTURE_TYPE_BMP;
                        break;
                    default:
                        throw new IOException("不支持的图片格式: " + ext);
                }

                run.addPicture(new ByteArrayInputStream(bytes), pictureType, "image." + ext,
                        Units.toEMU(imageWidthPoints), Units.toEMU(imageHeightPoints));
            }
        } catch (Exception e) {
            insertTextInElement(null, paragraph, "图片插入失败: " + e.getMessage());
        }
    }

    /**
     * 插入图片到指定的单元格，并等比缩放以填满整个单元格（增大尺寸并添加边框）
     */
    private void insertImageInCell(XWPFTableCell cell, String imagePath) {
        if (imagePath == null || imagePath.trim().isEmpty()) {
            insertTextInCell(cell, "-");
            return;
        }

        try {
            // 处理网络图片地址
            if (isUrl(imagePath)) {
                imagePath = downloadImageFromUrl(imagePath);
            }

            File imageFile = new File(imagePath);
            if (!imageFile.exists() || !imageFile.canRead()) {
                insertTextInCell(cell, "文件不可读: " + imagePath);
                return;
            }

            int cellWidthTwips = 12000;
            if (cell.getCTTc().isSetTcPr() && cell.getCTTc().getTcPr().isSetTcW()) {
                BigInteger widthBigInt = cell.getCTTc().getTcPr().getTcW().getW();
                if (widthBigInt != null) {
                    cellWidthTwips = widthBigInt.intValue();
                }
            }

            BufferedImage bufferedImage = ImageIO.read(imageFile);
            int imageWidth = bufferedImage.getWidth();
            int imageHeight = bufferedImage.getHeight();

            double aspectRatio = (double) imageHeight / imageWidth;
            int scaledHeightTwips = (int) (cellWidthTwips * aspectRatio * 0.95);

            int widthEMU = Math.round(cellWidthTwips * 635);
            int heightEMU = Math.round(scaledHeightTwips * 635);

            XWPFParagraph paragraph = cell.addParagraph();
            paragraph.setAlignment(ParagraphAlignment.CENTER);
            XWPFRun run = paragraph.createRun();

            try (InputStream imageStream = new FileInputStream(imageFile)) {
                byte[] bytes = IOUtils.toByteArray(imageStream);
                String ext = imagePath.substring(imagePath.lastIndexOf(".") + 1).toLowerCase();
                int pictureType;

                switch (ext) {
                    case "png":
                        pictureType = XWPFDocument.PICTURE_TYPE_PNG;
                        break;
                    case "jpg":
                    case "jpeg":
                        pictureType = XWPFDocument.PICTURE_TYPE_JPEG;
                        break;
                    case "gif":
                        pictureType = XWPFDocument.PICTURE_TYPE_GIF;
                        break;
                    case "bmp":
                        pictureType = XWPFDocument.PICTURE_TYPE_BMP;
                        break;
                    default:
                        throw new IOException("不支持的图片格式: " + ext);
                }

                run.addPicture(new ByteArrayInputStream(bytes), pictureType, "image." + ext, widthEMU, heightEMU);
            }
        } catch (Exception e) {
            insertTextInCell(cell, "图片插入失败: " + e.getMessage());
        }
    }

    /**
     * 通用方法，在单元格或段落中插入文本（适配 cell 和 paragraph 不同情况）
     */
    private static void insertTextInElement(XWPFTableCell cell, XWPFParagraph paragraph, String text) {
        try {
            if (cell != null) {
                insertTextInCell(cell, text);
            } else if (paragraph != null) {
                XWPFRun run = paragraph.createRun();
                run.setText(text);
            }
        } catch (Exception e) {
            // 即使在这里也确保不抛出异常
        }
    }

    /**
     * 使用 PDDocument 加载 PDF（适配 PDFBox 3.0.0）
     */
    private static PDDocument loadPdfDocument(String pdfPath) {
        try {
            return Loader.loadPDF(new File(pdfPath));
        } catch (IOException e) {
            System.err.println("加载PDF文件时发生错误：" + e.getMessage());
            return null;
        }
    }

    /**
     * 在单元格中插入PDF，并返回插入是否成功
     */
    private boolean insertPdfAsImagesInCell(XWPFTableCell cell, String pdfPath) {
        try {
            if (pdfPath == null || !new File(pdfPath).exists()) {
                System.err.println("【文件相关】PDF文件不存在，路径：" + pdfPath);
                insertTextInCell(cell, "PDF文件不存在");
                return false;
            }

            File tempDir = new File(TEMP_IMAGE_DIR);
            if (!tempDir.exists()) {
                System.out.println("【文件相关】临时目录不存在，创建临时目录：" + TEMP_IMAGE_DIR);
                tempDir.mkdirs();
            }

            String baseFileName = System.currentTimeMillis() + "_" + new Random().nextInt(1000);
            List<String> tempImagePaths = new ArrayList<>();

            try (PDDocument document = loadPdfDocument(pdfPath)) {
                if (document == null) {
                    insertTextInCell(cell, "PDF文件加载失败");
                    return false;
                }

                PDFRenderer renderer = new PDFRenderer(document);
                int pageCount = document.getNumberOfPages();

                if (pageCount > MAX_PAGES_PER_CELL) {
                    System.out.println("【文件相关】PDF页数过多(" + pageCount + "页)，无法放入单元格，将在其他位置插入");
                    return false;
                }

                for (int page = 0; page < pageCount; page++) {
                    String tempImagePath = TEMP_IMAGE_DIR + baseFileName + "_page" + (page + 1) + "." + IMAGE_FORMAT;
                    tempImagePaths.add(tempImagePath);

                    BufferedImage image = renderer.renderImageWithDPI(page, 300, ImageType.RGB);
                    ImageIO.write(image, IMAGE_FORMAT, new File(tempImagePath));
                }
            }

            int totalHeight = 0;
            for (String imagePath : tempImagePaths) {
                BufferedImage img = ImageIO.read(new File(imagePath));
                totalHeight += img.getHeight() * 15;
            }

            int cellHeight = getCellHeightInTwips(cell);
            if (totalHeight > cellHeight * MAX_HEIGHT_RATIO) {
                System.out.println("【文件相关】图片总高度(" + totalHeight + "twips)超过单元格高度(" + cellHeight + "twips)的" + MAX_HEIGHT_RATIO + "倍，将在其他位置插入");
                return false;
            }

            for (String imagePath : tempImagePaths) {
                insertImageInCell(cell, imagePath);
            }

            return true;
        } catch (Exception e) {
            System.err.println("【文件相关】PDF处理过程中发生未知异常");
            e.printStackTrace();
            insertTextInCell(cell, "PDF处理失败: " + e.getMessage());
            return false;
        }
    }

    /**
     * 在单元格中插入文本
     */
    private static void insertTextInCell(XWPFTableCell cell, String text) {
        try {
            XWPFParagraph p = cell.addParagraph();
            p.setAlignment(ParagraphAlignment.CENTER);
            XWPFRun run = p.createRun();
            run.setText(text);
        } catch (Exception e) {
            // 确保不抛出异常
        }
    }

    /**
     * 替换占位符核心逻辑
     */
    private static String replacePlaceholders(String text, Map<String, String> dataMap) {
        if (text == null || text.isEmpty()) return text;
        if (dataMap == null) return text; // 新增：处理dataMap为null的情况

        // 支持@、.、_等特殊字符的占位符（如{{mAP@0.5_1}}、{{@PR_curve_1}}）
        Pattern pattern = Pattern.compile("\\{\\{(@?[a-zA-Z0-9_@.]+?)\\}\\}");
        Matcher matcher = pattern.matcher(text);
        StringBuffer result = new StringBuffer();

        while (matcher.find()) {
            String key = matcher.group(1); // 先获取group(1)
            if (key == null) continue;     // 新增：处理group(1)为null的罕见情况

            key = key.trim();
            // 无对应值时返回"--"
            String replacement = dataMap.getOrDefault(key, "--");

            // 确保replacement不为null（虽然getOrDefault应该保证了这一点）
            replacement = replacement != null ? replacement : "--";

            try {
                // 使用quoteReplacement处理特殊字符
                matcher.appendReplacement(result, Matcher.quoteReplacement(replacement));
            } catch (Exception e) {
                // 记录异常，防止因单个替换失败导致整个过程中断
                System.err.println("Error replacing placeholder '" + key + "': " + e.getMessage());
                // 回退方案：直接使用原始占位符
                matcher.appendReplacement(result, "{{" + key + "}}");
            }
        }

        matcher.appendTail(result);
        return result.toString();
    }

    /**
     * 获取单元格高度（以twips为单位）
     */
    private static int getCellHeightInTwips(XWPFTableCell cell) {
        try {
            int defaultHeight = 1000;

            int estimatedHeight = 0;
            for (XWPFParagraph paragraph : cell.getParagraphs()) {
                estimatedHeight += estimateParagraphHeight(paragraph);
            }

            return Math.max(estimatedHeight, defaultHeight);
        } catch (Exception e) {
            return 1000;
        }
    }

    /**
     * 估算段落高度（以twips为单位）
     */
    private static int estimateParagraphHeight(XWPFParagraph paragraph) {
        try {
            int lineSpacing = 240;

            if (paragraph.getCTP().isSetPPr() && paragraph.getCTP().getPPr().isSetSpacing()) {
                org.openxmlformats.schemas.wordprocessingml.x2006.main.CTPPr pPr = paragraph.getCTP().getPPr();
                if (pPr.getSpacing().isSetLine()) {
                    lineSpacing = pPr.getSpacing().getLine().intValue();
                }
            }

            int lineCount = 1;
            String text = paragraph.getText();
            if (text != null) {
                lineCount = Math.max(1, (text.length() + 99) / 100);
            }

            return lineCount * lineSpacing;
        } catch (Exception e) {
            return 240;
        }
    }

    /**
     * 清理临时图片文件（包括本地临时文件和网络图片临时文件）
     */
    private void cleanTempImages() {
        try {
            File tempDir = new File(TEMP_IMAGE_DIR);
            if (tempDir.exists() && tempDir.isDirectory()) {
                File[] files = tempDir.listFiles();
                if (files != null) {
                    for (File file : files) {
                        // 只删除以network_开头的网络图片临时文件和PDF处理生成的临时文件
                        if (file.getName().startsWith(NETWORK_IMAGE_PREFIX) ||
                                file.getName().contains("_page") ||
                                file.getName().startsWith(PDF_IMAGE_PLACEHOLDER_PREFIX)) {
                            file.delete();
                        }
                    }
                }
            }
        } catch (Exception e) {
            System.err.println("清理临时文件时发生错误：" + e.getMessage());
        }
    }

    /**
     * 替换表格中的占位符（递归处理每个单元格的段落和嵌套表格）
     */
    private void replacePlaceholdersInTable(XWPFTable table, Map<String, String> dataMap) {
        try {
            // 使用安全的行遍历方式
            List<XWPFTableRow> rows = new ArrayList<>(table.getRows());
            for (int i = 0; i < rows.size(); i++) {
                XWPFTableRow row = rows.get(i);
                if (row == null) continue;

                // 使用安全的列遍历方式
                List<XWPFTableCell> cells = new ArrayList<>(row.getTableCells());
                for (int j = 0; j < cells.size(); j++) {
                    XWPFTableCell cell = cells.get(j);
                    if (cell == null) continue;

                    // 处理单元格中的段落
                    List<XWPFParagraph> paragraphs = new ArrayList<>(cell.getParagraphs());
                    for (XWPFParagraph paragraph : paragraphs) {
                        if (paragraph == null) continue;
                        replacePlaceholdersInParagraphWithImage(paragraph, dataMap, cell);
                    }

                    // 处理嵌套表格（修复：递归处理嵌套表格）
                    List<XWPFTable> nestedTables = new ArrayList<>(cell.getTables());
                    for (XWPFTable nestedTable : nestedTables) {
                        if (nestedTable != null) {
                            System.out.println("发现嵌套表格，继续处理...");
                            replacePlaceholdersInTable(nestedTable, dataMap);
                        }
                    }
                }
            }
        } catch (Exception e) {
            System.err.println("处理表格占位符时发生错误：" + e.getMessage());
            e.printStackTrace();
        }
    }


    /**
     * 检测路径是否为网络URL
     */
    private boolean isUrl(String path) {
        if (path == null || path.trim().isEmpty()) {
            return false;
        }
        String urlPattern = "^(https?|ftp|file)://[-a-zA-Z0-9+&@#/%?=~_|!:,.;]*[-a-zA-Z0-9+&@#/%=~_|]";
        return Pattern.matches(urlPattern, path);
    }

    /**
     * 从网络URL下载图片并保存为临时文件
     */
    private String downloadImageFromUrl(String imageUrl) throws IOException {
        if (!isUrl(imageUrl)) {
            return imageUrl; // 不是URL则直接返回原路径
        }

        // 生成临时文件名
        String fileName = NETWORK_IMAGE_PREFIX + System.currentTimeMillis() + "_" + new Random().nextInt(1000) + ".jpg";
        String tempFilePath = TEMP_IMAGE_DIR + fileName;

        // 确保临时目录存在
        File tempDir = new File(TEMP_IMAGE_DIR);
        if (!tempDir.exists()) {
            tempDir.mkdirs();
        }

        // 下载图片并保存（增加超时设置）
        URL url = new URL(imageUrl);
        HttpURLConnection connection = (HttpURLConnection) url.openConnection();
        connection.setConnectTimeout(5000); // 连接超时5秒
        connection.setReadTimeout(10000); // 读取超时10秒
        connection.setRequestMethod("GET");

        try (InputStream in = connection.getInputStream();
             OutputStream out = new FileOutputStream(tempFilePath)) {
            byte[] buffer = new byte[8192];
            int bytesRead;
            while ((bytesRead = in.read(buffer)) != -1) {
                out.write(buffer, 0, bytesRead);
            }
        } finally {
            // 断开连接
            if (connection != null) {
                connection.disconnect();
            }
        }

        return tempFilePath;
    }

    /**
     * 克隆表格
     *
     * @param table 要克隆的表格
     * @return 克隆后的表格
     */
    private XWPFTable cloneTable(XWPFTable table) {
        return cloneTable(table, currentDocument);
    }

    /**
     * 克隆表格并正确添加到文档
     *
     * @param sourceTable 要克隆的源表格
     * @param document    目标文档
     * @return 克隆并添加到文档后的新表格
     */
    private XWPFTable cloneTable(XWPFTable sourceTable, XWPFDocument document) {
        // 确保源表格至少有一行
        if (sourceTable.getRows().isEmpty()) {
            return document.createTable(0, 0);
        }

        // 获取源表格列数
        int columnCount = sourceTable.getRow(0).getTableCells().size();

        // 使用标准API创建新表格（关键修改点）
        XWPFTable newTable = document.createTable(sourceTable.getNumberOfRows(), columnCount);

        // 复制表格属性
        newTable.getCTTbl().setTblPr(sourceTable.getCTTbl().getTblPr());

        // 复制每个单元格的内容
        for (int i = 0; i < sourceTable.getRows().size(); i++) {
            XWPFTableRow sourceRow = sourceTable.getRow(i);
            XWPFTableRow targetRow = newTable.getRow(i);

            // 复制行属性
            if (sourceRow != null && sourceRow.getCtRow().isSetTrPr()) {
                targetRow.getCtRow().setTrPr(sourceRow.getCtRow().getTrPr());
            }

            // 确保目标行有足够的单元格
            while (targetRow.getTableCells().size() < columnCount) {
                targetRow.addNewTableCell();
            }

            int safeColumnCount = Math.min(
                    sourceRow != null ? sourceRow.getTableCells().size() : 0,
                    columnCount
            );

            for (int j = 0; j < safeColumnCount; j++) {
                XWPFTableCell sourceCell = sourceRow.getCell(j);
                XWPFTableCell targetCell = targetRow.getCell(j);

                // 复制单元格属性
                if (sourceCell != null && sourceCell.getCTTc().isSetTcPr()) {
                    targetCell.getCTTc().setTcPr(sourceCell.getCTTc().getTcPr());
                }

                // 清空目标单元格的默认段落（关键修改点）
                while (targetCell.getParagraphs().size() > 0) {
                    targetCell.removeParagraph(0);
                }

                // 复制源单元格的所有段落
                if (sourceCell != null) {
                    for (XWPFParagraph sourcePara : sourceCell.getParagraphs()) {
                        XWPFParagraph targetPara = targetCell.addParagraph();

                        // 复制段落属性
                        targetPara.getCTP().setPPr(sourcePara.getCTP().getPPr());

                        // 复制段落中的所有运行
                        for (XWPFRun sourceRun : sourcePara.getRuns()) {
                            XWPFRun targetRun = targetPara.createRun();

                            // 复制运行属性
                            if (sourceRun.getCTR().isSetRPr()) {
                                targetRun.getCTR().setRPr(sourceRun.getCTR().getRPr());
                            }

                            // 复制文本
                            targetRun.setText(sourceRun.getText(0));
                        }
                    }
                }
            }
        }

        return newTable;
    }


}