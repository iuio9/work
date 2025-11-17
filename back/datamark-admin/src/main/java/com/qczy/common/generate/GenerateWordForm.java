package com.qczy.common.generate;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
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

@Component
public class GenerateWordForm {

    @Autowired
    private ModelAssessTaskMapper modelAssessTaskMapper;
    @Autowired
    private ModelAssessConfigMapper modelAssessConfigMapper;
    @Autowired
    private ModelBaseMapper modelBaseMapper;
    @Autowired
    private ModelConfigureMapper modelConfigureMapper;

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
    private static final String TASK_TEMPLATE = "111222.docx";
    // 测试指标 Word 模板文件名（放在 resources 目录）
    private static final String TEST_INDICATOR_TEMPLATE = "templates/avgIndexTemplate.docx";

    // 持有当前处理的文档引用
    private XWPFDocument currentDocument;

    public void downloadWord(Integer taskId, HttpServletRequest request, HttpServletResponse response) {
        try {
            // 1. 加载模板文件
            InputStream is = GenerateWordForm.class.getClassLoader().getResourceAsStream(TASK_TEMPLATE);
            if (is == null) {
                response.sendError(HttpServletResponse.SC_NOT_FOUND, "模板文件未找到");
                return;
            }

            currentDocument = new XWPFDocument(is);

            // 2. 准备数据，提前处理PDF
            Map<String, String> dataMap = getDynamicData(taskId);

            // 3. 替换占位符（包括PDF占位符）
            replaceAllPlaceholders(dataMap);

            // 4. 设置响应头
            response.setContentType("application/vnd.openxmlformats-officedocument.wordprocessingml.document");
            response.setCharacterEncoding("UTF-8");

            // 5. 生成文件名并处理中文编码
            String fileName = "测评报告.docx";
            String userAgent = request.getHeader("User-Agent");
            String encodedFileName;

            if (userAgent != null && userAgent.matches("(?i).*(MSIE|Trident|Edge).*")) {
                // IE/Edge浏览器
                encodedFileName = URLEncoder.encode(fileName, "UTF-8").replace("+", "%20");
            } else {
                // 其他现代浏览器
                encodedFileName = new String(fileName.getBytes("UTF-8"), "ISO-8859-1");
            }

            response.setHeader("Content-Disposition", "attachment; filename=\"" + encodedFileName + "\"");

            // 6. 强制清除缓存头
            response.setHeader("Cache-Control", "no-cache, no-store, must-revalidate");
            response.setHeader("Pragma", "no-cache");
            response.setDateHeader("Expires", 0);

            // 7. 将文档内容写入响应输出流
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
            // 清理临时文件
            cleanTempImages();
        }
    }


    public void generateToFile(Integer taskId, File outputFile) throws Exception {
        InputStream is = null;
        XWPFDocument document = null;
        try {
            // 1. 加载模板文件
            String templateName = "111222.docx";
            is = GenerateWordForm.class.getClassLoader().getResourceAsStream(templateName);
            if (is == null) {
                throw new FileNotFoundException("模板文件未找到: " + templateName);
            }

            document = new XWPFDocument(is);

            // 2. 准备数据，提前处理PDF
            Map<String, String> dataMap = getDynamicData(taskId);

            // 3. 替换占位符（包括PDF占位符）
            replaceAllPlaceholders(document, dataMap); // 注意：这里假设原方法是静态的，或改为实例方法

            // 4. 确保输出目录存在
            File parentDir = outputFile.getParentFile();
            if (parentDir != null && !parentDir.exists()) {
                if (!parentDir.mkdirs()) {
                    throw new IOException("无法创建输出目录: " + parentDir.getAbsolutePath());
                }
            }

            // 5. 将文档内容写入指定文件
            try (FileOutputStream fos = new FileOutputStream(outputFile)) {
                document.write(fos);
                fos.flush();
            }

        } catch (Exception e) {
        /*    System.err.println("❌ 文档处理过程中发生错误：");
            e.printStackTrace();
            throw new Exception("生成文档失败", e);*/
            System.out.println("❌ 文档处理过程中发生错误：" + e.getMessage());
        } finally {
            // 关闭资源
            if (is != null) {
                try {
                    is.close();
                } catch (IOException e) {
                    //e.printStackTrace();
                    System.out.println("❌ 文档处理过程中发生错误：" + e.getMessage());
                }
            }
            // 清理临时文件
            cleanTempImages();
        }
    }

    /**
     * 生成测试指标 Word 文件到指定路径
     */
    public void generateTestIndicatorWord(File outputFile, Map<String, String> dataMap) throws IOException {
        // 1. 加载测试指标模板
        try (InputStream is = GenerateWordForm.class.getClassLoader().getResourceAsStream(TEST_INDICATOR_TEMPLATE)) {
            if (is == null) {
                throw new IOException("测试指标模板文件未找到: " + TEST_INDICATOR_TEMPLATE);
            }
            XWPFDocument document = new XWPFDocument(is);

            // 2. 替换占位符（使用统一的 replacePlaceholders 方法）
            replaceAllPlaceholders(document, dataMap);

            // 3. 写入文件
            try (FileOutputStream fos = new FileOutputStream(outputFile)) {
                document.write(fos);
            }
        }
    }


    /**
     * 获取动态数据，提前处理PDF文件
     */
    private Map<String, String> getDynamicData(Integer taskId) {
        Map<String, String> dataMap = new HashMap<>();
        try {
            ModelAssessTaskEntity modelAssessTaskEntity = modelAssessTaskMapper.selectById(taskId);

            if (modelAssessTaskEntity == null || modelAssessTaskEntity.getModelBaseId() == null) {
                return dataMap;
            }

            dataMap.putAll(EntityUtils.convertToMap(modelAssessTaskEntity));

            if (!StringUtils.isEmpty(modelAssessTaskEntity.getTaskResult())) {
                try {
                    dataMap.putAll(objectMapper.readValue(modelAssessTaskEntity.getTaskResult(), Map.class));
                } catch (JsonProcessingException e) {
                    System.err.println("任务结果JSON解析失败，使用默认值");
                    dataMap.put("taskResult", "-");
                }
            }

            ModelBaseEntity modelBaseEntity = modelBaseMapper.selectOne(
                    new LambdaQueryWrapper<ModelBaseEntity>()
                            .eq(ModelBaseEntity::getId, modelAssessTaskEntity.getModelBaseId())
            );
            if (modelBaseEntity != null) {
                dataMap.putAll(EntityUtils.convertToMap(modelBaseEntity));
                // 修改一下模型名称
                if (!StringUtils.isEmpty(modelBaseEntity.getModelName())) {
                    boolean endsWith = modelBaseEntity.getModelName().trim().endsWith("模型");
                    dataMap.put("modelName", endsWith ? modelBaseEntity.getModelName() : modelBaseEntity.getModelName() + "模型");
                }

                ModelConfigureEntity modelConfigureEntity = modelConfigureMapper.selectOne(
                        new LambdaQueryWrapper<ModelConfigureEntity>()
                                .eq(ModelConfigureEntity::getModelBaseId, modelBaseEntity.getId())
                );
                if (modelConfigureEntity != null) {
                    dataMap.putAll(EntityUtils.convertToMap(modelConfigureEntity));
                    if (!StringUtils.isEmpty(modelConfigureEntity.getModelInterfaceDesc())) {
                        dataMap.put("v1", "详见附表1");
                    }
                    if (!StringUtils.isEmpty(modelConfigureEntity.getModelCase())) {
                        dataMap.put("v2", "详见附表2");
                    }
                }

            }

            // 处理PDF文件
            String pdfPath = modelBaseEntity != null ? modelBaseEntity.getApplyForPdf() : "";
            if (StringUtils.isNotEmpty(pdfPath) && new File(pdfPath).exists()) {
                System.out.println("【文件相关】PDF文件存在，路径：" + pdfPath + "，开始预处理");

                // 生成PDF图片并获取占位符映射
                Map<String, List<String>> pdfPlaceholders = processPdfToImages(pdfPath);

                // 将占位符映射存入数据Map
                for (Map.Entry<String, List<String>> entry : pdfPlaceholders.entrySet()) {
                    try {
                        dataMap.put(entry.getKey(), objectMapper.writeValueAsString(entry.getValue()));
                    } catch (JsonProcessingException e) {
                        System.err.println("【文件相关】PDF占位符JSON序列化失败：" + e.getMessage());
                        dataMap.put(entry.getKey(), "");
                    }
                }

                dataMap.put("pdfPath", pdfPath);
            } else {
                dataMap.put("pdfPath", "");
            }

        } catch (Exception e) {
            System.err.println("获取动态数据时发生错误：" + e.getMessage());
            dataMap.put("error", "数据获取失败");
        }

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
     * 替换所有占位符，包括PDF占位符
     */
    private void replaceAllPlaceholders(Map<String, String> dataMap) {
        try {
            // 先处理段落中的占位符
            for (XWPFParagraph paragraph : currentDocument.getParagraphs()) {
                replacePlaceholdersInParagraphWithImage(paragraph, dataMap, null);
            }

            // 再处理表格中的占位符
            List<XWPFTable> tablesCopy = new ArrayList<>(currentDocument.getTables());
            for (XWPFTable table : tablesCopy) {
                replacePlaceholdersInTable(table, dataMap);
            }
        } catch (Exception e) {
            System.err.println("替换占位符时发生错误：" + e.getMessage());
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
            // 处理段落中的占位符（将原currentDocument改为传入的document）
            for (XWPFParagraph paragraph : document.getParagraphs()) {
                replacePlaceholdersInParagraphWithImage(paragraph, dataMap, null);
            }

            // 处理表格中的占位符（同理替换为document）
            List<XWPFTable> tablesCopy = new ArrayList<>(document.getTables());
            for (XWPFTable table : tablesCopy) {
                replacePlaceholdersInTable(table, dataMap);
            }
        } catch (Exception e) {
            System.err.println("替换占位符时发生错误：" + e.getMessage());
            //e.printStackTrace(); // 增加堆栈信息，便于调试
        }
    }

    /**
     * 替换段落中的占位符，并处理图片/PDF插入
     */
    private void replacePlaceholdersInParagraphWithImage(XWPFParagraph paragraph, Map<String, String> dataMap, XWPFTableCell cell) {
        try {
            List<XWPFRun> runs = paragraph.getRuns();
            if (runs == null || runs.isEmpty()) return;

            StringBuilder fullText = new StringBuilder();
            for (XWPFRun run : runs) {
                String text = run.getText(0);
                if (text != null) fullText.append(text);
            }

            String originalText = fullText.toString();

            // 清空原有文本
            while (runs.size() > 1) {
                paragraph.removeRun(1);
            }
            if (!runs.isEmpty()) {
                runs.get(0).setText("", 0);
            }

            // 处理图片/PDF占位符 {{@key}}
            Matcher imgMatcher = Pattern.compile("\\{\\{@(.*?)\\}\\}").matcher(originalText);
            boolean hasImage = false;

            while (imgMatcher.find()) {
                hasImage = true;
                String imgKey = imgMatcher.group(1).trim();
                String resourcePath = dataMap.get(imgKey);

                if (resourcePath != null) {
                    // 处理PDF文件
                    if (imgKey.equals("pdfPath") && !resourcePath.isEmpty()) {
                        // 获取PDF占位符
                        String pdfPlaceholderKeyJson = dataMap.get("pdfPath_placeholder");
                        if (pdfPlaceholderKeyJson != null) {
                            try {
                                List<String> pdfPlaceholderKeys = objectMapper.readValue(pdfPlaceholderKeyJson, List.class);
                                if (!pdfPlaceholderKeys.isEmpty()) {
                                    String placeholderKey = pdfPlaceholderKeys.get(0);
                                    // 获取PDF图片路径列表
                                    String imagePathsJson = dataMap.get(placeholderKey);
                                    if (imagePathsJson != null) {
                                        List<String> imagePaths = objectMapper.readValue(imagePathsJson, List.class);
                                        // 在当前位置插入PDF图片
                                        insertPdfImagesAtCurrentPosition(paragraph, imagePaths);
                                        continue;
                                    }
                                }
                            } catch (JsonProcessingException e) {
                                System.err.println("【文件相关】PDF占位符解析失败：" + e.getMessage());
                            }
                        }

                        // 如果占位符处理失败，则使用原始方法
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
                            insertImageInCell(cell, resourcePath);
                        } else {
                            insertImageInParagraph(paragraph, resourcePath);
                        }
                    }
                } else {
                    insertTextInElement(cell, paragraph, "");
                }
            }

            // 如果没有图片占位符，则替换普通文本占位符
            if (!hasImage) {
                String replacedText = replacePlaceholders(originalText, dataMap);
                if (!runs.isEmpty()) {
                    runs.get(0).setText(replacedText, 0);
                }
            }
        } catch (Exception e) {
            System.err.println("处理段落占位符时发生错误：" + e.getMessage());
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

        try {
            StringBuilder result = new StringBuilder();
            int lastEnd = 0;
            Matcher matcher = Pattern.compile("\\{\\{([^{}]+)\\}\\}").matcher(text);

            while (matcher.find()) {
                result.append(text, lastEnd, matcher.start());
                String key = matcher.group(1).trim();
                Object replacement = dataMap.getOrDefault(key, "");
                result.append(replacement);
                lastEnd = matcher.end();
            }

            result.append(text.substring(lastEnd));
            return result.toString();
        } catch (Exception e) {
            System.err.println("替换占位符时发生错误：" + e.getMessage());
            return text;
        }
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
            for (XWPFTableRow row : table.getRows()) {
                for (XWPFTableCell cell : row.getTableCells()) {
                    List<XWPFParagraph> paragraphs = new ArrayList<>(cell.getParagraphs());
                    for (XWPFParagraph paragraph : paragraphs) {
                        replacePlaceholdersInParagraphWithImage(paragraph, dataMap, cell);
                    }

                    List<XWPFTable> nestedTables = new ArrayList<>(cell.getTables());
                    for (XWPFTable nestedTable : nestedTables) {
                        replacePlaceholdersInTable(nestedTable, dataMap);
                    }
                }
            }
        } catch (Exception e) {
            System.err.println("处理表格占位符时发生错误：" + e.getMessage());
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
}