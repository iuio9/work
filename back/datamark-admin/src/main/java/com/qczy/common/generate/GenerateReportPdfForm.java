package com.qczy.common.generate;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.itextpdf.text.*;
import com.itextpdf.text.pdf.*;
import com.qczy.mapper.*;
import com.qczy.model.entity.ModelAssessConfigEntity;
import com.qczy.model.entity.ModelAssessTaskEntity;
import com.qczy.model.entity.ModelBaseEntity;
import com.qczy.model.entity.ModelConfigureEntity;
import com.qczy.utils.StringUtils;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;

import javax.servlet.ServletOutputStream;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import java.io.*;
import java.lang.reflect.Field;
import java.net.HttpURLConnection;
import java.net.MalformedURLException;
import java.net.URL;
import java.net.URLEncoder;
import java.text.SimpleDateFormat;
import java.util.*;

@Component
public class GenerateReportPdfForm {

    @Autowired
    private ModelBaseMapper modelBaseMapper;
    @Autowired
    private ModelConfigureMapper modelConfigureMapper;
    @Autowired
    private ModelAssessTaskMapper modelAssessTaskMapper;
    @Autowired
    private ModelAssessConfigMapper modelAssessConfigMapper;
    @Autowired
    private UserMapper userMapper;
    @Value("${upload.address}")
    private String uploadAddress;

    // JSON解析器
    private static final ObjectMapper objectMapper = new ObjectMapper();

    public void downloadPdf(Integer taskId, HttpServletRequest request, HttpServletResponse response) {
        try {
            ModelAssessTaskEntity modelAssessTaskEntity = modelAssessTaskMapper.selectById(taskId);
            if (modelAssessTaskEntity == null) {
                response.sendError(HttpServletResponse.SC_NOT_FOUND, "任务对象不存在");
                return;
            }
            ModelAssessConfigEntity modelAssessConfigEntity = modelAssessConfigMapper.selectOne(
                    new LambdaQueryWrapper<ModelAssessConfigEntity>()
                            .eq(ModelAssessConfigEntity::getAssessTaskId, modelAssessTaskEntity.getId())
            );
            if (modelAssessConfigEntity == null) {
                response.sendError(HttpServletResponse.SC_NOT_FOUND, "任务配置对象不存在");
                return;
            }

            ModelBaseEntity modelBaseEntity = modelBaseMapper.selectById(modelAssessTaskEntity.getModelBaseId());
            if (modelBaseEntity == null) {
                response.sendError(HttpServletResponse.SC_NOT_FOUND, "模型对象不存在");
                return;
            }

            response.setContentType("application/pdf");
            response.setCharacterEncoding("UTF-8");

            String fileName = "model_" + modelBaseEntity.getApplyForNum() + ".pdf";

            String userAgent = request.getHeader("User-Agent");
            String encodedFileName;
            if (userAgent != null && userAgent.matches("(?i).*(MSIE|Trident|Edge).*")) {
                encodedFileName = URLEncoder.encode(fileName, "UTF-8").replace("+", "%20");
            } else {
                encodedFileName = new String(fileName.getBytes("UTF-8"), "ISO-8859-1");
            }

            response.setHeader("Content-Disposition", "attachment; filename=\"" + encodedFileName + "\"");
            response.setHeader("Cache-Control", "no-cache, no-store, must-revalidate");
            response.setHeader("Pragma", "no-cache");
            response.setDateHeader("Expires", 0);

            try (ServletOutputStream out = response.getOutputStream()) {
                generatePad(modelBaseEntity, modelAssessTaskEntity, modelAssessConfigEntity, out);
                out.flush();
            }
        } catch (Exception e) {
            response.setStatus(HttpServletResponse.SC_INTERNAL_SERVER_ERROR);
        }
    }

    public void generatePad(ModelBaseEntity modelBaseEntity, ModelAssessTaskEntity modelAssessTaskEntity,
                            ModelAssessConfigEntity modelAssessConfigEntity, OutputStream outputStream) {
        ModelConfigureEntity modelConfigureEntity = modelConfigureMapper.selectOne(
                new LambdaQueryWrapper<ModelConfigureEntity>()
                        .eq(ModelConfigureEntity::getModelBaseId, modelBaseEntity.getId())
        );

        if (modelConfigureEntity == null) {
            return;
        }

        Document document = new Document(PageSize.A4);
        try {
            PdfWriter writer = PdfWriter.getInstance(document, outputStream);
            writer.setPageEvent(new PageNumberEvent());

            document.open();

            String fontPath = "/font/NotoSansSC-Regular.ttf";
            BaseFont bfChinese = BaseFont.createFont(fontPath, BaseFont.IDENTITY_H, BaseFont.EMBEDDED);
            Font titleFont = new Font(bfChinese, 18, Font.BOLD);
            Font normalFont = new Font(bfChinese, 12);
            Font sectionFont = new Font(bfChinese, 14, Font.BOLD);
            Font imageTitleFont = new Font(bfChinese, 12, Font.BOLD);
            Font imageSubtitleFont = new Font(bfChinese, 11, Font.BOLD);

            Paragraph title = new Paragraph("模型评估测试报告", titleFont);
            title.setAlignment(Element.ALIGN_CENTER);
            document.add(title);
            document.add(Chunk.NEWLINE);

            addSectionTitle(document, "一、模型基本信息", sectionFont);
            PdfPTable mainTable = createMainTable();
            fillMainTable(mainTable, modelBaseEntity, normalFont);
            document.add(mainTable);

            document.newPage();
            addSectionTitle(document, "二、模型准备工作", sectionFont);
            PdfPTable testTable = createTestTable();
            fillTestTable(testTable, modelConfigureEntity, normalFont, uploadAddress);
            document.add(testTable);

            Paragraph space = new Paragraph(" ");
            space.setLeading(20f);
            document.add(space);

            addSectionTitle(document, "三、任务信息", sectionFont);
            PdfPTable taskTable = createTaskTable();
            fillTaskTable(taskTable, modelAssessTaskEntity, modelAssessConfigEntity, normalFont);
            document.add(taskTable);

            document.newPage();
            addSectionTitle(document, "四、测试指标", sectionFont);
            PdfPTable metricsTable = createMetricsTable();
            // 解析任务结果中的JSON指标数据
            Map<String, Object> metricsData = new HashMap<>();
            String taskResult = modelAssessTaskEntity.getTaskResult();
            if (StringUtils.isNotEmpty(taskResult)) {
                try {
                    // 使用Jackson解析JSON字符串为Map
                    metricsData = objectMapper.readValue(taskResult, Map.class);
                } catch (IOException e) {
                    e.printStackTrace();
                    // 解析失败时使用默认值
                    metricsData.put("mPrecision", "-");
                    metricsData.put("mRecall", "-");
                    metricsData.put("mAP@0.5", "-");
                    metricsData.put("MissRate", "-");
                    metricsData.put("FalseAlarmRate", "-");
                    metricsData.put("mAccuracy", "-");
                    metricsData.put("PR_curve", null);
                    metricsData.put("confusion_matrix", null);
                }
            } else {
                // 任务结果为空时使用默认值
                metricsData.put("mPrecision", "-");
                metricsData.put("mRecall", "-");
                metricsData.put("mAP@0.5", "-");
                metricsData.put("MissRate", "-");
                metricsData.put("FalseAlarmRate", "-");
                metricsData.put("mAccuracy", "-");
                metricsData.put("PR_curve", null);
                metricsData.put("confusion_matrix", null);
            }

            // 填充测试指标表格
            fillMetricsTable(metricsTable, normalFont, metricsData);
            document.add(metricsTable);

            // 添加图片部分
            addImagesSection(document, metricsData, uploadAddress, imageSubtitleFont);

            document.close();
        } catch (Exception e) {
            e.printStackTrace();
            try {
                outputStream.close();
            } catch (IOException ex) {
                ex.printStackTrace();
            }
        }
    }

    private PdfPTable createMainTable() throws DocumentException {
        PdfPTable table = new PdfPTable(3);
        table.setWidthPercentage(90);
        table.setHorizontalAlignment(Element.ALIGN_CENTER);
        table.setWidths(new float[]{0.7f, 1.5f, 7.8f});
        return table;
    }

    private void fillMainTable(PdfPTable table, ModelBaseEntity entity, Font font) {
        addRow(table, "1", "申请单号", entity.getApplyForNum(), font);
        addRow(table, "2", "申请日期", formatDate(entity.getApplyForDate()), font);
        addRow(table, "3", "模型名称", entity.getModelName(), font);
        addRow(table, "4", "模型来源", entity.getModelSource(), font);
        addRow(table, "5", "模型类型", entity.getModelType(), font);
        addRow(table, "6", "模型功能", entity.getModelFunction(), font);

        addSimplifiedUnitRow(table, "7", "建设单位", entity, font, "build");
        addSimplifiedUnitRow(table, "8", "承建单位", entity, font, "bt");
    }

    private void addSimplifiedUnitRow(PdfPTable table, String num, String title, ModelBaseEntity entity, Font font, String prefix) {
        table.addCell(createCell(num, font));
        table.addCell(createCell(title, font));

        PdfPCell complexCell = new PdfPCell();
        PdfPTable subTable = new PdfPTable(2);
        try {
            subTable.setWidths(new float[]{2f, 8f});
        } catch (DocumentException e) {
            throw new RuntimeException(e);
        }
        subTable.getDefaultCell().setBorderWidth(0);

        subTable.addCell(createCell("单位名称", font));
        subTable.addCell(createCell(getValue(entity, prefix + "UnitName"), font));
        subTable.addCell(createCell("单位地址", font));
        subTable.addCell(createCell(getValue(entity, prefix + "UnitAddress"), font));
        subTable.addCell(createCell("联系人", font));
        subTable.addCell(createCell(getValue(entity, prefix + "UnitLeader"), font));
        subTable.addCell(createCell("联系电话", font));
        subTable.addCell(createCell(getValue(entity, prefix + "UnitContact"), font));

        complexCell.addElement(subTable);
        complexCell.setPadding(0);
        table.addCell(complexCell);
    }

    private PdfPTable createTestTable() throws DocumentException {
        PdfPTable table = new PdfPTable(3);
        table.setWidthPercentage(90);
        table.setHorizontalAlignment(Element.ALIGN_CENTER);
        table.setWidths(new float[]{0.7f, 2.3f, 7f});
        return table;
    }

    private void fillTestTable(PdfPTable table, ModelConfigureEntity entity, Font font, String uploadAddress) {
        addRow(table, "1", "文件模型名", entity.getModelFileName(), font);
        addRow(table, "2", "模型封装方式", entity.getModelEncapWay(), font);
        addRow(table, "3", "模型文件大小", entity.getModelFileSize(), font);
        addRow(table, "4", "模型部署位置", entity.getModelDeployAddr(), font);
        addRow(table, "5", "模型API接口说明", getUrlValue(entity.getModelInterfaceDesc(), uploadAddress), font);
        addRow(table, "6", "模型对外暴露端口", entity.getModelPort(), font);
        addRow(table, "7", "模型cuda版本", entity.getModelCudaVersion(), font);
        addRow(table, "8", "模型驱动版本", entity.getModelDriveVersion(), font);
        addRow(table, "9", "模型调用例", getUrlValue(entity.getModelCase(), uploadAddress), font);
    }

    private PdfPTable createTaskTable() throws DocumentException {
        PdfPTable table = new PdfPTable(3);
        table.setWidthPercentage(90);
        table.setHorizontalAlignment(Element.ALIGN_CENTER);
        table.setWidths(new float[]{0.7f, 2.3f, 7f});
        return table;
    }

    private void fillTaskTable(PdfPTable table, ModelAssessTaskEntity taskEntity,
                               ModelAssessConfigEntity configEntity, Font font) {
        addRow(table, "1", "任务名称", taskEntity.getTaskName(), font);
        addRow(table, "2", "任务类型", taskEntity.getTaskType() == 1 ? "测试" : "评估", font);
        addRow(table, "3", "任务描述", taskEntity.getTaskDesc(), font);
        addRow(table, "4", "任务版本", taskEntity.getTaskVersion(), font);
        addRow(table, "5", "版本描述", taskEntity.getVersionDesc(), font);
        addRow(table, "6", "模型接口地址", configEntity.getModelAddress(), font);
        addRow(table, "7", "模型传输方式", "POST", font);
        addRow(table, "8", "评估描述", configEntity.getAssessDesc(), font);
    }

    private PdfPTable createMetricsTable() throws DocumentException {
        // 3 列：序号、指标名称、指标值
        PdfPTable table = new PdfPTable(3);
        table.setWidthPercentage(90);
        table.setHorizontalAlignment(Element.ALIGN_CENTER);
        table.setWidths(new float[]{0.5f, 2f, 3f});
        return table;
    }

    private void fillMetricsTable(PdfPTable table, Font font, Map<String, Object> metricsData) {
        // 表头
        addHeaderRow(table, font, "序号", "指标名称", "指标值");

        // 按实际指标填充，确保 null 值显示为 "-"
        addRow(table, "1", "平均精度（mPrecision）",
                getMetricValue(metricsData.get("mPrecision")), font);
        addRow(table, "2", "平均召回率（mRecall）",
                getMetricValue(metricsData.get("mRecall")), font);
        addRow(table, "3", "均值平均精度（mAP@0.5）",
                getMetricValue(metricsData.get("mAP@0.5")), font);
        addRow(table, "4", "漏检率（MissRate）",
                getMetricValue(metricsData.get("MissRate")), font);
        addRow(table, "5", "虚警率（FalseAlarmRate）",
                getMetricValue(metricsData.get("FalseAlarmRate")), font);
        addRow(table, "6", "平均正确率（mAccuracy）",
                getMetricValue(metricsData.get("mAccuracy")), font);
    }

    // 辅助方法：获取指标值，处理 null 情况
    private String getMetricValue(Object value) {
        return value == null ? "-" : String.valueOf(value);
    }

    private void addHeaderRow(PdfPTable table, Font font, String... headers) {
        for (String header : headers) {
            PdfPCell cell = createCell(header, font);
            cell.setBackgroundColor(BaseColor.LIGHT_GRAY);
            cell.setHorizontalAlignment(Element.ALIGN_CENTER);
            table.addCell(cell);
        }
    }

    private void addRow(PdfPTable table, String num, String key, String value, Font font) {
        table.addCell(createCell(num, font));
        table.addCell(createCell(key, font));
        table.addCell(createCell(value, font));
    }

    private String formatDate(Date date) {
        return date != null ? new SimpleDateFormat("yyyy-MM-dd").format(date) : "-";
    }

    private String getValue(ModelBaseEntity entity, String field) {
        if (entity == null) {
            return "-";
        }
        try {
            Field fieldObj = entity.getClass().getDeclaredField(field);
            fieldObj.setAccessible(true);
            return (String) fieldObj.get(entity);
        } catch (Exception e) {
            return "-";
        }
    }

    private String getUrlValue(String value, String uploadAddress) {
        return StringUtils.isEmpty(value) ? "-" : uploadAddress + value;
    }

    private static void addSectionTitle(Document document, String title, Font font) throws DocumentException {
        Paragraph sectionTitle = new Paragraph(title, font);
        sectionTitle.setAlignment(Element.ALIGN_LEFT);

        PdfPTable titleContainer = new PdfPTable(1);
        titleContainer.setWidthPercentage(90);
        titleContainer.setHorizontalAlignment(Element.ALIGN_CENTER);
        titleContainer.getDefaultCell().setBorder(Rectangle.NO_BORDER);
        titleContainer.addCell(sectionTitle);

        document.add(titleContainer);
        Paragraph space = new Paragraph(" ");
        space.setLeading(10f);
        document.add(space);
    }

    private static PdfPCell createCell(String text, Font font) {
        PdfPCell cell = new PdfPCell(new Phrase(StringUtils.isEmpty(text) ? "-" : text, font));
        cell.setPadding(4);
        cell.setVerticalAlignment(Element.ALIGN_MIDDLE);
        cell.setMinimumHeight(25);
        return cell;
    }

    /**
     * 添加图片展示部分（优化版）
     */
    private void addImagesSection(Document document, Map<String, Object> metricsData, String uploadAddress, Font subtitleFont)
            throws DocumentException, IOException {
        // 移除大标题，直接添加图片小标题和内容
        String prCurvePath = getMetricValue(metricsData.get("PR_curve"));
        String confusionMatrixPath = getMetricValue(metricsData.get("confusion_matrix"));

        boolean hasImages = false;

        // 添加PR曲线图片（仅当路径有效时）
        if (!prCurvePath.equals("-")) {
            hasImages = true;
            addImageWithSubtitle(document, "PR曲线", prCurvePath, uploadAddress, subtitleFont);
            document.add(new Paragraph(" ", subtitleFont)); // 图片间空行
        }

        // 添加混淆矩阵图片（仅当路径有效时）
        if (!confusionMatrixPath.equals("-")) {
            hasImages = true;
            addImageWithSubtitle(document, "混淆矩阵", confusionMatrixPath, uploadAddress, subtitleFont);
        }

        // 如果没有图片，添加提示信息
        if (!hasImages) {
            Paragraph noImageText = new Paragraph("未找到评估结果图表", subtitleFont);
            noImageText.setAlignment(Element.ALIGN_CENTER);
            document.add(noImageText);
        }
    }

    /**
     * 添加带小标题的图片（隐藏路径，等比例缩放）
     */
    private void addImageWithSubtitle(Document document, String subtitle, String imagePath, String uploadAddress, Font subtitleFont)
            throws DocumentException, IOException {
        if (StringUtils.isEmpty(imagePath) || imagePath.equals("-")) {
            return; // 路径为空时不添加任何内容
        }

        try {
            // 处理URL中的换行符和空格
            String fullPath = imagePath.trim().replaceAll("\\s+", "");

            // 加载图片
            Image image = loadImageFromHttp(fullPath);
            if (image != null) {
                // 添加小标题
                Paragraph subTitle = new Paragraph(subtitle, subtitleFont);
                subTitle.setAlignment(Element.ALIGN_CENTER);
                document.add(subTitle);

                // 调整图片大小以适应页面（等比例缩放）
                float maxWidth = PageSize.A4.getWidth() - 50; // 最大宽度，留出边距
                float imgWidth = image.getWidth();
                float imgHeight = image.getHeight();

                // 计算缩放比例，保持宽高比
                float scaleRatio = 1.0f;
                if (imgWidth > maxWidth) {
                    scaleRatio = maxWidth / imgWidth;
                }

                // 应用缩放
                image.scalePercent(scaleRatio * 100);

                // 居中显示图片
                image.setAlignment(Element.ALIGN_CENTER);
                document.add(image);
            } else {
                Paragraph errorText = new Paragraph("图片加载失败", subtitleFont);
                errorText.setAlignment(Element.ALIGN_CENTER);
                document.add(errorText);
            }
        } catch (Exception e) {
            Paragraph errorText = new Paragraph("图片加载失败: " + e.getMessage(), subtitleFont);
            errorText.setAlignment(Element.ALIGN_CENTER);
            document.add(errorText);
        }
    }

    /**
     * 从HTTP URL加载图片
     */
    private Image loadImageFromHttp(String urlString) throws IOException, BadElementException {
        if (urlString == null || !urlString.startsWith("http")) {
            return null;
        }

        // 清理URL中的换行符和空格
        urlString = urlString.trim().replaceAll("\\s+", "");

        try {
            // 解码URL
            urlString = java.net.URLDecoder.decode(urlString, "UTF-8");

            // 建立HTTP连接
            URL url = new URL(urlString);
            HttpURLConnection connection = (HttpURLConnection) url.openConnection();
            connection.setRequestMethod("GET");
            connection.setConnectTimeout(10000); // 10秒连接超时
            connection.setReadTimeout(15000);    // 15秒读取超时

            // 处理重定向
            if (connection.getResponseCode() >= 300 && connection.getResponseCode() < 400) {
                String redirectUrl = connection.getHeaderField("Location");
                if (redirectUrl != null) {
                    connection.disconnect();
                    return loadImageFromHttp(redirectUrl);
                }
            }

            // 读取图片数据
            try (InputStream inputStream = connection.getInputStream();
                 ByteArrayOutputStream outputStream = new ByteArrayOutputStream()) {
                byte[] buffer = new byte[4096];
                int bytesRead;
                while ((bytesRead = inputStream.read(buffer)) != -1) {
                    outputStream.write(buffer, 0, bytesRead);
                }
                outputStream.flush();

                // 使用图片数据创建Image对象
                return Image.getInstance(outputStream.toByteArray());
            } finally {
                connection.disconnect();
            }
        } catch (MalformedURLException e) {
            throw new IOException("无效的URL: " + urlString, e);
        }
    }

    static class PageNumberEvent extends PdfPageEventHelper {
        Font pageNumFont;

        public PageNumberEvent() {
            try {
                String fontPath = "/font/NotoSansSC-Regular.ttf";
                BaseFont bf = BaseFont.createFont(fontPath, BaseFont.IDENTITY_H, BaseFont.EMBEDDED);
                pageNumFont = new Font(bf, 10);
            } catch (Exception e) {
                pageNumFont = new Font(Font.FontFamily.HELVETICA, 10);
            }
        }

        @Override
        public void onEndPage(PdfWriter writer, Document document) {
            PdfContentByte cb = writer.getDirectContent();
            cb.beginText();
            cb.setFontAndSize(pageNumFont.getBaseFont(), pageNumFont.getSize());
            float x = document.getPageSize().getWidth() / 2;
            float y = document.bottom() - 15;
            String pageNum = "第" + writer.getPageNumber() + "页";
            cb.showTextAligned(Element.ALIGN_CENTER, pageNum, x, y, 0);
            cb.endText();
        }
    }
}