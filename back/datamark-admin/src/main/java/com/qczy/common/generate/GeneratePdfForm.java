package com.qczy.common.generate;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.itextpdf.text.*;
import com.itextpdf.text.pdf.*;
import com.qczy.mapper.ModelBaseMapper;
import com.qczy.mapper.ModelConfigureMapper;
import com.qczy.mapper.UserMapper;
import com.qczy.model.entity.ModelBaseEntity;
import com.qczy.model.entity.ModelConfigureEntity;
import com.qczy.utils.StringUtils;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;

import javax.servlet.ServletOutputStream;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;

import java.io.IOException;
import java.io.OutputStream;
import java.lang.reflect.Field;
import java.net.URLEncoder;
import java.text.SimpleDateFormat;
import java.util.*;

import org.springframework.util.ReflectionUtils;

@Component
public class GeneratePdfForm {

    @Autowired
    private ModelBaseMapper modelBaseMapper;
    @Autowired
    private ModelConfigureMapper modelConfigureMapper;
    @Autowired
    private UserMapper userMapper;
    @Value("${upload.address}")
    private String uploadAddress;


    public void downloadPdf(Integer modelId, HttpServletRequest request, HttpServletResponse response) {
        try {
            // 1. 获取模型数据
            ModelBaseEntity modelBaseEntity = modelBaseMapper.selectById(modelId);
            if (modelBaseEntity == null) {
                response.sendError(HttpServletResponse.SC_NOT_FOUND, "模型对象不存在");
                return;
            }

            // 2. 设置响应头
            response.setContentType("application/pdf");
            response.setCharacterEncoding("UTF-8");

            // 3. 生成中文文件名
            String fileName = "model_" + modelBaseEntity.getApplyForNum() + ".pdf";

            // 4. 关键修复：三重编码方案（兼容所有浏览器和服务器）
            String userAgent = request.getHeader("User-Agent");
            // 改进后的文件名编码逻辑
            String encodedFileName;
            if (userAgent != null && userAgent.matches("(?i).*(MSIE|Trident|Edge).*")) {
                // IE/Edge浏览器
                encodedFileName = URLEncoder.encode(fileName, "UTF-8").replace("+", "%20");
            } else {
                // 其他现代浏览器
                encodedFileName = new String(fileName.getBytes("UTF-8"), "ISO-8859-1");
            }


            // 5. 设置Content-Disposition（强制下载）
            response.setHeader("Content-Disposition", "attachment; filename=\"" + encodedFileName + "\"");

            // 6. 强制清除缓存头
            response.setHeader("Cache-Control", "no-cache, no-store, must-revalidate");
            response.setHeader("Pragma", "no-cache");
            response.setDateHeader("Expires", 0);

            // 7. 写入PDF内容
            try (ServletOutputStream out = response.getOutputStream()) {
                generatePad(modelBaseEntity, out);
                out.flush();
            }
        } catch (Exception e) {
            response.setStatus(HttpServletResponse.SC_INTERNAL_SERVER_ERROR);
        }
    }


    public void generatePad(ModelBaseEntity modelBaseEntity, OutputStream outputStream) {
        ModelConfigureEntity modelConfigureEntity = modelConfigureMapper.selectOne(
                new LambdaQueryWrapper<ModelConfigureEntity>()
                        .eq(ModelConfigureEntity::getModelBaseId, modelBaseEntity.getId())
        );

        if (modelConfigureEntity == null) {
            return;
        }

        Document document = new Document(PageSize.A4);
        try {
            // 使用传入的 OutputStream 创建 PdfWriter
            PdfWriter writer = PdfWriter.getInstance(document, outputStream);
            writer.setPageEvent(new PageNumberEvent()); // 确保 PageNumberEvent 是静态内部类

            document.open();

            // 设置中文字体（保持不变）
            String fontPath = "/font/NotoSansSC-Regular.ttf";
            BaseFont bfChinese = BaseFont.createFont(fontPath, BaseFont.IDENTITY_H, BaseFont.EMBEDDED);
            Font titleFont = new Font(bfChinese, 18, Font.BOLD);
            Font normalFont = new Font(bfChinese, 12);
            Font sectionFont = new Font(bfChinese, 14, Font.BOLD);

            // 添加标题（保持不变）
            Paragraph title = new Paragraph("模型测试申请单", titleFont);
            title.setAlignment(Element.ALIGN_CENTER);
            document.add(title);
            document.add(Chunk.NEWLINE);

            // 第一页：基本信息
            addSectionTitle(document, "一、基本信息", sectionFont);
            PdfPTable mainTable = createMainTable();
            fillMainTable(mainTable, modelBaseEntity, normalFont);
            document.add(mainTable);

            // 第二页：准备工作
            document.newPage();
            addSectionTitle(document, "二、准备工作", sectionFont);
            PdfPTable testTable = createTestTable();
            fillTestTable(testTable, modelConfigureEntity, normalFont, uploadAddress);
            document.add(testTable);

            document.close(); // 关闭文档会自动刷新 OutputStream
        } catch (Exception e) {
            e.printStackTrace();
            try {
                outputStream.close(); // 确保异常时关闭流
            } catch (IOException ex) {
                ex.printStackTrace();
            }
        }
    }

    // 提取主表格创建逻辑
    private PdfPTable createMainTable() throws DocumentException {
        PdfPTable table = new PdfPTable(3);
        table.setWidthPercentage(90);
        table.setHorizontalAlignment(Element.ALIGN_CENTER);
        table.setWidths(new float[]{0.7f, 1.5f, 7.8f});
        return table;
    }

    // 填充主表格数据
    private void fillMainTable(PdfPTable table, ModelBaseEntity entity, Font font) {
        addRow(table, "1", "申请单号", entity.getApplyForNum(), font);
        addRow(table, "2", "申请日期", formatDate(entity.getApplyForDate()), font);
        addRow(table, "3", "模型名称", entity.getModelName(), font);
        addRow(table, "4", "模型来源", entity.getModelSource(), font);
        addRow(table, "5", "模型类型", entity.getModelType(), font);
        addRow(table, "6", "模型功能", entity.getModelFunction(), font);

        // 建设单位
        addComplexUnitRow(table, "7", "建设单位", entity, font, "build");
        // 承建单位
        addComplexUnitRow(table, "8", "承建单位", entity, font, "bt");
    }

    // 提取复杂单位行逻辑
    private void addComplexUnitRow(PdfPTable table, String num, String title, ModelBaseEntity entity, Font font, String prefix) {
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

        // 单位名称
        subTable.addCell(createCell("单位名称", font));
        subTable.addCell(createCell(getValue(entity, prefix + "UnitName"), font));
        // 单位地址
        subTable.addCell(createCell("单位地址", font));
        subTable.addCell(createCell(getValue(entity, prefix + "UnitAddress"), font));
        // 联系人
        subTable.addCell(createCell("联系人", font));
        subTable.addCell(createCell(getValue(entity, prefix + "UnitLeader"), font));
        // 联系电话
        subTable.addCell(createCell("联系电话", font));
        subTable.addCell(createCell(getValue(entity, prefix + "UnitContact"), font));

        // 创建一个包含签字和日期的单元格
        PdfPCell signDateCell = new PdfPCell();
        signDateCell.setColspan(2);
        signDateCell.setBorder(Rectangle.BOX);
        signDateCell.setPadding(10);
        signDateCell.setMinimumHeight(80); // 增加高度以容纳两行

        // 使用内嵌表格实现签字和日期的垂直排列
        PdfPTable innerTable = new PdfPTable(1);
        innerTable.setWidthPercentage(100);
        innerTable.getDefaultCell().setBorder(Rectangle.NO_BORDER);
        innerTable.getDefaultCell().setHorizontalAlignment(Element.ALIGN_CENTER);

        // 第一行：负责人签字
        PdfPCell signRow = new PdfPCell(new Phrase("负责人签字：                  ", font));
        signRow.setBorder(Rectangle.NO_BORDER); // 无边框
        signRow.setPaddingBottom(15);
        innerTable.addCell(signRow);

        // 第二行：日期
        PdfPCell dateRow = new PdfPCell(new Phrase("    日期：           年           月           日", font));
        dateRow.setBorder(Rectangle.NO_BORDER); // 无边框
        dateRow.setPaddingBottom(5);
        innerTable.addCell(dateRow);

        // 将内嵌表格添加到单元格
        signDateCell.addElement(innerTable);
        subTable.addCell(signDateCell);

        complexCell.addElement(subTable);
        complexCell.setPadding(0);
        table.addCell(complexCell);
    }

    // 提取第二页表格创建逻辑
    private PdfPTable createTestTable() throws DocumentException {
        PdfPTable table = new PdfPTable(3);
        table.setWidthPercentage(90);
        table.setHorizontalAlignment(Element.ALIGN_CENTER);
        table.setWidths(new float[]{0.7f, 2.3f, 7f});
        return table;
    }

    // 填充第二页数据
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

    // 通用行添加方法
    private void addRow(PdfPTable table, String num, String key, String value, Font font) {
        table.addCell(createCell(num, font));
        table.addCell(createCell(key, font));
        table.addCell(createCell(StringUtils.isEmpty(value) ? "-" : value, font));
    }


    // 日期格式化方法
    private String formatDate(Date date) {
        return date != null ? new SimpleDateFormat("yyyy-MM-dd").format(date) : "-";
    }

    // 获取属性值（带空值处理）
    private String getValue(ModelBaseEntity entity, String field) {
        return Optional.ofNullable(entity)
                .map(e -> (String) getFieldValue(e, field))
                .orElse("-");
    }

    // URL拼接方法
    private String getUrlValue(String value, String uploadAddress) {
        return StringUtils.isEmpty(value) ? "-" : uploadAddress + value;
    }


    public  Object getFieldValue(Object obj, String fieldName) {
        if (obj == null || fieldName == null) {
            return null;
        }
        try {
            Field field = obj.getClass().getDeclaredField(fieldName);
            field.setAccessible(true);
            return field.get(obj);
        } catch (NoSuchFieldException | IllegalAccessException e) {
            return null;
        }

    }



    // 添加章节标题（保持与表格左侧对齐且整体居中）
    private static void addSectionTitle(Document document, String title, Font font) throws DocumentException {
        Paragraph sectionTitle = new Paragraph(title, font);
        sectionTitle.setAlignment(Element.ALIGN_LEFT);

        PdfPTable titleContainer = new PdfPTable(1);
        titleContainer.setWidthPercentage(90);
        titleContainer.setHorizontalAlignment(Element.ALIGN_CENTER);
        titleContainer.getDefaultCell().setBorder(Rectangle.NO_BORDER);
        titleContainer.addCell(sectionTitle);

        document.add(titleContainer);

        // 添加少量间距
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

    // 页码事件类
    static class PageNumberEvent extends PdfPageEventHelper {
        Font pageNumFont;

        public PageNumberEvent() {
            try {
                // 设置页码中文字体
                String fontPath = "/font/NotoSansSC-Regular.ttf";
                BaseFont bf = BaseFont.createFont(fontPath, BaseFont.IDENTITY_H, BaseFont.EMBEDDED);
                pageNumFont = new Font(bf, 10);
            } catch (Exception e) {
                e.printStackTrace();
                // 使用默认字体
                pageNumFont = new Font(Font.FontFamily.HELVETICA, 10);
            }
        }

        @Override
        public void onEndPage(PdfWriter writer, Document document) {
            // 在每页底部添加页码
            PdfContentByte cb = writer.getDirectContent();
            cb.beginText();
            cb.setFontAndSize(pageNumFont.getBaseFont(), pageNumFont.getSize());

            // 计算页码位置（底部居中）
            float x = document.getPageSize().getWidth() / 2;
            float y = document.bottom() - 15;

            // 页码格式：第X页/共Y页
            String pageNum = "第" + writer.getPageNumber() + "页";
            cb.showTextAligned(Element.ALIGN_CENTER, pageNum, x, y, 0);
            cb.endText();
        }
    }


}
