package com.qczy.common.generate;

import com.itextpdf.text.*;
import com.itextpdf.text.pdf.*;
import java.io.FileOutputStream;

public class PdfGenerator {
    public static void main(String[] args) {
        Document document = new Document(PageSize.A4);
        try {
            // 获取PdfWriter实例并设置页面事件
            PdfWriter writer = PdfWriter.getInstance(document, new FileOutputStream("ModelTestApplication.pdf"));
            writer.setPageEvent(new PageNumberEvent());

            document.open();

            // 设置中文字体
            String fontPath = "/font/NotoSansSC-Regular.ttf";
            BaseFont bfChinese = BaseFont.createFont(fontPath, BaseFont.IDENTITY_H, BaseFont.EMBEDDED);
            Font titleFont = new Font(bfChinese, 18, Font.BOLD);
            Font normalFont = new Font(bfChinese, 12);
            Font sectionFont = new Font(bfChinese, 14, Font.BOLD); // 新增章节标题字体

            // 添加标题
            Paragraph title = new Paragraph("模型测试申请单", titleFont);
            title.setAlignment(Element.ALIGN_CENTER);
            document.add(title);
            document.add(Chunk.NEWLINE);

            // 添加“一、基本信息”标题
            addSectionTitle(document, "一、基本信息", sectionFont);

            // 主表格结构优化（3列实现紧凑布局）
            PdfPTable mainTable = new PdfPTable(3);
            mainTable.setWidthPercentage(90);
            mainTable.setHorizontalAlignment(Element.ALIGN_CENTER); // 居中对齐
            mainTable.setWidths(new float[]{0.7f, 1.5f, 7.3f}); // 列宽比例调整

            // 添加基础信息
            addCompactRow(mainTable, "1", "申请单号", "20250522326", normalFont);
            addCompactRow(mainTable, "2", "申请日期", "2025-05-27", normalFont);
            addCompactRow(mainTable, "3", "模型名称", "无法用模型", normalFont);
            addCompactRow(mainTable, "4", "模型来源", "（信通-数字2023-336)新技术及应用-2023年人工智能技术规模化应用-设计开发实施运营项目(现场作业智能管控规模化应用建设)-系统实施", normalFont);
            addCompactRow(mainTable, "5", "模型类型", "对基于视觉大模型研发违章识别、状态识别、 安全风险识别 3 大类场景模型进行检测", normalFont);
            addCompactRow(mainTable, "6", "模型功能", "对基于视觉大模型研发...（完整的密码）", normalFont);

            // 建设单位信息（改进合并方式）
            addComplexRow(mainTable, "7", "建设单位", new String[][]{
                    {"单位名称", "国际美颜信息公司"},
                    {"单位地址", "天津市河北区昆城路153号"},
                    {"联系人", "小红"},
                    {"联系电话", "17622229501"},
                    {"负责人签字：", null}
            }, normalFont);

            // 其他单位信息（紧凑布局）
            addComplexRow(mainTable, "8", "承建单位", new String[][]{
                    {"单位名称", "国电南瑞南京控制系统有限公司"},
                    {"单位地址", "南京市江宁经济技术开发区诚信大道19号"},
                    {"联系人", "小明"},
                    {"联系电话", "13212001628"},
                    {"负责人签字：", null}
            }, normalFont);

            document.add(mainTable);

            // 新增第二页
            document.newPage();

        // 添加“二、准备工作”标题
            addSectionTitle(document, "二、准备工作", sectionFont);

        // 创建第二页表格（注意这里使用 testTable 而非 mainTable）
            PdfPTable testTable = new PdfPTable(3); // 3列表格
            testTable.setWidthPercentage(90);
            testTable.setHorizontalAlignment(Element.ALIGN_CENTER);
            testTable.setWidths(new float[]{0.7f, 2.3f, 7f}); // 列宽比例调整

        // 添加测试信息表格内容（全部使用 testTable）
            addCompactRow(testTable, "1", "文件模型名", "20250522326", normalFont);
            addCompactRow(testTable, "2", "模型封装方式", "2025-05-27", normalFont);
            addCompactRow(testTable, "3", "模型文件大小", "无法用模型", normalFont);
            addCompactRow(testTable, "4", "模型部署位置", "无法用模型", normalFont);
            addCompactRow(testTable, "5", "模型API接口说明", "http://123.56.32:80801/api/1.txt", normalFont);
            addCompactRow(testTable, "6", "模型对外暴露端口", "无法用模型", normalFont);
            addCompactRow(testTable, "7", "模型cuda版本", "无法用模型", normalFont);
            addCompactRow(testTable, "8", "模型驱动版本", "无法用模型", normalFont);
            addCompactRow(testTable, "9", "模型调用例", "http://123.56.32:80801/api/2.txt", normalFont);

            document.add(testTable); // 添加 testTable 到文档

            document.close();
        } catch (Exception e) {
            e.printStackTrace();
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

    // 紧凑行添加方法
    private static void addCompactRow(PdfPTable table, String num, String key, String value, Font font) {
        table.addCell(createCell(num, font));
        table.addCell(createCell(key, font));
        PdfPCell valueCell = createCell(value, font);
        valueCell.setColspan(1);
        table.addCell(valueCell);
    }



    // 复杂行处理方法（建设单位）
    private static void addComplexRow(PdfPTable table, String num, String key, String[][] items, Font font) {
        // 序号和标题列
        table.addCell(createCell(num, font));
        table.addCell(createCell(key, font));

        // 合并内容列
        PdfPCell complexCell = new PdfPCell();
        PdfPTable subTable = new PdfPTable(2);
        try {
            subTable.setWidths(new float[]{2f, 8f});
            subTable.getDefaultCell().setBorderWidth(0);
        } catch (DocumentException e) {
            throw new RuntimeException(e);
        }

        for (String[] item : items) {
            // 特殊处理"负责人签字"项
            if ("负责人签字：".equals(item[0])) {
                PdfPCell signatureCell = new PdfPCell();
                signatureCell.setBorder(Rectangle.BOX);
                signatureCell.setPadding(10);

                Paragraph p = new Paragraph("负责人签字：", font);
                p.setAlignment(Element.ALIGN_CENTER);
                p.add(Chunk.NEWLINE);
                p.add(Chunk.NEWLINE);

                p.add(new Phrase("年          月          日", font));
                p.setAlignment(Element.ALIGN_CENTER);

                signatureCell.addElement(p);
                signatureCell.setColspan(2);
                signatureCell.setMinimumHeight(50);

                subTable.addCell(signatureCell);
                continue;
            }

            subTable.addCell(createCell(item[0], font));
            subTable.addCell(createCell(item[1], font));
        }

        complexCell.addElement(subTable);
        complexCell.setPadding(0);
        complexCell.setPaddingTop(0);
        complexCell.setPaddingBottom(0);
        table.addCell(complexCell);
    }

    private static PdfPCell createCell(String text, Font font) {
        PdfPCell cell = new PdfPCell(new Phrase(text, font));
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