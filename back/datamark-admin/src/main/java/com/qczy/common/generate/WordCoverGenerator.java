package com.qczy.common.generate;

import org.apache.poi.openxml4j.exceptions.InvalidFormatException;
import org.apache.poi.util.Units;
import org.apache.poi.wp.usermodel.HeaderFooterType;
import org.apache.poi.xwpf.usermodel.*;

import java.io.*;
import java.net.URL;
import java.math.BigInteger;

public class WordCoverGenerator {

    // 中文字体名称
    private static final String CHINESE_FONT = "SimHei"; // 或者 "SimSun"

    public static void main(String[] args) {
        try {
            System.out.println("开始生成国家电网测试报告封面...");

            XWPFDocument document = new XWPFDocument();

            // 添加页眉内容：Logo 左上角 + 报告编号 右上角
            addHeaderWithLogoAndReportNumber(document);

            // 添加主标题（居中）
            addMainTitle(document);

            // 添加底部信息（居中）
            addCommissionInfo(document);

            // 保存文档
            File outputFile = new File("国家电网模型测试报告封面.docx");
            try (FileOutputStream out = new FileOutputStream(outputFile)) {
                document.write(out);
            }

            System.out.println("✅ 封面生成成功！");
            System.out.println("路径: " + outputFile.getAbsolutePath());

        } catch (Exception e) {
            System.err.println("❌ 生成失败：" + e.getMessage());
            e.printStackTrace();
        }
    }

    /**
     * 添加页眉内容：左侧 Logo + 右侧报告编号
     */
    private static void addHeaderWithLogoAndReportNumber(XWPFDocument doc) throws IOException {
        // 创建默认页眉
        XWPFHeader header = doc.createHeader(HeaderFooterType.DEFAULT);

        // 创建一个一行两列的表格
        XWPFTable table = header.createTable(1, 2);
        table.setWidth("100%"); // 设置表格宽度为 100%

        // 获取两个单元格
        XWPFTableRow row = table.getRow(0);
        XWPFTableCell leftCell = row.getCell(0);
        XWPFTableCell rightCell = row.getCell(1);

        // 去除单元格边框（可选）
        removeTableCellBorders(leftCell);
        removeTableCellBorders(rightCell);

        // 左侧单元格：插入 Logo
        XWPFParagraph leftPara = leftCell.getParagraphs().get(0);
        leftPara.setAlignment(ParagraphAlignment.LEFT); // 左对齐

        XWPFRun leftRun = leftPara.createRun();

        // 获取资源路径下的 logo1.png
        URL resourceUrl = WordCoverGenerator.class.getClassLoader().getResource("images/logo1.png");
        if (resourceUrl == null) {
            System.err.println("⚠️ Logo未找到，跳过添加。请确认路径是否正确：resources/images/logo1.png");
        } else {
            try (InputStream is = resourceUrl.openStream()) {
                int format = XWPFDocument.PICTURE_TYPE_PNG; // PNG 格式
                leftRun.addPicture(is, format, "Logo", Units.toEMU(110), Units.toEMU(35));
            } catch (InvalidFormatException e) {
                throw new RuntimeException(e);
            }
        }

        // 右侧单元格：插入报告编号
        XWPFParagraph rightPara = rightCell.getParagraphs().get(0);
        rightPara.setAlignment(ParagraphAlignment.RIGHT); // 右对齐

        XWPFRun rightRun = rightPara.createRun();
        rightRun.setText("报告编号：SG2023-XXXXX");
        rightRun.setFontFamily(CHINESE_FONT);
        rightRun.setFontSize(14); // 四号字体
    }

    /**
     * 辅助方法：移除单元格边框
     */
    private static void removeTableCellBorders(XWPFTableCell cell) {
       /* cell.getCTTc().addNewTcPr().addNewTcBorders()
                .setSzArray(new BigInteger[]{BigInteger.ZERO, BigInteger.ZERO, BigInteger.ZERO, BigInteger.ZERO});*/
    }

    /**
     * 添加主标题（居中）
     */
    private static void addMainTitle(XWPFDocument doc) {
        XWPFParagraph paragraph = doc.createParagraph();
        paragraph.setAlignment(ParagraphAlignment.CENTER); // 居中
        paragraph.setSpacingAfter(200);                    // 段后间距

        XWPFRun run = paragraph.createRun();
        run.setText("电力系统稳定性模型测试报告");
        run.setFontFamily(CHINESE_FONT);
        run.setFontSize(36);
        run.setBold(true);
    }

    /**
     * 添加底部委托单位信息（居中）
     */
    private static void addCommissionInfo(XWPFDocument doc) {
        String[] lines = {
                "委托单位：________________________",
                "测评单位：________________________",
                "报告时间：________________________"
        };

        for (String line : lines) {
            XWPFParagraph para = doc.createParagraph();
            para.setAlignment(ParagraphAlignment.CENTER);
            para.setSpacingBefore(100);

            XWPFRun run = para.createRun();
            run.setText(line);
            run.setFontFamily(CHINESE_FONT);
            run.setFontSize(18);
        }
    }
}