package com.qczy.common.generate;

import org.apache.poi.xwpf.usermodel.*;
import org.openxmlformats.schemas.wordprocessingml.x2006.main.*;
import org.springframework.core.io.ClassPathResource;
import org.w3c.dom.Node;

import java.io.*;
import java.math.BigInteger;
import java.util.*;

public class PoiTlTest {
    // 字体样式常量
    private static final String FONT_FAMILY = "宋体";
    private static final int FONT_SIZE = 12;
    private static final int TITLE_FONT_SIZE = 14;

    // 调试开关
    private static final boolean DEBUG = true;
    // 标题特征匹配
    private static final String TITLE_PATTERN = "^[0-9]+\\. .+";
    // 测试结果标题文本（用于定位）
    private static final String TEST_RESULT_TITLE = "2.测试结果";
    // 测试数据集标题（模板中实际标题）
    private static final String TEST_DATASET_TITLE_TEMPLATE = "1.测试数据集";
    // 测试数据集表格占位符
    private static final String DATASET_TABLE_PLACEHOLDER = "{{staffTable}}";

    // 测试组数据模型
    static class TestGroup {
        String typeTitle;
        List<String[]> dataRows;
        List<SummaryRow> summaries;

        public TestGroup(String typeTitle, List<String[]> dataRows, List<SummaryRow> summaries) {
            this.typeTitle = typeTitle;
            this.dataRows = dataRows;
            this.summaries = summaries;
        }
    }

    // 汇总行数据模型
    static class SummaryRow {
        String label;
        List<String> values;

        public SummaryRow(String label, List<String> values) {
            this.label = label;
            this.values = values;
        }
    }

    public static void main(String[] args) {
        try {
            // 1. 准备数据
            Map<String, String> textData = prepareAllTextData();
            List<TestGroup> testGroups = prepareTestData();

            // 2. 读取模板
            ClassPathResource resource = new ClassPathResource("templates/test-template.docx");
            InputStream templateStream = resource.getInputStream();
            XWPFDocument doc = new XWPFDocument(templateStream);

            // 3. 替换文本占位符（标题、作者、日期等）
            replaceTextPlaceholders(doc, textData);

            // 4. 插入测试数据集表格到模板指定位置
            insertTestDatasetIntoTemplate(doc);

            // 5. 在"测试结果"标题后插入表格（使用新方法避免节点引用问题）
            insertTestTablesAfterTitleSafe(doc, testGroups);

            // 6. 保存文档
            File resultFile = new File("模型测试报告.docx");
            try (FileOutputStream out = new FileOutputStream(resultFile)) {
                doc.write(out);
            }

            System.out.println("文档生成成功: " + resultFile.getAbsolutePath());

        } catch (Exception e) {
            System.err.println("文档生成失败: " + e.getMessage());
            e.printStackTrace();
        }
    }

    // 准备文本替换数据（标题、作者、日期等）
    private static Map<String, String> prepareAllTextData() {
        Map<String, String> data = new HashMap<>();
        data.put("{{title}}", "模型测试结果分析报告");
        data.put("{{author}}", "技术研发部");
        data.put("{{date}}", "2025年7月31日");
        data.put("{{num1}}", "5");
        data.put("{{num2}}", "200");
        return data;
    }

    // 替换文档中的文本占位符（标题、作者等）
    private static void replaceTextPlaceholders(XWPFDocument doc, Map<String, String> data) {
        int replacedCount = 0;
        // 使用迭代器避免并发修改问题
        Iterator<XWPFParagraph> paraIterator = doc.getParagraphs().iterator();
        while (paraIterator.hasNext()) {
            XWPFParagraph para = paraIterator.next();
            String originalText = getParagraphTextSafe(para);
            if (originalText.isEmpty()) continue;

            String modifiedText = originalText;
            boolean isReplaced = false;

            for (Map.Entry<String, String> entry : data.entrySet()) {
                String placeholder = entry.getKey();
                String value = entry.getValue();

                if (modifiedText.contains(placeholder)) {
                    modifiedText = modifiedText.replace(placeholder, value);
                    isReplaced = true;
                    replacedCount++;
                    if (DEBUG) {
                        System.out.println("替换占位符: " + placeholder + " -> " + value);
                    }
                }
            }

            if (isReplaced) {
                clearParagraph(para);
                XWPFRun run = para.createRun();
                run.setText(modifiedText);
                if (modifiedText.matches(TITLE_PATTERN)) {
                    run.setBold(true);
                }
                applyRunStyle(run, para.getRuns());
            }
        }

        if (DEBUG) {
            System.out.println("共替换占位符: " + replacedCount + " 处");
        }
    }

    // 准备测试表格数据（通道施工、防山火、导线覆冰）
    private static List<TestGroup> prepareTestData() {
        List<TestGroup> testGroups = new ArrayList<>();

        // 通道施工机械识别组
        List<String[]> channelData = new ArrayList<>();
        channelData.add(new String[]{"序号", "样本标签", "标注数(M)", "发现数(Mo)", "正确检出数(Mc)", "发现率(R)", "误报率(F)", "误检比(C)"});
        channelData.add(new String[]{"1", "挖掘机", "302", "74", "65", "0.2152", "0.1216", "0.0298"});
        channelData.add(new String[]{"2", "水泥搅拌车", "285", "84", "75", "0.2947", "0.12", "0.0316"});
        List<SummaryRow> channelSummary = new ArrayList<>();
        channelSummary.add(new SummaryRow("单类型测试指标结果", Arrays.asList("", "", "", "", "0.255", "0.1208", "0.0307")));
        testGroups.add(new TestGroup("2.1 识别类型：通道施工机械识别", channelData, channelSummary));

        // 防山火智能识别组
        List<String[]> fireData = new ArrayList<>();
        fireData.add(new String[]{"序号", "样本标签", "标注数(M)", "发现数(Mo)", "正确检出数(Mc)", "发现率(R)", "误报率(F)", "误检比(C)"});
        fireData.add(new String[]{"1", "山火", "376", "56", "23", "0.0612", "0.5893", "0.0"});
        fireData.add(new String[]{"2", "烟雾", "235", "60", "27", "0.2553", "0.55", "0.1404"});
        List<SummaryRow> fireSummary = new ArrayList<>();
        fireSummary.add(new SummaryRow("单类型测试指标结果", Arrays.asList("", "", "", "", "0.1583", "0.5697", "0.0702")));
        testGroups.add(new TestGroup("2.2 识别类型：防山火智能识别", fireData, fireSummary));

        // 导线覆冰识别组
        List<String[]> iceData = new ArrayList<>();
        iceData.add(new String[]{"序号", "样本标签", "标注数(M)", "发现数(Mo)", "正确检出数(Mc)", "发现率(R)", "误报率(F)", "误检比(C)"});
        iceData.add(new String[]{"1", "轻度覆冰", "156", "42", "38", "0.2436", "0.0952", "0.0213"});
        iceData.add(new String[]{"2", "中度覆冰", "132", "36", "31", "0.2348", "0.1389", "0.0287"});
        iceData.add(new String[]{"3", "重度覆冰", "98", "28", "25", "0.2551", "0.1071", "0.0196"});
        List<SummaryRow> iceSummary = new ArrayList<>();
        iceSummary.add(new SummaryRow("单类型测试指标结果", Arrays.asList("", "", "", "", "0.2445", "0.1137", "0.0232")));
        testGroups.add(new TestGroup("2.3 识别类型：导线覆冰识别", iceData, iceSummary));

        return testGroups;
    }

    // 核心逻辑：插入测试数据集表格到模板中「1. 测试数据集」段落之后，替换占位符
    private static void insertTestDatasetIntoTemplate(XWPFDocument doc) throws Exception {
        // 1. 查找模板中「1. 测试数据集」段落
        XWPFParagraph datasetTitlePara = null;
        for (XWPFParagraph para : doc.getParagraphs()) {
            String text = getParagraphTextSafe(para);
            if (text.trim().equals(TEST_DATASET_TITLE_TEMPLATE)) {
                datasetTitlePara = para;
                break;
            }
        }

        if (datasetTitlePara == null) {
            throw new Exception("模板中未找到标题：" + TEST_DATASET_TITLE_TEMPLATE);
        }

        // 2. 查找表格占位符所在段落（在「1. 测试数据集」之后）
        XWPFParagraph tablePlaceholderPara = null;
        List<XWPFParagraph> paragraphs = new ArrayList<>(doc.getParagraphs()); // 创建副本避免并发问题
        int titleIndex = paragraphs.indexOf(datasetTitlePara);
        for (int i = titleIndex + 1; i < paragraphs.size(); i++) {
            XWPFParagraph para = paragraphs.get(i);
            if (getParagraphTextSafe(para).contains(DATASET_TABLE_PLACEHOLDER)) {
                tablePlaceholderPara = para;
                break;
            }
        }

        if (tablePlaceholderPara == null) {
            throw new Exception("模板中未找到表格占位符：" + DATASET_TABLE_PLACEHOLDER);
        }

        // 3. 在主文档中直接构建测试数据集表格
        XWPFTable datasetTable = doc.createTable(13, 6);
        buildTestDatasetTableContent(datasetTable);

        // 4. 获取DOM节点并执行替换
        Node placeholderNode = tablePlaceholderPara.getCTP().getDomNode();
        Node parentNode = placeholderNode.getParentNode();

        // 删除占位符段落
        parentNode.removeChild(placeholderNode);

        // 在数据集标题后插入新表格
        Node titleNode = datasetTitlePara.getCTP().getDomNode();
        insertNodeAfter(parentNode, titleNode, datasetTable.getCTTbl().getDomNode());

        if (DEBUG) {
            System.out.println("测试数据集表格已插入到 " + TEST_DATASET_TITLE_TEMPLATE + " 之后");
        }
    }

    // 填充表格内容
    private static void buildTestDatasetTableContent(XWPFTable table) {
        table.setWidthType(TableWidthType.AUTO);
        applyTableBorder(table);

        // 表头数据
        String[] header = {"序号", "识别类型", "模型功能", "样本标签", "样本数", "正负样本比例"};
        XWPFTableRow headerRow = table.getRow(0);
        for (int i = 0; i < header.length; i++) {
            XWPFTableCell cell = headerRow.getCell(i);
            clearParagraph(cell.getParagraphs().get(0));
            XWPFRun run = cell.getParagraphs().get(0).createRun();
            run.setText(header[i]);
            run.setFontFamily(FONT_FAMILY);
            run.setFontSize(FONT_SIZE);
            run.setBold(true);
            cell.getParagraphs().get(0).setAlignment(ParagraphAlignment.CENTER);
        }

        // 表格数据
        String[][] tableData = {
                {"1", "通道施工机械识别", "通道施工机械-挖掘机", "挖掘机", "350", "1:1.2"},
                {"2", "通道施工机械识别", "通道施工机械-水泥搅拌车", "水泥搅拌车", "320", "1:1.1"},
                {"3", "通道施工机械识别", "通道施工机械-吊车", "吊车", "280", "1:1.0"},
                {"4", "通道施工机械识别", "通道施工机械-塔吊", "塔吊", "260", "1:0.9"},
                {"5", "通道施工机械识别", "通道施工机械-打桩机", "打桩机", "240", "1:0.8"},
                {"6", "通道施工机械识别", "通道施工机械-铲车", "铲车", "220", "1:0.9"},
                {"7", "通道施工机械识别", "通道施工机械-水泥泵车", "水泥泵车", "200", "1:1.0"},
                {"8", "防山火智能识别", "山火-山火", "山火", "400", "1:2.0"},
                {"9", "防山火智能识别", "山火-烟雾", "烟雾", "380", "1:1.8"},
                {"10", "导线覆冰识别", "导线覆冰-轻度", "轻度覆冰", "300", "1:1.5"},
                {"11", "导线覆冰识别", "导线覆冰-中度", "中度覆冰", "280", "1:1.4"},
                {"12", "导线覆冰识别", "导线覆冰-重度", "重度覆冰", "250", "1:1.3"}
        };

        for (int i = 0; i < tableData.length; i++) {
            XWPFTableRow row = table.getRow(i + 1);
            String[] rowData = tableData[i];
            for (int j = 0; j < rowData.length; j++) {
                XWPFTableCell cell = row.getCell(j);
                clearParagraph(cell.getParagraphs().get(0));
                XWPFRun run = cell.getParagraphs().get(0).createRun();
                run.setText(rowData[j]);
                run.setFontFamily(FONT_FAMILY);
                run.setFontSize(FONT_SIZE);
                cell.getParagraphs().get(0).setAlignment(ParagraphAlignment.CENTER);
            }
        }
    }

    // 安全版本：在"测试结果"标题后插入表格，避免节点引用问题
    private static void insertTestTablesAfterTitleSafe(XWPFDocument doc, List<TestGroup> testGroups) {
        // 1. 查找"测试结果"标题段落（使用安全方法）
        XWPFParagraph targetPara = null;
        for (XWPFParagraph para : doc.getParagraphs()) {
            try {
                if (getParagraphTextSafe(para).contains(TEST_RESULT_TITLE)) {
                    targetPara = para;
                    break;
                }
            } catch (Exception e) {
                // 跳过无效段落
                continue;
            }
        }

        // 获取文档的XML节点
        CTDocument1 ctDoc = doc.getDocument();
        CTBody body = ctDoc.getBody();
        Node bodyNode = body.getDomNode();

        // 2. 确定插入位置
        Node insertAfterNode = null;
        if (targetPara != null) {
            insertAfterNode = targetPara.getCTP().getDomNode();
        } else {
            System.err.println("未找到标题：" + TEST_RESULT_TITLE + "，表格将插入到文档末尾");
            Node lastChild = bodyNode.getLastChild();
            insertAfterNode = lastChild != null ? lastChild : null;
        }

        // 3. 逐个插入测试组内容
        for (TestGroup group : testGroups) {
            // a. 创建组标题段落并插入
            XWPFParagraph groupPara = doc.createParagraph();
            XWPFRun groupRun = groupPara.createRun();
            groupRun.setText(group.typeTitle);
            groupRun.setFontFamily(FONT_FAMILY);
            groupRun.setFontSize(TITLE_FONT_SIZE);
            groupRun.setBold(true);
            Node paraNode = groupPara.getCTP().getDomNode();
            insertNodeAfter(bodyNode, insertAfterNode, paraNode);
            insertAfterNode = paraNode;

            // b. 创建表格并插入
            int colCount = group.dataRows.get(0).length;
            int rowCount = group.dataRows.size() + group.summaries.size();
            XWPFTable table = doc.createTable(rowCount, colCount);
            table.setWidthType(TableWidthType.AUTO);
            applyTableBorder(table);

            // 填充表头和数据行
            for (int i = 0; i < group.dataRows.size(); i++) {
                XWPFTableRow row = table.getRow(i);
                String[] rowData = group.dataRows.get(i);
                for (int j = 0; j < colCount; j++) {
                    XWPFTableCell cell = row.getCell(j);
                    clearParagraph(cell.getParagraphs().get(0));
                    XWPFRun run = cell.getParagraphs().get(0).createRun();
                    run.setText(rowData[j] != null ? rowData[j] : "");
                    cell.getParagraphs().get(0).setAlignment(ParagraphAlignment.CENTER);
                    run.setFontFamily(FONT_FAMILY);
                    run.setFontSize(FONT_SIZE);
                    if (i == 0) run.setBold(true);
                }
            }

            // 填充汇总行
            int summaryStart = group.dataRows.size();
            for (int i = 0; i < group.summaries.size(); i++) {
                SummaryRow summary = group.summaries.get(i);
                XWPFTableRow row = table.getRow(summaryStart + i);
                mergeCells(row, 0, colCount - 1);

                XWPFTableCell cell = row.getCell(0);
                clearParagraph(cell.getParagraphs().get(0));
                XWPFRun run = cell.getParagraphs().get(0).createRun();
                run.setText(summary.label);
                cell.getParagraphs().get(0).setAlignment(ParagraphAlignment.CENTER);
                run.setFontFamily(FONT_FAMILY);
                run.setFontSize(FONT_SIZE);
                run.setBold(true);

                for (int j = 0; j < summary.values.size(); j++) {
                    if (j + 1 < row.getTableCells().size()) {
                        XWPFTableCell valCell = row.getCell(j + 1);
                        clearParagraph(valCell.getParagraphs().get(0));
                        XWPFRun valRun = valCell.getParagraphs().get(0).createRun();
                        valRun.setText(summary.values.get(j) != null ? summary.values.get(j) : "");
                        valCell.getParagraphs().get(0).setAlignment(ParagraphAlignment.CENTER);
                        valRun.setFontFamily(FONT_FAMILY);
                        valRun.setFontSize(FONT_SIZE);
                        valRun.setBold(true);
                    }
                }
            }

            // 插入表格到文档
            Node tableNode = table.getCTTbl().getDomNode();
            insertNodeAfter(bodyNode, insertAfterNode, tableNode);
            insertAfterNode = tableNode;
        }
    }

    // DOM节点操作：在指定节点后插入新节点
    private static void insertNodeAfter(Node parent, Node referenceNode, Node newNode) {
        if (referenceNode == null) {
            parent.insertBefore(newNode, parent.getFirstChild());
        } else {
            Node nextSibling = referenceNode.getNextSibling();
            if (nextSibling == null) {
                parent.appendChild(newNode);
            } else {
                parent.insertBefore(newNode, nextSibling);
            }
        }
    }

    // 合并行内单元格
    private static void mergeCells(XWPFTableRow row, int startCol, int endCol) {
        if (startCol < 0 || endCol >= row.getTableCells().size() || startCol >= endCol) {
            return;
        }

        XWPFTableCell startCell = row.getCell(startCol);
        CTTcPr startTcPr = startCell.getCTTc().getTcPr() != null ? startCell.getCTTc().getTcPr() : startCell.getCTTc().addNewTcPr();
        startTcPr.addNewHMerge().setVal(STMerge.RESTART);

        for (int col = startCol + 1; col <= endCol; col++) {
            XWPFTableCell cell = row.getCell(col);
            CTTcPr tcPr = cell.getCTTc().getTcPr() != null ? cell.getCTTc().getTcPr() : cell.getCTTc().addNewTcPr();
            tcPr.addNewHMerge().setVal(STMerge.CONTINUE);
        }
    }

    // 应用段落样式
    @SuppressWarnings("deprecation")
    private static void applyRunStyle(XWPFRun run, List<XWPFRun> originalRuns) {
        if (originalRuns != null && !originalRuns.isEmpty()) {
            XWPFRun originalRun = originalRuns.get(0);
            String fontFamily = originalRun.getFontFamily();
            run.setFontFamily(fontFamily != null ? fontFamily : FONT_FAMILY);

            Integer fontSize = originalRun.getFontSize();
            run.setFontSize(fontSize != -1 ? fontSize : FONT_SIZE);

            run.setBold(originalRun.isBold());
            run.setItalic(originalRun.isItalic());
            run.setUnderline(originalRun.getUnderline());
        } else {
            run.setFontFamily(FONT_FAMILY);
            run.setFontSize(FONT_SIZE);
        }
    }

    // 为表格添加边框
    private static void applyTableBorder(XWPFTable table) {
        CTTbl tbl = table.getCTTbl();
        CTTblPr tblPr = tbl.getTblPr() != null ? tbl.getTblPr() : tbl.addNewTblPr();
        CTBorder border = CTBorder.Factory.newInstance();
        border.setColor("auto");
        border.setSz(BigInteger.valueOf(2));
        border.setVal(STBorder.SINGLE);

        CTTblBorders tblBorders = tblPr.getTblBorders() != null ? tblPr.getTblBorders() : tblPr.addNewTblBorders();
        tblBorders.setTop(border);
        tblBorders.setLeft(border);
        tblBorders.setBottom(border);
        tblBorders.setRight(border);
        tblBorders.setInsideH(border);
        tblBorders.setInsideV(border);
    }

    // 安全版本：获取段落文本内容，处理可能的断开连接异常
    private static String getParagraphTextSafe(XWPFParagraph para) {
        try {
            StringBuilder text = new StringBuilder();
            for (XWPFRun run : para.getRuns()) {
                // 检查run是否有效
                if (run == null) continue;

                String runText;
                try {
                    runText = run.getText(0);
                } catch (Exception e) {
                    // 处理断开连接的run
                    continue;
                }

                if (runText != null) {
                    text.append(runText);
                }
            }
            return text.toString();
        } catch (Exception e) {
            // 段落已失效，返回空字符串
            return "";
        }
    }

    // 清空段落内容
    private static void clearParagraph(XWPFParagraph para) {
        try {
            while (para.getRuns().size() > 0) {
                para.removeRun(0);
            }
        } catch (Exception e) {
            // 忽略已失效段落的清空操作
        }
    }
}
