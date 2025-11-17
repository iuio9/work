package com.qczy.common.word;

import org.apache.commons.compress.utils.IOUtils;
import org.apache.poi.util.Units;
import org.apache.poi.xwpf.usermodel.*;
import org.apache.xmlbeans.XmlCursor;
import org.openxmlformats.schemas.wordprocessingml.x2006.main.*;
import org.springframework.core.io.ClassPathResource;
import org.springframework.core.io.Resource;

import java.io.*;
import java.math.BigInteger;
import java.util.*;

public class WordTemplateMainTest1 {

    // 表格核心配置
    private static final int TARGET_COLUMN_COUNT = 15;
    private static final int SEQ_COLUMN = 1;
    private static final int CATEGORY_COLUMN = 1;
    private static final int STATE_GRID_COLUMNS = 7;
    private static final int GENERAL_COLUMNS = 6;
    private static final int FONT_SIZE = 10;
    private static final String FONT_FAMILY = "宋体";
    private static final boolean DEBUG = true;

    // 表格宽度与标题
    private static final int NORMAL_TABLE_WIDTH = 9000;
    private static final int TARGET_TABLE_WIDTH = 14000;
    private static final String DYNAMIC_TABLE_PLACEHOLDER = "{{testIndexTableData}}";
    private static final String TRAINING_TABLE_HEADER = "序号";
    private static final String TARGET_SECTION_TITLE = "四、测试指标要求";
    private static final String TARGET_TABLE_TITLE = "模型测试指标要求详情";  // 已移除该行，仅作为标识保留
    private static final String API_TABLE_TITLE = "附表1 模型API接口说明及模型调用例";

    // 表头与说明行
    private static final List<String> TABLE_HEADER = Arrays.asList(
            "序号", "模型识别类别",
            // 国网企标指标
            "召回率/发现率/检出率", "误检比", "误报率/误检率",
            "平均精度AP", "F1-分数", "识别时间", "IOU平均值",
            // 通用指标
            "平均精度(mPrecision)", "平均召回率(mRecall)", "均值平均精度(mAP@0.5)",
            "漏检率(MissRate)", "虚警率(FalseAlarmRate)", "平均正确率(mAccuracy)"
    );

    private static final List<String> TABLE_DESCRIPTION = Arrays.asList(
            "说明", "",
            // 国网企标说明
            "模型正确识别的样本数与实际存在的样本数之比",
            "误检数量与实际存在的样本数之比",
            "误报数量与总检测数量之比",
            "在一定的IOU值下，某类别识别中的P-R曲线下的面积",
            "精确率和召回率的调和平均数",
            "模型完成一次识别所需的平均时间",
            "交并比的平均值，检测框与真实框重叠程度的指标",
            // 通用指标说明
            "各类别精确率的平均值",
            "各类别召回率的平均值",
            "在IOU=0.5条件下各类别平均精度的平均值",
            "未检测到的真实样本数与总真实样本数之比",
            "误报数量与总样本数之比",
            "各类别正确率的平均值"
    );

    public static void main(String[] args) {
        try {
            // 1. 读取/创建模板
            Resource resource = new ClassPathResource("templates/申请单模板.docx");
            XWPFDocument document;
            if (resource.exists()) {
                document = new XWPFDocument(resource.getInputStream());
            } else {
                System.out.println("模板不存在，新建文档");
                document = new XWPFDocument();
            }

            // 2. 构造测试数据
            Map<String, Object> allData = getAllTestData();

            // 3. 替换占位符（先不处理动态表格占位符）
            replaceAllPlaceholders(document, allData);

            // 4. 插入训练表格
            String[][] trainingTableData = (String[][]) allData.get("trainingTableData");
            insertTrainingTable(document, trainingTableData);

            // 5. 插入动态测试表格（优先使用占位符位置）
            @SuppressWarnings("unchecked")
            List<Map<String, String>> dynamicTableData =
                    (List<Map<String, String>>) allData.get("dynamicTestTableData");
            insertDynamicTestTable(document, dynamicTableData);

            // 6. 清理残留占位符和可能的重复表格
            forceRemovePlaceholders(document, DYNAMIC_TABLE_PLACEHOLDER);
            removeDuplicateDynamicTables(document);

            // 7. 设置页面横向
            setPageLandscape(document);

            // 8. 插入图片
            String imagePath = "C:\\Users\\c\\Desktop\\1752457998361.jpg";
            File imageFile = new File(imagePath);
            if (imageFile.exists() && imageFile.isFile()) {
                XWPFParagraph apiTitlePara = findOrCreateSectionTitle(document, API_TABLE_TITLE);
                int titlePos = document.getPosOfParagraph(apiTitlePara);
                insertImage(document, imagePath, 600, 400, titlePos + 1);
                System.out.println("图片插入成功: " + imagePath);
            } else {
                System.err.println("图片文件不存在，跳过插入: " + imagePath);
            }

            // 9. 保存文档
            String outputPath = "generated_document.docx";
            try (FileOutputStream out = new FileOutputStream(outputPath)) {
                document.write(out);
            }
            document.close();

            System.out.println("生成成功：" + new File(outputPath).getAbsolutePath());

        } catch (Exception e) {
            System.err.println("生成失败：" + e.getMessage());
            if (e.getMessage() != null && e.getMessage().contains("图片")) {
                System.err.println("请检查图片路径是否正确或图片是否损坏");
            }
            e.printStackTrace();
        }
    }

    /**
     * 移除重复的动态表格（保留第一个，删除后续的）
     */
    private static void removeDuplicateDynamicTables(XWPFDocument document) {
        List<XWPFTable> dynamicTables = new ArrayList<>();

        // 收集所有动态表格（通过标题识别）
        for (XWPFTable table : document.getTables()) {
            if (table.getRow(0) != null && !table.getRow(0).getTableCells().isEmpty()) {
                String firstCellText = getParagraphFullText(table.getRow(0).getCell(0).getParagraphs().get(0));
                if (TARGET_TABLE_TITLE.equals(firstCellText)) {
                    dynamicTables.add(table);
                }
            }
        }

        // 如果找到多个相同表格，删除多余的
        if (dynamicTables.size() > 1) {
            if (DEBUG) {
                System.out.println("发现重复动态表格，共 " + dynamicTables.size() + " 个，保留第一个，删除其余");
            }
            // 从后往前删，避免索引变化影响
            for (int i = dynamicTables.size() - 1; i > 0; i--) {
                XWPFTable tableToRemove = dynamicTables.get(i);
                int pos = document.getPosOfTable(tableToRemove);
                if (pos != -1) {
                    document.removeBodyElement(pos);
                    if (DEBUG) {
                        System.out.println("已删除重复表格，位置：" + pos);
                    }
                }
            }
        }
    }

/*    *//**
     * 打印文档结构日志，辅助排查位置问题
     *//*
    private static void logDocumentStructure(String stage, XWPFDocument document) {
        if (!DEBUG) return;

        System.out.println("\n===== " + stage + " =====");
        System.out.println("文档元素总数: " + document.getBodyElements().size());

        for (int i = 0; i < document.getBodyElements().size(); i++) {
            IBodyElement elem = document.getBodyElements().get(i);
            String elemType = elem instanceof XWPFParagraph ? "段落" : "表格";
            String content = "";

            if (elem instanceof XWPFParagraph) {
                content = getParagraphFullText((XWPFParagraph) elem);
                // 只显示前50个字符，避免日志过长
                if (content.length() > 50) {
                    content = content.substring(0, 50) + "...";
                }
            } else if (elem instanceof XWPFTable) {
                XWPFTable table = (XWPFTable) elem;
                if (table.getRow(0) != null && !table.getRow(0).getTableCells().isEmpty()) {
                    content = getParagraphFullText(table.getRow(0).getCell(0).getParagraphs().get(0));
                }
            }

        }
    }*/

    /**
     * 查找或创建章节标题段落
     */
    private static XWPFParagraph findOrCreateSectionTitle(XWPFDocument document, String titleText) {
        XWPFParagraph para = findSectionTitleParagraph(document, titleText);
        if (para == null) {
            para = document.createParagraph();
            XWPFRun run = para.createRun();
            run.setText(titleText);
            run.setBold(true);
            run.setFontSize(FONT_SIZE + 2);
        }
        return para;
    }

    /**
     * 插入动态测试指标表格（优先使用占位符位置）
     */
    private static void insertDynamicTestTable(XWPFDocument document, List<Map<String, String>> dataList) throws Exception {
        // 1. 优先查找占位符位置
        XWPFParagraph placeholderPara = findPlaceholderParagraph(document, DYNAMIC_TABLE_PLACEHOLDER);
        int insertPosition = -1;

        if (placeholderPara != null) {
            // 找到占位符，使用其位置
            insertPosition = document.getPosOfParagraph(placeholderPara);
            if (DEBUG) {
                System.out.println("找到动态表格占位符，位置：" + insertPosition
                        + "，段落内容：" + getParagraphFullText(placeholderPara));
            }
        } else {
            // 未找到占位符，使用标题位置
            XWPFParagraph sectionTitlePara = findOrCreateSectionTitle(document, TARGET_SECTION_TITLE);
            insertPosition = document.getPosOfParagraph(sectionTitlePara) + 1;
            if (DEBUG) {
                System.out.println("未找到占位符，使用标题后位置：" + insertPosition);
            }
        }

        XWPFTable newTable = null;
        // 2. 根据不同场景创建表格
        if (placeholderPara != null) {
            // 有占位符时，先创建临时表格用于构建内容
            newTable = createDynamicTableStructure(dataList);

            // 使用cursor定位到占位符段落后面
            XmlCursor cursor = placeholderPara.getCTP().newCursor();
            cursor.toNextSibling(); // 移动到占位符段落之后

            // 删除占位符段落
            int pos = document.getPosOfParagraph(placeholderPara);
            document.removeBodyElement(pos);

            // 插入表格到cursor位置
            XWPFTable insertedTable = document.insertNewTbl(cursor);
            // 复制表格内容到插入的表格
            copyTableContent(newTable, insertedTable);

            cursor.dispose(); // 释放资源

            if (DEBUG) {
                System.out.println("已在占位符位置插入动态表格");
            }
        } else {
            // 无占位符时，直接在指定位置创建表格
            newTable = createDynamicTableStructure(dataList);
            document.insertTable(insertPosition, newTable);
            if (DEBUG) {
                System.out.println("已在标题后插入动态表格，位置：" + insertPosition);
            }
        }
    }

    /**
     * 创建动态表格结构（不直接添加到文档，避免自动添加到末尾）
     */
    private static XWPFTable createDynamicTableStructure(List<Map<String, String>> dataList) {
        // 修改：移除总标题行后，初始行数 = 分组标题行(1) + 表头行(1) + 说明行(1) + 数据行数
        int initialRowCount = 1 + 1 + 1 + dataList.size();
        // 使用空文档创建临时表格，避免直接添加到目标文档末尾
        XWPFDocument tempDoc = new XWPFDocument();
        XWPFTable table = tempDoc.createTable(initialRowCount, TARGET_COLUMN_COUNT);

        // 修改：移除原总标题行相关代码
        // setupMainTitleRow(table, 0);  // 已删除该行

        // 分组标题行（原索引1 -> 新索引0）
        setupGroupTitleRow(table, 0);

        // 表头行（原索引2 -> 新索引1）
        fillHeaderRow(table, 1);

        // 合并单元格（调整行索引）
        mergeFirstTwoGroups(table);

        // 说明行（原索引3 -> 新索引2）
        fillDescriptionRow(table, 2);

        // 数据行（原起始索引4 -> 新起始索引3）
        fillDynamicDataRows(table, 3, dataList);

        // 应用样式
        applyTableStyle(table, true);

        return table;
    }

    /**
     * 复制表格内容（结构、样式和数据）
     */
    private static void copyTableContent(XWPFTable source, XWPFTable target) {
        // 复制表格属性
        target.getCTTbl().setTblPr(source.getCTTbl().getTblPr());

        // 清除目标表格的默认行
        while (target.getRows().size() > 0) {
            target.removeRow(0);
        }

        // 复制行和单元格
        for (XWPFTableRow sourceRow : source.getRows()) {
            XWPFTableRow targetRow = target.createRow();
            // 确保目标行有足够的单元格
            while (targetRow.getTableCells().size() < sourceRow.getTableCells().size()) {
                targetRow.createCell();
            }

            for (int i = 0; i < sourceRow.getTableCells().size(); i++) {
                XWPFTableCell sourceCell = sourceRow.getCell(i);
                XWPFTableCell targetCell = targetRow.getCell(i);
                if (targetCell == null) {
                    targetCell = targetRow.createCell();
                }

                // 清除目标单元格的默认段落
                while (targetCell.getParagraphs().size() > 0) {
                    targetCell.removeParagraph(0);
                }

                // 复制单元格内容
                for (XWPFParagraph para : sourceCell.getParagraphs()) {
                    XWPFParagraph newPara = targetCell.addParagraph();
                    copyParagraphContent(para, newPara);
                }

                // 复制单元格属性（如合并、对齐）
                if (sourceCell.getCTTc().getTcPr() != null) {
                    targetCell.getCTTc().setTcPr((CTTcPr)sourceCell.getCTTc().getTcPr().copy());
                }
            }
        }
    }

    /**
     * 复制段落内容（包括样式和文本）
     */
    private static void copyParagraphContent(XWPFParagraph source, XWPFParagraph target) {
        // 复制段落样式
        if (source.getCTP().getPPr() != null) {
            target.getCTP().setPPr((CTPPr)source.getCTP().getPPr().copy());
        }

        // 清除目标段落的现有内容
        while (target.getRuns().size() > 0) {
            target.removeRun(0);
        }

        // 复制文本和样式
        for (XWPFRun run : source.getRuns()) {
            XWPFRun newRun = target.createRun();
            newRun.getCTR().set((CTR)run.getCTR().copy());
        }
    }

    /**
     * 设置总标题行（已移除，仅作为历史代码保留）
     */
    @Deprecated
    private static void setupMainTitleRow(XWPFTable table, int rowIndex) {
        XWPFTableRow titleRow = table.getRow(rowIndex);

        while (titleRow.getTableCells().size() < TARGET_COLUMN_COUNT) {
            titleRow.createCell();
        }

        XWPFTableCell titleCell = titleRow.getCell(0);
        CTTcPr tcPr = titleCell.getCTTc().getTcPr();
        if (tcPr == null) {
            tcPr = titleCell.getCTTc().addNewTcPr();
        }
        tcPr.addNewGridSpan().setVal(BigInteger.valueOf(TARGET_COLUMN_COUNT));

        setTableCellValue(titleCell, TARGET_TABLE_TITLE, true);

        for (int i = TARGET_COLUMN_COUNT - 1; i > 0; i--) {
            titleRow.removeCell(i);
        }
    }

    /**
     * 设置分组标题行（调整行索引适配移除总标题行后的结构）
     */
    private static void setupGroupTitleRow(XWPFTable table, int rowIndex) {
        XWPFTableRow groupRow = table.getRow(rowIndex);

        while (groupRow.getTableCells().size() < TARGET_COLUMN_COUNT) {
            groupRow.createCell();
        }

        // 序号列
        XWPFTableCell seqCell = groupRow.getCell(0);
        setTableCellValue(seqCell, "序号", true);

        // 模型识别类别列
        XWPFTableCell categoryCell = groupRow.getCell(1);
        setTableCellValue(categoryCell, "模型识别类别", true);

        // 国网企标列（合并）
        XWPFTableCell stateGridCell = groupRow.getCell(2);
        CTTcPr sgTcPr = stateGridCell.getCTTc().getTcPr();
        if (sgTcPr == null) {
            sgTcPr = stateGridCell.getCTTc().addNewTcPr();
        }
        sgTcPr.addNewGridSpan().setVal(BigInteger.valueOf(STATE_GRID_COLUMNS));
        setTableCellValue(stateGridCell, "国网企标", true);

        // 删除被合并的列
        for (int i = 2 + STATE_GRID_COLUMNS - 1; i > 2; i--) {
            groupRow.removeCell(i);
        }

        // 通用指标列（合并）
        int generalStartIndex = 2 + 1;
        while (groupRow.getTableCells().size() <= generalStartIndex) {
            groupRow.createCell();
        }

        XWPFTableCell generalCell = groupRow.getCell(generalStartIndex);
        CTTcPr gTcPr = generalCell.getCTTc().getTcPr();
        if (gTcPr == null) {
            gTcPr = generalCell.getCTTc().addNewTcPr();
        }
        gTcPr.addNewGridSpan().setVal(BigInteger.valueOf(GENERAL_COLUMNS));
        setTableCellValue(generalCell, "通用指标", true);

        // 删除被合并的列
        for (int i = generalStartIndex + GENERAL_COLUMNS - 1; i > generalStartIndex; i--) {
            if (i < groupRow.getTableCells().size()) {
                groupRow.removeCell(i);
            }
        }
    }

    /**
     * 合并序号和模型识别类别列（调整行索引适配移除总标题行后的结构）
     */
    private static void mergeFirstTwoGroups(XWPFTable table) {
        // 合并分组标题行(0)和表头行(1)的序号列和模型识别类别列
        mergeCellsVertically(table, 0, 0, 1);  // 原1-2行 -> 现0-1行
        mergeCellsVertically(table, 1, 0, 1);  // 原1-2行 -> 现0-1行
    }

    /**
     * 垂直合并单元格
     */
    private static void mergeCellsVertically(XWPFTable table, int col, int fromRow, int toRow) {
        for (int rowIndex = fromRow; rowIndex <= toRow; rowIndex++) {
            XWPFTableCell cell = table.getRow(rowIndex).getCell(col);
            if (cell == null) {
                cell = table.getRow(rowIndex).createCell();
            }

            CTTcPr tcPr = cell.getCTTc().getTcPr();
            if (tcPr == null) {
                tcPr = cell.getCTTc().addNewTcPr();
            }
            CTVMerge vmerge = tcPr.getVMerge();
            if (vmerge == null) {
                vmerge = tcPr.addNewVMerge();
            }
            vmerge.setVal(rowIndex == fromRow ? STMerge.RESTART : STMerge.CONTINUE);
        }
    }

    /**
     * 填充表头行
     */
    private static void fillHeaderRow(XWPFTable table, int rowIndex) {
        if (rowIndex >= table.getNumberOfRows()) return;
        XWPFTableRow row = table.getRow(rowIndex);

        while (row.getTableCells().size() > TARGET_COLUMN_COUNT) {
            row.removeCell(TARGET_COLUMN_COUNT);
        }
        while (row.getTableCells().size() < TARGET_COLUMN_COUNT) {
            row.createCell();
        }

        for (int i = 0; i < TARGET_COLUMN_COUNT; i++) {
            XWPFTableCell cell = row.getCell(i);
            setTableCellValue(cell, TABLE_HEADER.get(i), true);
        }
    }

    /**
     * 填充说明行
     */
    private static void fillDescriptionRow(XWPFTable table, int rowIndex) {
        if (rowIndex >= table.getNumberOfRows()) return;
        XWPFTableRow row = table.getRow(rowIndex);

        while (row.getTableCells().size() > TARGET_COLUMN_COUNT) {
            row.removeCell(TARGET_COLUMN_COUNT);
        }
        while (row.getTableCells().size() < TARGET_COLUMN_COUNT) {
            row.createCell();
        }

        for (int i = 0; i < TARGET_COLUMN_COUNT; i++) {
            XWPFTableCell cell = row.getCell(i);
            setTableCellValue(cell, TABLE_DESCRIPTION.get(i), false);
        }
    }

    /**
     * 填充数据行
     */
    private static void fillDynamicDataRows(XWPFTable table, int startRowIndex, List<Map<String, String>> dataList) {
        for (int i = 0; i < dataList.size(); i++) {
            int rowIndex = startRowIndex + i;
            XWPFTableRow row = table.getRow(rowIndex);
            if (row == null) {
                row = table.createRow();
            }

            while (row.getTableCells().size() > TARGET_COLUMN_COUNT) {
                row.removeCell(TARGET_COLUMN_COUNT);
            }
            while (row.getTableCells().size() < TARGET_COLUMN_COUNT) {
                row.createCell();
            }

            Map<String, String> rowData = dataList.get(i);
            for (int col = 0; col < TARGET_COLUMN_COUNT; col++) {
                XWPFTableCell cell = row.getCell(col);
                String header = TABLE_HEADER.get(col);
                setTableCellValue(cell, rowData.getOrDefault(header, "-"), false);
            }
        }
    }

    private static void replaceAllPlaceholders(XWPFDocument document, Map<String, Object> data) {
        replaceParagraphPlaceholders(document, data);
        replaceTablePlaceholders(document, data);
    }

    private static void replaceParagraphPlaceholders(XWPFDocument document, Map<String, Object> data) {
        int replacedCount = 0;
        for (XWPFParagraph para : new ArrayList<>(document.getParagraphs())) {
            String originalText = getParagraphFullText(para);
            if (originalText.isEmpty()) continue;

            // 跳过动态表格占位符，避免提前替换
            if (originalText.contains(DYNAMIC_TABLE_PLACEHOLDER)) {
                continue;
            }

            String modifiedText = originalText;
            boolean isReplaced = false;

            for (Map.Entry<String, Object> entry : data.entrySet()) {
                if (entry.getValue() instanceof String[][] || entry.getValue() instanceof List) {
                    continue;
                }

                String placeholder = "{{" + entry.getKey() + "}}";
                if (modifiedText.contains(placeholder)) {
                    modifiedText = modifiedText.replace(placeholder, entry.getValue().toString());
                    isReplaced = true;
                    replacedCount++;
                    if (DEBUG) {
                        System.out.println("段落替换占位符: " + placeholder + " -> " + entry.getValue());
                    }
                }
            }

            if (isReplaced) {
                clearParagraph(para);
                XWPFRun run = para.createRun();
                run.setText(modifiedText);
                applyRunStyle(run);
            }
        }

        if (DEBUG) {
            System.out.println("段落占位符替换完成，共" + replacedCount + "处");
        }
    }

    private static void replaceTablePlaceholders(XWPFDocument document, Map<String, Object> data) {
        int replacedCount = 0;
        for (XWPFTable table : document.getTables()) {
            for (XWPFTableRow row : table.getRows()) {
                for (XWPFTableCell cell : row.getTableCells()) {
                    for (XWPFParagraph para : cell.getParagraphs()) {
                        String originalText = getParagraphFullText(para);
                        if (originalText.isEmpty()) continue;

                        String modifiedText = originalText;
                        boolean isReplaced = false;

                        for (Map.Entry<String, Object> entry : data.entrySet()) {
                            if (entry.getValue() instanceof String[][] || entry.getValue() instanceof List) {
                                continue;
                            }

                            String placeholder = "{{" + entry.getKey() + "}}";
                            if (modifiedText.contains(placeholder)) {
                                modifiedText = modifiedText.replace(placeholder, entry.getValue().toString());
                                isReplaced = true;
                                replacedCount++;
                                if (DEBUG) {
                                    System.out.println("表格替换占位符: " + placeholder + " -> " + entry.getValue());
                                }
                            }
                        }

                        if (isReplaced) {
                            clearParagraph(para);
                            XWPFRun run = para.createRun();
                            run.setText(modifiedText);
                            applyRunStyle(run);
                        }
                    }
                }
            }
        }

        if (DEBUG) {
            System.out.println("表格占位符替换完成，共" + replacedCount + "处");
        }
    }

    private static void setPageLandscape(XWPFDocument document) {
        CTDocument1 ctDoc = document.getDocument();
        CTBody ctBody = ctDoc.getBody();
        if (ctBody == null) {
            ctBody = ctDoc.addNewBody();
        }

        CTSectPr sectPr = ctBody.getSectPr();
        if (sectPr == null) {
            sectPr = ctBody.addNewSectPr();
        }

        CTPageSz pageSz = sectPr.getPgSz();
        if (pageSz == null) {
            pageSz = sectPr.addNewPgSz();
        }
        pageSz.setOrient(STPageOrientation.LANDSCAPE);
        pageSz.setW(BigInteger.valueOf(842 * 20));
        pageSz.setH(BigInteger.valueOf(595 * 20));

        CTPageMar pageMar = sectPr.getPgMar();
        if (pageMar == null) {
            pageMar = sectPr.addNewPgMar();
        }
        pageMar.setTop(BigInteger.valueOf(720));
        pageMar.setRight(BigInteger.valueOf(720));
        pageMar.setBottom(BigInteger.valueOf(720));
        pageMar.setLeft(BigInteger.valueOf(720));
    }

    private static Map<String, Object> getAllTestData() {
        Map<String, Object> data = new HashMap<>();

        // 1. 基本信息
        data.put("modelName", "近义词模型");
        data.put("modelSource", "国网天津电力-人工智能项目");
        data.put("testDemandDesc", "模型性能测试");
        data.put("modelFunction", "BERT(NLP双向变换器模型)");
        data.put("buildUnitName", "天津电力公司信息通信公司");
        data.put("buildUnitAddress", "天津市河北区昆纬路153号");
        data.put("buildUnitLeader", "孙妍");
        data.put("buildUnitContact", "17622323501");
        data.put("btUnitName", "北京中电普华信息技术有限公司");
        data.put("btUnitAddress", "北京市昌平区清河路嘉铭奥森产业园");
        data.put("btUnitLeader", "陈树森");
        data.put("btUnitContact", "15210497989");

        // 2. 准备工作信息
        data.put("modelFileName", "nlp_synonym.tar");
        data.put("modelEncapWay", "Docker镜像");
        data.put("modelFileSize", "1.45G");
        data.put("modelHashValue", "H2595697445687");
        data.put("modelDeployAddr", "天津信通公司");
        data.put("modelPort", "18888");
        data.put("modelCudaVersion", "10.1");
        data.put("modelDriveVersion", "440.64.00");

        // 3. 训练样本表格数据
        String[][] trainingTableArray = {
                {"序号", "样本集/模型训练代码文件名称", "数量", "备注"},
                {"1", "样本集_分类模型", "100", "训练样本"},
                {"2", "训练代码_train.py", "1", "模型训练代码"},
                {"3", "测试用例.csv", "50", "验证数据"},
        };
        data.put("trainingTableData", trainingTableArray);

        // 4. 动态测试指标表格数据
        List<Map<String, String>> dynamicTableData = new ArrayList<>();
        dynamicTableData.add(createDynamicTableRow(
                "1", "近义词识别",
                "≥0.90", "≤0.03", "≤0.05",
                "≥0.85", "≥0.92", "≤0.5s", "≥0.88",
                "≥0.91", "≥0.90", "≥0.87", "≤0.05",
                "≤0.04", "≥0.92"
        ));
        dynamicTableData.add(createDynamicTableRow(
                "2", "反义词识别",
                "≥0.88", "≤0.05", "≤0.07",
                "≥0.82", "≥0.90", "≤0.6s", "≥0.85",
                "≥0.89", "≥0.88", "≥0.84", "≤0.07",
                "≤0.06", "≥0.90"
        ));
        dynamicTableData.add(createDynamicTableRow(
                "3", "语义相似度",
                "≥0.92", "≤0.02", "≤0.04",
                "≥0.88", "≥0.93", "≤0.4s", "≥0.90",
                "≥0.92", "≥0.91", "≥0.89", "≤0.04",
                "≤0.03", "≥0.94"
        ));
        data.put("dynamicTestTableData", dynamicTableData);

        return data;
    }

    /**
     * 创建动态表格行数据
     */
    private static Map<String, String> createDynamicTableRow(
            String id, String category,
            String recall, String falseDetectionRatio, String falseAlarmRatio,
            String ap, String f1, String recognitionTime, String iouAvg,
            String mPrecision, String mRecall, String mAP, String missRate,
            String falseAlarmRate, String mAccuracy) {

        Map<String, String> row = new HashMap<>();
        row.put("序号", id);
        row.put("模型识别类别", category);

        // 国网企标指标
        row.put("召回率/发现率/检出率", recall);
        row.put("误检比", falseDetectionRatio);
        row.put("误报率/误检率", falseAlarmRatio);
        row.put("平均精度AP", ap);
        row.put("F1-分数", f1);
        row.put("识别时间", recognitionTime);
        row.put("IOU平均值", iouAvg);

        // 通用指标
        row.put("平均精度(mPrecision)", mPrecision);
        row.put("平均召回率(mRecall)", mRecall);
        row.put("均值平均精度(mAP@0.5)", mAP);
        row.put("漏检率(MissRate)", missRate);
        row.put("虚警率(FalseAlarmRate)", falseAlarmRate);
        row.put("平均正确率(mAccuracy)", mAccuracy);

        return row;
    }

    private static void insertTrainingTable(XWPFDocument document, String[][] tableData) {
        XWPFTable targetTable = findTableByHeader(document, TRAINING_TABLE_HEADER);

        if (targetTable == null) {
            if (DEBUG) System.out.println("创建训练表格");
            XWPFParagraph titlePara = document.createParagraph();
            XWPFRun titleRun = titlePara.createRun();
            titleRun.setText("训练样本表格");
            titleRun.setBold(true);
            titleRun.setFontSize(FONT_SIZE + 1);
            targetTable = document.createTable(tableData.length, tableData[0].length);
        } else {
            clearTableRows(targetTable, 1);
        }

        // 填充表格数据
        for (int i = 0; i < tableData.length; i++) {
            XWPFTableRow row = (i < targetTable.getNumberOfRows()) ? targetTable.getRow(i) : targetTable.createRow();
            String[] rowData = tableData[i];
            for (int j = 0; j < rowData.length; j++) {
                XWPFTableCell cell = (j < row.getTableCells().size()) ? row.getCell(j) : row.createCell();
                setTableCellValue(cell, rowData[j], i == 0);
            }
        }

        applyTableStyle(targetTable, false);
    }

    private static XWPFParagraph findSectionTitleParagraph(XWPFDocument document, String titleText) {
        for (XWPFParagraph para : document.getParagraphs()) {
            if (getParagraphFullText(para).trim().equals(titleText)) {
                return para;
            }
        }
        return null;
    }

    private static XWPFTable findTableByHeader(XWPFDocument document, String headerText) {
        for (XWPFTable table : document.getTables()) {
            if (table.getRow(0) != null && !table.getRow(0).getTableCells().isEmpty()) {
                XWPFTableCell firstCell = table.getRow(0).getCell(0);
                if (firstCell != null && getParagraphFullText(firstCell.getParagraphs().get(0)).equals(headerText)) {
                    return table;
                }
            }
        }
        return null;
    }

    /**
     * 增强占位符查找逻辑：使用更灵活的匹配方式
     */
    private static XWPFParagraph findPlaceholderParagraph(XWPFDocument document, String placeholder) {
        for (XWPFParagraph para : document.getParagraphs()) {
            String paraText = getParagraphFullText(para).trim(); // 去除首尾空格
            // 忽略占位符内外的空格，灵活匹配
            String normalizedText = paraText.replaceAll("\\s+", "");
            String normalizedPlaceholder = placeholder.replaceAll("\\s+", "");
            if (normalizedText.contains(normalizedPlaceholder)) {
                return para;
            }
        }
        return null;
    }

    private static void clearParagraph(XWPFParagraph para) {
        while (para.getRuns().size() > 0) {
            para.removeRun(0);
        }
    }

    private static void clearTableRows(XWPFTable table, int keepRows) {
        int rowCount = table.getNumberOfRows();
        for (int i = rowCount - 1; i >= keepRows; i--) {
            table.removeRow(i);
        }
    }

    private static void setTableCellValue(XWPFTableCell cell, String value, boolean isHeader) {
        if (value == null || value.trim().isEmpty()) {
            value = "";
        }

        XWPFParagraph para = cell.getParagraphs().get(0);
        clearParagraph(para);
        XWPFRun run = para.createRun();
        run.setText(value);
        run.setFontFamily(FONT_FAMILY);
        run.setFontSize(FONT_SIZE);
        if (isHeader) {
            run.setBold(true);
            run.setFontSize(FONT_SIZE + 1);
        }
        para.setAlignment(ParagraphAlignment.CENTER);
        cell.setVerticalAlignment(XWPFTableCell.XWPFVertAlign.CENTER);
    }

    private static void applyTableStyle(XWPFTable table, boolean isTargetTable) {
        CTTblPr tblPr = table.getCTTbl().getTblPr() != null ? table.getCTTbl().getTblPr() : table.getCTTbl().addNewTblPr();

        // 设置表格宽度
        CTTblWidth tblWidth = tblPr.getTblW() != null ? tblPr.getTblW() : tblPr.addNewTblW();
        tblWidth.setW(BigInteger.valueOf(isTargetTable ? TARGET_TABLE_WIDTH : NORMAL_TABLE_WIDTH));
        tblWidth.setType(STTblWidth.DXA);

        // 设置表格边框
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

        // 单元格对齐
        for (XWPFTableRow row : table.getRows()) {
            for (XWPFTableCell cell : row.getTableCells()) {
                cell.setVerticalAlignment(XWPFTableCell.XWPFVertAlign.CENTER);
                for (XWPFParagraph para : cell.getParagraphs()) {
                    para.setAlignment(ParagraphAlignment.CENTER);
                }
            }
        }
    }

    private static String getParagraphFullText(XWPFParagraph para) {
        if (para == null) return "";
        StringBuilder fullText = new StringBuilder();
        for (XWPFRun run : para.getRuns()) {
            String runText = run.getText(0);
            if (runText != null) fullText.append(runText);
        }
        return fullText.toString().trim();
    }

    private static void applyRunStyle(XWPFRun run) {
        run.setFontFamily(FONT_FAMILY);
        run.setFontSize(FONT_SIZE);
    }

    private static void forceRemovePlaceholders(XWPFDocument document, String... placeholders) {
        List<XWPFParagraph> paragraphs = new ArrayList<>(document.getParagraphs());
        for (XWPFParagraph para : paragraphs) {
            String text = getParagraphFullText(para);
            for (String placeholder : placeholders) {
                if (text.contains(placeholder)) {
                    int pos = document.getPosOfParagraph(para);
                    if (pos != -1) {
                        document.removeBodyElement(pos);
                        if (DEBUG) {
                            System.out.println("强制删除残留占位符，位置：" + pos);
                        }
                    }
                    break;
                }
            }
        }
    }

    // 图片插入方法
    private static void insertImage(XWPFDocument document, String imagePath, int width, int height, int position) throws Exception {
        InputStream is;
        try {
            Resource resource = new ClassPathResource(imagePath);
            is = resource.getInputStream();
        } catch (Exception e) {
            is = new FileInputStream(new File(imagePath));
        }

        String fileExt = imagePath.substring(imagePath.lastIndexOf(".") + 1).toLowerCase();
        int pictureType;
        switch (fileExt) {
            case "jpg":
            case "jpeg":
                pictureType = XWPFDocument.PICTURE_TYPE_JPEG;
                break;
            case "png":
                pictureType = XWPFDocument.PICTURE_TYPE_PNG;
                break;
            case "gif":
                pictureType = XWPFDocument.PICTURE_TYPE_GIF;
                break;
            case "bmp":
                pictureType = XWPFDocument.PICTURE_TYPE_BMP;
                break;
            default:
                throw new IllegalArgumentException("不支持的图片格式：" + fileExt);
        }

        byte[] bytes = IOUtils.toByteArray(is);
        is.close();

        XWPFParagraph picturePara = new XWPFParagraph(CTP.Factory.newInstance(), document);
        picturePara.setAlignment(ParagraphAlignment.CENTER);
        XWPFRun run = picturePara.createRun();

        run.addPicture(
                new ByteArrayInputStream(bytes),
                pictureType,
                "image." + fileExt,
                Units.toEMU(width),
                Units.toEMU(height)
        );

        List<IBodyElement> elements = document.getBodyElements();
        XmlCursor cursor = null;

        try {
            if (position >= 0 && position <= elements.size()) {
                if (position < elements.size()) {
                    IBodyElement element = elements.get(position);
                    if (element instanceof XWPFParagraph) {
                        cursor = ((XWPFParagraph) element).getCTP().newCursor();
                    } else if (element instanceof XWPFTable) {
                        cursor = ((XWPFTable) element).getCTTbl().newCursor();
                    }
                } else {
                    cursor = document.getDocument().getBody().newCursor();
                    cursor.toEndToken();
                }
            } else {
                cursor = document.getDocument().getBody().newCursor();
                cursor.toEndToken();
            }

            if (cursor != null) {
                XWPFParagraph newPara = document.insertNewParagraph(cursor);
                newPara.getCTP().setPPr(picturePara.getCTP().getPPr());
                for (XWPFRun sourceRun : picturePara.getRuns()) {
                    XWPFRun targetRun = newPara.createRun();
                    targetRun.getCTR().set(sourceRun.getCTR());
                }
            }
        } finally {
            if (cursor != null) {
                cursor.dispose();
            }
        }
    }
}