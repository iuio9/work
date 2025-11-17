package com.qczy.common.word;

import org.apache.commons.compress.utils.IOUtils;
import org.apache.poi.util.Units;
import org.apache.poi.xwpf.usermodel.*;
import org.apache.xmlbeans.XmlCursor;
import org.openxmlformats.schemas.wordprocessingml.x2006.main.*;
import org.springframework.core.io.ClassPathResource;
import org.springframework.core.io.Resource;

import javax.servlet.http.HttpServletResponse;
import java.io.*;
import java.math.BigInteger;
import java.net.URLEncoder;
import java.util.*;

/**
 * Word文档生成工具类 - 修复"暂无数据"重复显示问题
 */
public class WordTemplateMainTest {

    // 表格核心配置
    private static final int FONT_SIZE = 10;
    private static final String FONT_FAMILY = "宋体";

    // 表格宽度与标题
    private static final int NORMAL_TABLE_WIDTH = 9000;
    private static final int TARGET_TABLE_WIDTH = 14000;
    private static final String DYNAMIC_TABLE_PLACEHOLDER = "{{testIndexTableData}}";
    private static final String TRAINING_TABLE_HEADER = "序号";
    private static final String TARGET_SECTION_TITLE = "四、测试指标要求";
    private static final String TARGET_TABLE_TITLE = "模型测试指标要求详情";
    private static final String API_TABLE_TITLE = "附表1 模型API接口说明及模型调用例";
    private static final String NO_DATA_TEXT = "暂无数据"; // 无指标时显示的文本

    // 表头与说明行映射关系
    private static final Map<String, String> PROP_TO_HEADER = new HashMap<String, String>() {{
        put("recall", "召回率/发现率/检出率");
        put("falseAlarmRate", "误检比");
        put("falseAlarmRate1", "误报率/误检率");
        put("ap", "平均精度AP");
        put("f1", "F1-分数");
        put("time", "识别时间");
        put("iou", "IOU平均值");
        put("mPrecision", "平均精度");
        put("mRecall", "平均召回率");
        put("mAP@0.5", "均值平均精度");
        put("MissRate", "漏检率");
        put("FalseAlarmRate", "虚警率");
        put("mAccuracy", "平均正确率");
    }};

    private static final Map<String, String> PROP_TO_DESCRIPTION = new HashMap<String, String>() {{
        put("recall", "模型正确识别的样本数与实际存在的样本数之比");
        put("falseAlarmRate", "误报数量与总检测数量之比");
        put("falseAlarmRate1", "误检数量与实际存在的样本数之比");
        put("ap", "在一定的IOU值下，某类别识别中的P-R曲线下的面积");
        put("f1", "精确率和召回率的调和平均数");
        put("time", "模型完成一次识别所需的平均时间");
        put("iou", "交并比的平均值，检测框与真实框重叠程度的指标");
        put("mPrecision", "各类别精确率的平均值");
        put("mRecall", "各类别召回率的平均值");
        put("mAP@0.5", "在IOU=0.5条件下各类别平均精度的平均值");
        put("MissRate", "未检测到的真实样本数与总真实样本数之比");
        put("FalseAlarmRate", "误报数量与总样本数之比");
        put("mAccuracy", "各类别正确率的平均值");
    }};

    /**
     * 生成Word文档并通过HTTP响应输出（供浏览器下载）
     */
    public static void generateDocument(InputStream templateIs, Map<String, Object> data, HttpServletResponse response) throws Exception {
        try (XWPFDocument document = new XWPFDocument(templateIs);
             OutputStream out = response.getOutputStream()) {

            log("模板加载成功，段落数：" + document.getParagraphs().size() + "，表格数：" + document.getTables().size());

            // 1. 替换占位符（先不处理动态表格占位符）
            replaceAllPlaceholders(document, data);

            // 2. 插入训练表格
            String[][] trainingTableData = (String[][]) data.get("trainingTableData");
            insertTrainingTable(document, trainingTableData);

            // 3. 插入动态测试表格（优先使用占位符位置）
            @SuppressWarnings("unchecked")
            List<Map<String, Object>> dynamicTableData = (List<Map<String, Object>>) data.get("dynamicTestTableData");
            insertDynamicTestTable(document, dynamicTableData);

            // 4. 清理残留占位符和可能的重复内容
            forceRemovePlaceholders(document, DYNAMIC_TABLE_PLACEHOLDER);
            removeDuplicateDynamicTables(document);
            removeDuplicateNoDataParagraphs(document); // 新增：移除重复的"暂无数据"

            // 5. 设置页面横向
            setPageLandscape(document);

            // 6. 插入图片
            String imagePath = (String) data.getOrDefault("apiImagePath", "");
            if (!imagePath.isEmpty()) {
                File imageFile = new File(imagePath);
                if (imageFile.exists() && imageFile.isFile()) {
                    XWPFParagraph apiTitlePara = findOrCreateSectionTitle(document, API_TABLE_TITLE);
                    int titlePos = document.getPosOfParagraph(apiTitlePara);
                    insertImage(document, imagePath, 600, 400, titlePos + 1);
                    log("图片插入成功: " + imagePath);
                } else {
                    log("图片文件不存在，跳过插入: " + imagePath);
                }
            }

            // 7. 设置响应头，支持浏览器下载
            response.setContentType("application/vnd.openxmlformats-officedocument.wordprocessingml.document");
            String fileName = URLEncoder.encode("模型申请单.docx", "UTF-8");
            response.setHeader("Content-Disposition", "attachment; filename*=UTF-8''" + fileName);

            // 8. 写入响应流
            document.write(out);
            out.flush();
            log("文档生成成功并写入响应流");
        }
    }

    /**
     * 新增：移除重复的"暂无数据"段落，只保留第一个
     */
    private static void removeDuplicateNoDataParagraphs(XWPFDocument document) {
        List<XWPFParagraph> noDataParagraphs = new ArrayList<>();

        // 收集所有包含"暂无数据"的段落
        for (XWPFParagraph para : document.getParagraphs()) {
            String text = getParagraphFullText(para).trim();
            if (NO_DATA_TEXT.equals(text)) {
                noDataParagraphs.add(para);
            }
        }

        // 如果找到多个，保留第一个，删除其余
        if (noDataParagraphs.size() > 1) {
            log("发现重复的" + NO_DATA_TEXT + "段落，共 " + noDataParagraphs.size() + " 个，保留第一个");
            // 从后往前删，避免索引变化影响
            for (int i = noDataParagraphs.size() - 1; i > 0; i--) {
                XWPFParagraph paraToRemove = noDataParagraphs.get(i);
                int pos = document.getPosOfParagraph(paraToRemove);
                if (pos != -1) {
                    document.removeBodyElement(pos);
                    log("已删除重复的" + NO_DATA_TEXT + "段落，位置：" + pos);
                }
            }
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
            log("发现重复动态表格，共 " + dynamicTables.size() + " 个，保留第一个，删除其余");
            // 从后往前删，避免索引变化影响
            for (int i = dynamicTables.size() - 1; i > 0; i--) {
                XWPFTable tableToRemove = dynamicTables.get(i);
                int pos = document.getPosOfTable(tableToRemove);
                if (pos != -1) {
                    document.removeBodyElement(pos);
                    log("已删除重复表格，位置：" + pos);
                }
            }
        }
    }

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
            run.setFontFamily(FONT_FAMILY);
            log("创建新章节标题：" + titleText);
        } else {
            log("找到章节标题：" + titleText);
        }
        return para;
    }

    /**
     * 插入动态测试指标表格（优先使用占位符位置）
     */
    private static void insertDynamicTestTable(XWPFDocument document, List<Map<String, Object>> dataList) throws Exception {
        // 检查是否已存在"暂无数据"段落，如果有则直接返回，避免重复处理
        if (hasNoDataParagraph(document)) {
            log("已存在" + NO_DATA_TEXT + "段落，跳过重复处理");
            return;
        }

        if (dataList == null || dataList.isEmpty()) {
            log("动态表格数据为空，跳过插入");
            return;
        }

        // 先删除模板中已有的测试指标表格，避免残留
        deleteExistingTestTables(document);

        // 1. 获取配置行（第一个元素是配置）
        Map<String, Object> configMap = dataList.get(0);
        @SuppressWarnings("unchecked")
        List<String> gridCheckedProps = (List<String>) configMap.getOrDefault("gridCheckedProps", new ArrayList<>());
        @SuppressWarnings("unchecked")
        List<String> commonCheckedProps = (List<String>) configMap.getOrDefault("commonCheckedProps", new ArrayList<>());
        log("动态表格配置：国网列数=" + gridCheckedProps.size() + "，通用列数=" + commonCheckedProps.size());

        // 2. 判断是否有选择的指标（国网或通用至少有一个）
        boolean hasSelectedIndicators = !gridCheckedProps.isEmpty() || !commonCheckedProps.isEmpty();

        // 3. 确定插入位置（无论有无数据都需要确定位置）
        XWPFParagraph placeholderPara = findPlaceholderParagraph(document, DYNAMIC_TABLE_PLACEHOLDER);
        int insertPosition = -1;

        if (placeholderPara != null) {
            insertPosition = document.getPosOfParagraph(placeholderPara);
            log("找到动态表格占位符，位置：" + insertPosition
                    + "，段落内容：" + getParagraphFullText(placeholderPara));
        } else {
            XWPFParagraph sectionTitlePara = findOrCreateSectionTitle(document, TARGET_SECTION_TITLE);
            insertPosition = document.getPosOfParagraph(sectionTitlePara) + 1;
            log("未找到占位符，使用标题后位置：" + insertPosition);
        }

        // 4. 处理无选择指标的情况：显示"暂无数据"
        if (!hasSelectedIndicators) {
            handleNoIndicatorsCase(document, placeholderPara, insertPosition);
            return;
        }

        // 5. 有选择的指标：正常创建表格
        int totalColumns = 2 + gridCheckedProps.size() + commonCheckedProps.size();
        if (totalColumns < 2) {
            log("无效的列配置，总列数不足");
            return;
        }

        // 创建表格结构
        XWPFTable newTable = createDynamicTableStructure(
                dataList.subList(1, dataList.size()), // 排除配置行
                gridCheckedProps,
                commonCheckedProps,
                totalColumns
        );

        // 插入表格
        if (placeholderPara != null) {
            // 使用cursor定位到占位符段落后面
            XmlCursor cursor = placeholderPara.getCTP().newCursor();
            cursor.toNextSibling(); // 移动到占位符段落之后

            // 删除占位符段落
            int pos = document.getPosOfParagraph(placeholderPara);
            document.removeBodyElement(pos);

            // 插入表格到cursor位置
            XWPFTable insertedTable = document.insertNewTbl(cursor);
            copyTableContent(newTable, insertedTable);
            cursor.dispose();
            log("已在占位符位置插入动态表格");
        } else {
            // 无占位符时，直接在指定位置创建表格
            document.insertTable(insertPosition, newTable);
            log("已在标题后插入动态表格，位置：" + insertPosition);
        }

        // 验证插入结果
        validateTableInsertion(document, insertPosition);
    }

    /**
     * 检查文档中是否已存在"暂无数据"段落
     */
    private static boolean hasNoDataParagraph(XWPFDocument document) {
        for (XWPFParagraph para : document.getParagraphs()) {
            if (NO_DATA_TEXT.equals(getParagraphFullText(para).trim())) {
                return true;
            }
        }
        return false;
    }

    /**
     * 处理无选择指标的情况：确保"暂无数据"只显示一次且在正确位置
     */
    private static void handleNoIndicatorsCase(XWPFDocument document, XWPFParagraph placeholderPara, int insertPosition) {
        // 双重检查，避免重复创建
        if (hasNoDataParagraph(document)) {
            log("已存在" + NO_DATA_TEXT + "段落，无需重复创建");
            // 如果存在占位符，仍然需要删除
            if (placeholderPara != null) {
                int pos = document.getPosOfParagraph(placeholderPara);
                if (pos != -1) {
                    document.removeBodyElement(pos);
                    log("已删除动态表格占位符");
                }
            }
            return;
        }

        log("未选择任何指标，显示：" + NO_DATA_TEXT);

        // 创建显示"暂无数据"的段落
        XWPFParagraph noDataPara = document.createParagraph();
        XWPFRun run = noDataPara.createRun();
        run.setText(NO_DATA_TEXT);
        run.setFontFamily(FONT_FAMILY);
        run.setFontSize(FONT_SIZE);
        noDataPara.setAlignment(ParagraphAlignment.CENTER); // 居中显示

        // 优先使用占位符位置
        if (placeholderPara != null) {
            // 使用占位符的cursor插入新段落
            XmlCursor cursor = placeholderPara.getCTP().newCursor();
            cursor.toNextSibling(); // 移动到占位符后面

            // 插入新段落
            XWPFParagraph insertedPara = document.insertNewParagraph(cursor);
            // 复制样式和内容
            insertedPara.getCTP().setPPr(noDataPara.getCTP().getPPr());
            for (XWPFRun sourceRun : noDataPara.getRuns()) {
                XWPFRun targetRun = insertedPara.createRun();
                targetRun.getCTR().set(sourceRun.getCTR());
            }

            // 删除原始占位符段落
            int pos = document.getPosOfParagraph(placeholderPara);
            if (pos != -1) {
                document.removeBodyElement(pos);
                log("已删除动态表格占位符并在其位置插入" + NO_DATA_TEXT);
            }
            cursor.dispose();
        } else if (insertPosition >= 0 && insertPosition <= document.getBodyElements().size()) {
            // 无占位符但有明确位置时，插入到指定位置
            XmlCursor cursor = null;
            try {
                if (insertPosition < document.getBodyElements().size()) {
                    IBodyElement element = document.getBodyElements().get(insertPosition);
                    if (element instanceof XWPFParagraph) {
                        cursor = ((XWPFParagraph) element).getCTP().newCursor();
                    } else if (element instanceof XWPFTable) {
                        cursor = ((XWPFTable) element).getCTTbl().newCursor();
                    }
                    cursor.toPrevSibling(); // 移动到目标元素前面
                } else {
                    cursor = document.getDocument().getBody().newCursor();
                    cursor.toEndToken();
                }

                if (cursor != null) {
                    XWPFParagraph insertedPara = document.insertNewParagraph(cursor);
                    insertedPara.getCTP().setPPr(noDataPara.getCTP().getPPr());
                    for (XWPFRun sourceRun : noDataPara.getRuns()) {
                        XWPFRun targetRun = insertedPara.createRun();
                        targetRun.getCTR().set(sourceRun.getCTR());
                    }
                    log("已在指定位置" + insertPosition + "插入" + NO_DATA_TEXT);
                }
            } finally {
                if (cursor != null) {
                    cursor.dispose();
                }
            }
        } else {
            //  fallback: 添加到文档末尾（但会在后续清理中检查重复）
           // document.addParagraph(noDataPara);
            log("已在文档末尾插入" + NO_DATA_TEXT);
        }
    }

    /**
     * 删除文档中已存在的测试指标表格
     */
    private static void deleteExistingTestTables(XWPFDocument document) {
        List<XWPFTable> tablesToRemove = new ArrayList<>();

        // 收集所有需要删除的表格
        for (XWPFTable table : document.getTables()) {
            if (table.getRow(0) != null && !table.getRow(0).getTableCells().isEmpty()) {
                String firstCellText = getParagraphFullText(table.getRow(0).getCell(0).getParagraphs().get(0));
                if (TARGET_TABLE_TITLE.equals(firstCellText) ||
                        firstCellText.contains("测试指标") ||
                        firstCellText.contains("国网企标")) {
                    tablesToRemove.add(table);
                }
            }
        }

        // 从后往前删除，避免索引问题
        for (int i = tablesToRemove.size() - 1; i >= 0; i--) {
            XWPFTable table = tablesToRemove.get(i);
            int pos = document.getPosOfTable(table);
            if (pos != -1) {
                document.removeBodyElement(pos);
                log("已删除模板中存在的测试表格，位置：" + pos);
            }
        }
    }

    /**
     * 验证表格是否成功插入
     */
    private static void validateTableInsertion(XWPFDocument document, int expectedPos) {
        if (expectedPos >= 0 && expectedPos < document.getBodyElements().size()) {
            IBodyElement element = document.getBodyElements().get(expectedPos);
            if (element instanceof XWPFTable) {
                XWPFTable table = (XWPFTable) element;
                log("验证：表格插入成功，行数=" + table.getNumberOfRows() + "，列数=" +
                        (table.getRow(0) != null ? table.getRow(0).getTableCells().size() : 0));
            } else {
                log("验证：插入位置不是表格，实际是：" + element.getClass().getSimpleName());
            }
        }
    }

    /**
     * 创建动态表格结构
     */
    private static XWPFTable createDynamicTableStructure(
            List<Map<String, Object>> dataRows,
            List<String> gridProps,
            List<String> commonProps,
            int totalColumns) {

        // 总行数：分组标题行(1) + 表头行(1) + 说明行(1) + 数据行数
        int totalRows = 3 + dataRows.size();
        XWPFDocument tempDoc = new XWPFDocument();
        XWPFTable table = tempDoc.createTable(totalRows, totalColumns);
        log("创建表格：" + totalRows + "行，" + totalColumns + "列");

        // 设置表格宽度
        CTTblWidth tblWidth = table.getCTTbl().getTblPr().addNewTblW();
        tblWidth.setW(BigInteger.valueOf(TARGET_TABLE_WIDTH));
        tblWidth.setType(STTblWidth.DXA);

        // 1. 分组标题行（第一行）- 动态列数
        setupGroupTitleRow(table, 0, gridProps, commonProps);

        // 2. 表头行（第二行）
        fillHeaderRow(table, 1, gridProps, commonProps);

        // 3. 说明行（第三行）
        fillDescriptionRow(table, 2, gridProps, commonProps);

        // 4. 垂直合并序号和类别列（0-2行）
        mergeFirstTwoColumns(table, 0, 2);

        // 5. 填充数据行（从第四行开始）
        fillDynamicDataRows(table, 3, dataRows, gridProps, commonProps);

        // 6. 应用样式（确保边框可见）
        applyTableStyle(table, true);

        // 7. 最终检查并清理所有行的多余列（特殊处理第一行）
        int hasGrid = gridProps.isEmpty() ? 0 : 1;
        int hasCommon = commonProps.isEmpty() ? 0 : 1;
        int firstRowExpectedColumns = 2 + hasGrid + hasCommon;
        ensureNoExtraColumns(table, totalColumns, firstRowExpectedColumns);

        return table;
    }

    /**
     * 确保表格所有行都没有多余列（适配第一行列数动态变化）
     */
    private static void ensureNoExtraColumns(XWPFTable table, int totalColumns, int firstRowExpectedColumns) {
        for (int i = 0; i < table.getNumberOfRows(); i++) {
            XWPFTableRow row = table.getRow(i);
            if (row == null) continue;

            // 第一行按动态计算的列数处理，其他行按总列数处理
            int rowExpectedColumns = (i == 0) ? firstRowExpectedColumns : totalColumns;

            // 移除多余的列
            while (row.getTableCells().size() > rowExpectedColumns) {
                row.removeCell(row.getTableCells().size() - 1);
                log("清理第" + i + "行多余列，当前列数：" + row.getTableCells().size());
            }

            // 补充不足的列
            while (row.getTableCells().size() < rowExpectedColumns) {
                row.createCell();
                log("补充第" + i + "行不足列，当前列数：" + row.getTableCells().size());
            }
        }
    }

    /**
     * 复制表格内容（结构、样式和数据）
     */
    private static void copyTableContent(XWPFTable source, XWPFTable target) {
        // 复制表格属性
        if (source.getCTTbl().getTblPr() != null) {
            target.getCTTbl().setTblPr((CTTblPr) source.getCTTbl().getTblPr().copy());
        }

        // 清除目标表格的默认行
        while (target.getRows().size() > 0) {
            target.removeRow(0);
        }

        // 复制行和单元格
        for (XWPFTableRow sourceRow : source.getRows()) {
            XWPFTableRow targetRow = target.createRow();

            // 确保目标行列数与源行一致
            while (targetRow.getTableCells().size() < sourceRow.getTableCells().size()) {
                targetRow.createCell();
            }
            while (targetRow.getTableCells().size() > sourceRow.getTableCells().size()) {
                targetRow.removeCell(targetRow.getTableCells().size() - 1);
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
                    targetCell.getCTTc().setTcPr((CTTcPr) sourceCell.getCTTc().getTcPr().copy());
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
            target.getCTP().setPPr((CTPPr) source.getCTP().getPPr().copy());
        }

        // 清除目标段落的现有内容
        while (target.getRuns().size() > 0) {
            target.removeRun(0);
        }

        // 复制文本和样式
        for (XWPFRun run : source.getRuns()) {
            XWPFRun newRun = target.createRun();
            newRun.getCTR().set((CTR) run.getCTR().copy());
        }
    }

    /**
     * 设置分组标题行（第一行）- 动态列数与合并
     */
    private static void setupGroupTitleRow(XWPFTable table, int rowIndex, List<String> gridProps, List<String> commonProps) {
        XWPFTableRow groupRow = table.getRow(rowIndex);
        // 计算第一行应有的列数：2（序号+类别） + 存在的指标类型数（国网/通用）
        int hasGrid = gridProps.isEmpty() ? 0 : 1;
        int hasCommon = commonProps.isEmpty() ? 0 : 1;
        int maxColumns = 2 + hasGrid + hasCommon;

        // 1. 清理/补充列数到maxColumns
        while (groupRow.getTableCells().size() > maxColumns) {
            groupRow.removeCell(groupRow.getTableCells().size() - 1);
            log("清理分组标题行多余列，当前列数：" + groupRow.getTableCells().size());
        }
        while (groupRow.getTableCells().size() < maxColumns) {
            groupRow.createCell();
            log("补充分组标题行列数，当前列数：" + groupRow.getTableCells().size());
        }

        // 2. 清空单元格，准备填充标题
        for (int i = 0; i < groupRow.getTableCells().size(); i++) {
            setTableCellValue(groupRow.getCell(i), "", true);
        }

        // 3. 固定列标题：序号、模型识别类别
        setTableCellValue(groupRow.getCell(0), "序号", true);
        setTableCellValue(groupRow.getCell(1), "模型识别类别", true);

        // 4. 动态填充指标类型列（仅为存在的类型创建）
        int currentColIndex = 2; // 从第三列开始填充指标类型

        // 若存在国网企标，创建对应列并合并
        if (hasGrid > 0) {
            XWPFTableCell gridCell = groupRow.getCell(currentColIndex);
            CTTcPr tcPr = gridCell.getCTTc().getTcPr() != null
                    ? gridCell.getCTTc().getTcPr()
                    : gridCell.getCTTc().addNewTcPr();
            // 合并列数 = 国网指标的数量
            tcPr.addNewGridSpan().setVal(BigInteger.valueOf(gridProps.size()));
            setTableCellValue(gridCell, "国网企标", true);
            log("国网企标列：合并跨度=" + gridProps.size() + "，覆盖" + gridProps.size() + "个指标列");
            currentColIndex++; // 移动到下一列
        }

        // 若存在通用指标，创建对应列并合并
        if (hasCommon > 0) {
            XWPFTableCell commonCell = groupRow.getCell(currentColIndex);
            CTTcPr tcPr = commonCell.getCTTc().getTcPr() != null
                    ? commonCell.getCTTc().getTcPr()
                    : commonCell.getCTTc().addNewTcPr();
            // 合并列数 = 通用指标的数量
            tcPr.addNewGridSpan().setVal(BigInteger.valueOf(commonProps.size()));
            setTableCellValue(commonCell, "通用指标", true);
            log("通用指标列：合并跨度=" + commonProps.size() + "，覆盖" + commonProps.size() + "个指标列");
        }

        // 5. 隐藏可能的多余列（若有）
        for (int i = 0; i < groupRow.getTableCells().size(); i++) {
            XWPFTableCell cell = groupRow.getCell(i);
            CTTc tc = cell.getCTTc();
            CTTcPr tcPr = tc.getTcPr() != null ? tc.getTcPr() : tc.addNewTcPr();

            // 超过应有列数的列隐藏
            if (i >= maxColumns) {
                CTTblWidth width = tcPr.isSetTcW() ? tcPr.getTcW() : tcPr.addNewTcW();
                width.setW(BigInteger.ZERO);
                width.setType(STTblWidth.DXA);
                log("隐藏第一行第" + (i + 1) + "列（多余列）");
            }
        }

        log("分组标题行设置完成，列数=" + maxColumns + "，国网=" + hasGrid + "，通用=" + hasCommon);
    }

    /**
     * 垂直合并序号和模型识别类别列（覆盖0-2行）
     */
    private static void mergeFirstTwoColumns(XWPFTable table, int fromRow, int toRow) {
        mergeCellsVertically(table, 0, fromRow, toRow);  // 序号列
        mergeCellsVertically(table, 1, fromRow, toRow);  // 模型识别类别列
        log("完成序号列和类别列的垂直合并（行 " + fromRow + " 到 " + toRow + "）");
    }

    /**
     * 垂直合并单元格（确保行有足够的单元格）
     */
    private static void mergeCellsVertically(XWPFTable table, int col, int fromRow, int toRow) {
        if (col < 0 || fromRow < 0 || toRow >= table.getNumberOfRows() || fromRow > toRow) {
            log("垂直合并参数无效：col=" + col + ", fromRow=" + fromRow + ", toRow=" + toRow);
            return;
        }

        for (int rowIndex = fromRow; rowIndex <= toRow; rowIndex++) {
            XWPFTableRow row = table.getRow(rowIndex);
            if (row == null) {
                row = table.createRow(); // 行不存在则创建
            }

            // 确保行有足够的单元格（至少col+1个）
            while (row.getTableCells().size() <= col) {
                row.createCell(); // 不足则创建新单元格
            }

            // 获取或创建目标单元格
            XWPFTableCell cell = row.getCell(col);
            if (cell == null) {
                cell = row.createCell();
            }

            // 设置垂直合并属性
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
     * 填充表头行（第二行）- 适配动态列数
     */
    private static void fillHeaderRow(XWPFTable table, int rowIndex, List<String> gridProps, List<String> commonProps) {
        if (rowIndex >= table.getNumberOfRows()) return;
        XWPFTableRow row = table.getRow(rowIndex);
        int totalCells = 2 + gridProps.size() + commonProps.size();

        // 确保列数正确
        while (row.getTableCells().size() > totalCells) {
            row.removeCell(row.getTableCells().size() - 1);
        }
        while (row.getTableCells().size() < totalCells) {
            row.createCell();
        }

        // 固定列留空（已合并）
        setTableCellValue(row.getCell(0), "", true);
        setTableCellValue(row.getCell(1), "", true);

        // 填充指标列标题（按实际存在的类型顺序）
        int colIndex = 2;
        // 先填充国网指标（若存在）
        for (String prop : gridProps) {
            String header = PROP_TO_HEADER.getOrDefault(prop, prop);
            setTableCellValue(row.getCell(colIndex++), header, true);
            log("国网企标列头：" + header);
        }
        // 再填充通用指标（若存在）
        for (String prop : commonProps) {
            String header = PROP_TO_HEADER.getOrDefault(prop, prop);
            setTableCellValue(row.getCell(colIndex++), header, true);
            log("通用指标列头：" + header);
        }
    }

    /**
     * 填充说明行（第三行）
     */
    private static void fillDescriptionRow(XWPFTable table, int rowIndex, List<String> gridProps, List<String> commonProps) {
        if (rowIndex >= table.getNumberOfRows()) return;
        XWPFTableRow row = table.getRow(rowIndex);
        int totalCells = 2 + gridProps.size() + commonProps.size();

        // 确保列数正确
        while (row.getTableCells().size() > totalCells) {
            row.removeCell(row.getTableCells().size() - 1);
        }
        while (row.getTableCells().size() < totalCells) {
            row.createCell();
        }

        // 说明列
        setTableCellValue(row.getCell(0), "说明", false);
        setTableCellValue(row.getCell(1), "", false);

        // 填充国网指标说明
        int colIndex = 2;
        for (String prop : gridProps) {
            String desc = PROP_TO_DESCRIPTION.getOrDefault(prop, "");
            setTableCellValue(row.getCell(colIndex++), desc, false);
        }

        // 填充通用指标说明
        for (String prop : commonProps) {
            String desc = PROP_TO_DESCRIPTION.getOrDefault(prop, "");
            setTableCellValue(row.getCell(colIndex++), desc, false);
        }
    }

    /**
     * 填充数据行（从第四行开始）
     */
    private static void fillDynamicDataRows(
            XWPFTable table,
            int startRowIndex,
            List<Map<String, Object>> dataRows,
            List<String> gridProps,
            List<String> commonProps) {

        int totalCells = 2 + gridProps.size() + commonProps.size();

        for (int i = 0; i < dataRows.size(); i++) {
            int rowIndex = startRowIndex + i;
            XWPFTableRow row = table.getRow(rowIndex);
            if (row == null) {
                row = table.createRow();
                log("数据行" + (i + 1) + "不存在，创建新行");
            }

            // 确保列数正确
            while (row.getTableCells().size() > totalCells) {
                row.removeCell(row.getTableCells().size() - 1);
            }
            while (row.getTableCells().size() < totalCells) {
                row.createCell();
            }

            Map<String, Object> rowData = dataRows.get(i);
            String index = rowData.getOrDefault("序号", (i + 1) + "").toString();
            String category = rowData.getOrDefault("类别", "").toString();

            // 序号和类别
            setTableCellValue(row.getCell(0), index, false);
            setTableCellValue(row.getCell(1), category, false);
            log("数据行" + index + "：类别=" + category);

            // 填充国网指标数据
            @SuppressWarnings("unchecked")
            Map<String, Object> gridData = (Map<String, Object>) rowData.getOrDefault("gridData", new HashMap<>());
            int colIndex = 2;
            for (String prop : gridProps) {
                Object value = gridData.get(prop);
                String displayValue = (value != null && !value.toString().isEmpty()) ? value.toString() : "-";
                setTableCellValue(row.getCell(colIndex++), displayValue, false);
            }

            // 填充通用指标数据
            @SuppressWarnings("unchecked")
            Map<String, Object> commonData = (Map<String, Object>) rowData.getOrDefault("commonData", new HashMap<>());
            for (String prop : commonProps) {
                Object value = commonData.get(prop);
                String displayValue = (value != null && !value.toString().isEmpty()) ? value.toString() : "-";
                setTableCellValue(row.getCell(colIndex++), displayValue, false);
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
                    log("替换段落占位符：" + placeholder + " -> " + entry.getValue());
                }
            }

            if (isReplaced) {
                clearParagraph(para);
                XWPFRun run = para.createRun();
                run.setText(modifiedText);
                applyRunStyle(run);
            }
        }

        log("段落占位符替换完成，共" + replacedCount + "处");
    }

    private static void replaceTablePlaceholders(XWPFDocument document, Map<String, Object> data) {
        int replacedCount = 0;
        for (XWPFTable table : document.getTables()) {
            // 跳过测试指标表格的占位符替换，避免干扰
            if (table.getRow(0) != null && !table.getRow(0).getTableCells().isEmpty()) {
                String firstCellText = getParagraphFullText(table.getRow(0).getCell(0).getParagraphs().get(0));
                if (TARGET_TABLE_TITLE.equals(firstCellText)) {
                    continue;
                }
            }

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

        log("表格占位符替换完成，共" + replacedCount + "处");
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
        pageSz.setW(BigInteger.valueOf(842 * 20));  // 宽
        pageSz.setH(BigInteger.valueOf(595 * 20));  // 高

        // 缩小边距，避免表格超出页面
        CTPageMar pageMar = sectPr.getPgMar();
        if (pageMar == null) {
            pageMar = sectPr.addNewPgMar();
        }
        pageMar.setTop(BigInteger.valueOf(500));
        pageMar.setRight(BigInteger.valueOf(500));
        pageMar.setBottom(BigInteger.valueOf(500));
        pageMar.setLeft(BigInteger.valueOf(500));
        log("设置页面为横向布局，调整边距");
    }

    private static void insertTrainingTable(XWPFDocument document, String[][] tableData) {
        if (tableData == null || tableData.length == 0) {
            log("训练表格数据为空，跳过插入");
            return;
        }

        XWPFTable targetTable = findTableByHeader(document, TRAINING_TABLE_HEADER);

        if (targetTable == null) {
            log("未找到训练表格，创建新表格");
            XWPFParagraph titlePara = document.createParagraph();
            XWPFRun titleRun = titlePara.createRun();
            titleRun.setText("训练样本表格");
            titleRun.setBold(true);
            titleRun.setFontSize(FONT_SIZE + 1);
            targetTable = document.createTable(tableData.length, tableData[0].length);
        } else {
            log("找到训练表格，清理旧数据");
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
        log("训练表格填充完成，行数：" + tableData.length);
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
            String paraText = getParagraphFullText(para).trim();
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

    /**
     * 应用表格样式（确保边框可见）
     */
    private static void applyTableStyle(XWPFTable table, boolean isTargetTable) {
        CTTblPr tblPr = table.getCTTbl().getTblPr() != null ? table.getCTTbl().getTblPr() : table.getCTTbl().addNewTblPr();

        // 设置表格宽度
        CTTblWidth tblWidth = tblPr.getTblW() != null ? tblPr.getTblW() : tblPr.addNewTblW();
        tblWidth.setW(BigInteger.valueOf(isTargetTable ? TARGET_TABLE_WIDTH : NORMAL_TABLE_WIDTH));
        tblWidth.setType(STTblWidth.DXA);

        // 设置表格边框（使用黑色边框确保可见）
        CTBorder border = CTBorder.Factory.newInstance();
        border.setColor("000000"); // 黑色边框
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
                        log("强制删除残留占位符，位置：" + pos);
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

    private static void log(String message) {
        System.out.println("[WordTemplate] " + message);
    }
}
