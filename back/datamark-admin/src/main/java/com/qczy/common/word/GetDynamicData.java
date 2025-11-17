package com.qczy.common.word;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.DeserializationFeature;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.qczy.mapper.ModelBaseMapper;
import com.qczy.mapper.ModelConfigureMapper;
import com.qczy.model.entity.ModelBaseEntity;
import com.qczy.model.entity.ModelConfigureEntity;
import com.qczy.utils.EntityUtils;
import com.qczy.utils.StringUtils;
import org.apache.poi.ss.usermodel.*;
import org.apache.poi.xssf.usermodel.XSSFCell;
import org.apache.poi.xssf.usermodel.XSSFRow;
import org.apache.poi.xssf.usermodel.XSSFSheet;
import org.apache.poi.xssf.usermodel.XSSFWorkbook;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.core.io.ClassPathResource;
import org.springframework.core.io.Resource;
import org.springframework.stereotype.Service;
import org.springframework.util.CollectionUtils;

import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import java.io.*;
import java.net.URLEncoder;
import java.text.SimpleDateFormat;
import java.util.*;

/**
 * 动态数据获取服务类 - 支持浏览器下载Word文档
 * 功能优化：只获取选中指标，汇总所有识别类型的选中指标作为表头，无值时显示 "-"
 */
@Service
public class GetDynamicData {

    @Autowired
    private ModelBaseMapper modelBaseMapper;

    @Autowired
    private ModelConfigureMapper modelConfigureMapper;

    // 模板文件路径
    private static final String TEMPLATE_PATH = "templates/申请单模板.docx";


    /**
     * 下载模型申请单Word文档到浏览器（核心入口）
     */
    public void downloadWord(Integer modelId, HttpServletRequest request, HttpServletResponse response) {
        try {
            // 1. 获取动态数据
            Map<String, Object> dynamicData = getDynamicData(modelId);

            // 2. 获取模板输入流
            try (InputStream templateIs = getTemplateInputStream()) {
                if (templateIs == null) {
                    throw new RuntimeException("模板输入流获取失败");
                }

                // 3. 调用文档生成工具类生成并下载文档
                WordTemplateMainTest.generateDocument(templateIs, dynamicData, response);
            }
        } catch (Exception e) {
            System.err.println("文档处理错误：" + e.getMessage());
            e.printStackTrace();
            try {
                if (!response.isCommitted()) {
                    response.sendError(HttpServletResponse.SC_INTERNAL_SERVER_ERROR, "生成文档失败：" + e.getMessage());
                }
            } catch (IOException ioException) {
                ioException.printStackTrace();
            }
        }
    }


    /**
     * 获取模板输入流（供文档生成工具类使用）
     */
    public InputStream getTemplateInputStream() {
        try {
            Resource resource = new ClassPathResource(TEMPLATE_PATH);
            if (resource.exists()) {
                return resource.getInputStream();
            } else {
                throw new FileNotFoundException("模板文件不存在：" + TEMPLATE_PATH);
            }
        } catch (IOException e) {
            System.err.println("加载模板失败：" + e.getMessage());
            return null;
        }
    }


    /**
     * 核心：获取动态数据（适配WordTemplateMainTest所需格式）
     */
    private Map<String, Object> getDynamicData(Integer modelId) {
        Map<String, Object> data = new HashMap<>();

        // 1. 基础信息（模型基本信息）
        ModelBaseEntity modelBaseEntity = modelBaseMapper.selectById(modelId);
        if (modelBaseEntity == null) {
            throw new RuntimeException("未找到模型ID为" + modelId + "的基础信息");
        }
        data.putAll(EntityUtils.convertToMap(modelBaseEntity)); // 转换实体为Map

        // 2. 配置信息（模型配置参数）
        ModelConfigureEntity modelConfigureEntity = modelConfigureMapper.selectOne(
                new LambdaQueryWrapper<ModelConfigureEntity>()
                        .eq(ModelConfigureEntity::getModelBaseId, modelId)
        );
        if (modelConfigureEntity != null) {
            data.putAll(EntityUtils.convertToMap(modelConfigureEntity));
        } else {
            throw new RuntimeException("未找到模型ID为" + modelId + "的配置信息");
        }

        // 3. 训练样本表格数据
        data.put("trainingTableData", getTrainingTableData(modelConfigureEntity));

        // 4. 动态测试指标表格数据
        List<TestIndexCategory> categories = parseTestIndexJson(modelConfigureEntity.getModelClass());
        List<Map<String, Object>> dynamicTestTableData = convertToDynamicTableData(categories);
        data.put("dynamicTestTableData", dynamicTestTableData);


        // 5. 附表一 模型API接口说明及模型调用例数据
        data.put("apiImagePath", modelConfigureEntity.getModelInterfaceDesc());

        return data;
    }


    /**
     * 解析测试指标JSON字符串为实体列表
     */
    private List<TestIndexCategory> parseTestIndexJson(String jsonStr) {
        if (jsonStr == null || jsonStr.trim().isEmpty()) {
            throw new RuntimeException("测试指标JSON字符串为空");
        }
        try {
            ObjectMapper objectMapper = new ObjectMapper();
            objectMapper.configure(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES, false);
            return objectMapper.readValue(jsonStr, new TypeReference<List<TestIndexCategory>>() {});
        } catch (Exception e) {
            throw new RuntimeException("解析测试指标JSON失败：" + e.getMessage(), e);
        }
    }


    /**
     * 转换为动态表格数据格式
     * 优化点：汇总所有选中指标作为表头，无值显示 "-"
     */
    private List<Map<String, Object>> convertToDynamicTableData(List<TestIndexCategory> categories) {
        List<Map<String, Object>> result = new ArrayList<>();
        if (CollectionUtils.isEmpty(categories)) {
            return result;
        }

        // 1. 汇总所有识别类型的选中指标（去重）
        Set<String> allGridCheckedSet = new LinkedHashSet<>();
        Set<String> allCommonCheckedSet = new LinkedHashSet<>();

        for (TestIndexCategory category : categories) {
            if (!CollectionUtils.isEmpty(category.getGridCheckedRowKeys())) {
                allGridCheckedSet.addAll(category.getGridCheckedRowKeys());
            }
            if (!CollectionUtils.isEmpty(category.getCommonCheckedRowKeys())) {
                allCommonCheckedSet.addAll(category.getCommonCheckedRowKeys());
            }
        }

        List<String> gridCheckedProps = new ArrayList<>(allGridCheckedSet);
        List<String> commonCheckedProps = new ArrayList<>(allCommonCheckedSet);


        // 2. 配置行（存储所有选中的列信息）
        Map<String, Object> configRow = new HashMap<>();
        configRow.put("gridCheckedProps", gridCheckedProps);
        configRow.put("commonCheckedProps", commonCheckedProps);
        result.add(configRow);


        // 3. 数据行（处理无值时显示 "-"）
        int index = 1;
        for (TestIndexCategory category : categories) {
            Map<String, Object> dataRow = new HashMap<>();
            dataRow.put("序号", index++);
            dataRow.put("类别", category.getClassName());

            // 3.1 国网企标数据（关键修复：处理Object转String）
            Map<String, Object> gridData = new HashMap<>();
            for (String prop : gridCheckedProps) {
                gridData.put(prop, "-"); // 默认值
            }
            if (!CollectionUtils.isEmpty(category.getModelGridData())) {
                for (TestIndexItem item : category.getModelGridData()) {
                    String prop = item.getProp();
                    Object valueObj = item.getValue(); // 先以Object接收
                    String value = (valueObj != null) ? valueObj.toString() : "-"; // 强制转为String

                    if (gridCheckedProps.contains(prop) && !value.trim().isEmpty()) {
                        gridData.put(prop, value);
                    }
                }
            }
            dataRow.put("gridData", gridData);


            // 3.2 通用指标数据（关键修复：处理Object转String）
            Map<String, Object> commonData = new HashMap<>();
            for (String prop : commonCheckedProps) {
                commonData.put(prop, "-"); // 默认值
            }
            if (!CollectionUtils.isEmpty(category.getModelCommonData())) {
                for (TestIndexItem item : category.getModelCommonData()) {
                    String prop = item.getProp();
                    Object valueObj = item.getValue(); // 先以Object接收
                    String value = (valueObj != null) ? valueObj.toString() : "-"; // 强制转为String

                    if (commonCheckedProps.contains(prop) && !value.trim().isEmpty()) {
                        commonData.put(prop, value);
                    }
                }
            }
            dataRow.put("commonData", commonData);


            result.add(dataRow);
        }

        return result;
    }


    /**
     * 生成训练样本表格数据
     * 从配置的Excel文件中读取数据并转换为二维数组格式
     * @param config 模型配置实体，包含测试用例文件路径
     * @return 训练样本表格数据（二维字符串数组），读取失败返回null
     */
    private String[][] getTrainingTableData(ModelConfigureEntity config) {
        // 验证配置及文件路径有效性
        if (config != null && !StringUtils.isEmpty(config.getTestCase())) {
            String filePath = config.getTestCase();
            // 从Excel文件读取数据
            List<String[]> excelData = readTrainingDataFromExcel(filePath);
            // 转换为二维数组并返回
            if (excelData != null && !excelData.isEmpty()) {
                return excelData.toArray(new String[0][]);
            }
        }
        // 读取失败返回null
        return null;
    }

    /**
     * 从Excel文件读取训练样本数据
     * 读取Excel第一列作为序号，依次读取后续列作为名称、数量和备注
     * @param filePath Excel文件路径
     * @return 包含表头和数据的字符串数组列表，读取失败返回null
     */
    private List<String[]> readTrainingDataFromExcel(String filePath) {
        File file = new File(filePath);
        // 验证文件存在性和有效性
        if (!file.exists() || !file.isFile() || !isValidExcelFile(file)) {
            return null;
        }

        List<String[]> dataList = new ArrayList<>();
        try (FileInputStream fis = new FileInputStream(file);
             Workbook workbook = WorkbookFactory.create(fis)) { // 兼容.xls和.xlsx格式

            // 获取第一个工作表
            Sheet sheet = workbook.getSheetAt(0);
            if (sheet == null) return null;

            // 读取表头（第一行）
            Row headerRow = sheet.getRow(0);
            if (headerRow != null) {
                String[] header = new String[4];
                header[0] = getCellValue(headerRow.getCell(0)); // 序号表头
                header[1] = getCellValue(headerRow.getCell(1)); // 名称表头
                header[2] = getCellValue(headerRow.getCell(2)); // 数量表头
                header[3] = getCellValue(headerRow.getCell(3)); // 备注表头
                dataList.add(header);
            }

            // 读取数据行（从第二行开始）
            int rowEnd = sheet.getLastRowNum();
            for (int i = 1; i <= rowEnd; i++) {
                Row row = sheet.getRow(i);
                if (row == null) continue; // 跳过空行

                String[] rowData = new String[4];
                rowData[0] = getCellValue(row.getCell(0)); // 序号（使用Excel第一列值）
                rowData[1] = getCellValue(row.getCell(1)); // 样本集/文件名称
                rowData[2] = getCellValue(row.getCell(2)); // 数量
                rowData[3] = getCellValue(row.getCell(3)); // 备注

                // 过滤有效行（序号和名称至少有一个不为空）
                if (!StringUtils.isEmpty(rowData[0]) && !StringUtils.isEmpty(rowData[1])) {
                    dataList.add(rowData);
                }
            }
            return dataList;
        } catch (Exception e) {
            // 捕获所有Excel处理异常（文件损坏、格式错误等）
            return null;
        }
    }

    /**
     * 验证文件是否为有效的Excel文件
     * 通过文件扩展名和文件头标识双重验证
     * @param file 待验证文件
     * @return 有效返回true，否则返回false
     */
    private boolean isValidExcelFile(File file) {
        String name = file.getName().toLowerCase();
        // 验证文件扩展名
        if (!(name.endsWith(".xlsx") || name.endsWith(".xls"))) {
            return false;
        }
        // 验证文件头标识
        return checkFileHeader(file);
    }

    /**
     * 检查文件头标识以确认Excel文件有效性
     * .xlsx文件头为0x504B0304，.xls文件头为0xD0CF11E0
     * @param file 待检查文件
     * @return 符合Excel文件头标识返回true，否则返回false
     */
    private boolean checkFileHeader(File file) {
        try (FileInputStream fis = new FileInputStream(file)) {
            byte[] header = new byte[8]; // 读取前8字节文件头
            fis.read(header);
            String hex = bytesToHex(header); // 转换为十六进制字符串
            // 验证Excel文件头标识
            return hex.startsWith("504b0304") || hex.startsWith("d0cf11e0");
        } catch (Exception e) {
            return false;
        }
    }

    /**
     * 将字节数组转换为十六进制字符串
     * 用于文件头标识验证
     * @param bytes 字节数组
     * @return 十六进制字符串
     */
    private String bytesToHex(byte[] bytes) {
        StringBuilder sb = new StringBuilder();
        for (byte b : bytes) {
            sb.append(String.format("%02x", b)); // 格式化每个字节为两位十六进制
        }
        return sb.toString();
    }

    /**
     * 获取Excel单元格的值并转换为字符串
     * 处理不同数据类型（字符串、数字、日期、布尔值等）
     * @param cell Excel单元格对象
     * @return 单元格值的字符串表示，空单元格返回空字符串
     */
    private String getCellValue(Cell cell) {
        if (cell == null) return "";

        // 获取单元格类型（处理公式单元格的缓存结果）
        CellType type = cell.getCellType();
        if (type == CellType.FORMULA) {
            type = cell.getCachedFormulaResultType();
        }

        switch (type) {
            case STRING:
                return cell.getStringCellValue().trim();
            case NUMERIC:
                // 处理日期类型
                if (DateUtil.isCellDateFormatted(cell)) {
                    return new SimpleDateFormat("yyyy-MM-dd").format(cell.getDateCellValue());
                } else {
                    // 处理数字类型（避免长数字科学计数法）
                    double val = cell.getNumericCellValue();
                    return val == (long) val ? String.valueOf((long) val) : String.valueOf(val);
                }
            case BOOLEAN:
                return String.valueOf(cell.getBooleanCellValue());
            default:
                return "";
        }
    }
















    // ------------------------------ 内部静态类（修正value类型） ------------------------------

    /**
     * 测试指标类别
     */
    private static class TestIndexCategory {
        private String className;
        private List<TestIndexItem> modelGridData;
        private List<TestIndexItem> modelCommonData;
        private List<String> gridCheckedRowKeys;
        private List<String> commonCheckedRowKeys;

        // Getters and Setters
        public String getClassName() { return className; }
        public void setClassName(String className) { this.className = className; }
        public List<TestIndexItem> getModelGridData() { return modelGridData; }
        public void setModelGridData(List<TestIndexItem> modelGridData) { this.modelGridData = modelGridData; }
        public List<TestIndexItem> getModelCommonData() { return modelCommonData; }
        public void setModelCommonData(List<TestIndexItem> modelCommonData) { this.modelCommonData = modelCommonData; }
        public List<String> getGridCheckedRowKeys() { return gridCheckedRowKeys; }
        public void setGridCheckedRowKeys(List<String> gridCheckedRowKeys) { this.gridCheckedRowKeys = gridCheckedRowKeys; }
        public List<String> getCommonCheckedRowKeys() { return commonCheckedRowKeys; }
        public void setCommonCheckedRowKeys(List<String> commonCheckedRowKeys) { this.commonCheckedRowKeys = commonCheckedRowKeys; }
    }

    /**
     * 测试指标项（关键修正：value改为Object类型，兼容数字/字符串）
     */
    private static class TestIndexItem {
        private String prop;
        private Object value; // 改为Object类型，支持数字和字符串

        // Getters and Setters
        public String getProp() { return prop; }
        public void setProp(String prop) { this.prop = prop; }
        public Object getValue() { return value; }
        public void setValue(Object value) { this.value = value; } // 接收任意类型的值
    }
}