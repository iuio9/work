package com.qczy.utils;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.apache.poi.ss.usermodel.*;
import org.apache.poi.xssf.usermodel.XSSFWorkbook;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;
import java.io.InputStream;
import java.util.HashMap;
import java.util.Map;

/**
 * @Author: hh
 * @Version: 1.0
 * @Date: 2025/6/6 14:40
 * @Description:
 */
public class ParamsUtils {

    private static final ObjectMapper objectMapper = new ObjectMapper();


    // 读取JSON文件并转换为Map
    public static  Map<String, Object> readJsonFile(MultipartFile file) throws IOException {
        return objectMapper.readValue(file.getInputStream(), Map.class);
    }


    // 将XLSX文件转换为Map - 第一列为参数名，第二列为值
    public static Map<String, Object> convertXlsxToMap(MultipartFile file) throws IOException {
        Map<String, Object> data = new HashMap<>();
        if (file == null || file.isEmpty()) {
            return data;
        }

        try (InputStream is = file.getInputStream();
             Workbook workbook = new XSSFWorkbook(is)) {

            Sheet sheet = workbook.getSheetAt(0);
            int rowCount = sheet.getLastRowNum();

            // 从第1行(index=0)开始读取数据
            for (int i = 0; i <= rowCount; i++) {
                Row row = sheet.getRow(i);
                if (row == null) continue;

                // 获取第一列作为参数名
                Cell paramNameCell = row.getCell(0);
                if (paramNameCell == null) continue;

                String paramName = getCellValueAsString(paramNameCell);
                if (paramName == null || paramName.trim().isEmpty()) continue;

                // 获取第二列作为参数值
                Cell paramValueCell = row.getCell(1);
                Object paramValue = paramValueCell != null ?
                        getCellValue(paramValueCell) : null;

                data.put(paramName, paramValue);
            }
        }

        return data;
    }


    // 获取单元格值
    public static Object getCellValue(Cell cell) {
        if (cell == null) return null;

        switch (cell.getCellType()) {
            case STRING:
                return cell.getStringCellValue();
            case NUMERIC:
                if (DateUtil.isCellDateFormatted(cell)) {
                    return cell.getDateCellValue();
                } else {
                    double numericValue = cell.getNumericCellValue();
                    // 检查是否为整数
                    if (numericValue == (int) numericValue) {
                        return (int) numericValue;
                    }
                    return numericValue;
                }
            case BOOLEAN:
                return cell.getBooleanCellValue();
            case FORMULA:
                return cell.getCellFormula();
            default:
                return null;
        }
    }

    // 获取单元格值并转换为字符串
    public static String getCellValueAsString(Cell cell) {
        if (cell == null) return null;

        switch (cell.getCellType()) {
            case STRING:
                return cell.getStringCellValue();
            case NUMERIC:
                if (DateUtil.isCellDateFormatted(cell)) {
                    return cell.getDateCellValue().toString();
                } else {
                    double numericValue = cell.getNumericCellValue();
                    // 检查是否为整数
                    if (numericValue == (int) numericValue) {
                        return String.valueOf((int) numericValue);
                    }
                    return String.valueOf(numericValue);
                }
            case BOOLEAN:
                return String.valueOf(cell.getBooleanCellValue());
            case FORMULA:
                return cell.getCellFormula();
            default:
                return null;
        }

    }






}
