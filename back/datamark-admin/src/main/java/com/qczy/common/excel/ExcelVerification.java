package com.qczy.common.excel;

import com.qczy.utils.StringUtils;
import org.apache.poi.ss.usermodel.*;
import org.apache.poi.xssf.usermodel.XSSFWorkbook;
import org.springframework.web.multipart.MultipartFile;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;


/**
 * @Author: hh
 * @Version: 1.0
 * @Date: 2025/4/7 15:07
 * @Description: excel 格式校验
 */
public class ExcelVerification {
    private static final String RGB_COLOR_REGEX = "^[（(]R=(0|[1-9]\\d?|1\\d\\d|2[0-4]\\d|25[0-5]),G=(0|[1-9]\\d?|1\\d\\d|2[0-4]\\d|25[0-5]),B=(0|[1-9]\\d?|1\\d\\d|2[0-4]\\d|25[0-5])[）)]$";
    private static final String CODE_REGEX = "^[a-zA-Z0-9/_-]+$";

    public static boolean validateExcel(String filePath) {
        StringBuilder validationResult = new StringBuilder();
        try (FileInputStream fis = new FileInputStream(new File(filePath));
             Workbook workbook = new XSSFWorkbook(fis)) {
            Sheet sheet = workbook.getSheetAt(0);
            for (Row row : sheet) {
                if (row.getRowNum() == 0) {
                    continue; // 跳过表头
                }
                // 校验标签编码
                Cell codeCell = row.getCell(0);
                String codeValue = getCellValueAsString(codeCell);
                if (codeValue.isEmpty()) {
                    continue;
                }

                // 检查标签编码是否仅包含数字和字母
                if (!codeValue.matches(CODE_REGEX)) {
                    return false;
                }

                // 检查标签编码中 / 的数量
                if (countOccurrences(codeValue, '/') > 1) {
                    return false;
                }

                // 校验标签名
                Cell nameCell = row.getCell(1);
                String nameValue = getCellValueAsString(nameCell);
                if (nameValue.isEmpty()) {
                    continue;
                }
                // 检查标签名中 / 的数量
                if (countOccurrences(nameValue, '/') > 1) {
                    return false;
                }

                // 校验颜色
                Cell colorCell = row.getCell(2);
                String colorValue = getCellValueAsString(colorCell);
                if (colorValue.isEmpty()){
                    continue;
                }
                if (!colorValue.matches(RGB_COLOR_REGEX)) {
                    return false;
                }
            }
           /* if (validationResult.length() == 0) {
                return true;
            }*/

            return true;
        } catch (IOException e) {
            validationResult.append("读取文件时发生错误: ").append(e.getMessage());
        }
        return false;
    }


    private static String getCellValueAsString(Cell cell) {
        if (cell == null) {
            return "";
        }
        switch (cell.getCellType()) {
            case STRING:
                return cell.getStringCellValue();
            case NUMERIC:
                return String.valueOf(cell.getNumericCellValue());
            case BOOLEAN:
                return String.valueOf(cell.getBooleanCellValue());
            case FORMULA:
                return cell.getCellFormula();
            default:
                return "";
        }
    }

    private static int countOccurrences(String str, char c) {
        int count = 0;
        for (int i = 0; i < str.length(); i++) {
            if (str.charAt(i) == c) {
                count++;
            }
        }
        return count;
    }

}
