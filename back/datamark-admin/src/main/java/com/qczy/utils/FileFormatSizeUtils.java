package com.qczy.utils;

import cn.hutool.core.util.NumberUtil;
import org.springframework.stereotype.Component;

import java.math.BigDecimal;
import java.text.DecimalFormat;

/**
 * @Author: hh
 * @Version: 1.0
 * @Date: 2024/8/7 20:56
 * @Description:
 */
@Component
public class FileFormatSizeUtils {

    public static void main(String[] args) {
        BigDecimal bigDecimal = NumberUtil.toBigDecimal("2.561765376E13");
        System.out.println(bigDecimal);
        System.out.println(bigDecimal.longValue());
        System.out.println(formatSize(bigDecimal.longValue()));
    }

    public static String formatSize(long fileS) {
        DecimalFormat df = new DecimalFormat("#.00");
        String fileSizeString = "";
        String wrongSize = "0B";
        if (fileS == 0) {
            return wrongSize;
        }
        if (fileS < 1024) {
            fileSizeString = df.format((double) fileS) + "B";
        } else if (fileS < 1048576) {
            fileSizeString = df.format((double) fileS / 1024) + "KB";
        } else if (fileS < 1073741824) {
            fileSizeString = df.format((double) fileS / 1048576) + "MB";
        } else {
            fileSizeString = df.format((double) fileS / 1073741824) + "GB";
        }
        return fileSizeString;
    }

    // 动态转换 CPU 频率单位
    public static String convertFrequency(long hz) {
        if (hz >= 1_000_000_000) {
            // 转换为 GHz
            double ghz = hz / 1_000_000_000.0;
            return String.format("%.2f GHz", ghz);
        } else if (hz >= 1_000_000) {
            // 转换为 MHz
            double mhz = hz / 1_000_000.0;
            return String.format("%.2f MHz", mhz);
        } else {
            // 保持 Hz
            return hz + " Hz";
        }
    }

    public static String calculatePercentage(long part, long total) {
        if (total == 0) {
            return "0.00%"; // 避免除以零
        }

        // 计算百分比，转为小数并乘以 100
        double percentage = (double) part / total * 100;

        // 格式化为保留两位小数的字符串
        return String.format("%.2f%%", percentage);
    }


}
