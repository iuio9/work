package com.qczy.common.markInfo;

import java.io.File;

/**
 * @Author: hh
 * @Version: 1.0
 * @Date: 2025/1/17 10:02
 * @Description: json 格式校验
 */
public class JsonFormatCheck {


    public static int jsonCheck(String jsonPath) {
        File file = new File(jsonPath);
        if (!file.exists()) {
            return 0;
        }
        String str  = file.getAbsolutePath();
        String[] array = str.split(",");
        for (String s : array) {
            //if ()
        }
        return 1;
    }

}
