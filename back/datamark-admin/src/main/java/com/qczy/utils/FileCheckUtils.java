package com.qczy.utils;


import javax.json.Json;
import javax.json.JsonArray;
import javax.json.JsonObject;
import javax.json.JsonReader;
import java.io.*;


/**
 * @Author: hh
 * @Version: 1.0
 * @Date: 2024/12/26 9:56
 * @Description:
 */
public class FileCheckUtils {


    // 对json文件进行校验
    public static boolean checkJson(String filePath) {

        try {

            File jsonFile = new File(filePath);
            // 判断文件是否存在
            if (!jsonFile.exists()) {
                return false;
            }
            // 文件为空
            if (jsonFile.length() == 0) {
                return false;
            }

            JsonReader reader = Json.createReader(new FileReader(filePath));
            JsonObject jsonObject = reader.readObject();

            JsonArray images = jsonObject.getJsonArray("images");
            JsonArray annotations = jsonObject.getJsonArray("annotations");
            JsonArray categories = jsonObject.getJsonArray("categories");

            // 验证 images 数组
            if (images == null || images.isEmpty()) {
                return false;
            }

            // 验证 annotations 数组
            if (annotations == null || annotations.isEmpty()) {
                return false;
            }

            // 验证 categories 数组
            return categories != null && !categories.isEmpty();

        } catch (Exception e) {
            return false;
        }
    }

    public static void main(String[] args) {
        boolean b = checkJson("C:\\Users\\c\\Desktop\\文档\\超大图片2\\coco_info.json");
        System.out.println(b);
    }


}
