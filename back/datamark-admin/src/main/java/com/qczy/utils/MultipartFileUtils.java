package com.qczy.utils;

import org.springframework.web.multipart.MultipartFile;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Base64;

public class MultipartFileUtils {

    /**
     * 将MultipartFile转换为Base64编码字符串（不带MIME类型前缀）
     * @param file 上传的文件
     * @return Base64编码字符串
     * @throws IOException 如果读取文件内容失败
     */
    public static String toBase64(MultipartFile file) throws IOException {
        byte[] fileContent = file.getBytes();
        return Base64.getEncoder().encodeToString(fileContent);
    }
    
    /**
     * 如果需要，可以从带MIME前缀的Base64字符串中提取实际编码部分
     * @param base64String 带MIME前缀的Base64字符串
     * @return 实际的Base64编码部分
     */
    public static String extractBase64Content(String base64String) {
        if (base64String == null) {
            return null;
        }
        int commaIndex = base64String.indexOf(',');
        if (commaIndex != -1) {
            return base64String.substring(commaIndex + 1);
        }
        return base64String;
    }

    /**
     * 获取 MultipartFile 的文件后缀（包含点号）
     * @param file 上传的文件
     * @return 文件后缀（例如：.jpg、.txt），若无后缀则返回空字符串
     */
    public static String getFileExtension(MultipartFile file) {
        if (file == null || file.getOriginalFilename() == null) {
            return "";
        }

        String fileName = file.getOriginalFilename();
        int lastDotIndex = fileName.lastIndexOf('.');

        return lastDotIndex >= 0 ? fileName.substring(lastDotIndex) : "";
    }

    /**
     * 获取 MultipartFile 的文件后缀（不包含点号）
     * @param file 上传的文件
     * @return 文件后缀（例如：jpg、txt），若无后缀则返回空字符串
     */
    public static String getFileExtensionWithoutDot(MultipartFile file) {
        String extension = getFileExtension(file);
        return extension.isEmpty() ? "" : extension.substring(1);
    }


    /**
     * 根据图片文件路径生成Base64编码字符串
     * @param imagePath 图片文件的完整路径（例如：/path/to/image.jpg）
     * @return Base64编码字符串，读取失败时返回null
     */
    public static String imagePathToBase64(String imagePath) {
        if (imagePath == null || imagePath.trim().isEmpty()) {
            return null;
        }

        try {
            // 方式1：使用Java NIO Files API（推荐，简洁高效）
            byte[] imageBytes = Files.readAllBytes(Paths.get(imagePath));
            return Base64.getEncoder().encodeToString(imageBytes);

            /* 方式2：使用传统FileInputStream（兼容旧版本Java）
            try (FileInputStream fis = new FileInputStream(imagePath)) {
                byte[] buffer = new byte[(int) new File(imagePath).length()];
                fis.read(buffer);
                return Base64.getEncoder().encodeToString(buffer);
            }
            */
        } catch (IOException e) {
            System.err.println("读取图片文件失败：" + imagePath);
            e.printStackTrace();
            return null;
        }
    }




}