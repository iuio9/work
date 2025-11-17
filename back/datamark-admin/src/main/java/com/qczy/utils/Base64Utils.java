package com.qczy.utils;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.util.Base64;
import java.util.UUID;

public class Base64Utils {

    /**
     * 从 Base64 字符串中提取图像信息（文件名、宽度、高度）
     * @param base64Data 完整的 Base64 字符串（支持带或不带 data URI 前缀）
     * @return ImageMeta 对象，包含文件名、宽度、高度
     * @throws IOException 解析图像失败时抛出
     */
    public static ImageMeta getImageMeta(String base64Data) throws IOException {
        // 1. 解析 Base64 数据（分离前缀和内容）
        String pureBase64 = extractPureBase64(base64Data);
        String mimeType = getMimeTypeFromBase64(base64Data);
        
        // 2. 解码 Base64 为字节数组
        byte[] imageBytes = Base64.getDecoder().decode(pureBase64);
        
        // 3. 获取图像尺寸
        int[] dimensions = getImageDimensions(imageBytes);
        
        // 4. 生成文件名（从 MIME 类型推断扩展名，或使用默认）
        String extension = getFileExtensionFromMimeType(mimeType);
        String fileName = generateFileName(extension);
        
        return new ImageMeta(fileName, dimensions[0], dimensions[1]);
    }

    /**
     * 提取纯 Base64 内容（移除 data URI 前缀）
     */
    private static String extractPureBase64(String base64Data) {
        if (base64Data == null || !base64Data.startsWith("data:")) {
            return base64Data;
        }
        int commaIndex = base64Data.indexOf(',');
        return commaIndex != -1 ? base64Data.substring(commaIndex + 1) : base64Data;
    }

    /**
     * 从 data URI 中提取 MIME 类型（如 image/png）
     */
    private static String getMimeTypeFromBase64(String base64Data) {
        if (base64Data == null || !base64Data.startsWith("data:image/")) {
            return "image/jpeg"; // 默认类型
        }
        int start = "data:image/".length();
        int end = base64Data.indexOf(';', start);
        return end != -1 ? base64Data.substring(start, end) : "image/jpeg";
    }

    /**
     * 根据 MIME 类型获取文件扩展名（如 png、jpg）
     */
    private static String getFileExtensionFromMimeType(String mimeType) {
        if (mimeType == null) return ".jpg";
        String[] parts = mimeType.split("/");
        if (parts.length < 2) return ".jpg";
        
        String type = parts[1].toLowerCase();
        switch (type) {
            case "jpeg": return ".jpg";
            case "png": return ".png";
            case "gif": return ".gif";
            case "bmp": return ".bmp";
            case "webp": return ".webp";
            default: return "." + type; // 未知类型保留原始后缀
        }
    }

    /**
     * 生成随机文件名（如：a1b2c3.jpg）
     */
    private static String generateFileName(String extension) {
        return UUID.randomUUID().toString().substring(0, 8) + extension;
    }

    /**
     * 解析图像字节数据获取宽度和高度
     */
    private static int[] getImageDimensions(byte[] imageBytes) throws IOException {
        try (ByteArrayInputStream bis = new ByteArrayInputStream(imageBytes)) {
            BufferedImage image = ImageIO.read(bis);
            if (image == null) {
                throw new IOException("不是有效的图像文件");
            }
            return new int[]{image.getWidth(), image.getHeight()};
        }
    }

    /**
     * 图像元数据封装类
     */
    public static class ImageMeta {
        private final String fileName;  // 文件名（如：image.jpg）
        private final int width;        // 宽度（像素）
        private final int height;       // 高度（像素）

        public ImageMeta(String fileName, int width, int height) {
            this.fileName = fileName;
            this.width = width;
            this.height = height;
        }

        // Getter 方法
        public String getFileName() { return fileName; }
        public int getWidth() { return width; }
        public int getHeight() { return height; }

        @Override
        public String toString() {
            return "ImageMeta{" +
                    "fileName='" + fileName + '\'' +
                    ", width=" + width +
                    ", height=" + height +
                    '}';
        }
    }

    // 测试示例
    public static void main(String[] args) {
        try {
            // 带 data URI 前缀的 Base64 字符串（示例为 PNG 图像）
            String base64Data = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAAAJElEQVQ4T2NkYGAQYcAP3uCTZhw1gGGYhAGBZIA/nYDCgBDAm9BGDWAAJyRCgLaBCAAgXwixzAS0pgAAAABJRU5ErkJggg==";
            
            ImageMeta meta = getImageMeta(base64Data);
            System.out.println("文件名：" + meta.getFileName());   // 输出类似：a1b2c3.png
            System.out.println("宽度：" + meta.getWidth());     // 输出：20
            System.out.println("高度：" + meta.getHeight());    // 输出：20
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}