package com.qczy.utils;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.*;
import java.nio.file.Files;

public class ImageUtils {

    /**
     * 获取图片的宽度和高度
     *
     * @param imagePath 图片路径
     * @return 一个数组，数组的第一个元素是宽度，第二个元素是高度
     * @throws IOException 如果读取图片失败
     */
    public static int[] getImageDimensions(String imagePath) throws IOException {

        String fileType = ImgeMimeTypeUtil.getMimeType(imagePath);
        if (fileType != null && "image/webp".equals(fileType)) {
            FileInputStream file = new FileInputStream(imagePath);
            byte[] bytes = new byte[64];
            file.read(bytes, 0, bytes.length);
            int width = ((int) bytes[27] & 0xff) << 8 | ((int) bytes[26] & 0xff);
            int height = ((int) bytes[29] & 0xff) << 8 | ((int) bytes[28] & 0xff);
            return new int[]{width, height};
        } else {
            File file = new File(imagePath);
            BufferedImage image = ImageIO.read(file);
            return new int[]{image.getWidth(),image.getHeight()};
        }
    }

    // 支持的图片扩展名
    private static final String[] IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp",
            ".JPG", ".JPEG", ".PNG", ".GIF", ".BMP", ".WEBP"};

    // 支持上传的文件
    private static final String[] FILE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp",
            ".JPG", ".JPEG", ".PNG", ".GIF", ".BMP", ".WEBP",".json", ".JSON", ".xml", ".XML", ".xlsx" ,".XLSX"};

    // 支持的图片MIME类型
    private static final String[] IMAGE_MIME_TYPES = {
            "image/jpeg",
            "image/png",
            "image/gif",
            "image/bmp",
            "image/webp"
    };

    /**
     * 判断文件是否为图片
     *
     * @param filePath 文件路径
     * @return true 如果文件是图片，false 否则
     */
    public static boolean isImage(String filePath) {
        File file = new File(filePath);

        // 检查文件是否存在
      /*  if (!file.exists() || file.isDirectory()) {
            return false;
        }*/

        // 检查扩展名
        if (hasImageExtension(file)) {
            return true;
        }

        // 检查MIME类型
        return isImageMimeType(file);
    }

    /**
     * 判断文件是否为 json 或者 xml
     *
     * @param filePath 文件路径
     * @return true 如果文件是json 或者 xml，false 否则
     */
    public static boolean isFile(String filePath) {
        File file = new File(filePath);

        // 检查文件是否存在
      /*  if (!file.exists() || file.isDirectory()) {
            return false;
        }*/

        // 检查扩展名
        return hasFileExtension(file);
    }

    /**
     * 检查文件扩展名
     *
     * @param file 文件
     * @return true 如果文件扩展名是图片类型，false 否则
     */
    private static boolean hasImageExtension(File file) {
        String fileName = file.getName().toLowerCase();
        for (String extension : IMAGE_EXTENSIONS) {
            if (fileName.endsWith(extension)) {
                return true;
            }
        }
        return false;
    }

    /**
     * 检查文件扩展名
     *
     * @param file 文件
     * @return true 如果文件扩展名是图片类型，false 否则
     */
    private static boolean hasFileExtension(File file) {
        String fileName = file.getName().toLowerCase();
        for (String extension : FILE_EXTENSIONS) {
            if (fileName.endsWith(extension)) {
                return true;
            }
        }
        return false;
    }

    /**
     * 检查文件的MIME类型
     *
     * @param file 文件
     * @return true 如果文件的MIME类型是图片类型，false 否则
     */
    private static boolean isImageMimeType(File file) {
        try {
            String mimeType = Files.probeContentType(file.toPath());
            for (String type : IMAGE_MIME_TYPES) {
                if (type.equals(mimeType)) {
                    return true;
                }
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        return false;
    }

    /**
     * 检查图片是否损坏（文件路径版本）
     *
     * @param imagePath 图片路径
     * @return true 如果图片损坏，false 否则
     */
    public static boolean isImageCorrupted(String imagePath) {
        if (imagePath == null || imagePath.isEmpty()) {
            return true;
        }

        File file = new File(imagePath);
        if (!file.exists() || file.isDirectory()) {
            return true;
        }

        try {
            String fileType = ImgeMimeTypeUtil.getMimeType(imagePath);
            if (fileType != null && "image/webp".equals(fileType)) {
                return isWebPCorrupted(file);
            } else {
                return isImageCorrupted(file);
            }
        } catch (Exception e) {
            return true;
        }
    }

    /**
     * 检查图片文件是否损坏（文件对象版本）
     *
     * @param file 图片文件
     * @return true 如果图片损坏，false 否则
     */
    private static boolean isImageCorrupted(File file) {
        try (InputStream is = new FileInputStream(file)) {
            BufferedImage image = ImageIO.read(is);
            if (image == null) {
                return true;
            }

            // 额外检查：尝试获取图片宽度和高度
            int width = image.getWidth();
            int height = image.getHeight();

            return width <= 0 || height <= 0;
        } catch (Exception e) {
            return true;
        }
    }

    /**
     * 检查WebP图片是否损坏
     *
     * @param file WebP图片文件
     * @return true 如果图片损坏，false 否则
     */
    private static boolean isWebPCorrupted(File file) {
        try (RandomAccessFile raf = new RandomAccessFile(file, "r")) {
            byte[] header = new byte[12];
            raf.readFully(header);

            // 检查RIFF头
            if (!"RIFF".equals(new String(header, 0, 4))) {
                return true;
            }

            // 检查WEBP头
            if (!"WEBP".equals(new String(header, 8, 4))) {
                return true;
            }

            // 检查VP8、VP8L或VP8X格式
            byte[] chunkHeader = new byte[8];
            raf.readFully(chunkHeader);

            String chunkType = new String(chunkHeader, 0, 4);
            if (!"VP8 ".equals(chunkType) && !"VP8L".equals(chunkType) && !"VP8X".equals(chunkType)) {
                return true;
            }

            return false;
        } catch (Exception e) {
            return true;
        }
    }

    /**
     * 检查文件是否损坏（通用版本）
     *
     * @param filePath 文件路径
     * @return true 如果文件损坏，false 否则
     */
    public static boolean isFileCorrupted(String filePath) {
        if (filePath == null || filePath.isEmpty()) {
            return true;
        }

        File file = new File(filePath);
        if (!file.exists() || file.isDirectory()) {
            return true;
        }

        // 对于图片文件，使用专门的图片检查方法
        if (isImage(filePath)) {
            return isImageCorrupted(filePath);
        }

        // 对于其他文件，检查文件大小是否为0
        return file.length() <= 0;
    }

    public static void main(String[] args) {
        try {
            String imagePath = "path/to/your/image.jpg"; // 替换为你的图片路径
            int[] dimensions = getImageDimensions(imagePath);
            System.out.println("Width: " + dimensions[0] + ", Height: " + dimensions[1]);

            boolean isCorrupted = isImageCorrupted(imagePath);
            System.out.println("Image corrupted: " + isCorrupted);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}