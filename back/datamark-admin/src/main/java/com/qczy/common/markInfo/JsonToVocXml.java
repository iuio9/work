package com.qczy.common.markInfo;

import cn.hutool.json.JSONArray;
import cn.hutool.json.JSONObject;
import com.qczy.model.entity.MarkInfoEntity;
import com.qczy.utils.StringUtils;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;

/**
 * 使用IO流直接写入XML，避免Transformer的路径问题
 */
public class JsonToVocXml {

    /**
     * 转换JSON标注为VOC格式XML（使用IO流直接写入）
     */
    public static void convertToVocXml(String fileName, String jsonStr, MarkInfoEntity entity, String outputFilePath) {
        BufferedWriter writer = null;
        try {
            System.out.println("开始转换JSON到XML，输入文件名: " + fileName);
            System.out.println("目标输出路径: " + outputFilePath);

            // 验证输入参数
            if (jsonStr == null || jsonStr.isEmpty()) {
                throw new IllegalArgumentException("JSON字符串不能为空");
            }

            // 解析目标路径，确定输出文件
            Path targetPath = Paths.get(outputFilePath);
            File outputFile;
            File parentDir;

            // 判断目标路径是文件还是目录
            if (isFilePath(outputFilePath)) {
                // 目标路径是文件（含文件名）
                outputFile = targetPath.toFile();
                parentDir = outputFile.getParentFile();
            } else {
                // 目标路径是目录，拼接文件名
                if (StringUtils.isEmpty(fileName)) {
                    throw new IllegalArgumentException("文件名不能为空（目标路径为目录时）");
                }
                String xmlFileName = fileName.endsWith(".xml") ? fileName : fileName + ".xml";
                outputFile = new File(targetPath.toFile(), xmlFileName);
                parentDir = outputFile.getParentFile();
            }

            // 创建父目录
            if (parentDir != null && !parentDir.exists() && !parentDir.mkdirs()) {
                throw new IOException("无法创建目录: " + parentDir.getAbsolutePath());
            }

            System.out.println("最终写入路径: " + outputFile.getAbsolutePath());

            // 解析JSON
            JSONObject jsonObject = new JSONObject(jsonStr);
            int imageWidth = jsonObject.getInt("imageWidth");
            int imageHeight = jsonObject.getInt("imageHeight");
            JSONArray shapesArray = jsonObject.getJSONArray("shapes");

            // 手动构建XML内容（核心：不依赖Transformer）
            StringBuilder xmlBuilder = new StringBuilder();
            xmlBuilder.append("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
            xmlBuilder.append("<annotation>\n");
            xmlBuilder.append("\t<filename>").append(escapeXml(fileName)).append("</filename>\n"); // 保留原始文件名

            // 图片尺寸
            xmlBuilder.append("\t<size>\n");
            xmlBuilder.append("\t\t<width>").append(imageWidth).append("</width>\n");
            xmlBuilder.append("\t\t<height>").append(imageHeight).append("</height>\n");
            xmlBuilder.append("\t\t<depth>3</depth>\n");
            xmlBuilder.append("\t</size>\n");

            // 标注数量
            xmlBuilder.append("\t<object_num>").append(shapesArray.size()).append("</object_num>\n");

            // 标注对象
            for (int i = 0; i < shapesArray.size(); i++) {
                JSONObject shape = shapesArray.getJSONObject(i);
                String label = shape.getStr("label");
                JSONArray points = shape.getJSONArray("points");

                // 提取边界框坐标（假设是四边形，取第0和第2个点）
                double xmin = points.getJSONArray(0).getDouble(0);
                double ymin = points.getJSONArray(0).getDouble(1);
                double xmax = points.getJSONArray(2).getDouble(0);
                double ymax = points.getJSONArray(2).getDouble(1);

                // 拼接object节点
                xmlBuilder.append("\t<object>\n");
                xmlBuilder.append("\t\t<name>").append(escapeXml(label)).append("</name>\n");
                xmlBuilder.append("\t\t<pose>Unspecified</pose>\n");
                xmlBuilder.append("\t\t<truncated>0</truncated>\n");
                xmlBuilder.append("\t\t<difficult>0</difficult>\n");
                xmlBuilder.append("\t\t<bndbox>\n");
                xmlBuilder.append("\t\t\t<xmin>").append(xmin).append("</xmin>\n");
                xmlBuilder.append("\t\t\t<ymin>").append(ymin).append("</ymin>\n");
                xmlBuilder.append("\t\t\t<xmax>").append(xmax).append("</xmax>\n");
                xmlBuilder.append("\t\t\t<ymax>").append(ymax).append("</ymax>\n");
                xmlBuilder.append("\t\t</bndbox>\n");
                xmlBuilder.append("\t</object>\n");
            }

            xmlBuilder.append("</annotation>");

            // 使用IO流直接写入文件（关键：替换Transformer的核心步骤）
            writer = new BufferedWriter(new FileWriter(outputFile));
            writer.write(xmlBuilder.toString());
            writer.flush();

            System.out.println("XML文件写入成功: " + outputFile.getAbsolutePath());

        } catch (Exception e) {
            System.err.println("XML转换失败: " + e.getMessage());
            e.printStackTrace();
            throw new RuntimeException("XML转换失败: " + e.getMessage(), e);
        } finally {
            // 关闭流
            if (writer != null) {
                try {
                    writer.close();
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }
    }

    /**
     * 判断路径是否为文件路径（含扩展名）
     */
    private static boolean isFilePath(String path) {
        if (path == null) return false;
        int lastDotIndex = path.lastIndexOf('.');
        int lastSeparatorIndex = Math.max(path.lastIndexOf('/'), path.lastIndexOf('\\'));
        return lastDotIndex > lastSeparatorIndex; // 扩展名在最后一个分隔符之后
    }

    /**
     * 转义XML特殊字符（避免标签被破坏）
     */
    private static String escapeXml(String content) {
        if (content == null) return "";
        return content.replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
                .replace("\"", "&quot;")
                .replace("'", "&apos;");
    }
}