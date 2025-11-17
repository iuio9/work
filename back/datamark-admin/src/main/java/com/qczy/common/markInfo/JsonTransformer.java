package com.qczy.common.markInfo;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.reflect.TypeToken;
import com.qczy.mapper.DataSonLabelMapper;
import com.qczy.mapper.LabelMapper;
import com.qczy.model.entity.DataSonLabelEntity;
import com.qczy.model.entity.LabelEntity;
import com.qczy.model.entity.MarkInfoEntity;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.scheduling.annotation.Async;
import org.springframework.stereotype.Service;
import org.springframework.util.CollectionUtils;

import java.io.BufferedWriter;
import java.io.IOException;
import java.lang.reflect.Type;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;
import java.util.List;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Collectors;

/**
 * @Author: hh
 * @Version: 1.0
 * @Date: 2025/5/14 10:26
 * @Description:
 */
@Service
public class JsonTransformer {

    @Autowired
    private DataSonLabelMapper dataSonLabelMapper;

    @Autowired
    private LabelMapper labelMapper;

    // 使用 ConcurrentHashMap 确保线程安全
    private Map<String, String> labelMap = new ConcurrentHashMap<>();

    @Async
    public void transformJson(MarkInfoEntity markInfoEntity, String filePath, String sonId) {
        try {
            // 解析输入的 JSON 数组
            Gson gson = new Gson();
            Type listType = new TypeToken<List<InputItem>>() {}.getType();
            List<InputItem> inputItems = gson.fromJson(markInfoEntity.getMarkInfo(), listType);

            // 转换为输出格式
            OutputResult outputResult = new OutputResult();
            outputResult.labels = new ArrayList<>();

            // 获取文件的正常宽高
            Integer width = markInfoEntity.getWidth();
            Integer height = markInfoEntity.getHeight();
            // 获取文件的缩放宽高
            Integer operateWidth = markInfoEntity.getOperateWidth();
            Integer operateHeight = markInfoEntity.getOperateHeight();

            for (InputItem item : inputItems) {
                System.out.println("item: " + item);
                Point label = new Point();
                label.name = setEnglishLabelName(sonId, item.props.name);
                System.out.println(" label.name =" + label.name);
                // 根据不同类型处理
                if ("RECT".equalsIgnoreCase(item.type)) {
                    // 处理矩形
                    label.x1 = (item.shape.x) * (width / (double) operateWidth);
                    label.y1 = (item.shape.y) * (height / (double) operateHeight);
                    label.x2 = (item.shape.x + item.shape.width) * (width / (double) operateWidth);
                    label.y2 = (item.shape.y + item.shape.height) * (height / (double) operateHeight);
                } else if ("POLYGON".equalsIgnoreCase(item.type)) {
                    // 处理多边形
                    PolygonBounds bounds = calculatePolygonBounds(item.shape.points, width, height, operateWidth, operateHeight);
                    label.x1 = bounds.minX;
                    label.y1 = bounds.minY;
                    label.x2 = bounds.maxX;
                    label.y2 = bounds.maxY;
                } else {
                    // 其他类型可以在这里扩展处理逻辑
                    System.out.println("Unsupported type: " + item.type);
                    continue;
                }

                outputResult.labels.add(label);
            }

            // 生成输出 JSON 字符串（美化格式）
            Gson prettyGson = new GsonBuilder().setPrettyPrinting().create();
            String json = prettyGson.toJson(outputResult);
            writeJsonToFile(json, filePath);

        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    // 计算多边形的边界矩形
    private PolygonBounds calculatePolygonBounds(List<PointData> points, Integer width, Integer height, Integer operateWidth, Integer operateHeight) {
        double minX = Double.MAX_VALUE;
        double minY = Double.MAX_VALUE;
        double maxX = Double.MIN_VALUE;
        double maxY = Double.MIN_VALUE;

        // 找出多边形所有顶点中的最小和最大坐标值
        for (PointData point : points) {
            double x = point.x * (width / (double) operateWidth);
            double y = point.y * (height / (double) operateHeight);

            minX = Math.min(minX, x);
            minY = Math.min(minY, y);
            maxX = Math.max(maxX, x);
            maxY = Math.max(maxY, y);
        }

        return new PolygonBounds(minX, minY, maxX, maxY);
    }

    // 多边形边界辅助类
    private static class PolygonBounds {
        double minX;
        double minY;
        double maxX;
        double maxY;

        public PolygonBounds(double minX, double minY, double maxX, double maxY) {
            this.minX = minX;
            this.minY = minY;
            this.maxX = maxX;
            this.maxY = maxY;
        }
    }

    // 输入数据模型类 - 修改后支持多种类型
    static class InputItem {
        String type; // 新增类型字段
        Props props;
        Shape shape;
        List<PointData> points; // 多边形顶点列表

        @Override
        public String toString() {
            return "InputItem{" +
                    "type='" + type + '\'' +
                    ", props=" + props +
                    ", shape=" + shape +
                    ", points=" + points +
                    '}';
        }
    }

    static class Props {
        String name;

        @Override
        public String toString() {
            return "Props{" +
                    "name='" + name + '\'' +
                    '}';
        }
    }

    // 矩形形状数据
    static class Shape {
        double x;
        double y;
        double width;
        double height;
        List<PointData> points; // 多边形顶点列表

        @Override
        public String toString() {
            return "Shape{" +
                    "x=" + x +
                    ", y=" + y +
                    ", width=" + width +
                    ", height=" + height +
                    ", points=" + points +
                    '}';
        }
    }

    // 多边形顶点数据
    static class PointData {
        double x;
        double y;

        @Override
        public String toString() {
            return "PointData{" +
                    "x=" + x +
                    ", y=" + y +
                    '}';
        }
    }

    // 输出数据模型类
    static class OutputResult {
        List<Point> labels;
    }

    static class Point {
        String name; // 标签名称
        double x1;
        double y1;
        double x2;
        double y2;
    }

    // 将 JSON 字符串写入指定文件
    public static void writeJsonToFile(String jsonString, String filePath) throws IOException {
        // 创建文件路径
        Path path = Paths.get(filePath);

        // 创建父文件夹（如果不存在）
        Path parentDir = path.getParent();
        if (parentDir != null) {
            Files.createDirectories(parentDir);
        }

        // 使用 try-with-resources 自动关闭资源
        try (BufferedWriter writer = Files.newBufferedWriter(path)) {
            writer.write(jsonString);
        }
    }

    public String setEnglishLabelName(String sonId, String labelName) {
        // 使用 computeIfAbsent 确保线程安全且懒加载
        return labelMap.computeIfAbsent(labelName.trim(), key -> {
            // 进行查询
            List<DataSonLabelEntity> list = dataSonLabelMapper.selectList(
                    new LambdaQueryWrapper<DataSonLabelEntity>()
                            .eq(DataSonLabelEntity::getSonId, sonId)
            );

            if (CollectionUtils.isEmpty(list)) {
                return null;
            }

            // 提取所有的 labelId
            List<Integer> labelIds = list.stream()
                    .map(DataSonLabelEntity::getLabelId)
                    .collect(Collectors.toList());

            // 批量查询标签信息
            List<LabelEntity> labelEntities = labelMapper.selectByIds(labelIds);

            // 构建局部映射，不更新全局 map
            // 构建局部映射，将所有可能的 null 值替换为字符串 "null"
            Map<String, String> tempMap = labelEntities.stream()
                    .filter(Objects::nonNull) // 过滤掉列表中的 null 元素
                    .collect(Collectors.toMap(
                            e -> Optional.ofNullable(e.getLabelName())
                                    .map(String::trim)
                                    .orElse("null"), // 将 null 标签名转换为 "null"
                            e -> Optional.ofNullable(e.getEnglishLabelName())
                                    .orElse("null"), // 将 null 英文标签名转换为 "null"
                            (existing, replacement) -> existing, // 处理键冲突的策略
                            HashMap::new // 显式指定 Map 实现
                    ));

            // 直接返回当前查询的值，不调用 putAll()
            return tempMap.getOrDefault(key, null);
        });
    }
}