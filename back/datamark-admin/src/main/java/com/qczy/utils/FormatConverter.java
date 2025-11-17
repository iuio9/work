package com.qczy.utils;

import cn.hutool.core.util.NumberUtil;
import cn.hutool.core.util.ObjectUtil;
import cn.hutool.json.JSONUtil;
import com.qczy.model.entity.*;
import org.springframework.util.ObjectUtils;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

// Labelme格式的类


// 工具类
public class FormatConverter {

    // 将Labelme格式转换为Web端格式
    public static WebRectangleShape convertLabelmeToWeb(LabelmeShape labelmeShape, String openId, String id,MarkInfoEntity markInfoEntity) {
        WebRectangleShape webShape = new WebRectangleShape();
        webShape.setOpenId(openId);
        webShape.setId(id);
        if(labelmeShape.getShape_type().equalsIgnoreCase("rect")){
            webShape.setType(labelmeShape.getShape_type());// 假设将Labelme的形状视为矩形

        }else {
            webShape.setType("POLYGON");// 假设将Labelme的形状视为矩形
        }
        // Props字段的转换
        WebProps webProps = new WebProps();
        webProps.setName(labelmeShape.getLabel().split(" ")[0]);
        webShape.setProps(webProps);

        // 形状字段的转换
        List<Map<String, Double>> shapePoints = webShape.getShapePoints();

        Map<String,Object> shape =  new HashMap<String,Object>();

        if(webShape.getType().equalsIgnoreCase("POLYGON")) {
            if (ObjectUtil.isNotEmpty(labelmeShape.getPoints())) {
                for (int i = 0; i < labelmeShape.getPoints().size(); i++) {
                    Map<String, Double> shapeMap = new HashMap<>();
                    shapeMap.put("x", (double) (labelmeShape.getPoints().get(i)[0]) / NumberUtil.div((int) markInfoEntity.getWidth(), (int) markInfoEntity.getOperateWidth()));
                    shapeMap.put("y", (double) (labelmeShape.getPoints().get(i)[1]) / NumberUtil.div((int) markInfoEntity.getWidth(), (int) markInfoEntity.getOperateWidth()));
//            shapeMap.put("width", (double) (labelmeShape.getPoints().get(i)[0] - labelmeShape.getPoints().get(i)[0]));
//            shapeMap.put("height", (double) (labelmeShape.getPoints().get(i)[1] - labelmeShape.getPoints().get(i)[1]));
                    shapePoints.add(shapeMap);
                    shape.put("points",shapePoints);
                    webShape.setShape(shape);
                }
            }
        }else if (webShape.getType().equalsIgnoreCase("RECT")) {  // 检查是否为矩形
            if (ObjectUtil.isNotEmpty(labelmeShape.getPoints())) {  // 判断是否有点数据
                // 获取矩形的左上角 (x1, y1) 和右下角 (x2, y2)
                double x1 = (double) labelmeShape.getPoints().get(0)[0] / NumberUtil.div((int) markInfoEntity.getWidth(), (int) markInfoEntity.getOperateWidth());
                double y1 = (double) labelmeShape.getPoints().get(0)[1] / NumberUtil.div((int) markInfoEntity.getWidth(), (int) markInfoEntity.getOperateWidth());
                double x2 = (double) labelmeShape.getPoints().get(2)[0] / NumberUtil.div((int) markInfoEntity.getWidth(), (int) markInfoEntity.getOperateWidth());
                double y2 = (double) labelmeShape.getPoints().get(2)[1] / NumberUtil.div((int) markInfoEntity.getWidth(), (int) markInfoEntity.getOperateWidth());

                // 计算矩形的宽度和高度
                double width = x2 - x1;
                double height = y2 - y1;

                // 创建 shapeMap 并存储 x, y, 宽度和高度
                Map<String, Double> shapeMap = new HashMap<>();
                shapeMap.put("x", x1);
                shapeMap.put("y", y1);
                shapeMap.put("width", width);
                shapeMap.put("height", height);
                shapePoints.add(shapeMap);  // 将计算好的矩形添加到结果列表
                webShape.setShape(shapeMap);
            }
        }

        // 样式字段的转换

        WebShapeStyle style = new WebShapeStyle();
        style.setOpacity(1.0);
        style.setFillStyle("#D91515");
        style.setLineWidth(0);
        style.setStrokeStyle( "rgba(77, 101, 170)");
        style.setFill(true);
        style.setGlobalAlpha(0.6);
        webShape.setStyle(style);

        return webShape;
    }

    // 将Web端格式转换为Labelme格式（修复类型转换问题）
    public static LabelmeShape convertWebToLabelme(WebRectangleShape webShape, MarkInfoEntity markInfoEntity) {
        if (webShape == null || markInfoEntity == null) {
            throw new IllegalArgumentException("webShape or markInfoEntity cannot be null");
        }

        LabelmeShape labelmeShape = new LabelmeShape();
        labelmeShape.setLabel(webShape.getProps() != null ? webShape.getProps().getName() : "unknown");

        String shapeType = webShape.getType();
        if (shapeType == null) {
            throw new IllegalArgumentException("Shape type cannot be null");
        }
        labelmeShape.setShape_type(shapeType);

        List<double[]> points = new ArrayList<>();
        try {
            if (shapeType.equalsIgnoreCase("rect")) {
                // 使用通配符Map处理任意类型的值
                if (!(webShape.getShape() instanceof Map)) {
                    throw new IllegalArgumentException("Invalid shape data for rectangle");
                }
                Map<?, ?> shape = (Map<?, ?>) webShape.getShape();

                // 检查关键字段
                checkShapeKeys(shape, "x", "y", "width", "height");

                // 计算缩放比例
                double widthScale = safeDiv(markInfoEntity.getWidth(), markInfoEntity.getOperateWidth());
                double heightScale = safeDiv(markInfoEntity.getHeight(), markInfoEntity.getOperateHeight());

                // 安全获取数值
                double x = getDoubleValue(shape, "x") * widthScale;
                double y = getDoubleValue(shape, "y") * heightScale;
                double width = getDoubleValue(shape, "width") * widthScale;
                double height = getDoubleValue(shape, "height") * heightScale;

                // 添加矩形四个顶点
                points.add(new double[]{x, y});
                points.add(new double[]{x + width, y});
                points.add(new double[]{x + width, y + height});
                points.add(new double[]{x, y + height});
            } else if (shapeType.equalsIgnoreCase("circle")) {
                // 安全处理圆形
                addCirclePoints(points, (Map<?, ?>) webShape.getShape(),
                        safeDiv(markInfoEntity.getWidth(), markInfoEntity.getOperateWidth()),
                        safeDiv(markInfoEntity.getHeight(), markInfoEntity.getOperateHeight()));
            } else if (shapeType.equalsIgnoreCase("polygon")) {
                // 安全处理多边形
                Object parsedShape = JSONUtil.parseObj(webShape.getShape());
                if (!(parsedShape instanceof Map) || !((Map<?, ?>) parsedShape).containsKey("points")) {
                    throw new IllegalArgumentException("Invalid polygon points data");
                }
                List<Map<?, ?>> pointsList = (List<Map<?, ?>>) ((Map<?, ?>) parsedShape).get("points");
                addPolygonPoints(points, pointsList,
                        safeDiv(markInfoEntity.getWidth(), markInfoEntity.getOperateWidth()),
                        safeDiv(markInfoEntity.getHeight(), markInfoEntity.getOperateHeight()));
            } else {
                System.err.println("Unsupported shape type: " + shapeType);
            }
        } catch (Exception e) {
            System.err.println("Error converting shape: " + e.getMessage());
            throw new RuntimeException("Failed to convert shape", e);
        }

        labelmeShape.setPoints(points);
        return labelmeShape;
    }

    /**
     * 检查 shape 是否包含必要的字段
     */
    private static void checkShapeKeys(Map<?, ?> shape, String... keys) {
        for (String key : keys) {
            if (!shape.containsKey(key) || shape.get(key) == null) {
                throw new IllegalArgumentException("Missing or invalid shape key: " + key);
            }
        }
    }

    /**
     * 安全的除法计算，避免除零错误
     */
    private static double safeDiv(Integer a, Integer b) {
        if (a == null || b == null || b == 0) {
            throw new IllegalArgumentException("Invalid division operands: a=" + a + ", b=" + b);
        }
        return (double) a / b;
    }

    /**
     * 将任意数值类型安全转换为double
     */
    private static double getDoubleValue(Map<?, ?> map, String key) {
        Object value = map.get(key);
        if (value == null) {
            throw new IllegalArgumentException("Missing key: " + key);
        }

        if (value instanceof Number) {
            return ((Number) value).doubleValue();
        }

        try {
            return Double.parseDouble(value.toString());
        } catch (NumberFormatException e) {
            throw new IllegalArgumentException("Invalid numeric value for key: " + key, e);
        }
    }

    /**
     * 处理圆形，生成36个点近似圆形
     */
    private static void addCirclePoints(List<double[]> points, Map<?, ?> shape, double xradio, double yradio) {
        double centerX = getDoubleValue(shape, "cx") * xradio;
        double centerY = getDoubleValue(shape, "cy") * yradio;
        double radius = getDoubleValue(shape, "r") * xradio; // 假设x和y方向的缩放比例相同
        final int numPoints = 36; // 用36个点来近似圆形

        for (int i = 0; i < numPoints; i++) {
            double angle = 2 * Math.PI * i / numPoints;
            double x = centerX + radius * Math.cos(angle);
            double y = centerY + radius * Math.sin(angle);
            points.add(new double[]{x, y});
        }
    }

    /**
     * 处理多边形点
     */
    private static void addPolygonPoints(List<double[]> points, List<Map<?, ?>> shapePoints, double xradio, double yradio) {
        for (Map<?, ?> point : shapePoints) {
            double x = getDoubleValue(point, "x") * xradio;
            double y = getDoubleValue(point, "y") * yradio;
            points.add(new double[]{x, y});
        }
    }


    // 如果需要处理整个Labelme文件，可以提供如下方法
    public static List<WebRectangleShape> convertLabelmeDataToWeb(LabelmeImageData labelmeData,MarkInfoEntity markInfoEntity) {
        List<WebRectangleShape> webShapes = new ArrayList<>();
        checkMarkInfoEntity(labelmeData,markInfoEntity);
        for (int i = 0; i < labelmeData.getShapes().size(); i++) {
            LabelmeShape labelmeShape = labelmeData.getShapes().get(i);
            WebRectangleShape webShape = convertLabelmeToWeb(labelmeShape, "openId-" + i, "id-" + i,markInfoEntity);
            webShapes.add(webShape);
        }
        return webShapes;
    }

    private static void checkMarkInfoEntity(LabelmeImageData labelmeData, MarkInfoEntity markInfoEntity) {
        // 计算矩形的宽度和高度
        if(ObjectUtils.isEmpty(markInfoEntity.getWidth())){
            markInfoEntity.setWidth(labelmeData.getImageWidth());
        }
        if(ObjectUtils.isEmpty(markInfoEntity.getHeight())){
            markInfoEntity.setHeight(labelmeData.getImageHeight());
        }
        if(ObjectUtils.isEmpty(markInfoEntity.getOperateWidth())){
            markInfoEntity.setOperateWidth(labelmeData.getImageWidth());
        }
        if(ObjectUtils.isEmpty(markInfoEntity.getOperateHeight())){
            markInfoEntity.setOperateHeight((labelmeData.getImageHeight()));
        }
    }

    public static LabelmeImageData convertWebDataToLabelme(List<WebRectangleShape> webShapes, String imagePath, int imageWidth, int imageHeight,MarkInfoEntity markInfoEntity) {
        LabelmeImageData labelmeData = new LabelmeImageData();
        labelmeData.setImagePath(imagePath);
        labelmeData.setImageWidth(imageWidth);
        labelmeData.setImageHeight(imageHeight);

        for (WebRectangleShape webShape : webShapes) {
            LabelmeShape labelmeShape = convertWebToLabelme(webShape,markInfoEntity);
            labelmeData.getShapes().add(labelmeShape);
        }

        return labelmeData;
    }


    public static void main(String[] args) {
//
//
//
//        String webMarkInfo = "[{\"openId\":\"1726740579600\",\"id\":\"1726740579600-zNgVUdoRM37kVVyQUimj-\",\"type\":\"POINT\",\"props\":{\"sign\":\"foreground_points\",\"operateWidth\":508,\"operateHeight\":381},\"shape\":{\"x\":195.04798988383504,\"y\":65.2269785031495,\"r\":5},\"style\":{\"opacity\":1,\"fillStyle\":\"#FF8C00\",\"lineWidth\":2,\"strokeStyle\":\"#000\",\"zIndex\":5}},{\"openId\":\"1726740582497\",\"id\":\"1726740582497-CmP8vhqgemsqNIy7k5z-b\",\"type\":\"POLYGON\",\"props\":{\"name\":\"\",\"textId\":\"autoAno\",\"deleteMarkerId\":\"autoAno\",\"isAutoFit\":true,\"operateWidth\":508,\"operateHeight\":381},\"shape\":{\"points\":[{\"x\":162.814,\"y\":226.822},{\"x\":163.322,\"y\":227.076},{\"x\":165.227,\"y\":226.94899999999998},{\"x\":164.465,\"y\":226.695}]},\"style\":{\"opacity\":1,\"fillStyle\":\"#D91515\",\"lineWidth\":0,\"strokeStyle\":\"rgba(77, 101, 170)\",\"fill\":true,\"globalAlpha\":0.6,\"stroke\":false}},{\"openId\":\"1726740582497\",\"id\":\"1726740582497-6dzk04CBPCvvw4y-xo-7Y\",\"type\":\"POLYGON\",\"props\":{\"name\":\"\",\"textId\":\"autoAno\",\"deleteMarkerId\":\"autoAno\",\"isAutoFit\":true,\"operateWidth\":508,\"operateHeight\":381},\"shape\":{\"points\":[{\"x\":171.958,\"y\":226.822},{\"x\":171.323,\"y\":226.441},{\"x\":170.053,\"y\":226.822}]},\"style\":{\"opacity\":1,\"fillStyle\":\"#D91515\",\"lineWidth\":0,\"strokeStyle\":\"rgba(77, 101, 170)\",\"fill\":true,\"globalAlpha\":0.6,\"stroke\":false}},{\"openId\":\"1726740582497\",\"id\":\"1726740582497-PoXNGaDXR6KxJc3zldyY_\",\"type\":\"POLYGON\",\"props\":{\"name\":\"\",\"textId\":\"autoAno\",\"deleteMarkerId\":\"autoAno\",\"isAutoFit\":true,\"operateWidth\":508,\"operateHeight\":381},\"shape\":{\"points\":[{\"x\":165.608,\"y\":226.94899999999998},{\"x\":167.894,\"y\":227.32999999999998},{\"x\":169.545,\"y\":226.822},{\"x\":168.148,\"y\":226.06},{\"x\":167.513,\"y\":226.06}]},\"style\":{\"opacity\":1,\"fillStyle\":\"#D91515\",\"lineWidth\":0,\"strokeStyle\":\"rgba(77, 101, 170)\",\"fill\":true,\"globalAlpha\":0.6,\"stroke\":false}},{\"openId\":\"1726740582498\",\"id\":\"1726740582498-W-o41q_MaQ14QLFjUifsz\",\"type\":\"POLYGON\",\"props\":{\"name\":\"\",\"textId\":\"autoAno\",\"deleteMarkerId\":\"autoAno\",\"isAutoFit\":true,\"operateWidth\":508,\"operateHeight\":381},\"shape\":{\"points\":[{\"x\":174.75199999999998,\"y\":224.917},{\"x\":175.26,\"y\":225.298},{\"x\":177.927,\"y\":224.917},{\"x\":177.292,\"y\":224.409},{\"x\":176.403,\"y\":224.28199999999998},{\"x\":175.26,\"y\":224.409}]},\"style\":{\"opacity\":1,\"fillStyle\":\"#D91515\",\"lineWidth\":0,\"strokeStyle\":\"rgba(77, 101, 170)\",\"fill\":true,\"globalAlpha\":0.6,\"stroke\":false}},{\"openId\":\"1726740582498\",\"id\":\"1726740582498-iu_5deoEVyIhl_6yEpJII\",\"type\":\"POLYGON\",\"props\":{\"name\":\"\",\"textId\":\"autoAno\",\"deleteMarkerId\":\"autoAno\",\"isAutoFit\":true,\"operateWidth\":508,\"operateHeight\":381},\"shape\":{\"points\":[{\"x\":178.816,\"y\":223.51999999999998},{\"x\":179.32399999999998,\"y\":223.774},{\"x\":180.46699999999998,\"y\":223.393},{\"x\":181.22899999999998,\"y\":223.393},{\"x\":183.388,\"y\":222.885},{\"x\":179.578,\"y\":222.75799999999998},{\"x\":179.197,\"y\":222.885}]},\"style\":{\"opacity\":1,\"fillStyle\":\"#D91515\",\"lineWidth\":0,\"strokeStyle\":\"rgba(77, 101, 170)\",\"fill\":true,\"globalAlpha\":0.6,\"stroke\":false}},{\"openId\":\"1726740582499\",\"id\":\"1726740582499-Ox-csGw-oBKO5BW-heFMl\",\"type\":\"POLYGON\",\"props\":{\"name\":\"\",\"textId\":\"autoAno\",\"deleteMarkerId\":\"autoAno\",\"isAutoFit\":true,\"operateWidth\":508,\"operateHeight\":381},\"shape\":{\"points\":[{\"x\":191.516,\"y\":218.948},{\"x\":191.262,\"y\":218.694},{\"x\":187.579,\"y\":218.948},{\"x\":187.452,\"y\":219.456}]},\"style\":{\"opacity\":1,\"fillStyle\":\"#D91515\",\"lineWidth\":0,\"strokeStyle\":\"rgba(77, 101, 170)\",\"fill\":true,\"globalAlpha\":0.6,\"stroke\":false}},{\"openId\":\"1726740582499\",\"id\":\"1726740582499-zlsntMygDJm6eWbMDfOpg\",\"type\":\"POLYGON\",\"props\":{\"name\":\"\",\"textId\":\"autoAno\",\"deleteMarkerId\":\"autoAno\",\"isAutoFit\":true,\"operateWidth\":508,\"operateHeight\":381},\"shape\":{\"points\":[{\"x\":177.79999999999998,\"y\":215.392},{\"x\":177.673,\"y\":216.408},{\"x\":177.292,\"y\":216.789},{\"x\":177.292,\"y\":217.17},{\"x\":177.79999999999998,\"y\":216.916}]},\"style\":{\"opacity\":1,\"fillStyle\":\"#D91515\",\"lineWidth\":0,\"strokeStyle\":\"rgba(77, 101, 170)\",\"fill\":true,\"globalAlpha\":0.6,\"stroke\":false}},{\"openId\":\"1726740582500\",\"id\":\"1726740582500-8ORVsLvK_09ye9krHPZWd\",\"type\":\"POLYGON\",\"props\":{\"name\":\"\",\"textId\":\"autoAno\",\"deleteMarkerId\":\"autoAno\",\"isAutoFit\":true,\"operateWidth\":508,\"operateHeight\":381},\"shape\":{\"points\":[{\"x\":195.072,\"y\":215.392},{\"x\":195.57999999999998,\"y\":215.519},{\"x\":195.96099999999998,\"y\":215.011},{\"x\":195.19899999999998,\"y\":215.011}]},\"style\":{\"opacity\":1,\"fillStyle\":\"#D91515\",\"lineWidth\":0,\"strokeStyle\":\"rgba(77, 101, 170)\",\"fill\":true,\"globalAlpha\":0.6,\"stroke\":false}},{\"openId\":\"1726740582500\",\"id\":\"1726740582500-xtTgVBZ-4NENk_9XPtKXA\",\"type\":\"POLYGON\",\"props\":{\"name\":\"\",\"textId\":\"autoAno\",\"deleteMarkerId\":\"autoAno\",\"isAutoFit\":true,\"operateWidth\":508,\"operateHeight\":381},\"shape\":{\"points\":[{\"x\":181.864,\"y\":210.31199999999998},{\"x\":181.356,\"y\":210.947},{\"x\":181.22899999999998,\"y\":211.45499999999998},{\"x\":181.737,\"y\":211.709},{\"x\":182.626,\"y\":211.07399999999998}]},\"style\":{\"opacity\":1,\"fillStyle\":\"#D91515\",\"lineWidth\":0,\"strokeStyle\":\"rgba(77, 101, 170)\",\"fill\":true,\"globalAlpha\":0.6,\"stroke\":false}},{\"openId\":\"1726740582501\",\"id\":\"1726740582501-2ludzlG27UBqZBVM9qqJR\",\"type\":\"POLYGON\",\"props\":{\"name\":\"\",\"textId\":\"autoAno\",\"deleteMarkerId\":\"autoAno\",\"isAutoFit\":true,\"operateWidth\":508,\"operateHeight\":381},\"shape\":{\"points\":[{\"x\":191.76999999999998,\"y\":194.945},{\"x\":191.516,\"y\":195.19899999999998},{\"x\":191.643,\"y\":195.707},{\"x\":191.897,\"y\":195.19899999999998}]},\"style\":{\"opacity\":1,\"fillStyle\":\"#D91515\",\"lineWidth\":0,\"strokeStyle\":\"rgba(77, 101, 170)\",\"fill\":true,\"globalAlpha\":0.6,\"stroke\":false}},{\"openId\":\"1726740582502\",\"id\":\"1726740582502-VGtSjR41SP3b10J7KsmCL\",\"type\":\"POLYGON\",\"props\":{\"name\":\"\",\"textId\":\"autoAno\",\"deleteMarkerId\":\"autoAno\",\"isAutoFit\":true,\"operateWidth\":508,\"operateHeight\":381},\"shape\":{\"points\":[{\"x\":212.21699999999998,\"y\":0},{\"x\":181.102,\"y\":0},{\"x\":180.975,\"y\":0.889},{\"x\":179.70499999999998,\"y\":1.905},{\"x\":179.32399999999998,\"y\":2.667},{\"x\":179.70499999999998,\"y\":4.191},{\"x\":180.34,\"y\":5.08},{\"x\":182.499,\"y\":6.476999999999999},{\"x\":184.404,\"y\":8.382},{\"x\":184.531,\"y\":9.017},{\"x\":183.388,\"y\":10.795},{\"x\":183.515,\"y\":12.318999999999999},{\"x\":183.769,\"y\":13.081},{\"x\":185.293,\"y\":14.224},{\"x\":185.928,\"y\":14.478},{\"x\":186.309,\"y\":15.113},{\"x\":185.674,\"y\":16.256},{\"x\":184.785,\"y\":17.145},{\"x\":181.864,\"y\":18.541999999999998},{\"x\":181.22899999999998,\"y\":19.177},{\"x\":180.34,\"y\":20.701},{\"x\":180.34,\"y\":21.717},{\"x\":180.721,\"y\":22.987},{\"x\":183.134,\"y\":25.146},{\"x\":184.912,\"y\":25.781},{\"x\":185.801,\"y\":26.416},{\"x\":185.801,\"y\":27.051},{\"x\":185.42,\"y\":27.432},{\"x\":184.277,\"y\":30.352999999999998},{\"x\":184.277,\"y\":30.988},{\"x\":185.293,\"y\":33.147},{\"x\":187.198,\"y\":34.417},{\"x\":187.071,\"y\":35.052},{\"x\":185.674,\"y\":35.814},{\"x\":184.15,\"y\":37.338},{\"x\":183.515,\"y\":37.719},{\"x\":182.499,\"y\":38.989},{\"x\":181.102,\"y\":40.259},{\"x\":180.975,\"y\":41.147999999999996},{\"x\":181.22899999999998,\"y\":42.291},{\"x\":181.864,\"y\":42.926},{\"x\":183.769,\"y\":43.942},{\"x\":186.055,\"y\":44.704},{\"x\":186.563,\"y\":45.211999999999996},{\"x\":186.817,\"y\":46.355},{\"x\":186.563,\"y\":46.863},{\"x\":184.912,\"y\":48.26},{\"x\":184.785,\"y\":48.768},{\"x\":185.039,\"y\":50.165},{\"x\":185.42,\"y\":50.8},{\"x\":187.579,\"y\":51.943},{\"x\":189.357,\"y\":52.196999999999996},{\"x\":191.135,\"y\":52.832},{\"x\":193.29399999999998,\"y\":52.958999999999996},{\"x\":193.929,\"y\":53.467},{\"x\":194.183,\"y\":53.975},{\"x\":194.183,\"y\":54.61},{\"x\":191.38899999999998,\"y\":56.769},{\"x\":187.452,\"y\":57.785},{\"x\":185.039,\"y\":58.800999999999995},{\"x\":183.134,\"y\":59.181999999999995},{\"x\":181.864,\"y\":60.198},{\"x\":181.864,\"y\":60.705999999999996},{\"x\":182.753,\"y\":61.722},{\"x\":183.769,\"y\":62.357},{\"x\":185.801,\"y\":63.119},{\"x\":189.23,\"y\":64.008},{\"x\":189.738,\"y\":64.51599999999999},{\"x\":189.48399999999998,\"y\":64.89699999999999},{\"x\":187.706,\"y\":65.913},{\"x\":186.055,\"y\":67.31},{\"x\":185.674,\"y\":67.945},{\"x\":185.674,\"y\":68.834},{\"x\":186.69,\"y\":69.977},{\"x\":189.103,\"y\":71.247},{\"x\":194.183,\"y\":72.517},{\"x\":194.31,\"y\":74.295},{\"x\":193.802,\"y\":74.92999999999999},{\"x\":192.91299999999998,\"y\":75.31099999999999},{\"x\":183.261,\"y\":76.581},{\"x\":182.626,\"y\":77.089},{\"x\":182.245,\"y\":77.978},{\"x\":182.245,\"y\":78.486},{\"x\":183.007,\"y\":79.62899999999999},{\"x\":183.642,\"y\":80.137},{\"x\":185.928,\"y\":81.407},{\"x\":188.468,\"y\":82.423},{\"x\":189.357,\"y\":83.185},{\"x\":190.754,\"y\":83.947},{\"x\":190.24599999999998,\"y\":84.582},{\"x\":189.611,\"y\":84.709},{\"x\":187.198,\"y\":86.106},{\"x\":186.944,\"y\":86.995},{\"x\":186.944,\"y\":88.011},{\"x\":187.198,\"y\":88.51899999999999},{\"x\":189.48399999999998,\"y\":89.789},{\"x\":193.548,\"y\":90.80499999999999},{\"x\":195.57999999999998,\"y\":91.059},{\"x\":196.72299999999998,\"y\":91.948},{\"x\":196.72299999999998,\"y\":92.583},{\"x\":195.57999999999998,\"y\":93.472},{\"x\":193.67499999999998,\"y\":93.726},{\"x\":189.992,\"y\":95.12299999999999},{\"x\":187.579,\"y\":95.377},{\"x\":184.404,\"y\":96.64699999999999},{\"x\":183.769,\"y\":97.155},{\"x\":183.388,\"y\":97.917},{\"x\":183.769,\"y\":98.679},{\"x\":187.071,\"y\":100.584},{\"x\":191.516,\"y\":101.6},{\"x\":193.167,\"y\":101.727},{\"x\":193.548,\"y\":102.108},{\"x\":193.421,\"y\":102.48899999999999},{\"x\":191.38899999999998,\"y\":102.86999999999999},{\"x\":188.214,\"y\":104.39399999999999},{\"x\":187.198,\"y\":105.664},{\"x\":187.198,\"y\":106.29899999999999},{\"x\":187.833,\"y\":106.934},{\"x\":191.76999999999998,\"y\":108.585},{\"x\":196.72299999999998,\"y\":109.093},{\"x\":197.48499999999999,\"y\":109.474},{\"x\":197.86599999999999,\"y\":110.23599999999999},{\"x\":197.48499999999999,\"y\":110.99799999999999},{\"x\":196.34199999999998,\"y\":111.75999999999999},{\"x\":191.262,\"y\":113.157},{\"x\":187.325,\"y\":113.919},{\"x\":185.42,\"y\":114.554},{\"x\":184.277,\"y\":115.443},{\"x\":184.15,\"y\":116.205},{\"x\":184.785,\"y\":116.967},{\"x\":185.674,\"y\":117.475},{\"x\":188.595,\"y\":118.237},{\"x\":193.167,\"y\":118.74499999999999},{\"x\":197.231,\"y\":118.74499999999999},{\"x\":197.739,\"y\":118.872},{\"x\":198.628,\"y\":119.634},{\"x\":199.009,\"y\":126.365},{\"x\":198.755,\"y\":128.016},{\"x\":198.12,\"y\":129.667},{\"x\":198.501,\"y\":132.715},{\"x\":198.501,\"y\":134.36599999999999},{\"x\":198.755,\"y\":135.509},{\"x\":198.628,\"y\":147.57399999999998},{\"x\":199.009,\"y\":148.844},{\"x\":200.787,\"y\":150.495},{\"x\":201.422,\"y\":151.511},{\"x\":201.295,\"y\":152.146},{\"x\":200.66,\"y\":152.654},{\"x\":199.136,\"y\":153.035},{\"x\":198.24699999999999,\"y\":153.543},{\"x\":197.10399999999998,\"y\":154.813},{\"x\":196.72299999999998,\"y\":155.956},{\"x\":196.977,\"y\":161.54399999999998},{\"x\":197.612,\"y\":162.30599999999998},{\"x\":198.628,\"y\":162.941},{\"x\":199.136,\"y\":163.957},{\"x\":200.025,\"y\":164.59199999999998},{\"x\":200.66,\"y\":165.35399999999998},{\"x\":200.914,\"y\":167.894},{\"x\":200.279,\"y\":168.656},{\"x\":198.882,\"y\":169.037},{\"x\":197.358,\"y\":169.164},{\"x\":196.469,\"y\":169.545},{\"x\":195.453,\"y\":170.307},{\"x\":194.945,\"y\":171.323},{\"x\":195.19899999999998,\"y\":172.212},{\"x\":196.596,\"y\":173.355},{\"x\":197.48499999999999,\"y\":174.75199999999998},{\"x\":198.24699999999999,\"y\":175.387},{\"x\":198.501,\"y\":176.022},{\"x\":198.628,\"y\":178.054},{\"x\":198.12,\"y\":184.023},{\"x\":198.374,\"y\":184.785},{\"x\":199.009,\"y\":185.674},{\"x\":198.628,\"y\":186.817},{\"x\":198.628,\"y\":187.706},{\"x\":199.263,\"y\":189.23},{\"x\":197.739,\"y\":191.00799999999998},{\"x\":197.48499999999999,\"y\":191.76999999999998},{\"x\":197.358,\"y\":193.802},{\"x\":196.34199999999998,\"y\":195.57999999999998},{\"x\":196.977,\"y\":197.10399999999998},{\"x\":196.85,\"y\":197.48499999999999},{\"x\":196.469,\"y\":197.86599999999999},{\"x\":195.19899999999998,\"y\":198.24699999999999},{\"x\":193.929,\"y\":199.39},{\"x\":193.548,\"y\":200.025},{\"x\":193.167,\"y\":200.025},{\"x\":192.405,\"y\":199.009},{\"x\":191.76999999999998,\"y\":198.755},{\"x\":191.00799999999998,\"y\":198.882},{\"x\":190.373,\"y\":199.644},{\"x\":190.62699999999998,\"y\":201.295},{\"x\":190.24599999999998,\"y\":201.93},{\"x\":190.119,\"y\":203.073},{\"x\":190.62699999999998,\"y\":203.581},{\"x\":192.786,\"y\":204.216},{\"x\":193.67499999999998,\"y\":205.105},{\"x\":193.67499999999998,\"y\":207.518},{\"x\":193.04,\"y\":209.042},{\"x\":193.167,\"y\":209.677},{\"x\":193.67499999999998,\"y\":209.677},{\"x\":193.802,\"y\":209.423},{\"x\":193.802,\"y\":206.88299999999998},{\"x\":194.183,\"y\":206.50199999999998},{\"x\":195.072,\"y\":206.375},{\"x\":195.57999999999998,\"y\":205.73999999999998},{\"x\":195.834,\"y\":202.057},{\"x\":196.088,\"y\":201.422},{\"x\":197.10399999999998,\"y\":200.152},{\"x\":197.993,\"y\":199.771},{\"x\":199.771,\"y\":199.517},{\"x\":200.406,\"y\":197.993},{\"x\":201.041,\"y\":197.231},{\"x\":200.914,\"y\":195.57999999999998},{\"x\":202.184,\"y\":194.691},{\"x\":202.311,\"y\":191.643},{\"x\":201.676,\"y\":189.86499999999998},{\"x\":201.803,\"y\":187.198},{\"x\":201.422,\"y\":185.293},{\"x\":201.803,\"y\":184.15},{\"x\":202.819,\"y\":182.753},{\"x\":203.581,\"y\":182.245},{\"x\":205.105,\"y\":182.499},{\"x\":205.613,\"y\":182.753},{\"x\":205.994,\"y\":183.261},{\"x\":206.50199999999998,\"y\":184.404},{\"x\":207.01,\"y\":187.579},{\"x\":207.01,\"y\":188.849},{\"x\":207.518,\"y\":190.5},{\"x\":209.296,\"y\":189.992},{\"x\":211.07399999999998,\"y\":189.992},{\"x\":211.582,\"y\":189.357},{\"x\":211.709,\"y\":187.706},{\"x\":211.45499999999998,\"y\":186.055},{\"x\":211.07399999999998,\"y\":184.912},{\"x\":210.31199999999998,\"y\":183.769},{\"x\":210.185,\"y\":182.245},{\"x\":209.93099999999998,\"y\":181.60999999999999},{\"x\":209.93099999999998,\"y\":179.70499999999998},{\"x\":209.54999999999998,\"y\":178.435},{\"x\":209.042,\"y\":177.79999999999998},{\"x\":209.54999999999998,\"y\":176.403},{\"x\":209.54999999999998,\"y\":175.26},{\"x\":209.93099999999998,\"y\":173.863},{\"x\":209.93099999999998,\"y\":171.577},{\"x\":209.042,\"y\":169.672},{\"x\":208.28,\"y\":169.291},{\"x\":207.772,\"y\":168.656},{\"x\":207.64499999999998,\"y\":167.767},{\"x\":207.01,\"y\":166.11599999999999},{\"x\":206.50199999999998,\"y\":165.73499999999999},{\"x\":206.50199999999998,\"y\":165.1},{\"x\":207.518,\"y\":162.68699999999998},{\"x\":208.02599999999998,\"y\":162.179},{\"x\":209.677,\"y\":161.92499999999998},{\"x\":211.07399999999998,\"y\":161.417},{\"x\":211.963,\"y\":160.147},{\"x\":212.09,\"y\":159.25799999999998},{\"x\":211.709,\"y\":158.115},{\"x\":211.45499999999998,\"y\":155.575},{\"x\":210.566,\"y\":155.067},{\"x\":209.423,\"y\":153.543},{\"x\":207.899,\"y\":153.162},{\"x\":206.756,\"y\":153.162},{\"x\":205.994,\"y\":153.416},{\"x\":204.97799999999998,\"y\":154.305},{\"x\":204.216,\"y\":152.781},{\"x\":204.216,\"y\":151.511},{\"x\":204.851,\"y\":149.85999999999999},{\"x\":205.994,\"y\":148.844},{\"x\":206.50199999999998,\"y\":147.95499999999998},{\"x\":206.50199999999998,\"y\":145.542},{\"x\":206.12099999999998,\"y\":143.256},{\"x\":205.73999999999998,\"y\":136.271},{\"x\":205.73999999999998,\"y\":129.921},{\"x\":205.35899999999998,\"y\":128.905},{\"x\":204.59699999999998,\"y\":128.016},{\"x\":204.343,\"y\":127.381},{\"x\":204.216,\"y\":119.88799999999999},{\"x\":204.97799999999998,\"y\":118.872},{\"x\":205.613,\"y\":118.618},{\"x\":207.137,\"y\":118.36399999999999},{\"x\":210.439,\"y\":118.237},{\"x\":212.725,\"y\":117.729},{\"x\":216.662,\"y\":116.332},{\"x\":217.678,\"y\":115.443},{\"x\":218.186,\"y\":114.173},{\"x\":218.059,\"y\":113.792},{\"x\":217.297,\"y\":113.03},{\"x\":213.35999999999999,\"y\":111.75999999999999},{\"x\":211.582,\"y\":111.506},{\"x\":207.137,\"y\":111.506},{\"x\":205.35899999999998,\"y\":111.125},{\"x\":204.59699999999998,\"y\":110.363},{\"x\":204.59699999999998,\"y\":109.855},{\"x\":205.613,\"y\":108.839},{\"x\":208.915,\"y\":108.331},{\"x\":211.963,\"y\":107.315},{\"x\":213.35999999999999,\"y\":106.67999999999999},{\"x\":213.995,\"y\":105.91799999999999},{\"x\":213.868,\"y\":104.267},{\"x\":213.35999999999999,\"y\":103.63199999999999},{\"x\":210.69299999999998,\"y\":102.86999999999999},{\"x\":206.756,\"y\":102.616},{\"x\":206.375,\"y\":102.48899999999999},{\"x\":206.12099999999998,\"y\":101.981},{\"x\":206.88299999999998,\"y\":101.6},{\"x\":211.201,\"y\":100.965},{\"x\":214.376,\"y\":99.187},{\"x\":215.392,\"y\":98.93299999999999},{\"x\":216.154,\"y\":98.425},{\"x\":217.297,\"y\":96.393},{\"x\":216.662,\"y\":95.50399999999999},{\"x\":215.138,\"y\":94.361},{\"x\":211.07399999999998,\"y\":93.472},{\"x\":205.35899999999998,\"y\":93.091},{\"x\":204.47,\"y\":92.456},{\"x\":204.343,\"y\":91.948},{\"x\":205.105,\"y\":91.059},{\"x\":206.12099999999998,\"y\":90.297},{\"x\":207.26399999999998,\"y\":89.789},{\"x\":209.16899999999998,\"y\":89.535},{\"x\":211.709,\"y\":88.392},{\"x\":212.725,\"y\":87.37599999999999},{\"x\":213.233,\"y\":86.614},{\"x\":213.233,\"y\":85.979},{\"x\":211.709,\"y\":84.963},{\"x\":209.93099999999998,\"y\":84.455},{\"x\":209.54999999999998,\"y\":84.074},{\"x\":209.804,\"y\":82.55},{\"x\":210.947,\"y\":81.788},{\"x\":213.233,\"y\":81.28},{\"x\":215.011,\"y\":80.264},{\"x\":216.408,\"y\":78.74},{\"x\":216.535,\"y\":77.724},{\"x\":216.281,\"y\":76.708},{\"x\":215.646,\"y\":75.946},{\"x\":212.852,\"y\":74.54899999999999},{\"x\":209.16899999999998,\"y\":73.66},{\"x\":205.486,\"y\":73.279},{\"x\":204.59699999999998,\"y\":73.02499999999999},{\"x\":203.962,\"y\":72.517},{\"x\":204.47,\"y\":72.009},{\"x\":207.391,\"y\":71.501},{\"x\":209.042,\"y\":70.993},{\"x\":211.328,\"y\":69.85},{\"x\":212.344,\"y\":68.834},{\"x\":212.471,\"y\":67.818},{\"x\":212.09,\"y\":66.675},{\"x\":211.582,\"y\":66.167},{\"x\":210.058,\"y\":65.27799999999999},{\"x\":208.78799999999998,\"y\":64.77},{\"x\":208.40699999999998,\"y\":64.389},{\"x\":208.40699999999998,\"y\":64.135},{\"x\":210.566,\"y\":62.864999999999995},{\"x\":213.487,\"y\":61.849},{\"x\":214.884,\"y\":60.96},{\"x\":215.519,\"y\":60.198},{\"x\":215.646,\"y\":58.800999999999995},{\"x\":215.392,\"y\":58.293},{\"x\":214.249,\"y\":57.785},{\"x\":209.423,\"y\":56.260999999999996},{\"x\":204.47,\"y\":55.245},{\"x\":203.454,\"y\":54.864},{\"x\":202.819,\"y\":54.229},{\"x\":203.073,\"y\":53.213},{\"x\":203.581,\"y\":52.577999999999996},{\"x\":208.78799999999998,\"y\":51.308},{\"x\":209.54999999999998,\"y\":50.927},{\"x\":211.07399999999998,\"y\":49.657},{\"x\":211.582,\"y\":48.894999999999996},{\"x\":211.582,\"y\":48.26},{\"x\":209.54999999999998,\"y\":45.592999999999996},{\"x\":208.534,\"y\":44.958},{\"x\":208.40699999999998,\"y\":44.577},{\"x\":208.661,\"y\":44.196},{\"x\":211.582,\"y\":42.926},{\"x\":212.725,\"y\":42.164},{\"x\":213.868,\"y\":41.147999999999996},{\"x\":214.376,\"y\":40.132},{\"x\":214.376,\"y\":38.735},{\"x\":213.487,\"y\":37.719},{\"x\":211.201,\"y\":36.321999999999996},{\"x\":209.16899999999998,\"y\":35.56},{\"x\":206.88299999999998,\"y\":35.052},{\"x\":206.375,\"y\":34.417},{\"x\":209.042,\"y\":32.385},{\"x\":210.185,\"y\":30.988},{\"x\":210.31199999999998,\"y\":28.701999999999998},{\"x\":209.804,\"y\":27.813},{\"x\":208.661,\"y\":26.797},{\"x\":208.661,\"y\":26.416},{\"x\":209.423,\"y\":25.781},{\"x\":211.83599999999998,\"y\":24.637999999999998},{\"x\":213.233,\"y\":23.241},{\"x\":213.487,\"y\":22.733},{\"x\":213.614,\"y\":20.701},{\"x\":212.471,\"y\":18.796},{\"x\":211.45499999999998,\"y\":17.907},{\"x\":208.915,\"y\":16.509999999999998},{\"x\":208.28,\"y\":16.383},{\"x\":206.50199999999998,\"y\":15.494},{\"x\":206.12099999999998,\"y\":15.113},{\"x\":206.12099999999998,\"y\":14.605},{\"x\":206.375,\"y\":14.350999999999999},{\"x\":209.042,\"y\":13.081},{\"x\":209.54999999999998,\"y\":12.446},{\"x\":209.804,\"y\":10.414},{\"x\":209.16899999999998,\"y\":8.89},{\"x\":209.16899999999998,\"y\":7.112},{\"x\":209.93099999999998,\"y\":6.35},{\"x\":211.582,\"y\":5.460999999999999},{\"x\":212.344,\"y\":4.445},{\"x\":212.852,\"y\":3.175},{\"x\":212.852,\"y\":2.54},{\"x\":212.21699999999998,\"y\":1.143}]},\"style\":{\"opacity\":1,\"fillStyle\":\"#D91515\",\"lineWidth\":0,\"strokeStyle\":\"rgba(77, 101, 170)\",\"fill\":true,\"globalAlpha\":0.6,\"stroke\":false}}]";
//
////        LabelmeImageData bean = JSONUtil.toBean(markInfo, LabelmeImageData.class);
////        System.out.println(bean);
////        List<WebRectangleShape> webRectangleShapes = FormatConverter.convertLabelmeDataToWeb(bean,null);
////        //(List<WebRectangleShape> webShapes, String imagePath, int imageWidth, int imageHeight) {
//        Image image = null;
//        List<WebRectangleShape> list = JSONUtil.toList(JSONUtil.parseArray(webMarkInfo), WebRectangleShape.class);
//
//        MarkInfoEntity markInfoEntity = new MarkInfoEntity();
//        markInfoEntity.setWidth(4000);
//        markInfoEntity.setHeight(3000);
//        markInfoEntity.setOperateWidth(508);
//        markInfoEntity.setOperateHeight(381);
//        LabelmeImageData labelmeImageData = FormatConverter.convertWebDataToLabelme(list, "123.jpg", 4000, 3000, markInfoEntity);
//        System.out.println(labelmeImageData);


//        System.out.println(JSONUtil.toJsonStr(webRectangleShapes));
        String jsonStr = "{\n" +
                "  \"version\": \"2.3.6\",\n" +
                "  \"flags\": {},\n" +
                "  \"shapes\": [\n" +
                "    {\n" +
                "      \"label\": \"insulator\",\n" +
                "      \"points\": [\n" +
                "        [\n" +
                "          2890.0,\n" +
                "          945.0\n" +
                "        ],\n" +
                "        [\n" +
                "          3345.0,\n" +
                "          945.0\n" +
                "        ],\n" +
                "        [\n" +
                "          3345.0,\n" +
                "          3140.0\n" +
                "        ],\n" +
                "        [\n" +
                "          2890.0,\n" +
                "          3140.0\n" +
                "        ]\n" +
                "      ],\n" +
                "      \"group_id\": null,\n" +
                "      \"description\": \"\",\n" +
                "      \"difficult\": false,\n" +
                "      \"shape_type\": \"rectangle\",\n" +
                "      \"flags\": {},\n" +
                "      \"attributes\": {}\n" +
                "    },\n" +
                "    {\n" +
                "      \"label\": \"insulator\",\n" +
                "      \"points\": [\n" +
                "        [\n" +
                "          4465.0,\n" +
                "          1140.0\n" +
                "        ],\n" +
                "        [\n" +
                "          5995.0,\n" +
                "          1140.0\n" +
                "        ],\n" +
                "        [\n" +
                "          5995.0,\n" +
                "          1925.0\n" +
                "        ],\n" +
                "        [\n" +
                "          4465.0,\n" +
                "          1925.0\n" +
                "        ]\n" +
                "      ],\n" +
                "      \"group_id\": null,\n" +
                "      \"description\": \"\",\n" +
                "      \"difficult\": false,\n" +
                "      \"shape_type\": \"rectangle\",\n" +
                "      \"flags\": {},\n" +
                "      \"attributes\": {}\n" +
                "    },\n" +
                "    {\n" +
                "      \"label\": \"damper\",\n" +
                "      \"points\": [\n" +
                "        [\n" +
                "          2805.0,\n" +
                "          3175.0\n" +
                "        ],\n" +
                "        [\n" +
                "          3015.0,\n" +
                "          3175.0\n" +
                "        ],\n" +
                "        [\n" +
                "          3015.0,\n" +
                "          3285.0\n" +
                "        ],\n" +
                "        [\n" +
                "          2805.0,\n" +
                "          3285.0\n" +
                "        ]\n" +
                "      ],\n" +
                "      \"group_id\": null,\n" +
                "      \"description\": \"\",\n" +
                "      \"difficult\": false,\n" +
                "      \"shape_type\": \"rectangle\",\n" +
                "      \"flags\": {},\n" +
                "      \"attributes\": {}\n" +
                "    },\n" +
                "    {\n" +
                "      \"label\": \"insulator\",\n" +
                "      \"points\": [\n" +
                "        [\n" +
                "          780.0,\n" +
                "          2810.0\n" +
                "        ],\n" +
                "        [\n" +
                "          985.0,\n" +
                "          2810.0\n" +
                "        ],\n" +
                "        [\n" +
                "          985.0,\n" +
                "          3455.0\n" +
                "        ],\n" +
                "        [\n" +
                "          780.0,\n" +
                "          3455.0\n" +
                "        ]\n" +
                "      ],\n" +
                "      \"group_id\": null,\n" +
                "      \"description\": \"\",\n" +
                "      \"difficult\": false,\n" +
                "      \"shape_type\": \"rectangle\",\n" +
                "      \"flags\": {},\n" +
                "      \"attributes\": {}\n" +
                "    },\n" +
                "    {\n" +
                "      \"label\": \"damper\",\n" +
                "      \"points\": [\n" +
                "        [\n" +
                "          2665.0,\n" +
                "          15.0\n" +
                "        ],\n" +
                "        [\n" +
                "          2870.0,\n" +
                "          15.0\n" +
                "        ],\n" +
                "        [\n" +
                "          2870.0,\n" +
                "          120.0\n" +
                "        ],\n" +
                "        [\n" +
                "          2665.0,\n" +
                "          120.0\n" +
                "        ]\n" +
                "      ],\n" +
                "      \"group_id\": null,\n" +
                "      \"description\": \"\",\n" +
                "      \"difficult\": false,\n" +
                "      \"shape_type\": \"rectangle\",\n" +
                "      \"flags\": {},\n" +
                "      \"attributes\": {}\n" +
                "    }\n" +
                "  ],\n" +
                "  \"imagePath\": \"AntiBird_1.jpg\",\n" +
                "  \"imageData\": null,\n" +
                "  \"imageHeight\": 4000,\n" +
                "  \"imageWidth\": 6000,\n" +
                "  \"text\": \"\"\n" +
                "}";

        LabelmeImageData bean = JSONUtil.toBean(jsonStr, LabelmeImageData.class);
        System.out.println(bean);
        MarkInfoEntity markEntity = new MarkInfoEntity();
        markEntity.setOperateWidth(6000);
        markEntity.setWidth(6000);
        List<WebRectangleShape> webRectangleShapes = FormatConverter.convertLabelmeDataToWeb(bean, markEntity);
        System.out.println(JSONUtil.toJsonStr(webRectangleShapes));


    }
}
