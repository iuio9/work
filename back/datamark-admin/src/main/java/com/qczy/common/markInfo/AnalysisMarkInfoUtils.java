package com.qczy.common.markInfo;

import cn.hutool.json.JSONUtil;
import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import com.qczy.model.entity.WebProps;
import com.qczy.model.entity.WebRectangleShape;
import com.qczy.model.entity.WebShapeStyle;
import com.qczy.utils.StringUtils;
import org.apache.commons.io.FilenameUtils;
import org.dom4j.DocumentException;
import org.dom4j.io.SAXReader;
import org.springframework.beans.BeanUtils;
import org.springframework.stereotype.Component;
import org.w3c.dom.Document;
import org.w3c.dom.Element;
import org.w3c.dom.Node;
import org.w3c.dom.NodeList;

import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.stream.Collectors;

/**
 * @Author: hh
 * @Version: 1.0
 * @Date: 2024/12/2 14:06
 * @Description: //解析mark信息 包括 json  xml
 */
@Component
public class AnalysisMarkInfoUtils {


/*
    private static Map<String, List<WebRectangleShape>> analysisXml(String filePath) {
        File file = new File(filePath);
        if (file.exists()) {
            throw new RuntimeException("文件不存在！");
        }
        if (!file.getName().split("\\.")[1].equalsIgnoreCase("xml")) {
            throw new RuntimeException("文件类型必须为xml文件");
        }

        try {
            //1.创建Reader对象
            SAXReader reader = new SAXReader();
            //2.加载xml
            Document document = reader.read(file);


        } catch (Exception e) {
            throw new RuntimeException(e);
        }


    }
*/


    // TODO 百度使用的json
    public static Map<String, List<WebRectangleShape>> analysisJson(String filePath) {
        File file = new File(filePath);
        if (!file.exists()) {
            throw new RuntimeException("文件不存在！");
        }
        if (!file.getName().split("\\.")[1].equalsIgnoreCase("json")) {
            throw new RuntimeException("文件类型必须为json文件");
        }


        try {
            String jsonStr = new String(Files.readAllBytes(Paths.get(filePath)));
            // 解析 JSON 字符串
            JsonObject jsonObject = JsonParser.parseString(jsonStr).getAsJsonObject();
            List<FileMarkLog> fileMarkLogList = new ArrayList<>();

            // 获取 images 和 annotations 和 categories 数组
            JsonArray images = jsonObject.getAsJsonArray("images");
            JsonArray annotations = jsonObject.getAsJsonArray("annotations");
            JsonArray categories = jsonObject.getAsJsonArray("categories");

            Map<Integer, String> categoryMap = new HashMap<>();

            // 遍历 categories 数组，填充 categoryMap
            for (JsonElement categoryElement : categories) {
                JsonObject categoryObject = categoryElement.getAsJsonObject();
                int categoryId = categoryObject.get("id").getAsInt();
                String categoryName = categoryObject.get("name").getAsString();
                categoryMap.put(categoryId, categoryName);
            }

            for (JsonElement annotation : annotations) {
                FileMarkLog fileMarkLog = new FileMarkLog();
                // 文件id
                int imageId = annotation.getAsJsonObject().get("image_id").getAsInt();
                fileMarkLog.setId(imageId);
                // 查找与 annotation 的 image_id 匹配的 image
                for (JsonElement image : images) {
                    JsonObject imageObject = image.getAsJsonObject();
                    int imageIdInImages = imageObject.get("id").getAsInt();  // 获取 image 的 id
                    // 如果 image_id 匹配，打印该 image 对应的 file_name width height
                    if (imageId == imageIdInImages) {
                        // 文件名称
                        fileMarkLog.setFileName(imageObject.get("file_name").getAsString());
                        fileMarkLog.setWidth(imageObject.get("width").getAsDouble());
                        fileMarkLog.setHeight(imageObject.get("height").getAsDouble());
                        break;  // 找到匹配的 image 后跳出循环
                    }
                }
                //  bbox`：对象的边界框，格式为`[x, y, width, height]`。
                fileMarkLog.setBbox(annotation.getAsJsonObject().getAsJsonArray("bbox"));
                // shape
                if (annotation.getAsJsonObject().has("shape") && !annotation.getAsJsonObject().get("shape").isJsonNull()) {
                    fileMarkLog.setShape(annotation.getAsJsonObject().get("shape").getAsString());
                } else {
                    fileMarkLog.setShape("RECT"); // 默认值
                }
                //fileMarkLog.setShape(annotation.getAsJsonObject().get("shape").getAsString());
                // 标签名称
                fileMarkLog.setLabelName(categoryMap.get(annotation.getAsJsonObject().get("category_id").getAsInt()));
                // 如果是多边形，获取点位
                if (!annotation.getAsJsonObject().get("segmentation").isJsonNull()) {
                    JsonArray segmentation = annotation.getAsJsonObject().getAsJsonArray("segmentation");
                    if (segmentation != null && segmentation.size() > 0) {
                        fileMarkLog.setSegmentation(segmentation.get(0).getAsJsonArray());
                    }
                }

                // 如果是圆形，获取 x ，y 面积
                if (annotation.getAsJsonObject().has("meta") && !annotation.getAsJsonObject().get("meta").isJsonNull()) {
                    fileMarkLog.setMeta(annotation.getAsJsonObject().get("meta").toString());
                }


                fileMarkLogList.add(fileMarkLog);

            }
            return setWebRectangleShapeMap(fileMarkLogList);
           /* for (Map.Entry<String, List<WebRectangleShape>> stringListEntry : stringListMap.entrySet()) {
                System.out.println(JSONUtil.toJsonStr(stringListEntry));
            }
*/
        } catch (IOException e) {
            throw new RuntimeException(e);
        }


    }

    // TODO 官网使用的json
    public static Map<String, List<WebRectangleShape>> analysisJson2(String filePath) {
      /*  File file = new File(filePath);
        if (!file.exists()) {
            throw new RuntimeException("文件不存在！");
        }
        if (!file.getName().split("\\.")[1].equalsIgnoreCase("json")) {
            throw new RuntimeException("文件类型必须为json文件");
        }*/


        try {
            // 读取文件
            String annotationFilePath = new String(Files.readAllBytes(Paths.get(filePath)));
            // 解析 JSON 字符串
            JsonObject jsonObject = JsonParser.parseString(annotationFilePath).getAsJsonObject();
            List<FileMarkLog> fileMarkLogList = new ArrayList<>();

            // 获取 images 和 annotations 和 categories 数组
            JsonArray images = jsonObject.getAsJsonArray("images");
            JsonArray annotations = jsonObject.getAsJsonArray("annotations");
            JsonArray categories = jsonObject.getAsJsonArray("categories");


            Map<Integer, String> categoryMap = new HashMap<>();

            // 遍历 categories 数组，填充 categoryMap
            for (JsonElement categoryElement : categories) {
                JsonObject categoryObject = categoryElement.getAsJsonObject();
                int categoryId = categoryObject.get("id").getAsInt();
                String categoryName = categoryObject.get("name").getAsString();
                categoryMap.put(categoryId, categoryName);
            }

            for (JsonElement imageElement : images) {
                JsonObject imageObject = imageElement.getAsJsonObject();
                FileMarkLog fileMarkLog = new FileMarkLog();
                fileMarkLog.setId(imageObject.get("id").getAsInt());
                fileMarkLog.setFileName(imageObject.get("file_name").getAsString());
                fileMarkLog.setWidth(imageObject.get("width").getAsDouble());
                fileMarkLog.setHeight(imageObject.get("height").getAsDouble());
                fileMarkLogList.add(fileMarkLog);
            }

            int result = checkJsonType(jsonObject);
            if (result == 1) {
                List<FileMarkLog> newFileMarkLogs = new ArrayList<>();
                // 目标检测
                for (JsonElement annotationElement : annotations) {
                    JsonObject annotationObject = annotationElement.getAsJsonObject();
                    int imageId = annotationObject.get("image_id").getAsInt();
                    // 循环
                    Iterator<FileMarkLog> iterator = fileMarkLogList.iterator();
                    while (iterator.hasNext()) {
                        FileMarkLog fileMarkLog = iterator.next();
                        if (fileMarkLog.getId() != imageId) {
                            continue;
                        }

                        JsonArray asJsonArray = annotationObject.get("segments_info").getAsJsonArray();
                        for (JsonElement jsonElement : asJsonArray) {
                            jsonObject = jsonElement.getAsJsonObject();
                            int categoryId = jsonObject.get("category_id").getAsInt();
                            // 拷贝
                            FileMarkLog markLog = new FileMarkLog();
                            BeanUtils.copyProperties(fileMarkLog, markLog);
                            markLog.setShape("RECT");
                            markLog.setBbox(jsonObject.get("bbox").getAsJsonArray());
                            markLog.setLabelName(categoryMap.get(categoryId));
                            newFileMarkLogs.add(fileMarkLog);
                        }
                    }
                }
                // 将修改后的元素添加到原列表中
                fileMarkLogList.addAll(newFileMarkLogs);
            } else if (result == 2) {
                List<FileMarkLog> newFileMarkLogs = new ArrayList<>();
                // 实例分割
                for (JsonElement annotationElement : annotations) {

                    JsonObject annotationObject = annotationElement.getAsJsonObject();
                    int imageId = annotationObject.get("image_id").getAsInt();
                    int categoryId = annotationObject.get("category_id").getAsInt();
                    // 循环
                    Iterator<FileMarkLog> iterator = fileMarkLogList.iterator();
                    while (iterator.hasNext()) {
                        FileMarkLog fileMarkLog = iterator.next();
                        if (fileMarkLog.getId() != imageId) {
                            continue;
                        }
                        if (annotationObject.get("segmentation") instanceof JsonArray) {
                            JsonArray asJsonArray = annotationObject.get("segmentation").getAsJsonArray();
                            for (JsonElement jsonElement : asJsonArray) {
                                fileMarkLog.setSegmentation(jsonElement.getAsJsonArray());
                            }
                        } else if (annotationObject.get("segmentation") instanceof JsonObject) {
                            JsonArray asJsonArray = annotationObject.get("segmentation").getAsJsonObject().get("counts").getAsJsonArray();
                            fileMarkLog.setSegmentation(asJsonArray);
                        }
                        fileMarkLog.setShape("BRUSH");
                        // fileMarkLog.setBbox(annotationObject.get("bbox").getAsJsonArray());
                        fileMarkLog.setLabelName(categoryMap.get(categoryId));
                        newFileMarkLogs.add(fileMarkLog);
                    }
                }
                // 将修改后的元素添加到原列表中
                fileMarkLogList.addAll(newFileMarkLogs);
            } else {
                // 暂无！！！
                return null;
            }


            return setWebRectangleShapeMap(fileMarkLogList);


        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    // TODO 算法使用的json
    public static Map<String, List<WebRectangleShape>> analysisJson3(String filePath) {
       /* File file = new File(filePath);
        if (!file.exists()) {
            throw new RuntimeException("文件不存在！");
        }
        if (!file.getName().split("\\.")[1].equalsIgnoreCase("json")) {
            throw new RuntimeException("文件类型必须为json文件");
        }*/

        try {
            // 读取文件
            String annotationFilePath = new String(Files.readAllBytes(Paths.get(filePath)));
            // 解析 JSON 字符串
            JsonObject jsonObject = JsonParser.parseString(annotationFilePath).getAsJsonObject();
            //System.out.println(jsonObject.toString());

            double imageWidth = jsonObject.get("imageWidth").getAsDouble();
            double imageHeight = jsonObject.get("imageHeight").getAsDouble();
            String fileName = jsonObject.get("imagePath").getAsString();


            List<FileMarkLog> fileMarkLogList = new ArrayList<>();
            int i = 1;
            JsonArray shapesList = jsonObject.getAsJsonArray("shapes");
            for (JsonElement shapeElement : shapesList) {
                FileMarkLog fileMarkLog = new FileMarkLog();
                fileMarkLog.setId(1);
                JsonObject shapeObject = shapeElement.getAsJsonObject();
                // 标签
                String label = shapeObject.get("label").getAsString();
                fileMarkLog.setLabelName(label);
                // 文件名
                fileMarkLog.setFileName(fileName);
                // 图片 长 - 宽
                fileMarkLog.setWidth(imageWidth);
                fileMarkLog.setHeight(imageHeight);
                // 图片点位
                fileMarkLog.setBbox(shapeObject.get("points").getAsJsonArray());
                // 标注类型
                fileMarkLog.setShape(shapeObject.get("shape_type").getAsString());

                fileMarkLogList.add(fileMarkLog);
                i++;
            }


            return setWebRectangleShapeMap(fileMarkLogList);


        } catch (IOException e) {
            throw new RuntimeException(e);
        }

    }

    // TODO 第三方平台的json
    public static Map<String, List<WebRectangleShape>> analysisJson4(String filePath, Map<String, Integer> imgMap) {
        File file = new File(filePath);
        if (!file.exists()) {
            throw new RuntimeException("文件不存在！");
        }

        // 获取文件后缀
        String extension = FilenameUtils.getExtension(file.getName());
        if (!extension.equalsIgnoreCase("json")) {
            throw new RuntimeException("文件类型必须为json文件");
        }


        try {
            // 读取文件
            String annotationFilePath = new String(Files.readAllBytes(Paths.get(filePath)));
            // 解析 JSON 字符串
            JsonObject jsonObject = JsonParser.parseString(annotationFilePath).getAsJsonObject();

            // 获取文件名
            // 1. 首先先获取当前json的文件名
            String fileName = checkFileNameInMap(file.getPath(), imgMap);
            if (StringUtils.isEmpty(fileName)) {
                return null;
            }


            List<FileMarkLog> fileMarkLogList = new ArrayList<>();
            int i = 1;
            JsonArray shapesList = jsonObject.getAsJsonArray("labels");
            for (JsonElement shapeElement : shapesList) {
                System.out.println("-----------" + shapeElement);
                FileMarkLog fileMarkLog = new FileMarkLog();
                fileMarkLog.setId(1);
                fileMarkLog.setFileName(fileName);
                JsonObject shapeObject = shapeElement.getAsJsonObject();
                // 标签
                String label = shapeObject.get("name").getAsString();
                fileMarkLog.setLabelName(label);


                // 提取矩形坐标
                JsonArray jsonArray = new JsonArray();
                jsonArray.add(shapeObject.get("x1").getAsDouble());
                jsonArray.add(shapeObject.get("y1").getAsDouble());
                jsonArray.add(shapeObject.get("x2").getAsDouble());
                jsonArray.add(shapeObject.get("y2").getAsDouble());
                fileMarkLog.setSegmentation(jsonArray);

                // 标注类型
                fileMarkLog.setShape("POLYGON1"); // 默认类型为多边形

                fileMarkLogList.add(fileMarkLog);
                i++;
            }

            System.out.println("执行了代码-----");


            return setWebRectangleShapeMap(fileMarkLogList);


        } catch (IOException e) {
            throw new RuntimeException(e);
        }

    }



    private static Map<String, List<WebRectangleShape>> setWebRectangleShapeMap(List<FileMarkLog> fileMarkLogList) {
        Map<String, List<WebRectangleShape>> map = new HashMap<>();
        // 根据 id 进行分组
        Map<Integer, List<FileMarkLog>> groupedMap = fileMarkLogList.stream()
                .collect(Collectors.groupingBy(FileMarkLog::getId));

        for (Map.Entry<Integer, List<FileMarkLog>> entry : groupedMap.entrySet()) {
            List<FileMarkLog> value = entry.getValue();
            // 每个分组创建一个新的列表
            List<WebRectangleShape> webRectangleShapeList = new ArrayList<>();

            for (int i = 0; i < value.size(); i++) {
                WebRectangleShape webRectangleShape = new WebRectangleShape();
                // 文件信息
                webRectangleShape.setOpenId("opId-" + i);
                webRectangleShape.setId("opId-" + i);

                // 判断标注类型
                if (!StringUtils.isEmpty(value.get(i).getShape())) {
                    if (value.get(i).getShape().equalsIgnoreCase("RECT")) { // 矩形
                      /*  webRectangleShape.setType("RECT");
                        Map<String, Double> pointMap = new HashMap<>();
                        pointMap.put("x", value.get(i).getBbox().get(0).getAsDouble());
                        pointMap.put("y", value.get(i).getBbox().get(1).getAsDouble());
                        pointMap.put("width", value.get(i).getBbox().get(2).getAsDouble());
                        pointMap.put("height", value.get(i).getBbox().get(3).getAsDouble());
                        webRectangleShape.setShape(pointMap);*/


                        webRectangleShape.setType("POLYGON");
                        FileMarkLog fileMarkLog = value.get(i);
                        Map<String, List<Map<String, Double>>> listMap = new HashMap<>();
                        List<Map<String, Double>> list = new ArrayList<>();
                        for (JsonElement bbox : fileMarkLog.getBbox()) {
                            Map<String, Double> pointMap = new HashMap<>();
                            pointMap.put("x", bbox.getAsJsonArray().get(0).getAsDouble());
                            pointMap.put("y", bbox.getAsJsonArray().get(1).getAsDouble());
                            list.add(pointMap);
                        }
                        listMap.put("points", list);
                        webRectangleShape.setShape(listMap);


                    } else if (value.get(i).getShape().equalsIgnoreCase("CIRCLE")) { // 圆形
                        webRectangleShape.setType("POLYGON");
                        FileMarkLog fileMarkLog = value.get(i);
                        Map<String, List<Map<String, Double>>> listMap = new HashMap<>();
                        List<Map<String, Double>> list = new ArrayList<>();
                        for (JsonElement bbox : fileMarkLog.getBbox()) {
                            Map<String, Double> pointMap = new HashMap<>();
                            pointMap.put("x", bbox.getAsJsonArray().get(0).getAsDouble());
                            pointMap.put("y", bbox.getAsJsonArray().get(1).getAsDouble());
                            list.add(pointMap);
                        }
                        listMap.put("points", list);
                        webRectangleShape.setShape(listMap);
                    } else if (value.get(i).getShape().equalsIgnoreCase("BRUSH")) { // 多边形
                        webRectangleShape.setType("POLYGON");
                        Map<String, List<Map<String, Double>>> listMap = new HashMap<>();
                        List<Map<String, Double>> pointList = new ArrayList<>();
                        JsonArray array = value.get(i).getSegmentation();
                        int sumCount = array.size();
                        int size = sumCount % 2 == 0 ? array.size() : array.size() - 1;
                        for (int j = 0; j < size; j += 2) {
                            Map<String, Double> coordinate = new HashMap<>();
                            coordinate.put("x", array.get(j).getAsDouble());
                            coordinate.put("y", array.get(j + 1).getAsDouble());
                            pointList.add(coordinate);
                        }
                        listMap.put("points", pointList);
                        webRectangleShape.setShape(listMap);
                    } else if (value.get(i).getShape().equalsIgnoreCase("RECTANGLE")) { //TODO 默认就按照多边形
                        webRectangleShape.setType("POLYGON");
                        FileMarkLog fileMarkLog = value.get(i);
                        Map<String, List<Map<String, Double>>> listMap = new HashMap<>();
                        List<Map<String, Double>> list = new ArrayList<>();
                        for (JsonElement bbox : fileMarkLog.getBbox()) {
                            Map<String, Double> pointMap = new HashMap<>();
                            pointMap.put("x", bbox.getAsJsonArray().get(0).getAsDouble());
                            pointMap.put("y", bbox.getAsJsonArray().get(1).getAsDouble());
                            list.add(pointMap);
                        }
                        listMap.put("points", list);
                        webRectangleShape.setShape(listMap);
                    } else if (value.get(i).getShape().equalsIgnoreCase("POLYGON")) { //默认按照多边形
                        webRectangleShape.setType("POLYGON");
                        FileMarkLog fileMarkLog = value.get(i);
                        Map<String, List<Map<String, Double>>> listMap = new HashMap<>();
                        List<Map<String, Double>> list = new ArrayList<>();
                        for (JsonElement bbox : fileMarkLog.getBbox()) {
                            Map<String, Double> pointMap = new HashMap<>();
                            pointMap.put("x", bbox.getAsJsonArray().get(0).getAsDouble());
                            pointMap.put("y", bbox.getAsJsonArray().get(1).getAsDouble());
                            list.add(pointMap);
                        }
                        listMap.put("points", list);
                        webRectangleShape.setShape(listMap);
                    }else if (value.get(i).getShape().equalsIgnoreCase("POLYGON1")){
                        webRectangleShape.setType("POLYGON");
                        FileMarkLog fileMarkLog = value.get(i);
                        Map<String, List<Map<String, Double>>> listMap = new HashMap<>();
                        // 创建四个点的列表（左上角、右上角、右下角、左下角）
                        List<Map<String, Double>> points = new ArrayList<>();

                        double x1 = fileMarkLog.getSegmentation().get(0).getAsDouble();
                        double y1 = fileMarkLog.getSegmentation().get(1).getAsDouble();
                        double x2 = fileMarkLog.getSegmentation().get(2).getAsDouble();
                        double y2 = fileMarkLog.getSegmentation().get(3).getAsDouble();

                        // 左上角
                        Map<String, Double> point1 = new HashMap<>();
                        point1.put("x", x1);
                        point1.put("y", y1);
                        points.add(point1);

                        // 右上角
                        Map<String, Double> point2 = new HashMap<>();
                        point2.put("x", x2);
                        point2.put("y", y1);
                        points.add(point2);

                        // 右下角
                        Map<String, Double> point3 = new HashMap<>();
                        point3.put("x", x2);
                        point3.put("y", y2);
                        points.add(point3);

                        // 左下角
                        Map<String, Double> point4 = new HashMap<>();
                        point4.put("x", x1);
                        point4.put("y", y2);
                        points.add(point4);

                        listMap.put("points", points);
                        webRectangleShape.setShape(listMap);

                    }

                }

                // 写入标签
                WebProps webProps = new WebProps();
                webProps.setName(value.get(i).getLabelName());
                webProps.setTextId("opId-" + i);
                webProps.setDeleteMarkerId("opId-" + i);
                webRectangleShape.setProps(webProps);
                // 样式字段的转换
                WebShapeStyle style = new WebShapeStyle();
                style.setOpacity(1.0);
                style.setFillStyle("#7C0DDD");
                style.setLineWidth(2);
                style.setStrokeStyle("#7C0DDD");
                style.setFill(true);
                style.setGlobalAlpha(0.3);
                webRectangleShape.setStyle(style);

                webRectangleShape.setEye(true);
                webRectangleShape.setOperateIdx(i);

                webRectangleShapeList.add(webRectangleShape);
            }

            // 确保每个分组的 fileName 对应唯一的 WebRectangleShape 列表
            map.put(value.get(0).getFileName(), webRectangleShapeList);
        }

        return map;
    }


    public static Map<String, List<WebRectangleShape>> analysisXml(String filePath, Map<String, Integer> imgMap) {
        File file = new File(filePath);
        if (!file.exists()) {
            throw new RuntimeException("文件不存在！");
        }
        // 获取文件后缀
        String extension = FilenameUtils.getExtension(file.getName());

        if (!extension.equalsIgnoreCase("xml")) {
            throw new RuntimeException("文件类型必须为xml文件");
        }

        List<FileMarkLog> fileMarkLogList = new ArrayList<>();

        try {
            FileMarkLog fileMarkLog = new FileMarkLog();
            // 创建DocumentBuilderFactory
            DocumentBuilderFactory factory = DocumentBuilderFactory.newInstance();
            // 创建DocumentBuilder
            DocumentBuilder builder = factory.newDocumentBuilder();
            // 解析XML文件
            Document document = builder.parse(new File(filePath));
            // 获取根元素
            Element root = document.getDocumentElement();
            // 获取filename节点
            //String filename = root.getElementsByTagName("filename").item(0).getTextContent();
            // 获取文件名
            String fileName = checkFileNameInMap(file.getPath(), imgMap);
            if (StringUtils.isEmpty(fileName)) {
                return null;
            }
            fileMarkLog.setFileName(fileName);
            // 获取size节点
            NodeList sizeNodeList = root.getElementsByTagName("size");
            if (sizeNodeList.getLength() > 0) {
                Element sizeElement = (Element) sizeNodeList.item(0);
                // 获取width和height值
                String width = sizeElement.getElementsByTagName("width").item(0).getTextContent();
                fileMarkLog.setWidth(Double.parseDouble(width));
                String height = sizeElement.getElementsByTagName("height").item(0).getTextContent();
                fileMarkLog.setHeight(Double.parseDouble(height));
            }
            // 获取object节点列表
            NodeList objectNodes = root.getElementsByTagName("object");

            // 遍历object节点
            for (int i = 0; i < objectNodes.getLength(); i++) {
                Node objectNode = objectNodes.item(i);
                if (objectNode.getNodeType() == Node.ELEMENT_NODE) {
                    Element objectElement = (Element) objectNode;
                    // 获取标签名name
                    String labelName = objectElement.getElementsByTagName("name").item(0).getTextContent();
                    System.out.println(labelName);
                    fileMarkLog.setLabelName(labelName);

                    // 获取bndbox
                    Element bndbox = (Element) objectElement.getElementsByTagName("bndbox").item(0);
                    JsonArray jsonArray = new JsonArray();

                    double x = Double.parseDouble(bndbox.getElementsByTagName("xmin").item(0).getTextContent());
                    double y = Double.parseDouble(bndbox.getElementsByTagName("ymin").item(0).getTextContent());
                    double x1 = Double.parseDouble(bndbox.getElementsByTagName("xmax").item(0).getTextContent());
                    double y1 = Double.parseDouble(bndbox.getElementsByTagName("ymax").item(0).getTextContent());
                    double width = x1 - x;
                    double height = y1 - y;
                    jsonArray.add(x);
                    jsonArray.add(y);
                    jsonArray.add(width);
                    jsonArray.add(height);
                    fileMarkLog.setBbox(jsonArray);
                    fileMarkLog.setId(i + 1);
                    fileMarkLog.setShape("RECT");


                    FileMarkLog newFileMarkLog = new FileMarkLog();
                    BeanUtils.copyProperties(fileMarkLog, newFileMarkLog);
                    fileMarkLogList.add(newFileMarkLog);
                }

            }


            return setWebRectangleShapeMapXml(fileMarkLogList);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }


    private static Map<String, List<WebRectangleShape>> setWebRectangleShapeMapXml(List<FileMarkLog> fileMarkLogList) {
        Map<String, List<WebRectangleShape>> map = new HashMap<>();
        // 根据 id 进行分组
        Map<Integer, List<FileMarkLog>> groupedMap = fileMarkLogList.stream()
                .collect(Collectors.groupingBy(FileMarkLog::getId));
        List<WebRectangleShape> webRectangleShapeList = new ArrayList<>();
        int d = 0;
        for (Map.Entry<Integer, List<FileMarkLog>> entry : groupedMap.entrySet()) {

            List<FileMarkLog> value = entry.getValue();
            for (int i = 0; i < value.size(); i++) {
                WebRectangleShape webRectangleShape = new WebRectangleShape();
                // 文件信息
                webRectangleShape.setOpenId("opId-" + d);
                webRectangleShape.setId("opId-" + d);

                // 判断标注类型
                if (value.get(i).getShape().equalsIgnoreCase("RECT")) { // 矩形
                    webRectangleShape.setType("RECT");
                    Map<String, Double> pointMap = new HashMap<>();
                    pointMap.put("x", value.get(i).getBbox().get(0).getAsDouble());
                    pointMap.put("y", value.get(i).getBbox().get(1).getAsDouble());
                    pointMap.put("width", value.get(i).getBbox().get(2).getAsDouble());
                    pointMap.put("height", value.get(i).getBbox().get(3).getAsDouble());
                    webRectangleShape.setShape(pointMap);
                } else if (value.get(i).getShape().equalsIgnoreCase("CIRCLE")) { // 圆形
                    webRectangleShape.setType("CIRCLE");
                    //  System.out.println("-------" + value.get(i).getMeta());
                    // 解析 JSON 字符串为 JsonObject
                    JsonObject jsonStr = JsonParser.parseString(value.get(i).getMeta()).getAsJsonObject();
                    JsonObject center = jsonStr.getAsJsonObject("center");
                    Map<String, Double> pointMap = new HashMap<>();
                    pointMap.put("cx", center.get("x").getAsDouble());
                    pointMap.put("cy", center.get("y").getAsDouble());
                    pointMap.put("r", jsonStr.get("radius").getAsDouble());
                    webRectangleShape.setShape(pointMap);
                } else if (value.get(i).getShape().equalsIgnoreCase("BRUSH")) { //多边形
                    webRectangleShape.setType("POLYGON");
                    Map<String, List<Map<String, Double>>> listMap = new HashMap<>();
                    List<Map<String, Double>> pointList = new ArrayList<>();
                    JsonArray array = value.get(i).getSegmentation();
                    // 遍历数组，每次取出两个值
                    for (int j = 0; j < array.size(); j += 2) {
                        Map<String, Double> coordinate = new HashMap<>();
                        // 将 x 和 y 存入 Map
                        coordinate.put("x", array.get(j).getAsDouble());
                        coordinate.put("y", array.get(j + 1).getAsDouble());
                        // 添加到 List 中
                        pointList.add(coordinate);
                    }
                    listMap.put("points", pointList);
                    webRectangleShape.setShape(listMap);
                }

                // 写入标签
                WebProps webProps = new WebProps();
                webProps.setName(value.get(i).getLabelName());
                webProps.setTextId("opId-" + d);
                webProps.setDeleteMarkerId("opId-" + d);
                webRectangleShape.setProps(webProps);
                // 样式字段的转换
                WebShapeStyle style = new WebShapeStyle();
                style.setOpacity(1.0);
                style.setFillStyle("#7C0DDD");
                style.setLineWidth(2);
                style.setStrokeStyle("#7C0DDD");
                style.setFill(true);
                style.setGlobalAlpha(0.3);
                webRectangleShape.setStyle(style);

                webRectangleShape.setEye(true);
                webRectangleShape.setOperateIdx(d);

                webRectangleShapeList.add(webRectangleShape);
                d++;
            }


            map.put(value.get(0).getFileName(), webRectangleShapeList);

        }

        return map;

    }


    public static int checkJsonType(JsonObject jsonObject) {
        boolean isInstanceSegmentation = false;
        boolean isObjectDetection = false;

        if (jsonObject.has("annotations")) {
            JsonArray annotations = jsonObject.getAsJsonArray("annotations");
            for (JsonElement annotation : annotations) {
                JsonObject annotationObj = annotation.getAsJsonObject();
                if (annotationObj.has("segmentation") && !annotationObj.has("segments_info")) {
                    isInstanceSegmentation = true;
                } else if (annotationObj.has("segments_info") && !annotationObj.has("segmentation")) {
                    isObjectDetection = true;
                }
            }
        }

        if (isObjectDetection) {
            System.out.println("The JSON data is for Object Detection.");
            return 1;
        } else if (isInstanceSegmentation) {
            System.out.println("The JSON data is for Instance Segmentation.");
            return 2;
        } else {
            System.out.println("The JSON data does not conform to the expected format for either Object Detection or Instance Segmentation.");
            return 0;
        }
    }


    public static String checkFileNameInMap(String filePath, Map<String, Integer> imgMap) {
        String fileName = new java.io.File(filePath).getName();
        System.out.println("==========文件名：" + fileName);

        // 获取文件前缀（不含扩展名）
        String filePrefix = fileName.split("\\.")[0];

        // 遍历Map的键，检查是否有以前缀开头的键
        for (String key : imgMap.keySet()) {
            // 获取Map键的前缀（不含扩展名）
            String keyPrefix = key.split("\\.")[0];
            if (keyPrefix.equals(filePrefix)) {
                return key;
            }
        }
        return null;
    }


}
