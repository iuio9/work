package com.qczy.common.markInfo;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.springframework.stereotype.Service;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.UUID;

@Service
public class JsonConverterService {

    public String convertPreToNext(String preJsonPath) throws IOException {
        ObjectMapper objectMapper = new ObjectMapper();
        File preJsonFile = new File(preJsonPath);
        PreJson preJson = objectMapper.readValue(preJsonFile, PreJson.class);

        List<NextJson> nextJsonList = new ArrayList<>();
        for (Shape shape : preJson.getShapes()) {
            NextJson nextJson = convertShapeToNextJson(shape);
            nextJsonList.add(nextJson);
        }

        return objectMapper.writeValueAsString(nextJsonList);
    }

    private NextJson convertShapeToNextJson(Shape shape) {
        NextJson nextJson = new NextJson();
        nextJson.setOpenId(UUID.randomUUID().toString());
        nextJson.setId(UUID.randomUUID().toString());
        nextJson.setType("POLYGON");

        Map<String, Object> props = new HashMap<>();
        props.put("name", shape.getLabel());
        props.put("textId", nextJson.getId());
        props.put("deleteMarkerId", nextJson.getId());

        double minX = Double.MAX_VALUE;
        double maxX = Double.MIN_VALUE;
        double minY = Double.MAX_VALUE;
        double maxY = Double.MIN_VALUE;
        for (List<Double> point : shape.getPoints()) {
            double x = point.get(0);
            double y = point.get(1);
            minX = Math.min(minX, x);
            maxX = Math.max(maxX, x);
            minY = Math.min(minY, y);
            maxY = Math.max(maxY, y);
        }
        props.put("operateWidth", maxX - minX);
        props.put("operateHeight", maxY - minY);
        nextJson.setProps(props);

        List<Map<String, Double>> points = new ArrayList<>();
        for (List<Double> point : shape.getPoints()) {
            Map<String, Double> pointMap = new HashMap<>();
            pointMap.put("x", point.get(0));
            pointMap.put("y", point.get(1));
            points.add(pointMap);
        }
        Map<String, List<Map<String, Double>>> shapeMap = new HashMap<>();
        shapeMap.put("points", points);
        nextJson.setShape(shapeMap);

        Map<String, Object> style = new HashMap<>();
        style.put("opacity", 1);
        style.put("fillStyle", "#C2DDC2");
        style.put("lineWidth", 2);
        style.put("strokeStyle", "#C2DDC2");
        style.put("fill", true);
        style.put("globalAlpha", 0.3);
        nextJson.setStyle(style);

        nextJson.setIsEye(true);
        nextJson.setOperateIdx(0);

        return nextJson;
    }

    static class PreJson {
        private int imageWidth;
        private String imagePath;
        private int imageHeight;
        private List<Shape> shapes;

        public int getImageWidth() {
            return imageWidth;
        }

        public void setImageWidth(int imageWidth) {
            this.imageWidth = imageWidth;
        }

        public String getImagePath() {
            return imagePath;
        }

        public void setImagePath(String imagePath) {
            this.imagePath = imagePath;
        }

        public int getImageHeight() {
            return imageHeight;
        }

        public void setImageHeight(int imageHeight) {
            this.imageHeight = imageHeight;
        }

        public List<Shape> getShapes() {
            return shapes;
        }

        public void setShapes(List<Shape> shapes) {
            this.shapes = shapes;
        }
    }

    static class Shape {
        private String label;
        private List<List<Double>> points;
        private String shape_type;

        public String getLabel() {
            return label;
        }

        public void setLabel(String label) {
            this.label = label;
        }

        public List<List<Double>> getPoints() {
            return points;
        }

        public void setPoints(List<List<Double>> points) {
            this.points = points;
        }

        public String getShape_type() {
            return shape_type;
        }

        public void setShape_type(String shape_type) {
            this.shape_type = shape_type;
        }
    }

    static class NextJson {
        private String openId;
        private String id;
        private String type;
        private Map<String, Object> props;
        private Map<String, List<Map<String, Double>>> shape;
        private Map<String, Object> style;
        private boolean isEye;
        private int operateIdx;

        public String getOpenId() {
            return openId;
        }

        public void setOpenId(String openId) {
            this.openId = openId;
        }

        public String getId() {
            return id;
        }

        public void setId(String id) {
            this.id = id;
        }

        public String getType() {
            return type;
        }

        public void setType(String type) {
            this.type = type;
        }

        public Map<String, Object> getProps() {
            return props;
        }

        public void setProps(Map<String, Object> props) {
            this.props = props;
        }

        public Map<String, List<Map<String, Double>>> getShape() {
            return shape;
        }

        public void setShape(Map<String, List<Map<String, Double>>> shape) {
            this.shape = shape;
        }

        public Map<String, Object> getStyle() {
            return style;
        }

        public void setStyle(Map<String, Object> style) {
            this.style = style;
        }

        public boolean isEye() {
            return isEye;
        }

        public void setIsEye(boolean eye) {
            isEye = eye;
        }

        public int getOperateIdx() {
            return operateIdx;
        }

        public void setOperateIdx(int operateIdx) {
            this.operateIdx = operateIdx;
        }
    }
}