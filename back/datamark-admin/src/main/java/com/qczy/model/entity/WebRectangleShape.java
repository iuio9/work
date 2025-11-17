package com.qczy.model.entity;

import lombok.Data;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

@Data
public class WebRectangleShape {
    String openId;
    String id;
    String type;
    WebProps props;
    Object shape;
    List<Map<String, Double>> shapePoints = new ArrayList<>();
    WebShapeStyle style;
    boolean isEye;  // true
    Integer operateIdx;

}
