package com.qczy.model.entity;

import lombok.Data;

import java.util.ArrayList;
import java.util.List;

@Data
public class LabelmeImageData {
    String imagePath;
    int imageWidth;
    int imageHeight;
    List<LabelmeShape> shapes = new ArrayList<>();

}
