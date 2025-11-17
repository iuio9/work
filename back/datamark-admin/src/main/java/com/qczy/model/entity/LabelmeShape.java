package com.qczy.model.entity;

import lombok.Data;

import java.util.List;
@Data
public class LabelmeShape {
    String label;
    List<double[]> points;
    String shape_type;
}
