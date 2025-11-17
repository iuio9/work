package com.qczy;

import java.util.List;

/**
 * @Author: hh
 * @Version: 1.0
 * @Date: 2025/1/13 16:20
 * @Description:
 */
public class AnnotationEntity {
    // 唯一 id
    private Integer id;
    // 文件信息
    private String fileName;
    // 文件宽
    private double width;
    // 文件高
    private double height;
    // 标注类型 矩形
    private String shape;
    // bbox：对象的边界框，格式为 [x, y, width, height]。
    private List<Double> bbox;
    // 多边形 - 点位
    private List<List<Double>> segmentation;
    // 圆形 - 点位
    private String meta;
    // 标签
    private String labelName;

    // 构造函数
    public AnnotationEntity(Integer id, String fileName, double width, double height, String shape, List<Double> bbox, List<List<Double>> segmentation, String meta, String labelName) {
        this.id = id;
        this.fileName = fileName;
        this.width = width;
        this.height = height;
        this.shape = shape;
        this.bbox = bbox;
        this.segmentation = segmentation;
        this.meta = meta;
        this.labelName = labelName;
    }

    // Getters 和 Setters
    public Integer getId() {
        return id;
    }

    public void setId(Integer id) {
        this.id = id;
    }

    public String getFileName() {
        return fileName;
    }

    public void setFileName(String fileName) {
        this.fileName = fileName;
    }

    public double getWidth() {
        return width;
    }

    public void setWidth(double width) {
        this.width = width;
    }

    public double getHeight() {
        return height;
    }

    public void setHeight(double height) {
        this.height = height;
    }

    public String getShape() {
        return shape;
    }

    public void setShape(String shape) {
        this.shape = shape;
    }

    public List<Double> getBbox() {
        return bbox;
    }

    public void setBbox(List<Double> bbox) {
        this.bbox = bbox;
    }

    public List<List<Double>> getSegmentation() {
        return segmentation;
    }

    public void setSegmentation(List<List<Double>> segmentation) {
        this.segmentation = segmentation;
    }

    public String getMeta() {
        return meta;
    }

    public void setMeta(String meta) {
        this.meta = meta;
    }

    public String getLabelName() {
        return labelName;
    }

    public void setLabelName(String labelName) {
        this.labelName = labelName;
    }

    @Override
    public String toString() {
        return "AnnotationEntity{" +
                "id=" + id +
                ", fileName='" + fileName + '\'' +
                ", width=" + width +
                ", height=" + height +
                ", shape='" + shape + '\'' +
                ", bbox=" + bbox +
                ", segmentation=" + segmentation +
                ", meta='" + meta + '\'' +
                ", labelName='" + labelName + '\'' +
                '}';
    }
}
