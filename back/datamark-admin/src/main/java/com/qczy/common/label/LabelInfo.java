package com.qczy.common.label;

import com.fasterxml.jackson.annotation.JsonProperty;

public class LabelInfo {
    @JsonProperty("labelId")
    private Integer labelId;
    @JsonProperty("onlyId")
    private String onlyId;
    @JsonProperty("labelName")
    private String labelName;
    @JsonProperty("englishLabelName")
    private String englishLabelName;
    @JsonProperty("labelColor")
    private String labelColor;
    @JsonProperty("labelCount")
    private Integer labelCount;
    @JsonProperty("labelGroupId")
    private Integer labelGroupId;
    @JsonProperty("twoLabelName")
    private String twoLabelName;
    @JsonProperty("labelSort")
    private Integer labelSort;
    @JsonProperty("mapLabel")
    private String mapLabel;

    // 必须提供无参构造函数（Jackson解析需要）
    public LabelInfo() {}

    // getter和setter方法
    public Integer getLabelId() {
        return labelId;
    }

    public void setLabelId(Integer labelId) {
        this.labelId = labelId;
    }

    public String getOnlyId() {
        return onlyId;
    }

    public void setOnlyId(String onlyId) {
        this.onlyId = onlyId;
    }

    public String getLabelName() {
        return labelName;
    }

    public void setLabelName(String labelName) {
        this.labelName = labelName;
    }

    public String getEnglishLabelName() {
        return englishLabelName;
    }

    public void setEnglishLabelName(String englishLabelName) {
        this.englishLabelName = englishLabelName;
    }

    public String getLabelColor() {
        return labelColor;
    }

    public void setLabelColor(String labelColor) {
        this.labelColor = labelColor;
    }

    public Integer getLabelCount() {
        return labelCount;
    }

    public void setLabelCount(Integer labelCount) {
        this.labelCount = labelCount;
    }

    public Integer getLabelGroupId() {
        return labelGroupId;
    }

    public void setLabelGroupId(Integer labelGroupId) {
        this.labelGroupId = labelGroupId;
    }

    public String getTwoLabelName() {
        return twoLabelName;
    }

    public void setTwoLabelName(String twoLabelName) {
        this.twoLabelName = twoLabelName;
    }

    public Integer getLabelSort() {
        return labelSort;
    }

    public void setLabelSort(Integer labelSort) {
        this.labelSort = labelSort;
    }

    public String getMapLabel() {
        return mapLabel;
    }

    public void setMapLabel(String mapLabel) {
        this.mapLabel = mapLabel;
    }
}