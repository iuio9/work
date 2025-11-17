package com.qczy.model.response;

import lombok.Data;

/**
 * @Author: hh
 * @Version: 1.0
 * @Date: 2024/8/22 15:13
 * @Description:
 */
@Data
public class DataSetLabelResponse {

    private Integer labelId;
    private String onlyId;
    private String labelName;
    private String englishLabelName;
    private String labelColor;
    private Integer labelCount;
    private Integer labelGroupId;
    // 二级标签
    private String twoLabelName;

    private Integer labelSort;


}
