package com.qczy.model.request;

import lombok.Data;

import java.util.List;
import java.util.Map;

/**
 * @author ：gwj
 * @date ：Created in 2024-09-04 16:32
 * @description：训练参数
 * @modified By：
 * @version: $
 */
@Data
public class DataEnhancementEntity {
    private String taskInputName;

    private String modelId;
    //请选择标注类型-数据类型
    private String dataEnhanceType;
    //请选择标注类型-标注类型
    private String dataEnhanceMarkType;
    //请选择数据集-数据输入
    private String datasetId;
    //请选择数据集-选择标签
    private List datasetTags;
    //请选择数据集-增强区域
    private String datasetEnhanceArea;
    //请选择算子任务
    private List<Map<String,Object>>  dataEnhanceLst; //Map<String,Object>包含algorithmId，参数key和值
    // 请选择算子处理策略（串行叠加:0  并行遍历:1）
    private String  dataEnhanceTactics;
    private String datasetOutId;;


}
