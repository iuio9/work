package com.qczy.model.entity;


import com.baomidou.mybatisplus.annotation.TableName;
import com.qczy.common.excel.ExcelImport;
import io.swagger.annotations.ApiModelProperty;
import lombok.Data;

@Data
@TableName("qczy_model_code")
public class ModelCodeEntity {


    /**
     * 自增id
     */
    @ApiModelProperty(value = "自增id")
    private Integer id;

    /**
     * 模型id
     */
    @ApiModelProperty(value = "模型id")
    private Integer modelBaseId;

    /**
     * code码
     */
    @ApiModelProperty(value = "code码")
    @ExcelImport("编码")
    private String code;

    /**
     * 算法
     */
    @ApiModelProperty(value = "算法")
    @ExcelImport("算法")
    private String algorithm;

    /**
     * 中文信息
     */
    @ApiModelProperty(value = "中文信息")
    @ExcelImport("中文信息")
    private String chineseInfo;



}
