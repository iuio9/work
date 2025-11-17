package com.qczy.model.entity;

import com.baomidou.mybatisplus.annotation.TableField;
import com.baomidou.mybatisplus.annotation.TableName;
import io.swagger.annotations.ApiModelProperty;
import lombok.Data;

import java.io.Serializable;

/**
 * @Author: hh
 * @Version: 1.0
 * @Date: 2025/5/16 10:51
 * @Description:
 */

@Data
@TableName("qczy_model_configure")
public class ModelConfigureEntity implements Serializable {

    /** 自增id */
    @ApiModelProperty(value = "自增id")
    private Integer id;

    /** 模型基础信息id */
    @ApiModelProperty(value = "模型基础信息id")
    private Integer modelBaseId;


    /** 模型封装方式 */
    @ApiModelProperty(value = "模型封装方式")
    private String modelEncapWay;

    /** 模型部署位置 */
    @ApiModelProperty(value = "模型部署位置")
    private String modelDeployAddr;

    /** 模型文件名称 */
    @ApiModelProperty(value = "模型文件名称")
    private String modelFileName;

    /** 模型文件大小 */
    @ApiModelProperty(value = "模型文件大小")
    private String modelFileSize;

    /** 模型API接口说明 */
    @ApiModelProperty(value = "模型API接口说明")
    private String modelInterfaceDesc;

    /** 模型对外暴露端口 */
    @ApiModelProperty(value = "模型对外暴露端口")
    private String modelPort;

    /** 模型cuda版本 */
    @ApiModelProperty(value = "模型cuda版本")
    private String modelCudaVersion;

    /** 模型驱动版本 */
    @ApiModelProperty(value = "模型驱动版本")
    private String modelDriveVersion;

    /** 模型调用例 */
    @ApiModelProperty(value = "模型调用例")
    private String modelCase;



    /** 模型检查场景 */
    @ApiModelProperty(value = "模型检查场景")
    private String modelScene;

    /** 训练样本  （数据集id） */
    @ApiModelProperty(value = "训练样本（数据集id）")
    private String sonId;

    /** 测试指标 */
    @ApiModelProperty(value = "测试指标")
    private String testIndic;

    @ApiModelProperty(value = "算法编码")
    private String modelAlgorithmCode;

    /** 测试指标2 */
    @ApiModelProperty(value = "测试指标2")
    private String testIndicMap;

    // SHA256校验
    private String modelHashValue;
    // MD5
    private String modelMd5Value;
    // 训练样本路径
    private String trainSample;

    // 测试指标
    private String testCase;

    //模型训练代码
    private String modelTrainCode;

    // 模型识别类别
    private String modelClass;

    // 评估图表
    private String assessChart;



}
