package com.qczy.model.request;

import com.qczy.model.entity.ModelBaseEntity;
import com.qczy.model.entity.ModelConfigureEntity;
import io.swagger.annotations.ApiModelProperty;
import lombok.Data;

import javax.validation.constraints.NotBlank;
import java.util.List;
import java.util.Map;

/**
 * @Author: hh
 * @Version: 1.0
 * @Date: 2025/5/16 11:26
 * @Description:
 */
@Data
public class ModelApplyForRequest {

    /** id */
    private Integer id;


    /** 模型名称 */
    @NotBlank(message = "模型名称不能为空")
    private String modelName;

    /** 模型来源 */
    @NotBlank(message = "模型来源不能为空")
    private String modelSource;

    /** 模型类型 */
    @NotBlank(message = "模型类型不能为空")
    private String modelType;


    /** 模型功能 */
    @NotBlank(message = "模型功能不能为空")
    private String modelFunction;

    /** 建设单位名称 */
    @NotBlank(message = "建设单位名称不能为空")
    private String buildUnitName;

    /** 建设单位名称 */
    @NotBlank(message = "建设单位地址不能为空")
    private String buildUnitAddress;

    /** 建设单位负责人 */
    @NotBlank(message = "建设单位负责人不能为空")
    private String buildUnitLeader;

    /** 建设单位联系方式 */
    @NotBlank(message = "建设单位联系方式不能为空")
    private String buildUnitContact;

    /** 承建单位名称 */
    @NotBlank(message = "承建单位名称不能为空")
    private String btUnitName;

    /** 承建单位地址 */
    @NotBlank(message = "承建单位地址不能为空")
    private String btUnitAddress;

    /** 承建单位负责人 */
    @NotBlank(message = "承建单位负责人不能为空")
    private String btUnitLeader;

    /** 建设单位联系方式 */
    @NotBlank(message = "开发单位负责人联系方式不能为空")
    private String btUnitContact;

    /** 申请类型
     *  1. 文本申请
     *  2. 系统申请
     * */
    private Integer applyForType;

    /**
     *  模型方式
     *  1.：测试
     *  2 ：评估
     */
    private Integer modelWay;

    private String modelWayStr;



    //-------------------------------------------------------------------------------------------------------------------------------------


    /** 模型封装方式 */
    @NotBlank(message = "模型封装方式不能为空")
    private String modelEncapWay;

    /** 模型部署位置 */
    @NotBlank(message = "模型部署位置不能为空")
    private String modelDeployAddr;

    /** 模型文件名称 */
    @NotBlank(message = "模型文件名称不能为空")
    private String modelFileName;

    /** 模型文件大小 */
    @NotBlank(message = "模型文件大小不能为空")
    private String modelFileSize;

    /** 模型API接口说明 */
    private String modelInterfaceDesc;
    private String modelInterfaceDescFileName;

    /** 模型对外暴露端口 */
    @NotBlank(message = "模型对外暴露端口不能为空")
    private String modelPort;

    /** 模型cuda版本 */
    @NotBlank(message = "模型cuda版本不能为空")
    private String modelCudaVersion;

    /** 模型驱动版本 */
    @NotBlank(message = "模型驱动版本不能为空")
    private String modelDriveVersion;

    /** 模型调用例 */
    private String modelCase;
    private String modelCaseFileName;

    /** 模型算法编码 */
    private String modelAlgorithmCode;
    private String modelAlgorithmCodeFileName;



    /** 模型检查场景 */
    //@ApiModelProperty(value = "模型检查场景")
    private String modelScene;



    /** 测试指标 */
   // @NotBlank(message = "测试指标不能为空")
    private String testIndic;

    /** 测试指标2 */
    private String testIndicMap;


    /** 文件id */
    private String fileId;


    // 测试需求简述
    private String testDemandDesc;
    // SHA256校验
    private String modelHashValue;
    // MD5
    private String modelMd5Value;

    // 训练样本路径 【数据集】
    private String trainSample;
    // 测试指标
    private String testCase;

    //模型训练代码
    private String modelTrainCode;

    // 模型识别类别
    private String modelClass;

    // 评估图表
    private String assessChart;

    // 数据集组名称
    private String groupNameAndVersion;
}
