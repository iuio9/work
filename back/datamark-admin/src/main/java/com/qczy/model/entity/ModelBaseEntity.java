package com.qczy.model.entity;

import com.baomidou.mybatisplus.annotation.TableName;
import io.swagger.annotations.ApiModelProperty;
import lombok.Data;
import org.jpedal.parser.shape.S;

import javax.validation.constraints.NotBlank;
import java.io.Serializable;
import java.util.Date;

/**
 * @Author: hh
 * @Version: 1.0
 * @Date: 2025/5/16 10:36
 * @Description:
 */
@Data
@TableName("qczy_model_base")
@SuppressWarnings("all")
public class ModelBaseEntity implements Serializable {

    /** 自增id */
    @ApiModelProperty(value = "自增id")
    private Integer id;

    /** 申请单号 */
    @ApiModelProperty(value = "申请单号")
    @NotBlank(message = "申请单号不能为空")
    private String applyForNum;

    /** 模型名称 */
    @ApiModelProperty(value = "模型名称")
    @NotBlank(message = "模型名称不能为空")
    private String modelName;

    /** 模型来源 */
    @ApiModelProperty(value = "模型来源")
    @NotBlank(message = "模型来源不能为空")
    private String modelSource;

    /** 模型类型 */
    @ApiModelProperty(value = "模型类型")
    @NotBlank(message = "模型类型不能为空")
    private String modelType;


    /** 模型功能 */
    @ApiModelProperty(value = "模型功能")
    @NotBlank(message = "模型功能不能为空")
    private String modelFunction;

    /** 建设单位名称 */
    @ApiModelProperty(value = "建设单位名称")
    @NotBlank(message = "建设单位名称不能为空")
    private String buildUnitName;

    /** 建设单位名称 */
    @ApiModelProperty(value = "建设单位地址")
    @NotBlank(message = "建设单位地址不能为空")
    private String buildUnitAddress;

    /** 建设单位负责人 */
    @ApiModelProperty(value = "建设单位负责人")
    @NotBlank(message = "建设单位负责人不能为空")
    private String buildUnitLeader;

    /** 建设单位联系方式 */
    @ApiModelProperty(value = "建设单位联系方式")
    @NotBlank(message = "建设单位联系方式不能为空")
    private String buildUnitContact;

    /** 承建单位名称 */
    @ApiModelProperty(value = "承建单位名称")
    @NotBlank(message = "承建单位名称不能为空")
    private String btUnitName;

    /** 承建单位地址 */
    @ApiModelProperty(value = "承建单位地址")
    @NotBlank(message = "承建单位地址不能为空")
    private String btUnitAddress;

    /** 承建单位负责人 */
    @ApiModelProperty(value = "承建单位负责人")
    @NotBlank(message = "承建单位负责人不能为空")
    private String btUnitLeader;

    /** 建设单位联系方式 */
    @ApiModelProperty(value = "开发单位负责人联系方式")
    @NotBlank(message = "开发单位负责人联系方式不能为空")
    private String btUnitContact;

    /** 申请类型
     *  1. 文本申请
     *  2. 系统申请
     * */
    @ApiModelProperty(value = "申请类型")
    private Integer applyForType;

    /**
     *  模型方式
     */
    @ApiModelProperty(value = "模型方式")
    private Integer modelWay;

    /** 文本申请pdf路径 */
    @ApiModelProperty(value = "文本申请pdf路径")
    private String applyForPdf;

    /** 申请状态->
        1：草稿 、
        2：审批中 、
        3：审批通过 、
        4：审批打回
        5：已完成
        */
    @ApiModelProperty(value = "申请状态")
    private Integer applyForStatus;

    /**  审批状态
     *   1：待审批
     *   2：已通过
     *   3：已退回
     *   4：处理中
     *   5：已完成
     */
    @ApiModelProperty(value = "审批状态")
    private Integer approveStatus;

    /** 申请日期 */
    @ApiModelProperty(value = "申请日期")
    private Date applyForDate;


    /** 用户id */
    @ApiModelProperty(value = "用户id")
    private Integer userId;

    /** 测试描述 */
    @ApiModelProperty(value = "测试描述")
    private String testDemandDesc;

}
