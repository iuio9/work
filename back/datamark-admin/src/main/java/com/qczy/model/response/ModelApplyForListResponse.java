package com.qczy.model.response;

import com.fasterxml.jackson.annotation.JsonFormat;
import lombok.Data;
import org.springframework.format.annotation.DateTimeFormat;

import java.util.Date;

/**
 * @Author: hh
 * @Version: 1.0
 * @Date: 2025/5/16 15:49
 * @Description:
 */
@Data
public class ModelApplyForListResponse {

    /**
     *  id
     */
    private Integer id;

    /**
     *  申请单号
     */
    private String applyForNum;

    /**
     *  模型名称
     */
    private String modelName;


    /**
     *  建设单位名称
     */
    private String buildUnitName;

    /**
     *  承建单位名称
     */
    private String btUnitName;

    /**
     *  模型类型
     */
    private String modelType;



    /** 申请类型
     *  1. 文本申请
     *  2. 系统申请
     * */
    private Integer applyForType;

    /** 申请状态
     * 1：草稿 、
     * 2：审批中 、
     * 3：审批通过 、
     * 4：审批打回
     * 5：已完成
     * */
    private Integer applyForStatus;

    /**
     *  申请日期
     */
    @DateTimeFormat(pattern = "yyyy-MM-dd")
    @JsonFormat(pattern = "yyyy-MM-dd", timezone = "GMT+8")
    private Date applyForDate;

    /**
     *  申请人
     */
    private String nickName;

    private String modelWayStr;

}
