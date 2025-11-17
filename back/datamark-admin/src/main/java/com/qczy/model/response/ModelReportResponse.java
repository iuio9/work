package com.qczy.model.response;

import com.fasterxml.jackson.annotation.JsonFormat;
import lombok.Data;
import org.springframework.format.annotation.DateTimeFormat;

import java.util.Date;

@Data
public class ModelReportResponse {

    /**
     * id
     */
    private Integer id;

    /**
     * 任务名称
     */
    private String taskName;

    /**
     * 任务类型
     */
    private String taskType;

    /**
     * 任务进度
     */
    private String taskProgress;

    /**
     * 任务状态
     */
    private Integer taskStatus;

    /**
     * 关联数据集
     */
    private String relatedDataset;

    /**
     * 任务id
     */
    private Integer taskId;

    /**
     * 申请编号
     */
    private String applyForNum;

    /**
     * 模型名称
     */
    private String modelName;

    /**
     * 模型类型
     */
    private String modelType;

    /**
     * 建设单位
     */
    private String buildUnitName;

    /**
     * 实施单位
     */
    private String btUnitName;

    /**
     * 申请类型
     */
    private Integer applyForType;

    /**
     *  申请日期
     */
    @DateTimeFormat(pattern = "yyyy-MM-dd")
    @JsonFormat(pattern = "yyyy-MM-dd", timezone = "GMT+8")
    private Date applyForTime;

    /**
     *  数据集Id
     */
    private String sonId;
}
