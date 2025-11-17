package com.qczy.model.entity;

import com.baomidou.mybatisplus.annotation.*;
import com.fasterxml.jackson.annotation.JsonFormat;
import io.swagger.annotations.ApiModelProperty;
import lombok.Data;
import org.springframework.format.annotation.DateTimeFormat;

import java.io.Serializable;
import java.util.Date;

/**
 * @Author: hh
 * @Version: 1.0
 * @Date: 2024/8/21 10:20
 * @Description:
 */
@Data
@TableName("qczy_algorithm_task_result")
public class AlgorithmTaskResultEntity implements Serializable {

    /** id */
    @ApiModelProperty(value = "id")
    private Long taskId;

    /** 算法任务执行结果 */
    @ApiModelProperty(value = "算法任务执行结果")
    private String taskResult;

    /** 算法任务参数 */
    @ApiModelProperty(value = "算法任务参数")
    private String taskParams;

    /** 创建时间 */
    @ApiModelProperty(value = "创建时间")
    @DateTimeFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss", timezone = "GMT+8")
    @TableField(fill = FieldFill.INSERT)
    private Date createTime;

    /** 更新时间 */
    @ApiModelProperty(value = "更新时间")
    @DateTimeFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss", timezone = "GMT+8")
    @TableField(fill = FieldFill.INSERT_UPDATE)
    private Date updateTime;

    // 前端额外传入的字段（这个字段没用）
    @TableField(exist = false)
    private Integer index;

}
