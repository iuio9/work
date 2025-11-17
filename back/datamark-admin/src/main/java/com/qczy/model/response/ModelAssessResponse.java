package com.qczy.model.response;

import com.fasterxml.jackson.annotation.JsonFormat;
import io.swagger.annotations.ApiModelProperty;
import lombok.Data;
import org.springframework.format.annotation.DateTimeFormat;

import java.util.Date;

/**
 * @Author: hh
 * @Version: 1.0
 * @Date: 2025/6/6 15:18
 * @Description:
 */
@Data
public class ModelAssessResponse {

    // id
    private Integer id;

    // 任务名称
    private String taskName;

    /** 任务类型
     1：测试
     2：评估
     * */
    private Integer taskType;

    // 模型名称
    private String modelName;

    // 任务进度
    private String taskProgress;


    /** 任务状态
     1：待执行
     2：执行中
     3：已完成
     4：任务失败
     5：终止
     6：继续
     * */
    private Integer taskStatus;

    // 关联数据集
    private String sonName;

    // 错误信息
    private String errorMessage;

    // 创建时间
    @DateTimeFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss", timezone = "GMT+8")
    private Date createTime;

}
