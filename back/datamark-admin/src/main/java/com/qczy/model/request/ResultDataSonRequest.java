package com.qczy.model.request;

import lombok.Data;

import javax.validation.constraints.NotBlank;

@Data
public class ResultDataSonRequest {
    private Integer taskId;
    private String groupName;
    // 数据集类型
    private Integer dataTypeId;
}
