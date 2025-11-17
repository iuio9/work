package com.qczy.model.request;

import lombok.Data;

import java.util.List;

/**
 * @Author: hh
 * @Version: 1.0
 * @Date: 2025/4/19 23:51
 * @Description:
 */
@Data
public class ExamineReturnRequest {

    // 任务id
    private Integer taskId;

    // 退回某个人员
    private List<Integer> ids;
}
