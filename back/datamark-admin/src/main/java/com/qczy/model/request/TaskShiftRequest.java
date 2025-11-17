package com.qczy.model.request;

import lombok.Data;

/**
 * @Author: hh
 * @Version: 1.0
 * @Date: 2025/2/28 15:42
 * @Description:
 */
@Data
public class TaskShiftRequest {

    /**
     *  任务id
     */
    private String id;

    /**
     *  当前用户
     */
    private Integer currentUserId;


    /**
     *  需要转交的用户
     */
    private Integer shiftId;
}
