package com.qczy.model.request;

import lombok.Data;

/**
 * @Author: hh
 * @Version: 1.0
 * @Date: 2024/8/1 13:30
 * @Description:
 */
@Data
public class DeleteRequest {

    private int[] ids;

    private Integer id;

    private Integer taskId;

    private Integer modelId;

    private String serverKey;


}
