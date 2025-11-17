package com.qczy.model.response;

import lombok.Data;

@Data
public class ModelDebugLogResponse {


    // 模型地址
    private String modelAddress;

    // 接口请求方式 - >
    //1：post
    //2:  get
    //3:  put
    private Integer requestType;

}
