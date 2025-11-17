package com.qczy.model.request;

import lombok.Data;
import org.springframework.web.multipart.MultipartFile;

import java.util.Map;

/**
 * @Author: hh
 * @Version: 1.0
 * @Date: 2025/6/9 10:18
 * @Description:
 */
@Data
public class DebugModelRequest {

    private String modelAddress;

    private Integer requestType;

    // 是否保存测试结果  1: 保存 、 0：不保存
    private String isSavaResult;

    private Integer applyForType;

    private MultipartFile modelFile;

    private MultipartFile debugFile;

    // 参数
    private Map<String, Object> params;

    private String paramName;
    // 判断是否是一键调试     // 0：不是  1：是
    private Integer isOneClickDebugging;
}
