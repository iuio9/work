package com.qczy.model.entity;

import lombok.Data;
import org.springframework.web.multipart.MultipartFile;

/**
 * @Author: hh
 * @Version: 1.0
 * @Date: 2025/5/30 11:11
 * @Description:
 */
@Data
public class DebugModelRequest {

    // 模型地址
    private String modelAddress;

    // 接口请求方式
    private Integer requestType;

    // 提交方式
    private Integer applyForType;

    // 参数文件
    private MultipartFile modelFile;

    // 测试文件路径
    private MultipartFile debugFile;

    private String paramName;

    // 是否保存测试结果  1: 保存 、 0：不保存
    private String isSavaResult;

}
