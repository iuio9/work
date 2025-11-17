package com.qczy.model.request;

import io.swagger.annotations.ApiModelProperty;
import lombok.Data;
import org.springframework.web.multipart.MultipartFile;

/**
 * @Author: hh
 * @Version: 1.0
 * @Date: 2025/6/5 11:09
 * @Description:
 */
@Data
public class AssessTaskRequest {


    // id
    private Integer id;

    // 任务名称
    private String taskName;

    /** 任务类型
     1：测试
     2：评估
     * */
    private Integer taskType;

    // 模型id
    private Integer modelId;

    // 任务描述
    private String taskDesc;

    // 任务版本
    private String taskVersion;

    // 版本描述
    private String versionDesc;

    // 数据集id
    private String sonId;

    // 模型接口地址
    private String modelAddress;

    // 模型传输方式(请求类型)  1：post 、 2：get
    private Integer requestType;

    // 模型参数文件名
    private String modelFileName;

    // 模型参数文件
    private MultipartFile modelParamsFile;

    // 评估描述
    private String assessDesc;

    // 评估描述
    private String assessTarget;

    // 评估描述
    private String assessChart;

    private String labelMap;
    
    private String assessTargetMap;

    private String modelParamsPath;

    private String modelClass;



}
