package com.qczy.model.entity;

import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;

import java.util.Date;

/**
 * @Author: hh
 * @Version: 1.0
 * @Date: 2025/6/2 19:15
 * @Description:
 */
@TableName("qczy_model_debug_log")
@Data
public class ModelDebugLog {

    // 自增id
    private Integer id;

    // 模型基础信息id
    private String modelBaseId;

    // 模型地址
    private String modelAddress;

    // 接口请求方式 - >
    //1：post
    //2:  get
    //3:  put
    private Integer requestType;

    // 提交方式 - > 1 代表 Json 、2 代表 Excel
    private Integer applyForType;

    // 测试文件地址
    private String testFileBase64;

    // 调试参数
    private String debugParams;

    // 调试结果
    private String debugResult;

    // 调试是否成功  -> 0: 失败 、 1：成功
    private Integer debugStatus;

    // 调试时间
    private Date debugTime;
}
