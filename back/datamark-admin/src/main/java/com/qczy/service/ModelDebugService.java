package com.qczy.service;

import com.baomidou.mybatisplus.extension.service.IService;
import com.qczy.model.request.DebugModelRequest;
import com.qczy.model.entity.ModelDebugLog;

import java.util.Map;

/**
 * @Author: hh
 * @Version: 1.0
 * @Date: 2025/5/30 11:17
 * @Description:
 */
public interface ModelDebugService extends IService<ModelDebugLog> {

     Map<String, Object> debugModel(DebugModelRequest debugModelRequest);

     int savaDebugLog(ModelDebugLog modelDebugLog);

     // 根据模型id查询最近一条调用日志
     ModelDebugLog getModelDebugLog(Integer id);

     // 根据最近一条调用日志，发送请求，开始进行测试
     String oneClickDebugging(ModelDebugLog modelDebugLog);
}
