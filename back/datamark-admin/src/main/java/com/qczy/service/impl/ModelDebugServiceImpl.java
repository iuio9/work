package com.qczy.service.impl;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.qczy.mapper.ModelDebugLogMapper;
import com.qczy.model.request.DebugModelRequest;
import com.qczy.model.entity.ModelDebugLog;
import com.qczy.service.ModelDebugService;
import com.qczy.utils.MultipartFileUtils;
import com.qczy.utils.ParamsUtils;
import com.qczy.utils.StringUtils;
import okhttp3.*;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import org.springframework.util.CollectionUtils;
import org.springframework.web.client.RestTemplate;
import org.springframework.web.multipart.MultipartFile;
import org.apache.poi.ss.usermodel.*;
import org.apache.poi.xssf.usermodel.XSSFWorkbook;

import java.io.IOException;
import java.io.InputStream;
import java.util.Date;
import java.util.HashMap;
import java.util.List;
import java.util.Map;


/**
 * @Author: hh
 * @Version: 1.0
 * @Date: 2025/5/30 11:17
 * @Description:
 */
@Service
public class ModelDebugServiceImpl extends ServiceImpl<ModelDebugLogMapper, ModelDebugLog> implements ModelDebugService {

    @Autowired
    private RestTemplate restTemplate;

    @Autowired
    private ModelDebugLogMapper modelDebugLogMapper;

    private final ObjectMapper objectMapper = new ObjectMapper();
    private final OkHttpClient client = new OkHttpClient();


    @Override
    public Map<String, Object> debugModel(DebugModelRequest debugModelRequest) {
        Map<String, Object> result = new HashMap<>();
        // 日志上传对象
        ModelDebugLog modelDebugLog = new ModelDebugLog();
        result.put("modelDebugLogEntity", modelDebugLog);

        try {
            modelDebugLog.setModelAddress(debugModelRequest.getModelAddress());
            modelDebugLog.setApplyForType(debugModelRequest.getApplyForType());
            modelDebugLog.setRequestType(debugModelRequest.getRequestType());
            modelDebugLog.setDebugTime(new Date());

            // 1. 解析参数
            // 判断参数是 json 对象  、  还是 from- data
            Map<String, Object> requestData = new HashMap<>();
            if (CollectionUtils.isEmpty(debugModelRequest.getParams())) {
                if (debugModelRequest.getApplyForType() != null && debugModelRequest.getApplyForType() == 1 && debugModelRequest.getModelFile() != null) {
                    // json文件
                    requestData = ParamsUtils.readJsonFile(debugModelRequest.getModelFile());
                } else {
                    // xlsx文件
                    requestData = ParamsUtils.convertXlsxToMap(debugModelRequest.getModelFile());
                }
            } else {
                requestData = debugModelRequest.getParams();
            }


            // 不是一键调试的时候， 进行解析和替换
            if (debugModelRequest.getIsOneClickDebugging() == null || debugModelRequest.getIsOneClickDebugging() == 0) {

                String base64 = null;
                // 记录解析的调式文件
                if (debugModelRequest.getDebugFile() != null) {
                    // 将 MultipartFile 转换为 base64
                    base64 = MultipartFileUtils.toBase64(debugModelRequest.getDebugFile());
                    modelDebugLog.setTestFileBase64(base64);
                }

                // 替换参数名
                if (!StringUtils.isEmpty(debugModelRequest.getParamName())) {
                    // 包含，进行替换value
                    if (base64 == null) {
                        result.put("success", false);
                        result.put("error", "调试文件不存在！");
                        return result;
                    }
                    requestData.put(debugModelRequest.getParamName(), base64);
                } else {
                    // 不包含
                    requestData.put("image_base64", base64);
                }

            }

            // 记录解析的参数
            if (StringUtils.isEmpty(modelDebugLog.getDebugParams())) {
                ObjectMapper objectMapper = new ObjectMapper();
                modelDebugLog.setDebugParams(objectMapper.writeValueAsString(requestData));
            }

            String requestType = "";
            switch (debugModelRequest.getRequestType()) {
                case 1:
                    requestType = "POST";
                    break;
                case 2:
                    requestType = "GET";
                    break;
                case 3:
                    requestType = "PUT";
                    break;
            }


            // 2. 根据请求方法构建HTTP请求
            Request request = buildHttpRequest(
                    debugModelRequest.getModelAddress(),
                    requestType,
                    requestData
            );


            // 3. 发送HTTP请求并获取响应
            try (Response response = client.newCall(request).execute()) {
                result.put("statusCode", response.code());

                if (!response.isSuccessful()) {
                    result.put("success", false);
                    result.put("error", "HTTP错误: " + response.code());
                    ResponseBody body = response.body();
                    if (body != null) {
                        modelDebugLog.setDebugStatus(0);
                        modelDebugLog.setDebugResult(body.string());
                    }
                    return result;
                }


                // 4. 解析响应体为JSON
                String responseBody = response.body().string();
                try {
                    // 尝试将响应解析为JSON对象
                    Map<String, Object> responseJson = objectMapper.readValue(responseBody, Map.class);
                    result.put("response", responseJson);
                    modelDebugLog.setDebugStatus(1);
                    modelDebugLog.setDebugResult(responseBody);
                } catch (Exception e) {
                    // 如果解析失败，作为普通文本返回
                    result.put("response", responseBody);
                    modelDebugLog.setDebugStatus(0);
                    modelDebugLog.setDebugResult(responseBody);
                }

                result.put("success", true);
                return result;
            }


        } catch (Exception e) {
            result.put("success", false);
            result.put("error", "调试过程中发生错误: " + e.getMessage());
            e.printStackTrace();
            log.error("调试发生错误，日志为，" + e.getMessage());
            modelDebugLog.setDebugStatus(0);
            modelDebugLog.setDebugResult("调试过程中发生错误: " + e.getMessage());
            log.error("调试发生错误，日志为，" + e.getMessage());
            return result;
        }
    }


    // 构建HTTP请求
    private Request buildHttpRequest(String url, String method, Map<String, Object> data) throws IOException {
        Request.Builder requestBuilder = new Request.Builder().url(url);

        switch (method) {
            case "GET":
                // 构建GET请求的查询参数
                HttpUrl.Builder httpUrlBuilder = HttpUrl.parse(url).newBuilder();
                if (data != null) {
                    for (Map.Entry<String, Object> entry : data.entrySet()) {
                        httpUrlBuilder.addQueryParameter(entry.getKey(), entry.getValue().toString());
                    }
                }
                return requestBuilder.url(httpUrlBuilder.build()).get().build();

            case "POST":
                return requestBuilder.post(buildRequestBody(data)).build();

            case "PUT":
                return requestBuilder.put(buildRequestBody(data)).build();

            default:
                throw new IllegalArgumentException("不支持的HTTP方法: " + method);
        }
    }

    // 构建请求体
    private RequestBody buildRequestBody(Map<String, Object> data) throws IOException {
        if (data == null || data.isEmpty()) {
            return RequestBody.create("", MediaType.get("application/json"));
        }

        // 将Map转换为JSON字符串
        String json = objectMapper.writeValueAsString(data);
        return RequestBody.create(json, MediaType.get("application/json"));
    }


    @Override
    public int savaDebugLog(ModelDebugLog modelDebugLog) {
        modelDebugLogMapper.delete(
                new LambdaQueryWrapper<ModelDebugLog>().eq(
                        ModelDebugLog::getModelBaseId, modelDebugLog.getModelBaseId()
                )
        );
        return modelDebugLogMapper.insert(modelDebugLog);
    }

    @Override
    public ModelDebugLog getModelDebugLog(Integer id) {
        List<ModelDebugLog> modelDebugLogList = modelDebugLogMapper.selectList(
                new LambdaQueryWrapper<ModelDebugLog>()
                        .eq(ModelDebugLog::getModelBaseId, id)
                        .orderByDesc(ModelDebugLog::getDebugTime)
        );
        return CollectionUtils.isEmpty(modelDebugLogList) ? null : modelDebugLogList.get(0);
    }

    @Override
    public String oneClickDebugging(ModelDebugLog modelDebugLog) {
        // 组织参数，进行调用
        DebugModelRequest debugModelRequest = new DebugModelRequest();
        debugModelRequest.setModelAddress(modelDebugLog.getModelAddress());
        debugModelRequest.setRequestType(modelDebugLog.getRequestType());
        debugModelRequest.setIsOneClickDebugging(1);
        if (!StringUtils.isEmpty(modelDebugLog.getDebugParams())) {
            try {
                Map<String, Object> params = objectMapper.readValue(modelDebugLog.getDebugParams(), Map.class);
                debugModelRequest.setParams(params);
            } catch (JsonProcessingException e) {
                throw new RuntimeException(e);
            }
        }

        Map<String, Object> map = this.debugModel(debugModelRequest);
        Map<String, Object> responseMap = (Map<String, Object>) map.get("response");
        if (CollectionUtils.isEmpty(responseMap) || responseMap.get("code") == null) {
            return "调试失败！";
        } else {
            String code = responseMap.get("code").toString();
            if (code.equals("200")) {
                return "调试成功！";
            } else {
                return "调试失败！";
            }
        }
    }

}


