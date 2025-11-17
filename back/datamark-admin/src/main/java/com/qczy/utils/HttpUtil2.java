package com.qczy.utils;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import okhttp3.*;
import org.springframework.context.annotation.Configuration;

import java.io.IOException;
import java.util.Map;

/**
 * HTTP请求工具类，支持JSON格式请求
 */
@Configuration
public class HttpUtil2 {

    private final ObjectMapper objectMapper = new ObjectMapper();
    private final OkHttpClient client = new OkHttpClient();

    /**
     * 构建HTTP请求（支持JSON格式）
     * @param url 请求地址
     * @param method 请求方法（POST/GET/PUT/DELETE）
     * @param params 请求参数（将转换为JSON或查询参数）
     * @return 构建好的Request对象
     */
    public Request buildHttpRequest(String url, String method, Map<String, Object> params) throws JsonProcessingException {
        Request.Builder requestBuilder = new Request.Builder()
                .url(url);

        // 处理不同请求方法
        if ("POST".equalsIgnoreCase(method)) {
            // 构建JSON请求体
            RequestBody body = buildRequestBody(params);
            requestBuilder.post(body);
            requestBuilder.addHeader("Content-Type", "application/json");
        } else if ("GET".equalsIgnoreCase(method)) {
            // 构建GET请求的查询参数
            HttpUrl.Builder urlBuilder = HttpUrl.parse(url).newBuilder();
            for (Map.Entry<String, Object> entry : params.entrySet()) {
                urlBuilder.addQueryParameter(entry.getKey(), entry.getValue().toString());
            }
            requestBuilder.url(urlBuilder.build());
            requestBuilder.get();
        } else if ("PUT".equalsIgnoreCase(method)) {
            // PUT请求（JSON格式）
            RequestBody body = buildRequestBody(params);
            requestBuilder.put(body);
            requestBuilder.addHeader("Content-Type", "application/json");
        } else if ("DELETE".equalsIgnoreCase(method)) {
            // DELETE请求（可选参数）
            if (!params.isEmpty()) {
                HttpUrl.Builder urlBuilder = HttpUrl.parse(url).newBuilder();
                for (Map.Entry<String, Object> entry : params.entrySet()) {
                    urlBuilder.addQueryParameter(entry.getKey(), entry.getValue().toString());
                }
                requestBuilder.url(urlBuilder.build());
            }
            requestBuilder.delete();
        }

        return requestBuilder.build();
    }

    /**
     * 构建JSON格式的请求体
     */
    public RequestBody buildRequestBody(Map<String, Object> data) throws JsonProcessingException {
        if (data == null || data.isEmpty()) {
            return RequestBody.create("", MediaType.get("application/json"));
        }
        // 将Map转换为JSON字符串
        String json = objectMapper.writeValueAsString(data);
        return RequestBody.create(json, MediaType.get("application/json"));
    }

    /**
     * 发送HTTP请求并获取响应
     */
    public String sendRequest(Request request) throws IOException {
        try (Response response = client.newCall(request).execute()) {
            if (!response.isSuccessful()) {
                throw new IOException("Unexpected response code: " + response.code());
            }
            return response.body().string();
        }
    }
}