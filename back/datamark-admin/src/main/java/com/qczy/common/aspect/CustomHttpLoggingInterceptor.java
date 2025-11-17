package com.qczy.common.aspect;

import com.qczy.common.aspect.ApiLogContext;
import okhttp3.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

public class CustomHttpLoggingInterceptor implements Interceptor {
    private final Logger logger = LoggerFactory.getLogger(this.getClass());

    @Override
    public Response intercept(Chain chain) throws IOException {
        Request request = chain.request();

        // 记录请求信息到上下文
        Map<String, Object> requestInfo = new HashMap<>();
        requestInfo.put("method", request.method());
        requestInfo.put("url", request.url());
        requestInfo.put("headers", request.headers());
        ApiLogContext.put("networkRequest", requestInfo);

        long startTime = System.currentTimeMillis();
        Response response = chain.proceed(request);
        long duration = System.currentTimeMillis() - startTime;

        // 记录响应信息到上下文
        Map<String, Object> responseInfo = new HashMap<>();
        responseInfo.put("statusCode", response.code());
        responseInfo.put("headers", response.headers());
        responseInfo.put("durationMs", duration);
        ApiLogContext.put("networkResponse", responseInfo);

        // 记录响应体（仅用于调试，大响应体可能导致内存问题）
        if (logger.isDebugEnabled()) {
            ResponseBody responseBody = response.body();
            if (responseBody != null) {
                MediaType contentType = responseBody.contentType();
                String body = responseBody.string();

                // 重新创建响应，因为body只能读取一次
                response = response.newBuilder()
                        .body(ResponseBody.create(body, contentType))
                        .build();

                ApiLogContext.put("responseBody", body);
            }
        }

        return response;
    }
}