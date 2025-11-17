package com.qczy.config;

/**
 * @author ：gwj
 * @date ：Created in 2024-09-04 17:02
 * @description：
 * @modified By：
 * @version: $
 */
import org.apache.http.client.config.RequestConfig;
import org.apache.http.impl.client.CloseableHttpClient;
import org.apache.http.impl.client.HttpClientBuilder;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.http.client.HttpComponentsClientHttpRequestFactory;
import org.springframework.web.client.RestTemplate;

@Configuration
public class RestTemplateConfig {

    @Bean
    public RestTemplate restTemplate() {
        // 设置超时时间
        int timeout =  Integer.MAX_VALUE; // 超时时间为 60 秒
//        int timeout = 2000; // 超时时间为 2 秒

        // 创建请求配置
        RequestConfig config = RequestConfig.custom()
                .setConnectTimeout(timeout)      // 连接超时时间
                .setConnectionRequestTimeout(timeout) // 请求连接池超时时间
                .setSocketTimeout(timeout)       // 读取数据超时时间
                .build();

        // 创建HttpClient并应用请求配置
        CloseableHttpClient client = HttpClientBuilder.create()
                .setDefaultRequestConfig(config)
                .build();

        // 使用HttpClient创建HttpComponentsClientHttpRequestFactory
        HttpComponentsClientHttpRequestFactory factory = new HttpComponentsClientHttpRequestFactory(client);

        // 创建RestTemplate并应用请求工厂
        return new RestTemplate(factory);
    }
}
