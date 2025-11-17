package com.qczy.config;

/**
 * @author ：gwj
 * @date ：Created in 2024-08-24 14:52
 * @description：全局配置 Jackson 忽略 JSONNull
 * 如果 JSONNull 的使用非常广泛，并且您不希望手动处理每一个字段，可以创建一个自定义的 Module，并在其中注册序列化器：
 * @modified By：
 * @version: $
 */
import com.fasterxml.jackson.databind.module.SimpleModule;
import cn.hutool.json.JSONNull;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

@Configuration
public class JacksonConfig {

    @Bean
    public ObjectMapper objectMapper() {
        ObjectMapper objectMapper = new ObjectMapper();

        SimpleModule module = new SimpleModule();
        module.addSerializer(JSONNull.class, new JSONNullSerializer());
        objectMapper.registerModule(module);
        return objectMapper;
    }
}

