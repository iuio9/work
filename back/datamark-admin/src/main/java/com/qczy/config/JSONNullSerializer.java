package com.qczy.config;

/**
 * @author ：gwj
 * @date ：Created in 2024-08-24 14:53
 * @description：
 * @modified By：
 * @version: $
 */
import com.fasterxml.jackson.core.JsonGenerator;
import com.fasterxml.jackson.databind.JsonSerializer;
import com.fasterxml.jackson.databind.SerializerProvider;
import cn.hutool.json.JSONNull;

import java.io.IOException;

public class JSONNullSerializer extends JsonSerializer<JSONNull> {
    @Override
    public void serialize(JSONNull value, JsonGenerator gen, SerializerProvider serializers) throws IOException {
        gen.writeNull();  // 将 JSONNull 序列化为标准的 null
    }
}
