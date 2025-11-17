package com.qczy.utils;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.JsonMappingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.google.gson.*;

import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;

/**
 * Json字符串工具类
 * @program: PostGirl-panent
 * @description: JsonUtil
 * @author: Cheng Zhi
 * @create: 2021-03-15 15:19
 **/
public class JsonUtil {

    /**
     * 递归替换json中的指定key的value
     * @param jsonElement
     * @param map
     * @return
     */
    public static JsonElement replaceJsonNode(JsonElement jsonElement, Map<String,String> map) {

        if (map.size() == 0) {
            return jsonElement;
        }

        // 判断如果是简单json串直接返回
        if (jsonElement.isJsonPrimitive()) {
            return jsonElement;
        }

        // 判断如果是数组类型字符串，则逐个解析
        if (jsonElement.isJsonArray()) {
            JsonArray jsonArray = jsonElement.getAsJsonArray();
            JsonArray jsonArryNew = new JsonArray();
            Iterator iterator = jsonArray.iterator();
            while (iterator.hasNext()) {
                // 递归调用
                jsonArryNew.add(replaceJsonNode((JsonElement) iterator.next(),map));
            }
            return jsonArryNew;
        }

        // 判断如果是key-value类型的
        if (jsonElement.isJsonObject()) {
            JsonObject object = jsonElement.getAsJsonObject();
            JsonObject objectNew = new JsonObject();
            Iterator iterator = object.keySet().iterator();
            while (iterator.hasNext()) {
                String key = (String) iterator.next();
                // 如果和目标字段匹配则更换value;
                if (map.keySet().contains(key)) {
                    String newValue = map.get(key);
                    object.addProperty(key,newValue);
                }
                JsonElement jsonEle = object.get(key);
                JsonElement jsonElementNew = replaceJsonNode(jsonEle, map);
                objectNew.add(key,jsonElementNew);
            }
            return  objectNew;
        }
        return jsonElement;
    }

    /**
     * 递归替换json中的指定key的value
     * @param jsonElement
     * @param map
     * @return
     */
    public static JsonElement MyReplaceJsonNode(JsonElement jsonElement, Map<String,String> map) {

        if (map.size() == 0) {
            return jsonElement;
        }

        // 判断如果是简单json串直接返回
        if (jsonElement.isJsonPrimitive()) {
            return jsonElement;
        }

        // 判断如果是数组类型字符串，则逐个解析
        if (jsonElement.isJsonArray()) {
            JsonArray jsonArray = jsonElement.getAsJsonArray();
            JsonArray jsonArryNew = new JsonArray();
            Iterator iterator = jsonArray.iterator();
            while (iterator.hasNext()) {
                // 递归调用
                jsonArryNew.add(replaceJsonNode((JsonElement) iterator.next(),map));
            }
            return jsonArryNew;
        }

        // 判断如果是key-value类型的
        if (jsonElement.isJsonObject()) {
            JsonObject object = jsonElement.getAsJsonObject();
            JsonObject objectNew = new JsonObject();
            Iterator iterator = object.keySet().iterator();
            while (iterator.hasNext()) {
                String key = (String) iterator.next();
                // 如果和目标字段匹配则更换value;
                if (map.keySet().contains(key)) {
                    String newValue = map.get(key);
                    object.addProperty(key,newValue);
                }
                JsonElement jsonEle = object.get(key);
                JsonElement jsonElementNew = replaceJsonNode(jsonEle, map);
                objectNew.add(key,jsonElementNew);
            }
            return  objectNew;
        }
        return jsonElement;
    }













     class User {
        private String name;
        private int age;
        private String email;

        // Getters and Setters
        public String getName() {
            return name;
        }

        public void setName(String name) {
            this.name = name;
        }

        public int getAge() {
            return age;
        }

        public void setAge(int age) {
            this.age = age;
        }

        public String getEmail() {
            return email;
        }

        public void setEmail(String email) {
            this.email = email;
        }
    }

    public static void main(String[] args) {

    }
}

