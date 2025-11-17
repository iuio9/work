package com.qczy.utils;

import org.apache.commons.beanutils.BeanUtils;
import java.util.HashMap;
import java.util.Map;

/**
 * 将实体对象自动转成 Map<String, String>
 */
public class EntityUtils {

    /**
     * 将实体对象转换为 Map<String, String>
     */
    public static Map<String, String> convertToMap(Object obj) {
        Map<String, String> map = new HashMap<>();
        if (obj == null) {
            return map;
        }

        try {
            // 利用 BeanUtils.describe 方法获取对象的所有属性及其值
            Map<String, String> describeResult = BeanUtils.describe(obj);

            // 描述方法会包含类名（"class"），我们通常不需要这个
            describeResult.remove("class");

            // 如果需要过滤掉一些特定的字段，可以在这里处理
            // ...

            return describeResult;
        } catch (Exception e) {
            e.printStackTrace();
            return map;
        }
    }


    // 在 EntityUtils 类中新增此方法
    public static Map<String, String> convertToMapWithSuffix(Object entity, int taskIndex) {
        Map<String, String> map = convertToMap(entity); // 调用原有 convertToMap 方法
        Map<String, String> result = new HashMap<>();
        for (Map.Entry<String, String> entry : map.entrySet()) {
            String keyWithSuffix = entry.getKey() + "_" + taskIndex;
            result.put(keyWithSuffix, entry.getValue());
        }
        return result;
    }
}