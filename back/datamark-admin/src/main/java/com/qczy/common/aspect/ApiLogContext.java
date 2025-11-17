package com.qczy.common.aspect;

import java.util.HashMap;
import java.util.Map;

public class ApiLogContext {
    private static final ThreadLocal<Map<String, Object>> context = new ThreadLocal<>();

    public static void init() {
        context.set(new HashMap<>());
    }

    public static void put(String key, Object value) {
        if (context.get() != null) {
            context.get().put(key, value);
        }
    }

    public static Map<String, Object> get() {
        return context.get();
    }

    public static void clear() {
        context.remove();
    }

    // 获取网络日志
    public static Map<String, Object> getNetworkLog() {
        Map<String, Object> log = new HashMap<>();
        Map<String, Object> ctx = context.get();
        if (ctx != null) {
            log.put("request", ctx.get("networkRequest"));
            log.put("response", ctx.get("networkResponse"));
            if (ctx.containsKey("responseBody")) {
                log.put("body", ctx.get("responseBody"));
            }
        }
        return log;
    }
}