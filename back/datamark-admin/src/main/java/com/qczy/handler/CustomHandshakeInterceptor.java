package com.qczy.handler;

import org.springframework.http.server.ServerHttpRequest;
import org.springframework.http.server.ServerHttpResponse;
import org.springframework.http.server.ServletServerHttpRequest;
import org.springframework.stereotype.Component;
import org.springframework.web.socket.WebSocketHandler;
import org.springframework.web.socket.server.HandshakeInterceptor;

import java.util.Map;

/**
 * @author ：gwj
 * @date ：Created in 2024-10-09 11:26
 * @description：
 * @modified By：
 * @version: $
 */
@Component
public class CustomHandshakeInterceptor  implements HandshakeInterceptor {
    @Override
    public boolean beforeHandshake(ServerHttpRequest request, ServerHttpResponse response,
                                   WebSocketHandler wsHandler, Map<String, Object> attributes) throws Exception {
        if (request instanceof ServletServerHttpRequest) {
            ServletServerHttpRequest servletRequest = (ServletServerHttpRequest) request;
            String uri = servletRequest.getServletRequest().getRequestURI();
            // 假设 WebSocket URL 格式为 /websocket/{bizName}/{id}
            String[] parts = uri.split("/");
            if (parts.length >= 3) {
                String bizName = parts[parts.length - 2];
                String id = parts[parts.length - 1];
                attributes.put("bizName", bizName);
                attributes.put("id", id);
            }
        }
        return true;
    }

    @Override
    public void afterHandshake(ServerHttpRequest request, ServerHttpResponse response,
                               WebSocketHandler wsHandler, Exception exception) {
        // 握手完成后不需要其他操作
    }
}
