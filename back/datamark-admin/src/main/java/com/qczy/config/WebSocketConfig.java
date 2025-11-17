package com.qczy.config;

import com.qczy.handler.CustomHandshakeInterceptor;
import com.qczy.handler.MyWebSocketHandler;
import com.qczy.handler.TaskWebSocketHandler;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.web.socket.config.annotation.EnableWebSocket;
import org.springframework.web.socket.config.annotation.WebSocketConfigurer;
import org.springframework.web.socket.config.annotation.WebSocketHandlerRegistry;
import org.springframework.web.socket.server.standard.ServletServerContainerFactoryBean;

@Configuration
@EnableWebSocket
public class WebSocketConfig implements WebSocketConfigurer {

    @Autowired
    private TaskWebSocketHandler taskWebSocketHandler;

    @Override
    public void registerWebSocketHandlers(WebSocketHandlerRegistry registry) {
        // 注册原有 WebSocket 端点
        registry.addHandler(new MyWebSocketHandler(), "/websocket/{bizName}/{id}")
                .setAllowedOrigins("*")
                .addInterceptors(new CustomHandshakeInterceptor());

        // 注册任务 WebSocket 端点（现在可以正确编译）
        registry.addHandler(taskWebSocketHandler, "/ws/task/{taskId}")
                .setAllowedOrigins("*")
                .addInterceptors(new CustomHandshakeInterceptor());
    }

    @Bean
    public ServletServerContainerFactoryBean createWebSocketContainer() {
        ServletServerContainerFactoryBean container = new ServletServerContainerFactoryBean();
        container.setMaxTextMessageBufferSize(8192);
        container.setMaxBinaryMessageBufferSize(8192);
        return container;
    }
}