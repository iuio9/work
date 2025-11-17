package com.qczy.handler;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Component;
import org.springframework.web.socket.CloseStatus;
import org.springframework.web.socket.TextMessage;
import org.springframework.web.socket.WebSocketSession;
import org.springframework.web.socket.handler.TextWebSocketHandler;

import java.io.IOException;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

@Component
public class MyWebSocketHandler extends TextWebSocketHandler {

    private static final Logger logger = LoggerFactory.getLogger(MyWebSocketHandler.class);

    // 用于存储 bizName:id:sessionId 对应的 WebSocketSession
    private static Map<String, WebSocketSession> userSessions = new ConcurrentHashMap<>();

    @Override
    public void afterConnectionEstablished(WebSocketSession session) throws Exception {
        String bizName = (String) session.getAttributes().get("bizName");
        String id = (String) session.getAttributes().get("id");
        String key = bizName + ":" + id + ":" + session.getId();
        userSessions.put(key, session);

        logger.info("User connected with key: " + key);
    }

    @Override
    public void handleTextMessage(WebSocketSession session, TextMessage message) throws Exception {
        String bizName = (String) session.getAttributes().get("bizName");
        String id = (String) session.getAttributes().get("id");
        String key = bizName + ":" + id + ":" + session.getId();
        String msg = message.getPayload();
        logger.info("Received message from " + key + ": " + msg);

        session.sendMessage(new TextMessage("Hello " + id + " in " + bizName + ", you sent: " + msg));
    }

    @Override
    public void afterConnectionClosed(WebSocketSession session, CloseStatus status) throws Exception {
        String bizName = (String) session.getAttributes().get("bizName");
        String id = (String) session.getAttributes().get("id");
        String key = bizName + ":" + id + ":" + session.getId();
        userSessions.remove(key);
        logger.info("User disconnected with key: " + key);
    }

    public void sendMessageToUser(String bizName, String id, String message) {
        userSessions.entrySet().stream()
                .filter(entry -> entry.getKey().startsWith(bizName + ":" + id))
                .forEach(entry -> {
                    try {
                        WebSocketSession session = entry.getValue();
                        if (session.isOpen()) {
                            session.sendMessage(new TextMessage(message));
                        }
                    } catch (IOException e) {
                        logger.error("Failed to send message to " + entry.getKey() + ": " + e.getMessage());
                    }
                });
    }

    public void disconnectUser(String bizName, String id) {
        userSessions.entrySet().stream()
                .filter(entry -> entry.getKey().startsWith(bizName + ":" + id))
                .forEach(entry -> {
                    try {
                        WebSocketSession session = entry.getValue();
                        session.close();
                        userSessions.remove(entry.getKey());
                        logger.info("Disconnected user: " + entry.getKey());
                    } catch (IOException e) {
                        logger.error("Failed to disconnect " + entry.getKey() + ": " + e.getMessage());
                    }
                });
    }
}