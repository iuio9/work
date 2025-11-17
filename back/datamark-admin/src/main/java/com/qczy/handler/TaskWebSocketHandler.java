package com.qczy.handler;

import com.alibaba.fastjson.JSONObject;
import com.qczy.model.entity.ModelAssessTaskEntity;
import com.qczy.service.ManufacturerService;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;
import org.springframework.web.socket.CloseStatus;
import org.springframework.web.socket.TextMessage;
import org.springframework.web.socket.WebSocketSession;
import org.springframework.web.socket.handler.TextWebSocketHandler;

import java.io.IOException;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * WebSocket处理器，支持通过任务ID建立连接
 */
@Component
public class TaskWebSocketHandler extends TextWebSocketHandler {

    private static final Logger logger = LoggerFactory.getLogger(TaskWebSocketHandler.class);

    // 存储任务ID到WebSocket会话的映射
    private static final ConcurrentHashMap<String, WebSocketSession> taskSessions = new ConcurrentHashMap<>();

    @Autowired
    private ManufacturerService manufacturerService;

    // 持有自身实例，用于静态方法
    private static TaskWebSocketHandler instance;

    @Autowired
    public void setInstance(TaskWebSocketHandler instance) {
        TaskWebSocketHandler.instance = instance;
    }

    /**
     * 建立WebSocket连接
     */
    @Override
    public void afterConnectionEstablished(WebSocketSession session) throws Exception {
        // 从URI中提取任务ID
        String taskId = extractTaskId(session.getUri().getPath());
        if (taskId != null) {
            taskSessions.put(taskId, session);
            logger.info("任务 {} 建立WebSocket连接", taskId);

            // 通知前端连接成功
            sendMessage(taskId, "connect", "WebSocket连接已建立");

            ModelAssessTaskEntity task = new ModelAssessTaskEntity();
            task.setId(Integer.parseInt(taskId));

            // 启动任务 (首先先去判断该任务是否执行过，如果执行过，则不在执行，并且打印出该任务已经执行过)
            if (!manufacturerService.isExecuteTask(task)) {
                manufacturerService.startManufacturer(task);
            } else {
                // 直接打印一条日志，告诉用户任务已经执行过
                sendMessage(taskId, "task_log", "当前任务已经执行过，请直接点击完成对接！");
            }
        }
    }

    /**
     * 接收消息
     */
    @Override
    protected void handleTextMessage(WebSocketSession session, TextMessage message) throws Exception {
        String payload = message.getPayload();
        String taskId = findTaskIdBySession(session);

        if (taskId != null) {
            logger.info("收到任务 {} 的消息: {}", taskId, payload);

            // 处理前端发来的消息，如控制任务停止等
            if ("stop".equals(payload)) {
                ModelAssessTaskEntity task = new ModelAssessTaskEntity();
                task.setId(Integer.parseInt(taskId));
                manufacturerService.stopManufacturer(task);
                sendMessage(taskId, "status", "任务已停止");
            }
        }
    }

    /**
     * 连接关闭
     */
    @Override
    public void afterConnectionClosed(WebSocketSession session, CloseStatus status) throws Exception {
        String taskId = findTaskIdBySession(session);
        if (taskId != null) {
            taskSessions.remove(taskId);
            logger.info("任务 {} 的WebSocket连接已关闭", taskId);

            // 任务结束
            ModelAssessTaskEntity task = new ModelAssessTaskEntity();
            task.setId(Integer.parseInt(taskId));
            manufacturerService.endManufacturer(task);
        }
    }

    /**
     * 发送消息给指定任务（静态方法）
     */
    public static void sendMessage(String taskId, String type, Object message) {
        if (instance == null) {
            logger.warn("TaskWebSocketHandler实例未初始化，无法发送消息");
            return;
        }

        WebSocketSession session = taskSessions.get(taskId);
        if (session != null && session.isOpen()) {
            try {
                // 构建消息格式
                JSONObject msg = new JSONObject();
                msg.put("type", type);
                msg.put("data", message);
                session.sendMessage(new TextMessage(msg.toJSONString()));
            } catch (IOException e) {
                logger.error("发送消息到任务 {} 失败", taskId, e);
            }
        } else {
            logger.warn("任务 {} 的WebSocket会话不存在或已关闭", taskId);
        }
    }

    /**
     * 从URI路径中提取任务ID
     */
    private String extractTaskId(String path) {
        if (path == null) return null;

        String[] parts = path.split("/");
        for (int i = 0; i < parts.length; i++) {
            if ("task".equals(parts[i]) && i + 1 < parts.length) {
                return parts[i + 1];
            }
        }
        return null;
    }

    /**
     * 根据会话查找任务ID
     */
    private String findTaskIdBySession(WebSocketSession session) {
        for (Map.Entry<String, WebSocketSession> entry : taskSessions.entrySet()) {
            if (entry.getValue().equals(session)) {
                return entry.getKey();
            }
        }
        return null;
    }
}