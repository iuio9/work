import { WebSocketClient } from '@/utils/ws';
import { defineStore } from 'pinia';
import { ref } from 'vue';

export const useWebSocketStore = defineStore('webSocket', () => {
    const items = ref<any>(["训练开始..."]);
    const logSocket = ref<any>();
    const sItems = ref<any>(["训练开始..."]);
    const sSocket = ref<any>();
    const connect = () => {
        // ----------------------error 控制台--------------------------------
        const socketUrl = `${import.meta.env.VITE_WS_BASE_URL}/websocket/exceptionTerminalProgress/1`;
        logSocket.value = new WebSocket(socketUrl);
        logSocket.value.onopen = () => {
            console.log("WebSocket连接已打开");
        };
        logSocket.value.onmessage = (event) => {
            console.log("🚀 ~ onMounted ~ event error异常:", event);
            items.value = [...items.value, event.data.trim()];
        };
        logSocket.value.onerror = (error) => {
            console.error("WebSocket错误:", error);
        };
        logSocket.value.onclose = () => {
            console.log("WebSocket连接已关闭");
        };

        // ----------------------success 控制台---------------------------------
        const socketUrl0 = `${import.meta.env.VITE_WS_BASE_URL}/websocket/terminalProgress/1`;
        sSocket.value = new WebSocket(socketUrl0);
        sSocket.value.onopen = () => {
            console.log("WebSocket连接已打开");
        };
        sSocket.value.onmessage = (event) => {
            console.log("🚀 ~ onMounted ~ event success成功:", event);
            sItems.value = [...sItems.value, event.data.trim()];
        };
        sSocket.value.onerror = (error) => {
            console.error("WebSocket错误:", error);
        };
        sSocket.value.onclose = () => {
            console.log("WebSocket连接已关闭");
        };
    };
    // 重置
    const reset = () => {
        // logSocket.value.close();
        items.value = ["训练开始..."];
        // sSocket.value.close();
        sItems.value = ["训练开始..."];
    };
    return {
        logSocket,
        items,
        sSocket,
        sItems,
        connect,
        reset
    };
});