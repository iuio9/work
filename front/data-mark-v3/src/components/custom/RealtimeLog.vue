<template>
  <div class="flex flex-col h-full bg-gray-900 rounded-lg shadow-xl overflow-hidden border border-gray-800">
    <!-- 头部工具栏 -->
    <div class="flex items-center justify-between p-3 bg-gray-800 border-b border-gray-700">
      <div class="flex items-center space-x-2">
        <div class="h-3 w-3 rounded-full bg-red-500"></div>
        <div class="h-3 w-3 rounded-full bg-yellow-500"></div>
        <div class="h-3 w-3 rounded-full bg-green-500"></div>
        <h3 class="ml-2 text-sm font-medium text-gray-300">实时日志监控</h3>
      </div>

      <div class="flex items-center space-x-2">
        <!-- 连接状态指示器 -->
        <div class="flex items-center">
          <span class="mr-1 text-xs text-gray-400">状态:</span>
          <span
            class="inline-block w-2 h-2 rounded-full mr-1"
            :class="connectionStatus === 'connected' ? 'bg-green-500' : 'bg-gray-500'"
          ></span>
          <span class="text-xs" :class="connectionStatus === 'connected' ? 'text-green-400' : 'text-gray-400'">
            {{ connectionStatus }}
          </span>
        </div>

        <!-- 过滤输入框 -->
        <div class="relative">
          <input
            type="text"
            v-model="filterText"
            placeholder="过滤日志..."
            class="w-48 px-3 py-1 text-xs rounded-md bg-gray-700 border border-gray-600 text-gray-300 focus:outline-none focus:ring-1 focus:ring-blue-500"
          />
          <i class="fa fa-filter absolute right-3 top-1/2 -translate-y-1/2 text-gray-400 text-xs"></i>
        </div>

        <!-- 清空按钮 -->
        <button
          @click="clearLogs"
          class="px-2 py-1 text-xs rounded-md bg-gray-700 hover:bg-gray-600 text-gray-300 transition-colors"
        >
          <i class="fa fa-trash mr-1"></i>清空
        </button>
      </div>
    </div>

    <!-- 日志内容区域 -->
    <div
      ref="logContainer"
      class="flex-1 overflow-y-auto p-3 bg-gray-900 text-gray-300 space-y-1"
      :style="{
        scrollBehavior: shouldAutoScroll ? 'smooth' : 'auto'
      }"
    >
      <div
        v-for="(log, index) in filteredLogs"
        :key="index"
        :class="[
          'p-1.5 rounded-md text-xs',
          log.type === 'error' ? 'bg-red-900/30 text-red-300' :
          log.type === 'warning' ? 'bg-yellow-900/30 text-yellow-300' :
          log.type === 'info' ? 'bg-blue-900/30 text-blue-300' :
          log.type === 'success' ? 'bg-green-900/30 text-green-300' :
          'bg-gray-800/50 text-gray-300'
        ]"
      >
        <span class="text-gray-400 mr-2">{{ formatTimestamp(log.timestamp) }}</span>
        <span class="font-medium mr-1.5">{{ log.type.toUpperCase() }}</span>
        <span>{{ log.message }}</span>
      </div>
    </div>

    <!-- 底部控制栏 -->
    <div class="p-2 bg-gray-800 border-t border-gray-700 flex items-center justify-between">
      <div class="flex items-center space-x-2">
        <span class="text-xs text-gray-400">日志数量: {{ logs.length }}</span>
        <span class="text-xs text-gray-400">显示: {{ filteredLogs.length }}</span>
      </div>

      <div class="flex items-center space-x-2">
        <button
          @click="toggleAutoScroll"
          class="px-2 py-1 text-xs rounded-md transition-colors"
          :class="shouldAutoScroll
            ? 'bg-blue-600 hover:bg-blue-700 text-white'
            : 'bg-gray-700 hover:bg-gray-600 text-gray-300'"
        >
          <i class="fa fa-chevron-down mr-1"></i>
          {{ shouldAutoScroll ? '自动滚动' : '固定位置' }}
        </button>

        <!-- <button
          @click="connectSocket"
          class="px-2 py-1 text-xs rounded-md transition-colors"
          :class="connectionStatus === 'connected'
            ? 'bg-gray-700 hover:bg-gray-600 text-gray-300'
            : 'bg-green-600 hover:bg-green-700 text-white'"
          :disabled="connectionStatus === 'connecting'"
        >
          <i class="fa fa-plug mr-1"></i>
          {{ connectionStatus === 'connected' ? '断开连接' : '连接' }}
        </button> -->
      </div>
    </div>
  </div>
</template>

<script lang="ts" setup>
import { ref, onMounted, onUnmounted, watch, computed, defineProps } from 'vue';

// 定义日志类型
type LogType = 'info' | 'warning' | 'error' | 'success' | string;

// 定义日志条目接口
interface LogEntry {
  timestamp: number;
  type: LogType;
  message: string;
}

// 定义连接状态类型
type ConnectionStatus = 'connected' | 'disconnected' | 'connecting';

// 组件属性定义
const props = defineProps<{
  socketUrl: string;
  maxLogs: number;
  autoConnect: boolean;
}>();

// 状态变量定义
const logContainer = ref<HTMLDivElement | null>(null);
const logs = ref<LogEntry[]>([]);
const filterText = ref<string>('');
const connectionStatus = ref<ConnectionStatus>('disconnected');
const shouldAutoScroll = ref<boolean>(true);
const webSocket = ref<WebSocket | null>(null);
const reconnectTimeout = ref<NodeJS.Timeout | null>(null);
const reconnectAttempts = ref<number>(0);
const MAX_RECONNECT_ATTEMPTS = ref<number>(10);
const RECONNECT_DELAY = ref<number>(2000); // 初始重连延迟(毫秒)

// 计算属性 - 过滤后的日志
const filteredLogs = computed(() => {
  if (!filterText.value) return logs.value;
  const filter = filterText.value.toLowerCase();
  return logs.value.filter(log =>
    log.message.toLowerCase().includes(filter) ||
    log.type.toLowerCase().includes(filter)
  );
});

// 格式化时间戳
const formatTimestamp = (timestamp: number): string => {
  const date = new Date(timestamp);
  return date.toLocaleTimeString([], {
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
    hour12: false
  });
};

// 连接WebSocket
const connectSocket = (): void => {
  if (connectionStatus.value === 'connected') {
    disconnectSocket();
    return;
  }

  connectionStatus.value = 'connecting';
  reconnectAttempts.value = 0;

  try {
    webSocket.value = new WebSocket(props.socketUrl);

    // 连接建立
    webSocket.value.onopen = () => {
      connectionStatus.value = 'connected';
      reconnectAttempts.value = 0;
      console.log('WebSocket连接已建立');
    };

    // 接收消息
    webSocket.value.onmessage = (event) => {
      try {
        // 尝试解析JSON数据
        const logData = JSON.parse(event.data);
        addLog(logData);
      } catch (error) {
        // 如果解析失败，将原始数据作为消息
        addLog({ type: 'info', message: event.data });
      }
    };

    // 连接关闭
    webSocket.value.onclose = (event) => {
      webSocket.value = null;

      if (connectionStatus.value === 'connected') {
        // 意外断开连接，尝试重连
        connectionStatus.value = 'disconnected';
        console.log('WebSocket连接已关闭，准备重连');
        scheduleReconnect();
      } else {
        // 用户主动断开连接
        connectionStatus.value = 'disconnected';
        console.log('WebSocket连接已关闭');
      }
    };

    // 错误处理
    webSocket.value.onerror = (error) => {
      console.error('WebSocket错误:', error);
      if (webSocket.value) {
        webSocket.value.close();
      }
    };
  } catch (error) {
    connectionStatus.value = 'disconnected';
    console.error('创建WebSocket连接失败:', error);
  }
};

// 断开WebSocket连接
const disconnectSocket = (): void => {
  // 取消重连定时器
  if (reconnectTimeout.value) {
    clearTimeout(reconnectTimeout.value);
    reconnectTimeout.value = null;
  }

  if (webSocket.value) {
    webSocket.value.close();
    webSocket.value = null;
  }
};

// 计划重连
const scheduleReconnect = (): void => {
  if (reconnectAttempts.value >= MAX_RECONNECT_ATTEMPTS.value) {
    console.log('达到最大重连尝试次数');
    return;
  }

  // 使用指数退避算法计算重连延迟
  const delay = RECONNECT_DELAY.value * Math.pow(2, reconnectAttempts.value);
  const jitter = Math.random() * 1000; // 添加随机抖动，避免多个客户端同时重连

  reconnectTimeout.value = setTimeout(() => {
    reconnectAttempts.value++;
    console.log(`尝试重连 (${reconnectAttempts.value}/${MAX_RECONNECT_ATTEMPTS.value})`);
    connectSocket();
  }, delay + jitter);
};

// 添加日志
const addLog = (logData: any): void => {
  // 确保日志对象包含必要的属性
  const log: LogEntry = {
    timestamp: Date.now(),
    type: logData.type || 'info',
    message: logData.message || JSON.stringify(logData)
  };

  // 添加新日志
  logs.value.push(log);

  // 限制日志数量
  if (logs.value.length > props.maxLogs) {
    logs.value.shift();
  }

  // 自动滚动到底部
  if (shouldAutoScroll.value) {
    scrollToBottom();
  }
};

// 清空日志
const clearLogs = (): void => {
  logs.value = [];
};

// 滚动到底部
const scrollToBottom = (): void => {
  if (logContainer.value) {
    logContainer.value.scrollTop = logContainer.value.scrollHeight;
  }
};

// 切换自动滚动
const toggleAutoScroll = (): void => {
  shouldAutoScroll.value = !shouldAutoScroll.value;
  if (shouldAutoScroll.value) {
    scrollToBottom();
  }
};

// 监听容器滚动，判断是否需要自动滚动
const handleScroll = (): void => {
  if (logContainer.value) {
    // 如果用户滚动到接近底部，则启用自动滚动
    const scrollTop = logContainer.value.scrollTop;
    const scrollHeight = logContainer.value.scrollHeight;
    const clientHeight = logContainer.value.clientHeight;

    shouldAutoScroll.value = (scrollHeight - scrollTop - clientHeight) < 50;
  }
};

// 生命周期钩子
onMounted(() => {
  if (props.autoConnect) {
    connectSocket();
  }

  if (logContainer.value) {
    logContainer.value.addEventListener('scroll', handleScroll);
  }
});

onUnmounted(() => {
  disconnectSocket();

  if (logContainer.value) {
    logContainer.value.removeEventListener('scroll', handleScroll);
  }
});

// 监听过滤文本变化
watch(filterText, () => {
  // 如果过滤后有内容且启用了自动滚动，则滚动到底部
  if (filteredLogs.value.length > 0 && shouldAutoScroll.value) {
    scrollToBottom();
  }
});
</script>

<style scoped>
/* 自定义滚动条样式 */
::-webkit-scrollbar {
  width: 6px;
  height: 6px;
}

::-webkit-scrollbar-track {
  background: theme('colors.gray.800');
  border-radius: 3px;
}

::-webkit-scrollbar-thumb {
  background: theme('colors.gray.600');
  border-radius: 3px;
}

::-webkit-scrollbar-thumb:hover {
  background: theme('colors.gray.500');
}
</style>
