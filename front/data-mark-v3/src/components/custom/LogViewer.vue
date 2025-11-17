<template>
  <div class="log-container bg-#000000" ref="containerRef" @scroll="handleScroll">
    <!-- 滚动内容区 -->
    <div class="log-content" ref="contentRef" :style="contentStyle">
      <!-- 实际渲染的日志行 -->
      <div
        v-for="item in visibleItems"
        :key="item.id"
        class="log-line"
        :style="{ color: item.color || 'inherit' }"
      >
        {{ item.data }}
      </div>
    </div>
  </div>
</template>

<script lang="ts" setup>
defineOptions({
  name: 'LogViewer',
})

import { ref, computed, onMounted, onUnmounted, watchEffect } from 'vue'
import type { Ref } from 'vue'

// 日志项类型定义
interface LogItem {
  id: string | number      // 唯一标识
  content: string          // 日志内容
  timestamp?: number       // 时间戳
  color?: string           // 文本颜色（可选）
  height?: number          // 行高（动态计算）
}

// 组件 props
const props = defineProps({
  // WebSocket 连接地址
  socketUrl: {
    type: String,
    required: true
  },
  // 容器高度
  containerHeight: {
    type: Number,
    default: 500
  },
  // 预估行高（用于初始计算）
  estimatedLineHeight: {
    type: Number,
    default: 22
  },
  // 最大日志条数（超出自动清理）
  maxLogs: {
    type: Number,
    default: 10000
  }
})

// 组件 emits
const emit = defineEmits(['connected', 'disconnected', 'error', 'message'])

// DOM 引用
const containerRef = ref<HTMLDivElement | null>(null)
const contentRef = ref<HTMLDivElement | null>(null)

// 状态管理
const socket = ref<WebSocket | null>(null)
const logs = ref<LogItem[]>([])
const scrollTop = ref(0)
const lineHeights = ref<Record<string | number, number>>({})
const isConnected = ref(false)
const lastMessageTime = ref(0)

// 可视区域计算
const visibleRange = computed(() => {
  if (!containerRef.value) return { start: 0, end: 0 }

  const { scrollTop, clientHeight } = containerRef.value
  const startIdx = Math.floor(scrollTop / props.estimatedLineHeight)
  const endIdx = Math.ceil((scrollTop + clientHeight) / props.estimatedLineHeight)

  return {
    start: Math.max(0, startIdx),
    end: Math.min(logs.value.length - 1, endIdx)
  }
})

// 可视区域内的日志项
const visibleItems = computed(() => {
  return logs.value.slice(visibleRange.value.start, visibleRange.value.end + 1)
})

// 内容样式（用于实现虚拟滚动）
const contentStyle = computed(() => {
  // 计算可视区域上方的总高度
  const topOffset = logs.value
    .slice(0, visibleRange.value.start)
    .reduce((sum, item) => sum + (lineHeights.value[item.id] || props.estimatedLineHeight), 0)

  // 计算总内容高度
  const totalHeight = logs.value
    .reduce((sum, item) => sum + (lineHeights.value[item.id] || props.estimatedLineHeight), 0)

  return {
    transform: `translateY(${topOffset}px)`,
    height: `${totalHeight}px`
  }
})

// 连接 WebSocket
const connectSocket = () => {
  if (socket.value) {
    try { socket.value.close() } catch (e) {}
  }

  socket.value = new WebSocket(props.socketUrl)

  socket.value.onopen = () => {
    isConnected.value = true
    emit('connected')
  }

  socket.value.onmessage = (event) => {
    lastMessageTime.value = Date.now()

    // 处理接收到的消息（这里假设是 JSON 格式的日志项）
    try {
      const logData = JSON.parse(event.data) as LogItem
      console.log('logData: ', logData);
      addLog(logData)
      emit('message', logData)
    } catch (e) {
      // 非 JSON 格式，作为普通文本处理
      addLog({
        id: Date.now().toString(),
        content: event.data.toString(),
        timestamp: Date.now()
      })
    }
  }

  socket.value.onclose = () => {
    isConnected.value = false
    emit('disconnected')

    // 自动重连（可配置）
    setTimeout(connectSocket, 3000)
  }

  socket.value.onerror = (error) => {
    console.error('WebSocket Error:', error)
    emit('error', error)
  }
}

// 添加日志项
const addLog = (logItem: LogItem) => {
  // 确保有唯一 ID
  const logWithId = {
    ...logItem,
    id: logItem.id || Date.now().toString()
  }

  // 添加到日志列表
  logs.value.push(logWithId)

  // 超出最大数量时清理旧日志
  if (logs.value.length > props.maxLogs) {
    const removedCount = logs.value.length - props.maxLogs
    logs.value = logs.value.slice(removedCount)

    // 调整滚动位置
    if (containerRef.value) {
      const heightToRemove = logs.value
        .slice(0, removedCount)
        .reduce((sum, item) => sum + (lineHeights.value[item.id] || props.estimatedLineHeight), 0)

      containerRef.value.scrollTop -= heightToRemove
    }
  }

  // 自动滚动到底部（仅当用户未主动滚动时）
  if (containerRef.value &&
      containerRef.value.scrollTop + containerRef.value.clientHeight >=
      containerRef.value.scrollHeight - 20) {
    scrollToBottom()
  }
}

// 滚动到底部
const scrollToBottom = () => {
  nextTick(() => {
    if (containerRef.value) {
      containerRef.value.scrollTop = containerRef.value.scrollHeight
    }
  })
}

// 处理滚动事件
const handleScroll = () => {
  if (containerRef.value) {
    scrollTop.value = containerRef.value.scrollTop
  }
}

// 测量实际行高
const measureLineHeights = () => {
  if (!contentRef.value) return

  const lineElements = contentRef.value.querySelectorAll('.log-line')
  lineElements.forEach((el, index) => {
    const logItem = visibleItems.value[index]
    if (!logItem) return

    const rect = el.getBoundingClientRect()
    if (rect.height && rect.height !== lineHeights.value[logItem.id]) {
      lineHeights.value[logItem.id] = rect.height
    }
  })
}

// 生命周期钩子
onMounted(() => {
  connectSocket()

  // 初始化测量
  const observer = new ResizeObserver(measureLineHeights)
  if (contentRef.value) {
    observer.observe(contentRef.value)
  }

  // 定时测量（处理动态内容）
  const measureInterval = setInterval(measureLineHeights, 1000)

  onUnmounted(() => {
    if (socket.value) {
      socket.value.close()
    }

    observer.disconnect()
    clearInterval(measureInterval)
  })
})

// 监听滚动位置变化，优化性能
watchEffect(() => {
  if (scrollTop.value > 0) {
    // 用户主动滚动时，减少测量频率
    setTimeout(measureLineHeights, 500)
  }
})
</script>

<style scoped>
.log-container {
  height: 100%;
  overflow-y: auto;
  background-color: #0f172a;
  color: #e2e8f0;
  font-family: 'Consolas', 'Monaco', monospace;
  font-size: 14px;
  border-radius: 4px;
  box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
}

.log-content {
  position: relative;
}

.log-line {
  padding: 4px 8px;
  white-space: pre-wrap;
  word-break: break-all;
  border-bottom: 1px solid rgba(255, 255, 255, 0.05);
  transition: background-color 0.2s;
}

.log-line:hover {
  background-color: rgba(255, 255, 255, 0.05);
}
</style>
