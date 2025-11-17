<template>
  <div class="flex flex-col h-full bg-gray-900 rounded-lg overflow-hidden shadow-lg">
    <!-- 头部 -->
    <div class="bg-gray-800 text-white px-4 py-2 flex flex-wrap justify-between items-center border-b border-gray-700">
      <h3 class="font-semibold">实时日志</h3>
      <div class="flex flex-wrap gap-2 mt-2 sm:mt-0">
        <div class="flex items-center">
          <input
            v-model="searchKeyword"
            type="text"
            placeholder="搜索日志..."
            class="px-3 py-1 bg-gray-700 border border-gray-600 rounded text-sm text-gray-300 focus:outline-none focus:ring-1 focus:ring-blue-500 w-48"
          >
          <button
            v-if="searchKeyword"
            @click="searchKeyword = ''"
            class="ml-1 text-gray-400 hover:text-gray-300"
          >
            <i class="fa fa-times-circle"></i>
          </button>
        </div>
        <select
          v-model="selectedLevel"
          class="px-3 py-1 bg-gray-700 border border-gray-600 rounded text-sm text-gray-300 focus:outline-none focus:ring-1 focus:ring-blue-500"
        >
          <option value="">全部级别</option>
          <option value="info">信息</option>
          <option value="warning">警告</option>
          <option value="error">错误</option>
          <option value="success">成功</option>
        </select>
        <button
          @click="clearLogs"
          class="px-3 py-1 bg-gray-700 hover:bg-gray-600 text-white rounded text-sm transition-colors duration-200"
        >
          清空
        </button>
        <label class="inline-flex items-center">
          <input
            type="checkbox"
            v-model="autoScroll"
            class="form-checkbox h-4 w-4 text-blue-500 rounded border-gray-300"
          >
          <span class="ml-1 text-sm text-gray-300">自动滚动</span>
        </label>
        <!-- 新增：日志推送速率控制 -->
        <div class="flex items-center text-xs">
          <span class="mr-1">推送速率:</span>
          <select
            v-model="logPushInterval"
            class="px-2 py-1 bg-gray-700 border border-gray-600 rounded text-xs text-gray-300 focus:outline-none focus:ring-1 focus:ring-blue-500"
          >
            <option value="10">极快</option>
            <option value="50">快速</option>
            <option value="100">中速</option>
            <option value="300">慢速</option>
            <option value="500">极慢</option>
          </select>
          <span class="ml-1">ms/条</span>
        </div>
      </div>
    </div>

    <!-- 日志列表 -->
    <div class="flex-1 overflow-hidden bg-gray-900 text-gray-300">
      <n-virtual-list
        ref="logVirtualList"
        :items="displayedLogs"
        :item-size="42"
        item-resizable
        class="w-full h-full"
      >
        <template #default="{ item }">
          <div
            :class="[
              'py-1.5 px-3 rounded my-1.5 transition-all duration-200 log-item',
              {
                'bg-gray-800/50 text-gray-300': item.level === 'info',
                'bg-yellow-900/30 text-yellow-300': item.level === 'warning',
                'bg-red-900/30 text-red-300': item.level === 'error',
                'bg-green-900/30 text-green-300': item.level === 'success'
              }
            ]"
          >
            <div class="flex flex-wrap">
              <span class="text-gray-400 mr-2 text-xs whitespace-nowrap">{{ item.timestamp }}</span>
              <span class="flex-1 break-all">
                {{ item.message }}
              </span>
            </div>
          </div>
        </template>
      </n-virtual-list>
    </div>

    <!-- 底部状态 -->
    <div class="bg-gray-800 text-white px-4 py-1 text-xs flex flex-wrap justify-between items-center border-t border-gray-700">
      <div>
        连接状态: <span :class="{ 'text-green-400': socketStatus === 'connected', 'text-red-400': socketStatus === 'disconnected' }">
          {{ socketStatusText }}
        </span>
      </div>
      <div>显示日志: {{ displayedLogs.length }} / 总日志: {{ logs.length }}</div>
      <!-- 新增：队列状态 -->
      <!-- <div>队列大小: {{ logQueue.length }}</div> -->
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted, onUnmounted, watch, computed, nextTick, toRaw, onActivated, onDeactivated } from 'vue'
import type { VirtualListInst } from 'naive-ui'
import type { LogTypes } from '@/types/log'

// 组件 props
const props = defineProps<{
  socketUrl: string
  maxLogs?: number
  autoConnect?: boolean
}>()

// 组件状态
const logVirtualList = ref<VirtualListInst | null>(null)
const logs = ref<LogTypes.LogItem[]>([])
const autoScroll = ref(true)
const socket = ref<WebSocket | null>(null)
const socketStatus = ref<'disconnected' | 'connecting' | 'connected'>('disconnected')
const isMobile = ref(window.innerWidth < 768)
const searchKeyword = ref('')
const selectedLevel = ref('')
const maxLogCount = props.maxLogs || 1000
const retryCount = ref(0) // 重试次数
const maxRetryCount = 5 // 最大重试次数
const isComponentVisible = ref(true) // 组件是否可见
const isUserScrolling = ref(false) // 用户是否正在滚动
const scrollTimeout = ref<NodeJS.Timeout | null>(null)
const scrollSpeed = ref(1000) // 滚动动画持续时间，单位ms

// 新增日志队列和速率控制相关状态
const logQueue = ref<{ message: string, level: LogTypes.LogItem['level'] }[]>([])
const isProcessingQueue = ref(false)
const logPushInterval = ref(300) // 日志推送间隔，单位ms
const maxQueueSize = ref(1000) // 最大队列大小

// 调试辅助函数
const logDebugInfo = () => {
  console.log('[日志调试信息]')
  console.log('logs.length:', logs.value.length)
  console.log('selectedLevel:', selectedLevel.value)
  console.log('searchKeyword:', searchKeyword.value)
  console.log('filteredLogs.length:', filteredLogs.value.length)
  console.log('displayedLogs.length:', displayedLogs.value.length)
  console.log('logQueue.length:', logQueue.value.length)

  if (logs.value.length > 0) {
    console.log('第一条日志:', toRaw(logs.value[0]))
  }
}

// 计算属性
const socketStatusText = computed(() => {
  return {
    'disconnected': '已断开',
    'connecting': '连接中...',
    'connected': '已连接'
  }[socketStatus.value]
})

// 过滤和搜索后的日志
const filteredLogs = computed(() => {
  const result = selectedLevel.value
    ? logs.value.filter(log => log.level === selectedLevel.value)
    : [...logs.value]

  console.log('[filteredLogs] 过滤后日志数量:', result.length)
  return result
})

const displayedLogs = computed(() => {
  const keyword = searchKeyword.value.toLowerCase().trim()

  if (!keyword) {
    console.log('[displayedLogs] 无搜索关键词，返回 filteredLogs')
    return filteredLogs.value
  }

  const result = filteredLogs.value.filter(log =>
    log.message.toLowerCase().includes(keyword)
  )

  console.log(`[displayedLogs] 搜索关键词: "${keyword}"，匹配结果数量:`, result.length)
  return result
})

// 判断是否在底部
const isAtBottom = () => {
  if (!logVirtualList.value) return true

  const { scrollTop, scrollHeight, clientHeight } = logVirtualList.value.$el as HTMLElement
  return scrollTop + clientHeight >= scrollHeight - 100 // 允许有100px的误差
}

// 生命周期钩子
onMounted(() => {
  window.addEventListener('resize', handleResize)

  // 监听滚动事件
  const virtualListEl = logVirtualList.value?.$el as HTMLElement
  if (virtualListEl) {
    virtualListEl.addEventListener('scroll', handleScroll)
  }

  if (props.autoConnect) {
    connectSocket()
  }

  // 初始日志调试信息
  setTimeout(logDebugInfo, 1000)
})

onUnmounted(() => {
  window.removeEventListener('resize', handleResize)

  const virtualListEl = logVirtualList.value?.$el as HTMLElement
  if (virtualListEl) {
    virtualListEl.removeEventListener('scroll', handleScroll)
  }

  disconnectSocket()
})

onActivated(() => {
  isComponentVisible.value = true
})

onDeactivated(() => {
  isComponentVisible.value = false
})

// 方法定义
const connectSocket = () => {
  if (socket.value) {
    disconnectSocket()
  }

  if (retryCount.value >= maxRetryCount) {
    console.error('达到最大重试次数，停止重试')
    return
  }

  socketStatus.value = 'connecting'
  retryCount.value++

  try {
    socket.value = new WebSocket(props.socketUrl)

    socket.value.onopen = () => {
      socketStatus.value = 'connected'
      retryCount.value = 0 // 重置重试次数
      addLog('WebSocket连接已建立', 'success')
    }

    socket.value.onmessage = (event) => {
      console.log('[WebSocket] 收到消息:', event.data)
      addLog(event.data)
    }

    socket.value.onclose = (event) => {
      socketStatus.value = 'disconnected'
      addLog('WebSocket连接已关闭', 'warning')

      setTimeout(() => {
        if (socketStatus.value === 'disconnected') {
          connectSocket()
        }
      }, 5000)
    }

    socket.value.onerror = (error) => {
      console.error('WebSocket错误:', error)
      addLog('WebSocket连接错误', 'error')
    }
  } catch (error) {
    console.error('创建WebSocket失败:', error)
    socketStatus.value = 'disconnected'
    addLog('创建WebSocket连接失败', 'error')
  }
}

const disconnectSocket = () => {
  if (socket.value) {
    socket.value.close()
    socket.value = null
  }
  socketStatus.value = 'disconnected'
}

const createLogItem = (message: string, level: LogTypes.LogItem['level']) => {
  return {
    id: Date.now().toString() + Math.random().toString(36).substr(2, 9),
    timestamp: new Date().toLocaleTimeString(),
    message,
    level
  }
}

const addLog = (message: string, level: LogTypes.LogItem['level'] = 'info') => {
  if (!isComponentVisible.value) return // 组件不可见时不添加日志

  // 如果队列过长，丢弃较早的日志
  if (logQueue.value.length >= maxQueueSize.value) {
    logQueue.value.shift()
  }

  // 将日志添加到队列而不是直接处理
  logQueue.value.push({ message, level })

  // 如果队列处理没有在进行，则启动处理
  if (!isProcessingQueue.value) {
    processLogQueue()
  }
}

// 处理日志队列
const processLogQueue = async () => {
  isProcessingQueue.value = true

  while (logQueue.value.length > 0) {
    const { message, level } = logQueue.value.shift()!

    let logMessage = message
    let logLevel = level

    // 优化的JSON解析逻辑
    try {
      if (message.trim().startsWith('{') || message.trim().startsWith('[')) {
        const parsed = JSON.parse(message)
        if (parsed.message) {
          logMessage = parsed.message
        }
        if (parsed.level) {
          logLevel = parsed.level
        }
      }
    } catch (e) {
      console.error('解析JSON消息失败，使用原始消息:', e)
    }

    const newLog = createLogItem(logMessage, logLevel)
    console.log('[addLog] 添加新日志:', newLog)

    // 使用 splice 方法优化数组更新
    logs.value.push(newLog)

    // 自动滚动逻辑优化
    if (autoScroll.value && !isUserScrolling.value) {
      nextTick(() => {
        scrollToBottom()
      })
    } else if (!isAtBottom()) {
      // 用户不在底部，添加提示
      addLog('有新日志，自动滚动已暂停', 'warning')
    }

    // 添加日志后打印调试信息
    logDebugInfo()

    // 等待指定间隔后处理下一条日志
    await new Promise(resolve => setTimeout(resolve, logPushInterval.value))
  }

  isProcessingQueue.value = false
}

// 修改后的滚动函数 - 增加延迟和动画效果
const scrollToBottom = () => {
  if (!logVirtualList.value) return

  // 平滑滚动到底部，增加滚动时间控制
  logVirtualList.value.scrollTo({
    position: 'bottom',
    behavior: 'smooth',
    duration: scrollSpeed.value // 自定义滚动速度
  })
}

// 处理用户滚动
const handleScroll = () => {
  if (!logVirtualList.value) return

  // 用户正在滚动
  isUserScrolling.value = true

  // 清除之前的定时器
  if (scrollTimeout.value) {
    clearTimeout(scrollTimeout.value)
  }

  // 设置新的定时器，3秒后认为用户停止滚动
  scrollTimeout.value = setTimeout(() => {
    isUserScrolling.value = false

    // 如果用户停止在底部，恢复自动滚动
    if (isAtBottom()) {
      autoScroll.value = true
    }
  }, 3000) as unknown as NodeJS.Timeout
}

const clearLogs = () => {
  logs.value = []
  console.log('[clearLogs] 日志已清空')
}

const handleResize = () => {
  isMobile.value = window.innerWidth < 768
}

// 监听过滤条件变化
watch([selectedLevel, searchKeyword], () => {
  console.log('[过滤条件变化]', { selectedLevel: selectedLevel.value, searchKeyword: searchKeyword.value })
  logDebugInfo()
})

// 监听自动滚动状态变化
watch(autoScroll, (newValue) => {
  if (newValue && !isAtBottom()) {
    // 如果开启自动滚动但不在底部，滚动到底部
    nextTick(() => {
      scrollToBottom()
    })
  }
})

// 新增日志推送速率控制方法
const setLogPushInterval = (interval: number) => {
  logPushInterval.value = interval
  console.log(`[日志控制] 设置日志推送间隔为 ${interval}ms`)
}

// 新增清空日志队列方法
const clearLogQueue = () => {
  logQueue.value = []
  console.log('[日志控制] 清空日志队列')
}


defineExpose({
  clearLogs
})
</script>

<style scoped>
/* 移动端优化 */
@media (max-width: 768px) {
  .text-sm {
    font-size: 12px;
  }

  .text-xs {
    font-size: 10px;
  }
}

/* 日志项动画 */
.log-item {
  opacity: 0;
  animation: fadeIn 0.3s ease forwards;
}

@keyframes fadeIn {
  from {
    opacity: 0;
  }
  to {
    opacity: 1;
  }
}
</style>
