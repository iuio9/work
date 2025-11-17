import { ref, watch, onUnmounted, Ref, computed, nextTick } from 'vue';

interface TaskItem {
  taskId: number | string;
  taskProgress?: number;
  [key: string]: any;
}

interface TaskProgressMessage {
  taskId: number | string;
  taskProgress?: string;
  [key: string]: any;
}

type ConnectionStatus = 'disconnected' | 'connecting' | 'connected' | 'closed' | 'error';

/**
 * 管理多个任务的WebSocket连接
 */
export function useMultiTaskWebSocket<T extends TaskItem>(
  options: {
    tasks: Ref<T[]>;
    getDataByPage: () => void | Promise<void>;
    autoReconnect?: boolean;
  }
) {
  const socketMap = ref<Map<string, WebSocket>>(new Map());
  const connectionStatusMap = ref<Map<string, ConnectionStatus>>(new Map());

  // 初始化单个任务的WebSocket连接
  const initSocket = (taskId: number | string) => {
    const taskIdStr = String(taskId);

    // 如果已存在该任务的连接，先关闭
    if (socketMap.value.has(taskIdStr)) {
      closeSocket(taskId);
    }

    connectionStatusMap.value.set(taskIdStr, 'connecting');

    try {
      const socketUrl = `${import.meta.env.VITE_WS_BASE_URL}/websocket/assessProgress/${taskId}`;
      const socket = new WebSocket(socketUrl);

      socketMap.value.set(taskIdStr, socket);

      // 连接成功回调
      socket.onopen = () => {
        connectionStatusMap.value.set(taskIdStr, 'connected');
        console.log(`WebSocket连接已打开`);
      };

      // 接收消息回调
      socket.onmessage = (event: MessageEvent) => {
        try {
          const obj = JSON.parse(event.data) as TaskProgressMessage;
          console.log('obj: ', obj);
          updateTaskProgress(obj);
        } catch (parseError) {
          console.error('解析WebSocket消息失败:', parseError);
        }
      };

      // 错误回调
      socket.onerror = (error: Event) => {
        connectionStatusMap.value.set(taskIdStr, 'error');
        console.error(`WebSocket错误(${taskIdStr}):`, error);
      };

      // 关闭连接回调
      socket.onclose = (event: CloseEvent) => {
        connectionStatusMap.value.set(taskIdStr, 'closed');
        console.log(`WebSocket连接已关闭(${taskIdStr}):`, event);

        // 从map中移除
        socketMap.value.delete(taskIdStr);

        // 如果不是主动关闭且启用了自动重连，则尝试重连
        const shouldReconnect = options.autoReconnect !== false && event.code !== 1000;
        if (shouldReconnect) {
          setTimeout(() => initSocket(taskId), 3000);
        }
      };

    } catch (error) {
      connectionStatusMap.value.set(taskIdStr, 'error');
      console.error(`创建WebSocket连接失败(${taskIdStr}):`, error);
    }
  };

  // 更新任务进度
  const updateTaskProgress = (obj: TaskProgressMessage) => {
    const taskIdStr = String(obj.id);
    const index = options.tasks.findIndex(task => {
      return String(task.id) === taskIdStr
    });

    if (index !== -1) {
      const progress = obj.taskProgress ? parseInt(obj.taskProgress, 10) : 0;

      // 使用Vue 3的响应式更新方式
      options.tasks = [
        ...options.tasks.slice(0, index),
        { ...options.tasks[index], taskProgress: progress },
        ...options.tasks.slice(index + 1)
      ];

      // 任务完成时关闭连接并调用刷新函数
      if (progress === 100) {
        closeSocket(obj.taskId);
        options.getDataByPage();
      }
    }
  };

  // 关闭单个任务的WebSocket连接
  const closeSocket = (taskId: number | string) => {
    const taskIdStr = String(taskId);
    const socket = socketMap.value.get(taskIdStr);

    if (socket) {
      socket.close(1000, '手动关闭连接');
      socketMap.value.delete(taskIdStr);
      connectionStatusMap.value.delete(taskIdStr);
    }
  };

  // 关闭所有WebSocket连接
  const closeAllSockets = () => {
    socketMap.value.forEach((socket, taskId) => {
      socket.close(1000, '手动关闭连接');
    });
    socketMap.value.clear();
    connectionStatusMap.value.clear();
  };

  // 计算属性：所有连接的总体状态
  const overallStatus = computed<ConnectionStatus>(() => {
    const statuses = Array.from(connectionStatusMap.value.values());

    if (statuses.some(s => s === 'connecting')) return 'connecting';
    if (statuses.some(s => s === 'error')) return 'error';
    if (statuses.some(s => s === 'connected')) return 'connected';
    if (statuses.some(s => s === 'closed')) return 'closed';

    return 'disconnected';
  });

  // 监听任务变化，动态调整WebSocket连接
  watch(options.tasks, (newTasks, oldTasks) => {
    newTasks = newTasks.map(item => {
      return {
        ...item,
        taskId: item.id
      }
    })
    console.log('newTasks: ', newTasks);
    // 使用nextTick确保DOM更新完成后再处理
    nextTick(() => {
      // 获取新旧任务ID集合
      const newTaskIds = new Set(newTasks.map(task => String(task.id)));
      const oldTaskIds = new Set(oldTasks?.map(task => String(task.id)) || []);

      // 找出需要新增的任务
      const tasksToAdd = [...newTaskIds].filter(id => !oldTaskIds.has(id));

      // 找出需要移除的任务
      const tasksToRemove = [...oldTaskIds].filter(id => !newTaskIds.has(id));

      // 为新增的任务创建WebSocket连接
      tasksToAdd.forEach(taskId => {
        initSocket(taskId);
      });

      // 移除不需要的任务的WebSocket连接
      tasksToRemove.forEach(taskId => {
        closeSocket(taskId);
      });
    });
  }, { deep: true, immediate: true });

  // 组件卸载时关闭所有连接
  onUnmounted(() => {
    closeAllSockets();
  });

  return {
    connectionStatusMap,
    overallStatus,
    initSocket,
    closeSocket,
    closeAllSockets
  };
}
