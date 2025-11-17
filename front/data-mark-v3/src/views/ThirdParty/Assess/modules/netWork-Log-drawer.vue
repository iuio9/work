<script setup lang="ts">
import { $t } from '@/locales';
import VueJsonPretty from 'vue-json-pretty'
import 'vue-json-pretty/lib/styles.css'
import LogViewer from "@/components/custom/LogViewer.vue"

defineOptions({
  name: 'NetWorkLogDrawer'
});

interface Emits {
  (e: 'submitted'): void;
}

// 网站访问日志
const websiteAccessLogs = [
  {
    id: 1,
    accessTime: '2025-05-07 10:00:00',
    visitorIP: '192.168.1.10',
    accessedPage: '/index.html',
    operation: 'GET',
    responseTime: 200
  },
  {
    id: 2,
    accessTime: '2025-05-07 10:00:10',
    visitorIP: '192.168.1.11',
    accessedPage: '/product-detail?id=123',
    operation: 'GET',
    responseTime: 180
  },
  {
    id: 3,
    accessTime: '2025-05-07 10:00:20',
    visitorIP: '192.168.1.12',
    accessedPage: '/contact.html',
    operation: 'POST（提交表单）',
    responseTime: 250
  },
  {
    id: 4,
    accessTime: '2025-05-07 10:00:30',
    visitorIP: '192.168.1.13',
    accessedPage: '/about.html',
    operation: 'GET',
    responseTime: 150
  }
];

const networkLogs = ref([
  {
    id: 1,
    logTime: '2025-05-07 12:00:00',
    sourceIP: '192.168.1.10',
    destinationIP: '10.0.0.1',
    action: '访问',
    details: '访问了目标服务器的 80 端口'
  },
  {
    id: 2,
    logTime: '2025-05-07 12:10:00',
    sourceIP: '192.168.1.11',
    destinationIP: '10.0.0.2',
    action: '数据传输',
    details: '向目标服务器传输了 1024 字节的数据'
  },
  {
    id: 3,
    logTime: '2025-05-07 12:20:00',
    sourceIP: '192.168.1.12',
    destinationIP: '10.0.0.3',
    action: '连接失败',
    details: '尝试连接目标服务器的 443 端口失败'
  }
]);

const trainingLogs = [
  {
    timestamp: "2025-04-28T12:34:56Z",
    modelName: "ModelA",
    status: "success",
    errorInfo: null,
    additionalInfo: "Training completed within the expected time frame."
  },
  {
    timestamp: "2025-04-29T08:15:23Z",
    modelName: "ModelB",
    status: "failure",
    errorInfo: "Memory allocation failed.",
    additionalInfo: "The model requires more memory than available."
  },
  {
    timestamp: "2025-04-30T14:47:32Z",
    modelName: "ModelC",
    status: "success",
    errorInfo: null,
    additionalInfo: "Model accuracy met the desired threshold."
  },
  {
    timestamp: "2025-05-01T11:22:45Z",
    modelName: "ModelD",
    status: "failure",
    errorInfo: "Data preprocessing error.",
    additionalInfo: "Invalid data format detected in input dataset."
  },
  {
    timestamp: "2025-05-02T16:56:12Z",
    modelName: "ModelE",
    status: "success",
    errorInfo: null,
    additionalInfo: "Training process was optimized for speed."
  },
  {
    timestamp: "2025-05-03T09:33:58Z",
    modelName: "ModelF",
    status: "failure",
    errorInfo: "Timeout during training.",
    additionalInfo: "The model took too long to train and timed out."
  },
  {
    timestamp: "2025-05-04T13:01:27Z",
    modelName: "ModelG",
    status: "success",
    errorInfo: null,
    additionalInfo: "No errors encountered during training."
  },
  {
    timestamp: "2025-05-05T17:45:03Z",
    modelName: "ModelH",
    status: "failure",
    errorInfo: "Gradient descent failure.",
    additionalInfo: "The gradient descent algorithm did not converge."
  },
  {
    timestamp: "2025-05-06T10:28:49Z",
    modelName: "ModelI",
    status: "success",
    errorInfo: null,
    additionalInfo: "All epochs completed successfully."
  },
  {
    timestamp: "2025-05-07T14:16:31Z",
    modelName: "ModelJ",
    status: "failure",
    errorInfo: "Hardware malfunction.",
    additionalInfo: "GPU overheated and stopped responding."
  }
];

const tabValue = ref<string>('1');

const emit = defineEmits<Emits>();

const visible = defineModel<boolean>('visible', {
  default: false
});

function closeDrawer() {
  visible.value = false;
}

function handleUpdateValue(value: string) {
  tabValue.value = value;
}

// log
const socketUrl = ref('ws://your-api-server/log-stream')
const logs = ref([])
const isConnected = ref(false)

const clearLogs = () => {
  logs.value = []
}

const scrollToBottom = () => {
  // 调用子组件方法（需添加 ref）
}

const onConnected = () => {
  isConnected.value = true
  console.log('WebSocket 连接成功')
}

const onDisconnected = () => {
  isConnected.value = false
  console.log('WebSocket 连接断开')
}

const onMessage = (message) => {
  logs.value.push(message)
}

const onError = (error) => {
  console.error('WebSocket 错误:', error)
}
</script>

<template>
  <NDrawer v-model:show="visible" display-directive="show" :width="800">
    <NDrawerContent title="日志列表" :native-scrollbar="false" closable>
      <div class="w-full h-full flex-col justify-start items-start">
        <n-tabs justify-content="space-evenly" type="line" @update:value="handleUpdateValue">
          <n-tab-pane name="1" tab="网络日志">
          </n-tab-pane>
          <n-tab-pane name="2" tab="JSON日志">
          </n-tab-pane>
        </n-tabs>
        <div class="content min-h-0 flex-1 w-full">
          <div class="w-full h-full bg-#000 " v-show="tabValue === '1'">
            <!-- <n-virtual-list ref="virtualListInst" class="!h-full !text-[#7d8799]" :item-size="64"
              :items="trainingLogs">
              <template #default="{ item, index }">
                <div :key="index" class="item box-border px-8px" style="height: 64px">
                  <div>{{ `时间: ${item.timestamp}, 模型名称: ${item.modelName}, 状态: ${item.status}, ` }}</div>
                  <div>{{ `错误信息: ${item.errorInfo}, 附加信息: ${item.additionalInfo}` }}</div>
                </div>
              </template>
</n-virtual-list> -->
            <LogViewer :socket-url="socketUrl" :container-height="500" @connected="onConnected"
              @disconnected="onDisconnected" @message="onMessage" @error="onError" />
          </div>
          <!-- JSON日志 -->
          <div class="json_pretty_container" v-show="tabValue === '2'">
            <VueJsonPretty path="res" :data="trainingLogs" :show-length="true" />
          </div>
        </div>
      </div>
      <template #footer>
        <NSpace :size="16">
          <NButton type="primary" @click="emit('submitted')">日志下载</NButton>
          <NButton @click="closeDrawer">{{ $t('common.cancel') }}</NButton>
        </NSpace>
      </template>
    </NDrawerContent>
  </NDrawer>
</template>

<style scoped lang="scss">
:deep(.n-scrollbar-content) {
  height: 100% !important;
}
</style>
