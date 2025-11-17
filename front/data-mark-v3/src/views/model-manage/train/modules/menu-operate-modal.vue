<script setup lang="ts">
import {$t} from "@/locales";
import {LogInst, VirtualListInst} from "naive-ui";

defineOptions({
  name: "MenuOperateModal",
});

export type OperateType = NaiveUI.TableOperateType | "addChild";

interface Props {
  taskId: string;
  trainType: string;
  items: any;
  sItems: any;
}

const props = defineProps<Props>();

interface Emits {
  (e: "submitted"): void;
}

const emit = defineEmits<Emits>();

const visible = defineModel<boolean>("visible", {
  default: false,
});

const virtualListInst = ref<VirtualListInst>();
let timer = null;
watchEffect(() => {
  const dataList = props.trainType == 0 ? props.sItems : props.items;
  const lenIdx = dataList.length - 1;
  let startLen = lenIdx - 10 < 0 ? 0 : lenIdx - 10;
  timer = setInterval(() => {
    // eslint-disable-next-line no-plusplus
    startLen++;
    if (startLen == lenIdx) {
      clearInterval(timer);
    }
    virtualListInst.value?.scrollTo({index: startLen});
  }, 30);
});

function closeDrawer() {
  clearInterval(timer);
  timer = null;
  visible.value = false;
};

// socket 任务进度
// const items = ref<any>([]);
// const logSocket = ref<WebSocket>();
// const sItems = ref<any>([]);
// const sSocket = ref<WebSocket>();
// let timer = null;
// let sTimer = null;
// const virtualListInst = ref<VirtualListInst>();
// const sVirtualListInst = ref<VirtualListInst>();
// onMounted(() => {
//   const socketUrl = `${import.meta.env.VITE_WS_BASE_URL}/websocket/exceptionTerminalProgress/1`;
//   logSocket.value = new WebSocket(socketUrl);
//   logSocket.value.onopen = () => {
//     console.log("WebSocket连接已打开");
//   };
//   logSocket.value.onmessage = (event) => {
//     console.log("🚀 ~ onMounted ~ event error异常:", event);
//     items.value = [...items.value, event.data.trim()];
//     // 定时器模拟无缝滚动效果;
//     const lenIdx = items.value.length - 1;
//     let startLen = lenIdx - 1 < 0 ? 0 : lenIdx - 1;
//     timer = setInterval(() => {
//       startLen++;
//       if (startLen == lenIdx) {
//         clearInterval(timer);
//       }
//       virtualListInst.value?.scrollTo({ index: startLen });
//     }, 30);
//   };
//   logSocket.value.onerror = (error) => {
//     console.error("WebSocket错误:", error);
//   };
//   logSocket.value.onclose = () => {
//     console.log("WebSocket连接已关闭");
//   };
//   // ----------------------success 控制台---------------------------------
//   const socketUrl0 = `${import.meta.env.VITE_WS_BASE_URL}/websocket/terminalProgress/1`;
//   sSocket.value = new WebSocket(socketUrl0);
//   sSocket.value.onopen = () => {
//     console.log("WebSocket连接已打开");
//   };
//   sSocket.value.onmessage = (event) => {
//     console.log("🚀 ~ onMounted ~ event success成功:", event);
//     sItems.value = [...sItems.value, event.data.trim()];
//     // 定时器模拟无缝滚动效果;
//     const lenIdx = sItems.value.length - 1;
//     let startLen = lenIdx - 1 < 0 ? 0 : lenIdx - 1;
//     sTimer = setInterval(() => {
//       startLen++;
//       if (startLen == lenIdx) {
//         clearInterval(sTimer);
//       }
//       sVirtualListInst.value?.scrollTo({ index: startLen });
//     }, 30);
//   };
//   sSocket.value.onerror = (error) => {
//     console.error("WebSocket错误:", error);
//   };
//   sSocket.value.onclose = () => {
//     console.log("WebSocket连接已关闭");
//   };
// });

// onUnmounted(() => {
//   logSocket.value?.close();
//   clearInterval(timer);
//   timer = null;
//   sSocket.value?.close();
//   clearInterval(sTimer);
//   sTimer = null;
// });

</script>

<template>
  <NModal v-model:show="visible" :title="props.trainType == 0 ? '正常控制台' : '异常控制台'" preset="card"
          class="w-1200px">
    <!--    <div class="h-480px pr-20px bg-[#1e1e1e]" v-if="props.trainType == 0">
          <n-virtual-list
            ref="sVirtualListInst"
            class="!h-full !text-[#7d8799]"
            :item-size="32"
            :items="props.trainType == 0 ? sItems : items">
            <template #default="{ item, index }">
              <div :key="index" class="item box-border px-8px" style="height: 32px">
                <pre class="px-0px">
                  <div>{{ item }}</div>
                </pre>
              </div>
            </template>
          </n-virtual-list>
        </div>
        <div class="h-480px pr-20px bg-[#1e1e1e]" v-else>
          <n-virtual-list
            ref="virtualListInst"
            class="!h-full !text-[#7d8799]"
            :item-size="32"
            :items="props.trainType == 0 ? sItems : items">
            <template #default="{ item, index }">
              <div :key="index" class="item box-border px-8px" style="height: 32px">
                <pre class="px-0px">
                  <div>{{ item }}</div>
                </pre>
              </div>
            </template>
          </n-virtual-list>
        </div>-->
    <div class="h-480px pr-20px bg-[#1e1e1e]">
      <n-virtual-list
        ref="virtualListInst"
        class="!h-full !text-[#7d8799]"
        :item-size="32"
        :items="props.trainType == 0 ? props.sItems : props.items">
        <template #default="{ item, index }">
          <div :key="index" class="item box-border px-8px" style="height: 32px">
            <pre class="px-0px">
              <div>{{ item }}</div>
            </pre>
          </div>
        </template>
      </n-virtual-list>
    </div>
    <template #footer>
      <NSpace justify="end" :size="16">
        <NButton @click="closeDrawer">{{ $t("common.cancel") }}</NButton>
      </NSpace>
    </template>
  </NModal>
</template>

<style scoped lang="scss">
:deep(.wrap_scrollMain) {
  .n-scrollbar-container {
    background: #1e1e1e !important;
  }
}
</style>
