<script setup lang="ts">
import { $t } from "@/locales";
import { LogInst, VirtualListInst } from "naive-ui";

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

function closeDrawer() {
  clearInterval(timer);
  timer = null;
  visible.value = false;
};

const handleOperate = (sign: 'stop' | 'abort' | 'again') => {
  if (sign === 'stop') {
  }
  if (sign ==='abort') {
  }
  if (sign ==='again') {
  }
}
</script>

<template>
  <NModal v-model:show="visible" title="模型测试" preset="card" class="w-1200px">
    <div class="h-480px pr-20px bg-[#1e1e1e]">
      <!-- <n-virtual-list ref="virtualListInst" class="!h-full !text-[#7d8799]" :item-size="32"
        :items="props.trainType == 0 ? props.sItems : props.items">
        <template #default="{ item, index }">
          <div :key="index" class="item box-border px-8px" style="height: 32px">
            <pre class="px-0px">
              <div>{{ item }}</div>
            </pre>
          </div>
        </template>
      </n-virtual-list> -->
    </div>
    <template #footer>
      <NSpace justify="end" :size="16">
        <NButton type="primary" @click="handleOperate('stop')">模型暂停</NButton>
        <NButton type="primary" @click="handleOperate('abort')">模型终止</NButton>
        <NButton type="primary" @click="handleOperate('again')">重新运行</NButton>
        <NButton @click="closeDrawer">关闭窗口</NButton>
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
