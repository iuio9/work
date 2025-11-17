<template>
  <n-modal
    v-model:visible="propsVisible"
    title="任务详情"
    :closable="true"
    @update:visible="handleClose"
  >
    <n-grid :gutter="20">
      <!-- 基本信息 -->
      <n-grid-item :span="12">
        <div class="info-item" v-for="(item, index) in basicInfo" :key="index">
          <span class="label">{{ item.label }}:</span>
          <span>{{ taskInfo[item.key] }}</span>
        </div>
      </n-grid-item>

      <n-grid-item :span="12">
        <div class="info-item" v-for="(item, index) in anotherBasicInfo" :key="index">
          <span class="label">{{ item.label }}:</span>
          <span>{{ taskInfo[item.key] }}</span>
        </div>
      </n-grid-item>

      <!-- 任务分配信息 -->
      <n-grid-item :span="24">
        <h4>任务分配信息</h4>
        <n-grid :gutter="20">
          <n-grid-item :span="12">
            <div class="info-item" v-for="(item, index) in assignInfo1" :key="index">
              <span class="label">{{ item.label }}:</span>
              <span>{{ taskInfo[item.key] }}</span>
            </div>
          </n-grid-item>

          <n-grid-item :span="12">
            <div class="info-item" v-for="(item, index) in assignInfo2" :key="index">
              <span class="label">{{ item.label }}:</span>
              <span>{{ taskInfo[item.key] }}</span>
            </div>
          </n-grid-item>
        </n-grid>
      </n-grid-item>
    </n-grid>

    <template #footer>
      <n-button type="primary" @click="handleClose">关闭</n-button>
    </template>
  </n-modal>
</template>

<script setup>
import { defineProps, defineEmits, ref } from 'vue';
import { NModal, NGrid, NGridItem, NButton } from 'naive-ui';

const props = defineProps({
  taskInfo: {
    type: Object,
    required: true
  },
  visible: {
    type: Boolean,
    default: false
  }
});

const emits = defineEmits(['update:visible']);

const propsVisible = ref(props.visible);

// 基本信息结构配置
const basicInfo = [
  { label: '任务名称', key: 'taskName' },
  { label: '分配数据类型', key: 'assignDataType' },
  { label: '任务截止时间', key: 'taskDeadline' }
];

const anotherBasicInfo = [
  { label: '待标注数据集', key: 'toBeLabeledDataset' },
  { label: '任务开始时间', key: 'taskStartDate' }
];

// 任务分配信息结构配置
const assignInfo1 = [
  { label: '任务分配信息', key: 'distributionInfo' },
  { label: '每人标注数量', key: 'perPersonNum' },
  { label: '任务截止时间', key: 'taskDeadline' }
];

const assignInfo2 = [
  { label: '标注团队名称', key: 'teamName' },
  { label: '成员权限范围', key: 'permissionScope' },
  { label: '标注完数据集', key: 'labeledDataset' }
];

const handleClose = () => {
  propsVisible.value = false;
  emits('update:visible', false);
};
</script>

<style scoped>
.info-item {
  margin-bottom: 12px;
  display: flex;
  align-items: center;
  gap: 8px;
}

.label {
  min-width: 80px;
  color: #666;
}
</style>
