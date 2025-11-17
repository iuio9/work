<template>
  <n-modal v-model:show="visible" preset="card" :close-on-esc="false" title="关联指标" class="w-1200px" @after-leave="handleAfterLeave">
    <NGrid x-gap="12" :cols="2">
      <n-gi>
        <div class="w-full flex-col items-center gap-4px">
          <span class="text-lg">国网企标</span>
          <NDataTable
            :columns="columns"
            :data="modelGridData"
            :row-key="(row: any) => row.prop"
            :checked-row-keys="gridCheckedRowKeys"
            @update:checked-row-keys="gridCheckedRowKeys = $event" />
        </div>
      </n-gi>
      <n-gi>
        <div class="w-full flex-col items-center gap-4px">
          <span class="text-lg">通用指标</span>
          <NDataTable
            :columns="columns"
            :data="modelCommonData"
            :row-key="(row: any) => row.prop"
            :checked-row-keys="commonCheckedRowKeys"
            @update:checked-row-keys="commonCheckedRowKeys = $event" />
        </div>
      </n-gi>
    </NGrid>
    <template #footer>
      <n-space justify="end" space="[16px]">
        <n-button @click="closeModal" class="bg-gray-200 text-gray-800 hover:bg-gray-300">关闭窗口</n-button>
      </n-space>
    </template>
  </n-modal>
</template>

<script setup lang="ts">
import { NInput, NDataTable, NSpace, NButton, NInputNumber } from 'naive-ui';

defineOptions({
  name: 'BindIndicatorModal'
});

interface Emits {
  (e: "afterLeave", params: any): void;
}

const emit = defineEmits<Emits>();
const visible = defineModel<boolean>('visible', {
  default: false
});

const classId = defineModel<string>('classId', {
  default: ''
});

// 国网企标指标
const modelGridData = defineModel<any>('modelGridData', {
  default: [
    { label: '召回率/发现率/检出率', prop: 'recall', value: "1", key: 0 },
    { label: '误检比', prop: 'falseAlarmRate', value: "1", key: 1 },
    { label: '误报率/误检率', prop: 'falseAlarmRate', value: "1", key: 2 },
    { label: '平均精度AP', prop: 'ap', value: "1", key: 3 },
    { label: 'F1-分数', prop: 'f1', value: "1", key: 4 },
    { label: '识别时间', prop: 'time', value: "1", key: 5 },
    { label: 'IOU平均值', prop: 'iou', value: "1", key: 6 }
  ]
});

const gridCheckedRowKeys = defineModel<any>('gridCheckedRowKeys', {
  default: []
});

// 通用指标
const modelCommonData = defineModel<any>('modelCommonData', {
  default: [
    { label: '平均精度 (mPrecision)', prop: 'mPrecision', value: "1", key: 0 },
    { label: '平均召回率 (mRecall)', prop: 'mRecall', value: "1", key: 1 },
    { label: '均值平均精度 (mAP@0.5)', prop: 'mAP@0.5', value: "1", key: 2 },
    { label: '漏检率 (MissRate)', prop: 'MissRate', value: "1", key: 3 },
    { label: '虚警率 (FalseAlarmRate)', prop: 'FalseAlarmRate', value: "1", key: 4 },
    { label: '平均正确率 (mAccuracy)', prop: 'mAccuracy', value: "1", key: 5 }
  ]
});

const commonCheckedRowKeys = defineModel<any>('commonCheckedRowKeys', {
  default: []
});

const columns = ref<any>([
  {
    type: 'selection',
    disabled: (row: any) => true
  },
  {
    title: '指标名称',
    key: 'label',
    width: '30%'
  },
  {
    title: '指标标识',
    key: 'prop',
    width: '30%'
  },
  {
    title: '指标值',
    key: 'value',
    render: (row: any) => {
      return h(NInputNumber, {
        value: Number(row.value),
        step: '0.2',
        disabled: true,
        onUpdateValue: (v) => {
          row.value = v // 直接通过index更新数据
        }
      })
    }
  }
]);

function closeModal() {
  visible.value = false;
}

function handleAfterLeave() {
  emit('afterLeave', {
    classId: classId.value,
    modelGridData: modelGridData.value,
    modelCommonData: modelCommonData.value,
    gridCheckedRowKeys: gridCheckedRowKeys.value,
    commonCheckedRowKeys: commonCheckedRowKeys.value,
  });
}
</script>

<style scoped></style>
