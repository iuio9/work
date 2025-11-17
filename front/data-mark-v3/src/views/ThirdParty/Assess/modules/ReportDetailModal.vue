<template>
  <NModal v-model:show="visible" title="评估报告" preset="card" style="width: 800px;">
    <NGrid :cols="24" :x-gap="12">
      <!-- 左侧列 -->
      <NGi :span="12">
        <div class="detail-section flex justify-start">
          <strong class="mr-4px">任务名称:</strong>
          {{ task.name }}
        </div>
        <div class="detail-section flex justify-start">
          <strong class="mr-4px">任务描述:</strong>
          {{ task.description }}
        </div>
        <div class="detail-section flex justify-start">
          <strong class="mr-4px">版本名称:</strong>
          {{ task.versionName }}
        </div>
      </NGi>

      <!-- 右侧列 -->
      <NGi :span="12">
        <div class="detail-section flex justify-start">
          <strong class="mr-4px">版本描述:</strong>
          {{ task.versionDescription }}
        </div>
        <div class="detail-section flex justify-start">
          <strong class="mr-4px">评估数据集:</strong>
          {{ task.evaluationDataset }}
        </div>
        <div class="detail-section flex justify-start">
          <strong class="mr-4px">评估模型:</strong>
          {{ task.evaluationModel }}
        </div>
      </NGi>

      <!--评估指标-->
      <NGi :span="24">
        <div class="evaluation-section flex justify-start">
          <strong class="mr-4px">评估指标:</strong>
          <n-checkbox-group v-model:value="indicators">
            <n-space item-style="display: flex;">
              <n-checkbox  v-for="song in options" :key="song.value" :value="song.value" :label="song.label" />
            </n-space>
          </n-checkbox-group>
        </div>
      </NGi>
    </NGrid>
    <template #footer>
      <NSpace justify="end" :size="16">
        <NButton type="primary">生成报告</NButton>
        <NButton @click="() => (visible = false)">关闭</NButton>
      </NSpace>
    </template>
  </NModal>
</template>

<script setup>
import { ref, defineProps, defineEmits } from 'vue';
import {NModal, NGrid, NGi, NButton} from 'naive-ui';

// 定义 props
const props = defineProps({
  modelValue: {
    type: Boolean,
    required: true,
  },
  task: {
    type: Object,
    required: true,
    // validator: (obj) => {
    //   return obj && 'name' in obj && 'description' in obj && 'versionName' in obj &&
    //     'versionDescription' in obj && 'evaluationDataset' in obj && 'evaluationModel' in obj;
    // }
  }
});

const indicators = ref(["1", "2", "3", "4", "5", "6"])

// 定义 emit
const emit = defineEmits(['update:modelValue']);

// 控制模态框显示状态的变量
const visible = ref(props.modelValue);

const options = [
  { value: 'mPrecision', label: 'mPrecision' },
  { value: 'mAP', label: 'mAP' },
  { value: 'mAccuracy', label: 'mAccuracy' },
  { value: 'mRecall', label: 'mRecall' },
  { value: '漏检率', label: '漏检率' },
  { value: '虚警率', label: '虚警率' }
];

// 监听父组件传递的 modelValue 变化
watch(() => props.modelValue, (newValue) => {
  visible.value = newValue;
});

// 当 visible 发生变化时更新父组件的状态
watch(visible, (newValue) => {
  emit('update:modelValue', newValue);
});
</script>

<style scoped>
.detail-section {
  margin-bottom: 16px;
}

.detail-section strong {
  display: block;
  margin-bottom: 4px;
}
</style>
