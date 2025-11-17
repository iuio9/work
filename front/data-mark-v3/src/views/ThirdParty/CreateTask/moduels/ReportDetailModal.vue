<template>
  <NModal v-model:show="visible" title="任务详情" preset="card" style="width: 600px;">
    <NGrid :cols="24" :x-gap="12">
      <!-- 左侧列 -->
      <NGi :span="12">
        <div class="detail-section">
          <strong>任务名称:</strong><br />
          {{ task.name }}
        </div>
        <div class="detail-section">
          <strong>任务描述:</strong><br />
          {{ task.description }}
        </div>
        <div class="detail-section">
          <strong>版本名称:</strong><br />
          {{ task.versionName }}
        </div>
      </NGi>

      <!-- 右侧列 -->
      <NGi :span="12">
        <div class="detail-section">
          <strong>版本描述:</strong><br />
          {{ task.versionDescription }}
        </div>
        <div class="detail-section">
          <strong>评估数据集:</strong><br />
          {{ task.evaluationDataset }}
        </div>
        <div class="detail-section">
          <strong>评估模型:</strong><br />
          {{ task.evaluationModel }}
        </div>
      </NGi>
    </NGrid>
  </NModal>
</template>

<script setup>
import { ref, defineProps, defineEmits } from 'vue';
import { NModal, NGrid, NGi } from 'naive-ui';

// 定义 props
const props = defineProps({
  modelValue: {
    type: Boolean,
    required: true,
  },
  task: {
    type: Object,
    required: true,
    validator: (obj) => {
      return obj && 'name' in obj && 'description' in obj && 'versionName' in obj &&
        'versionDescription' in obj && 'evaluationDataset' in obj && 'evaluationModel' in obj;
    }
  }
});

// 定义 emit
const emit = defineEmits(['update:modelValue']);

// 控制模态框显示状态的变量
const visible = ref(props.modelValue);

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
