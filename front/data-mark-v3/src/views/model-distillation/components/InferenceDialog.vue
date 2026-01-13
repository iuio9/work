<template>
  <n-modal
    v-model:show="showModal"
    preset="card"
    title="使用模型进行自动标注"
    :style="{ width: '600px' }"
    :mask-closable="false"
    :segmented="{ content: 'soft', footer: 'soft' }"
  >
    <n-form ref="formRef" :model="formData" :rules="rules" label-placement="left" label-width="120">
      <n-form-item label="训练任务" path="taskId">
        <n-input :value="taskInfo" disabled />
      </n-form-item>

      <n-form-item label="输入图像目录" path="inputDir">
        <n-input
          v-model:value="formData.inputDir"
          placeholder="/data/images"
          :disabled="loading"
        />
      </n-form-item>

      <n-form-item label="输出标注目录" path="outputDir">
        <n-input
          v-model:value="formData.outputDir"
          placeholder="/data/labels"
          :disabled="loading"
        />
      </n-form-item>

      <n-form-item label="批次大小" path="batchSize">
        <n-input-number
          v-model:value="formData.batchSize"
          :min="1"
          :max="128"
          :disabled="loading"
          style="width: 100%"
        />
      </n-form-item>

      <n-form-item label="数据集ID" path="datasetId">
        <n-input
          v-model:value="formData.datasetId"
          placeholder="可选，用于将结果保存到数据集"
          :disabled="loading"
        />
      </n-form-item>

      <n-form-item label="自动导入结果">
        <n-switch v-model:value="formData.autoImport" :disabled="loading" />
      </n-form-item>
    </n-form>

    <template #footer>
      <n-space justify="end">
        <n-button @click="handleCancel" :disabled="loading">取消</n-button>
        <n-button type="primary" @click="handleSubmit" :loading="loading">
          {{ loading ? '提交中...' : '开始推理' }}
        </n-button>
      </n-space>
    </template>
  </n-modal>
</template>

<script setup lang="ts">
import { ref, computed, watch } from 'vue';
import { useMessage } from 'naive-ui';
import { submitInferenceTask } from '@/service/api/model-distillation';

interface Props {
  show: boolean;
  task: any;
}

interface Emits {
  (e: 'update:show', value: boolean): void;
  (e: 'success', inferenceId: string): void;
}

const props = defineProps<Props>();
const emit = defineEmits<Emits>();
const message = useMessage();

const showModal = computed({
  get: () => props.show,
  set: (value) => emit('update:show', value)
});

const formRef = ref();
const loading = ref(false);

const formData = ref({
  inputDir: '',
  outputDir: '',
  batchSize: 8,
  datasetId: '',
  autoImport: true
});

const taskInfo = computed(() => {
  if (!props.task) return '';
  return `${props.task.taskName} (${props.task.taskId})`;
});

const rules = {
  inputDir: [
    { required: true, message: '请输入输入图像目录', trigger: 'blur' }
  ],
  outputDir: [
    { required: true, message: '请输入输出标注目录', trigger: 'blur' }
  ]
};

// 重置表单
watch(() => props.show, (newVal) => {
  if (newVal && props.task) {
    // 自动填充默认路径
    formData.value.inputDir = `/data/images/${props.task.taskId}`;
    formData.value.outputDir = `/data/labels/${props.task.taskId}`;
  }
});

const handleSubmit = async () => {
  try {
    console.log('开始提交推理任务...');
    console.log('props.task:', props.task);
    console.log('formData:', formData.value);

    // 表单验证
    await formRef.value?.validate();
    console.log('表单验证通过');

    loading.value = true;

    const submitData = {
      taskId: props.task.taskId,
      ...formData.value
    };

    console.log('提交数据:', submitData);

    const res = await submitInferenceTask(submitData);

    console.log('后端响应:', res);

    // 检查响应是否成功（兼容不同的响应格式）
    if (res.code === 200 || res.code === 0 || (res.data && !res.error)) {
      message.success('推理任务已提交！推理ID: ' + (res.data || ''));
      emit('success', res.data);
      showModal.value = false;
    } else {
      message.error(res.message || res.error || '提交失败');
    }
  } catch (error: any) {
    console.error('提交推理任务错误:', error);

    // 检查是否是表单验证错误
    if (error && error.constructor && error.constructor.name === 'Array') {
      console.log('表单验证失败');
      message.warning('请填写必填项');
      return;
    }

    message.error('提交失败：' + (error?.message || JSON.stringify(error) || '未知错误'));
  } finally {
    loading.value = false;
  }
};

const handleCancel = () => {
  showModal.value = false;
};
</script>

<style scoped>
/* 样式可以根据需要添加 */
</style>
