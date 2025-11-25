<script setup lang="ts">
import { computed, ref, onMounted } from 'vue';
import { useRoute } from 'vue-router';
import { $t } from '@/locales';
import { useFormRules, useNaiveForm } from '@/hooks/common/form';
import { enableStatusOptions, userGenderOptions } from '@/constants/business';
import { translateOptions } from '@/utils/common';
import { fetchCompletedDistillationModels } from '@/service/api';

defineOptions({
  name: 'UserSearch'
});

interface Emits {
  (e: 'reset'): void;
  (e: 'search'): void;
}

const emit = defineEmits<Emits>();
const route = useRoute();

const { formRef, validate, restoreValidation } = useNaiveForm();

const model = defineModel<Api.SystemManage.UserSearchParams>('model', { required: true });

type RuleKey = Extract<keyof Api.SystemManage.UserSearchParams, 'userEmail' | 'userPhone'>;

const rules = computed<Record<RuleKey, App.Global.FormRule>>(() => {
  const { patternRules } = useFormRules(); // inside computed to make locale reactive

  return {
    userEmail: patternRules.email,
    userPhone: patternRules.phone
  };
});

// 训练模型选择
const selectedTrainingModel = ref<string>('');
const trainingModelOptions = ref<any[]>([]);
const loadingModels = ref(false);

// 系统默认模型选项
const defaultModelOptions = [
  { label: '系统默认模型 1', value: 'default_model_1' },
  { label: '系统默认模型 2', value: 'default_model_2' },
  { label: '系统默认模型 3', value: 'default_model_3' }
];

// 示例数据：大小模型协同训练的已完成模型（用于演示）
const mockDistillationModels = [
  {
    taskId: 'TASK_001',
    taskName: '目标检测协同训练-YOLOv5',
    accuracy: 92.5,
    teacherModel: 'llama2-7b',
    studentModel: 'yolov5s'
  },
  {
    taskId: 'TASK_002',
    taskName: '图像分类协同训练-ResNet',
    accuracy: 88.3,
    teacherModel: 'qwen-7b',
    studentModel: 'resnet50'
  },
  {
    taskId: 'TASK_003',
    taskName: '语义分割协同训练-UNet',
    accuracy: 85.7,
    teacherModel: 'llama2-13b',
    studentModel: 'unet'
  },
  {
    taskId: 'TASK_005',
    taskName: '视觉Transformer协同训练',
    accuracy: 90.2,
    teacherModel: 'llama2-7b',
    studentModel: 'vit'
  }
];

// 加载大小模型协同训练的已完成模型
async function loadDistillationModels() {
  loadingModels.value = true;
  try {
    const res = await fetchCompletedDistillationModels({ minAccuracy: 70 });
    let distillationOptions = [];

    if (res.data && Array.isArray(res.data) && res.data.length > 0) {
      // 如果API返回了数据，使用API数据
      distillationOptions = res.data.map((task: any) => ({
        label: `[协同训练] ${task.taskName} (准确率: ${task.accuracy?.toFixed(2)}%)`,
        value: `distillation_${task.taskId}`,
        taskId: task.taskId,
        type: 'distillation'
      }));
    } else {
      // 如果API没有返回数据，使用示例数据（方便演示）
      distillationOptions = mockDistillationModels.map((task: any) => ({
        label: `[协同训练] ${task.taskName} (准确率: ${task.accuracy.toFixed(2)}%)`,
        value: `distillation_${task.taskId}`,
        taskId: task.taskId,
        type: 'distillation'
      }));
      console.log('使用示例数据展示大小模型协同训练模型');
    }

    // 合并系统默认模型和大小模型协同训练模型
    trainingModelOptions.value = [
      ...defaultModelOptions,
      ...distillationOptions
    ];
  } catch (error) {
    console.error('加载大小模型协同训练模型失败，使用示例数据:', error);

    // API调用失败时，使用示例数据
    const distillationOptions = mockDistillationModels.map((task: any) => ({
      label: `[协同训练] ${task.taskName} (准确率: ${task.accuracy.toFixed(2)}%)`,
      value: `distillation_${task.taskId}`,
      taskId: task.taskId,
      type: 'distillation'
    }));

    trainingModelOptions.value = [
      ...defaultModelOptions,
      ...distillationOptions
    ];
  } finally {
    loadingModels.value = false;
  }
}

// 处理模型选择变更
function handleModelChange(value: string) {
  selectedTrainingModel.value = value;
  // 将选中的模型ID传递给父组件
  (model.value as any).selectedModelId = value;
}

onMounted(() => {
  loadDistillationModels();

  // 检查路由参数，如果从大小模型协同训练页面跳转过来，自动选择模型
  const distillationModelId = route.query.distillationModelId as string;
  const distillationModelName = route.query.distillationModelName as string;

  if (distillationModelId) {
    selectedTrainingModel.value = `distillation_${distillationModelId}`;
    (model.value as any).selectedModelId = `distillation_${distillationModelId}`;

    if (distillationModelName) {
      window.$message?.success(`已自动选择模型: ${distillationModelName}`);
    }
  }
});

async function reset() {
  await restoreValidation();
  selectedTrainingModel.value = '';
  emit('reset');
}

async function search() {
  await validate();
  emit('search');
}
</script>

<template>
  <NCard :title="$t('common.search')" :bordered="false" size="small" class="card-wrapper">
    <NForm ref="formRef" :model="model" :rules="rules" label-placement="left" :label-width="80">
      <NGrid responsive="screen" item-responsive>
        <NFormItemGi span="24 s:12 m:6" label="任务名称" path="userName" class="pr-24px">
          <NInput v-model:value="model.taskName" placeholder="请输入任务名称" />
        </NFormItemGi>
        <NFormItemGi span="24 s:12 m:6" label="训练模型" path="selectedModelId" class="pr-24px">
          <NSelect
            v-model:value="selectedTrainingModel"
            :options="trainingModelOptions"
            :loading="loadingModels"
            placeholder="选择训练模型（包含大小模型协同训练）"
            clearable
            filterable
            @update:value="handleModelChange"
          >
            <template #empty>
              <div class="text-center py-4">
                <div>暂无可用模型</div>
                <div class="text-sm text-gray-500 mt-2">请先在"大小模型协同训练"模块训练模型</div>
              </div>
            </template>
          </NSelect>
        </NFormItemGi>
        <NFormItemGi span="24 s:12 m:6" label="时间段" path="status" class="pr-24px">
          <n-date-picker v-model:value="model.taskTimeArr" type="datetimerange" clearable />
        </NFormItemGi>
        <NFormItemGi span="24 m:6" class="pr-24px">
          <NSpace class="w-full" justify="end">
            <NButton @click="reset">
              <template #icon>
                <icon-ic-round-refresh class="text-icon" />
              </template>
              {{ $t('common.reset') }}
            </NButton>
            <NButton type="primary" ghost @click="search">
              <template #icon>
                <icon-ic-round-search class="text-icon" />
              </template>
              {{ $t('common.search') }}
            </NButton>
          </NSpace>
        </NFormItemGi>
      </NGrid>
    </NForm>
  </NCard>
</template>

<style scoped></style>
