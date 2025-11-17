<script setup lang="ts">
import { computed, reactive, watch, ref } from 'vue';
import { useFormRules, useNaiveForm } from '@/hooks/common/form';
import {
  NButton, NInput, NUpload, NModal, NForm, NFormItem,
  NSpace, NRadioGroup, NRadio
} from 'naive-ui';
import { startBlackWhiteTest } from '@/service/api/third';
import { useBoolean } from '~/packages/hooks/src';
import UploadOperateDrawer from "./upload-operate-drawer.vue"

defineOptions({
  name: 'BlackWhiteBoxOperateModal'
});

export type OperateType = 'add' | 'edit';

interface Props {
  operateType: OperateType;
  taskData?: any | null;
}

const props = defineProps<Props>();

interface Emits {
  (e: 'submitted'): void;
}

const emit = defineEmits<Emits>();

const visible = defineModel<boolean>('visible', {
  default: false
});

const { formRef, validate, restoreValidation } = useNaiveForm();
const { defaultRequiredRule } = useFormRules();

interface TaskModel {
  taskName: string | null;
  testType: 'black' | 'white' | null;
  attachments: any;
  taskDesc?: string | null;
}

const model: TaskModel = reactive<any>({
  taskName: null,
  testType: null,
  attachments: [],
  taskDesc: null,
  markStatus: "0",
  importMode: '0-1'
});

const rules = {
  taskName: defaultRequiredRule,
  testType: defaultRequiredRule
};

const testTypes = [
  { value: '0', label: '黑盒测试' },
  { value: '1', label: '白盒测试' }
];

const { bool: drawerVisible, setTrue: openDrawer } = useBoolean();
const isUpSuccess = ref<boolean>(false);

function handleInitModel() {
  Object.assign(model, {
    taskName: null,
    testType: null,
    attachments: [],
    taskDesc: null
  });
}

function closeDrawer() {
  visible.value = false;
}

async function handleSubmit() {
  await validate();
  console.log(model);
  const params = {
    taskName: model.taskName,
    testType: model.testType,
    taskDesc: model.taskDesc,
    fileId: model.attachments.map(val => val?.id)[0]
  };
  const res = await startBlackWhiteTest(params);
  if (res.data) {
    visible.value = false;
    emit('submitted')
  }
}

watch(visible, () => {
  if (visible.value) {
    handleInitModel();
    restoreValidation();
  }
});

function uploadSubmit(data: any) {
  model.attachments = data.fileList;
}
</script>

<template>
  <NModal v-model:show="visible" title="黑白盒测试" preset="card" class="w-800px">
    <NScrollbar class="h-360px pr-20px">
      <NForm ref="formRef" :model="model" :rules="rules" label-placement="left" :label-width="120">
        <NFormItem label="任务名称" path="taskName">
          <NInput v-model:value="model.taskName" placeholder="请输入任务名称" />
        </NFormItem>

        <NFormItem label="测试类型" path="testType">
          <NRadioGroup v-model:value="model.testType">
            <NSpace>
              <NRadio v-for="type in testTypes" :key="type.value" :value="type.value">
                {{ type.label }}
              </NRadio>
            </NSpace>
          </NRadioGroup>
        </NFormItem>

        <NFormItem label="上传附件" path="attachments">
          <n-button type="primary" @click="() => openDrawer()">上传附件</n-button>
          <div v-if="isUpSuccess" class="ml-16px flex items-center justify-start">
            <NSpin size="small" />
            <div class="ml-8px text-14px">
              文件异步上传中... 请稍等!!!
            </div>
          </div>
          <div v-show="model.attachments.length !== 0" class="ml-16px flex items-center justify-start">
            <span>已上传</span>
            <span>{{ model.attachments.length }}</span>
            <span>个文件</span>
          </div>
        </NFormItem>

        <NFormItem label="任务描述" path="taskDesc">
          <NInput type="textarea" v-model:value="model.taskDesc" placeholder="请输入任务描述" :rows="4" />
        </NFormItem>
      </NForm>
    </NScrollbar>

    <template #footer>
      <NSpace justify="end" :size="16">
        <NButton @click="closeDrawer">取消</NButton>
        <NButton type="primary" @click="handleSubmit">开始测试</NButton>
      </NSpace>
    </template>
    <UploadOperateDrawer ref="uploadRef" v-model:visible="drawerVisible" v-model:isUpSuccess="isUpSuccess"
      v-model:markStatus="model.markStatus" v-model:importMode="model.importMode" @submitted="uploadSubmit" />
  </NModal>
</template>

<style scoped>
/* 自定义样式 */
.n-upload-file-list {
  display: none;
  /* 隐藏默认文件列表 */
}
</style>
