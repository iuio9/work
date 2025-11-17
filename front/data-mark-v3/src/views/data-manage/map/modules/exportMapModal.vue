<script setup lang="ts">
import axios from 'axios';
import { downloadByData } from '@/utils/common';
import { useFormRules } from '@/hooks/common/form';
import { $t } from '@/locales';

defineOptions({
  name: 'MenuAuthModal'
});

interface Props {
  /** the roleId */
  sonId: number | string;
}

const props = defineProps<Props>();

const visible = defineModel<boolean>('visible', {
  default: false
});

const { defaultRequiredRule } = useFormRules();

const model = ref(createDefaultModel());
const isExport = ref<Boolean>(false);

const rules: Record<string, App.Global.FormRule> = {
  type: defaultRequiredRule,
  anoType: defaultRequiredRule
};

function createDefaultModel(): any {
  return {
    type: undefined,
    anoType: undefined,
  };
}

function closeModal() {
  visible.value = false;
}

function downloadPost(config: any) {
  return new Promise((resolve, reject) => {
    axios({
      url: config.url, // 请求地址
      method: 'post',
      data: config.data, // 参数
      responseType: 'blob' // 表明返回服务器返回的数据类型
    })
      .then(res => {
        resolve(res);
      })
      .catch(err => {
        reject(err);
      });
  });
}

async function handleSubmit() {
  // request
  isExport.value = true;
  const baseUrl = import.meta.env.VITE_SERVICE_BASE_URL;
  const config = {
    url: `${baseUrl}/file/download?sonId=${props.sonId}`,
    data: {
      sonId: props.sonId,
      type: model.value.type,
      anoType: model.value.anoType
    }
  };
  const fileName = `数据集${props.sonId}.zip`;
  const res: any = await downloadPost(config);
  if (res.data) {
    isExport.value = false;
    await downloadByData(res.data, fileName);
    closeModal();
  }
}
</script>

<template>
  <NModal v-model:show="visible" title="导出数据集" preset="card" class="w-600px" :close-on-esc="false">
    <NForm ref="formRef" :model="model" :rules="rules">
      <!--导出数据-->
      <n-form-item label="导出数据" path="exportData" class="w-[70%]">
        <n-radio-group v-model:value="model.type" name="type">
          <n-space>
            <n-radio value="1" class="flex items-center">
              <span> 导出全部数据</span>
            </n-radio>
            <n-radio value="2" class="flex items-center">
              <span> 仅导出源文件 </span>
            </n-radio>
          </n-space>
        </n-radio-group>
      </n-form-item>
      <!--标注格式-->
      <n-form-item label="标注格式" path="anoType" class="w-[70%]" v-show="model.type === '1'">
        <n-radio-group v-model:value="model.anoType" name="anoType">
          <n-space>
            <n-radio value="1" class="flex items-center">
              <span> json </span>
            </n-radio>
            <n-radio value="2" class="flex items-center">
              <span> xml </span>
            </n-radio>
            <n-radio value="3" class="flex items-center">
              <span> 矩形框标注（json） </span>
            </n-radio>
          </n-space>
        </n-radio-group>
      </n-form-item>
    </NForm>
    <template #footer>
      <NSpace justify="end">
        <NButton class="mt-16px" @click="closeModal">
          {{ $t('common.cancel') }}
        </NButton>
        <NButton type="primary" class="mt-16px" @click="handleSubmit" :loading="isExport">
          {{ $t('common.confirm') }}
        </NButton>
      </NSpace>
    </template>
  </NModal>
</template>

<style scoped></style>
