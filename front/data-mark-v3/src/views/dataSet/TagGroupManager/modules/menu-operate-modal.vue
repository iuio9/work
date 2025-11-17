<script setup lang="tsx">
import { ref, watch } from 'vue';
import { useFormRules, useNaiveForm } from '@/hooks/common/form';
import { $t } from '@/locales';
import { NButton, NModal, NForm, NFormItem, NInput, NSelect, NAlert, NSpace } from 'naive-ui';

defineOptions({
  name: 'MenuOperateModal'
});

interface Emits {
  (e: 'submitted'): void;
  (e: 'create'): void;
}

const emit = defineEmits<Emits>();

const visible = defineModel<boolean>('visible', {
  default: false
});

const tagOptions = ref([]);

const { formRef, validate, restoreValidation } = useNaiveForm();
const { defaultRequiredRule } = useFormRules();

const model = ref(createDefaultModel());

function createDefaultModel(): any {
  return {
    groupName: null
  };
}

function closeDrawer() {
  visible.value = false;
}

async function handleSubmit() {
  await validate();
  // request
  window.$message?.success($t('common.updateSuccess'));
  closeDrawer();
  emit('submitted');
}

watch(visible, () => {
  if (visible.value) {
    restoreValidation();
  }
});

function createTagGroup() {
  emit('create')
}
</script>

<template>
  <NModal v-model:show="visible" title="场景操作" preset="card" class="w-800px">
    <NScrollbar class="h-520px pr-20px">
      <NForm ref="formRef" :model="model" label-placement="left" :label-width="100">
        <NGrid responsive="screen" item-responsive :cols="1">
          <n-gi>
            <div class="text-20px font-600 mb-14px ml-16px">创建场景</div>
          </n-gi>
          <NFormItemGi span="24 m:24" label="场景名称" path="menuType">
            <div class="w-full flex items-center gap-24px">
              <NInput v-model:value="model.groupName" placeholder="请输入场景名称"/>
              <NButton type="primary" @click="createTagGroup">创建</NButton>
            </div>
          </NFormItemGi>
          <n-gi>
            <div class="text-20px font-600 mb-14px ml-16px">修改场景名称</div>
          </n-gi>
          <NFormItemGi span="24 m:24" label="场景名称" path="menuType">
            <div class="w-full flex items-center gap-24px">
              <NSelect
                v-model:value="model.userRoles"
                multiple
                :options="tagOptions"
                placeholder="请选择场景名称"
              />
              <NButton type="primary" @click="createTagGroup">创建</NButton>
            </div>
          </NFormItemGi>
          <NFormItemGi span="24 m:24" label="场景新名称" path="menuType">
            <div class="w-full flex items-center gap-24px">
              <NInput v-model:value="model.groupName" placeholder="请输入场景新名称"/>
              <NButton type="primary" @click="createTagGroup">确认修改</NButton>
            </div>
          </NFormItemGi>
          <NFormItemGi span="24 m:24" label="" path="menuType" class="w-full flex justify-center items-center">
            <!-- 提示信息 -->
            <n-space>
              <n-alert type="info" :show-icon="true">
                <span>选择您要修改的场景，然后在下面输入需要修改的场景名称，点击确认修改即可完成操作！</span>
              </n-alert>
            </n-space>
          </NFormItemGi>
          <n-gi>
            <div class="text-20px font-600 mb-14px ml-16px">删除场景</div>
          </n-gi>
          <NFormItemGi span="24 m:24" label="场景名称" path="menuType">
            <div class="w-full flex items-center gap-24px">
              <NSelect
                v-model:value="model.userRoles"
                multiple
                :options="tagOptions"
                placeholder="请选择场景名称"
              />
              <NButton type="primary" @click="createTagGroup">确认删除</NButton>
            </div>
          </NFormItemGi>
          <NFormItemGi span="24 m:24" label="" path="menuType" class="w-full flex justify-center items-center">
            <n-space>
              <n-alert type="warning" :show-icon="true">
                <span>所选场景删除之后，该场景里面所有标签将会全部删除，请谨慎操作！</span>
              </n-alert>
            </n-space>
          </NFormItemGi>
        </NGrid>
      </NForm>
    </NScrollbar>
    <template #footer>
      <NSpace justify="end" :size="16">
        <NButton type="default" @click="closeDrawer">关闭操作窗口</NButton>
      </NSpace>
    </template>
  </NModal>
</template>

<style scoped lang="scss">
:deep(.n-scrollbar-container) {
  border: none !important;
}
</style>
