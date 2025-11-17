<script setup lang="tsx">
import { ref, watch } from 'vue';
import { NAlert, NButton, NForm, NFormItem, NInput, NModal, NSelect, NSpace } from 'naive-ui';
import { useFormRules, useNaiveForm } from '@/hooks/common/form';
import { $t } from '@/locales';

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
  emit('create');
}
</script>

<template>
  <NModal v-model:show="visible" title="标签组操作" preset="card" class="w-800px">
    <NScrollbar class="h-520px pr-20px">
      <NForm ref="formRef" :model="model" label-placement="left" :label-width="100">
        <NGrid responsive="screen" item-responsive :cols="1">
          <!-- 创建标签组 -->
          <NGi>
            <div class="mb-14px ml-16px text-20px font-600">创建标签组</div>
          </NGi>
          <NFormItemGi span="24 m:12" label="选择数据集" path="menuType">
            <div class="w-full flex items-center gap-24px">
              <NCascader
                v-model:value="model.dataTypeId"
                clearable
                placeholder="请选择数据集"
                :options="setOptions"
                check-strategy="all"
              ></NCascader>
            </div>
          </NFormItemGi>
          <NFormItemGi span="24 m:24" label="标签组名称" path="menuType">
            <div class="w-full flex items-center gap-24px">
              <NInput v-model:value="model.groupName" placeholder="请输入标签组名称" />
              <NButton type="primary" @click="createTagGroup">创建</NButton>
            </div>
          </NFormItemGi>
          <!-- 修改标签组名称 -->
          <NGi>
            <div class="mb-14px ml-16px text-20px font-600">修改标签组名称</div>
          </NGi>
          <NFormItemGi span="24 m:12" label="选择数据集" path="menuType">
            <div class="w-full flex items-center gap-24px">
              <NCascader
                v-model:value="model.dataTypeId"
                clearable
                placeholder="请选择数据集"
                :options="setOptions"
                check-strategy="all"
              ></NCascader>
            </div>
          </NFormItemGi>
          <NFormItemGi span="24 m:24" label="标签组名称" path="menuType">
            <div class="w-full flex items-center gap-24px">
              <NSelect v-model:value="model.userRoles" multiple :options="tagOptions" placeholder="请选择标签组名称" />
            </div>
          </NFormItemGi>
          <NFormItemGi span="24 m:24" label="标签组新名称" path="menuType">
            <div class="w-full flex items-center gap-24px">
              <NInput v-model:value="model.groupName" placeholder="请输入标签组新名称" />
              <NButton type="primary" @click="createTagGroup">确认修改</NButton>
            </div>
          </NFormItemGi>
          <NFormItemGi span="24 m:24" label="" path="menuType" class="w-full flex items-center justify-center">
            <!-- 提示信息 -->
            <NSpace>
              <NAlert type="info" :show-icon="true">
                <span>选择您要修改的标签组，然后在下面输入需要修改的标签组名称，点击确认修改即可完成操作！</span>
              </NAlert>
            </NSpace>
          </NFormItemGi>
          <!-- 删除标签组 -->
          <NGi>
            <div class="mb-14px ml-16px text-20px font-600">删除标签组</div>
          </NGi>
          <NFormItemGi span="24 m:12" label="选择数据集" path="menuType">
            <div class="w-full flex items-center gap-24px">
              <NCascader
                v-model:value="model.dataTypeId"
                clearable
                placeholder="请选择数据集"
                :options="setOptions"
                check-strategy="all"
              ></NCascader>
            </div>
          </NFormItemGi>
          <NFormItemGi span="24 m:24" label="标签组名称" path="menuType">
            <div class="w-full flex items-center gap-24px">
              <NSelect v-model:value="model.userRoles" multiple :options="tagOptions" placeholder="请选择标签组名称" />
              <NButton type="primary" @click="createTagGroup">确认删除</NButton>
            </div>
          </NFormItemGi>
          <NFormItemGi span="24 m:24" label="" path="menuType" class="w-full flex items-center justify-center">
            <NSpace>
              <NAlert type="warning" :show-icon="true">
                <span>所选标签组删除之后，该标签组里面所有标签将会全部删除，请谨慎操作！</span>
              </NAlert>
            </NSpace>
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
