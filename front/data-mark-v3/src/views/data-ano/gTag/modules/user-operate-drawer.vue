<script setup lang="ts">
import { computed, reactive, ref, watch } from 'vue';
import { useFormRules, useNaiveForm } from '@/hooks/common/form';
import { $t } from '@/locales';
import { fetchLabelAdd, fetchLabelEdit } from "@/service/api/tag";

defineOptions({
  name: 'UserOperateDrawer'
});

interface Props {
  /** the type of operation */
  operateType: NaiveUI.TableOperateType;
  /** the edit row data */
  rowData?: Api.SystemManage.User | null;
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
const { defaultRequiredRule, formRules } = useFormRules();

const title = computed(() => {
  const titles: Record<NaiveUI.TableOperateType, string> = {
    add: "新增标签",
    edit: "编辑标签"
  };
  return titles[props.operateType];
});

type Model = Pick<
  Api.SystemManage.User,
  'userName' | 'userGender' | 'nickName' | 'userPhone' | 'userEmail' | 'userRoles' | 'status'
>;

const model: Model = reactive(createDefaultModel());

const route = useRoute();
function createDefaultModel(): any {
  return {
    onlyId: null,
    labelName: null,
    labelColor: "#000000",
    englishLabelName: null,
    labelSort: undefined
    // labelGroupId: route.query.id
  };
}

type RuleKey = Extract<keyof Model, 'labelName' | 'labelName'>;

const rules: Record<RuleKey, App.Global.FormRule> = {
  labelName: defaultRequiredRule,
  englishLabelName: formRules.english,
};

/** the enabled role options */

function handleInitModel() {
  Object.assign(model, createDefaultModel(), {
    // labelGroupId: route.query.id
    labelGroupId: JSON.parse(localStorage.getItem("row")).id
  });

  if (props.operateType === 'edit' && props.rowData) {
    Object.assign(model, props.rowData);
  }
}

function closeDrawer() {
  visible.value = false;
}

async function handleSubmit() {
  await validate();
  // request
  let res: any;
  const { index, ...rest } = model;
  if (props.operateType === 'edit') {
    res = await fetchLabelEdit(rest);
  }
  if (props.operateType === 'add') {
    res = await fetchLabelAdd(rest);
  }
  if (res.data >= 1) {
    window.$message?.success(props.operateType === 'add' ? "新增成功" : $t('common.updateSuccess'));
    closeDrawer();
    emit('submitted');
  }
}

watch(visible, () => {
  if (visible.value) {
    handleInitModel();
    restoreValidation();
  }
});
</script>

<template>
  <NDrawer v-model:show="visible" display-directive="show" :width="360">
    <NDrawerContent :title="title" :native-scrollbar="false" closable>
      <NForm ref="formRef" :model="model" :rules="rules">
        <!-- <NFormItem label="唯一ID" path="onlyId">
          <n-input
            :disabled="props.operateType === 'add' ? false : true"
            v-model:value="model.onlyId"
            placeholder="请输入唯一Id"
          />
        </NFormItem> -->
        <NFormItem label="标签名称" path="labelName">
          <n-input v-model:value="model.labelName" placeholder="请输入标签名称" />
        </NFormItem>
        <NFormItem label="标签名称(英文名)" path="englishLabelName">
          <n-input v-model:value="model.englishLabelName" placeholder="请输入标签名称（英文名）" />
        </NFormItem>
        <NFormItem label="标签颜色" path="labelColor">
          <n-color-picker v-model:value="model.labelColor" :show-alpha="false" />
        </NFormItem>
        <NFormItem label="标签排序号" path="labelSort">
          <n-input-number v-model:value="model.labelSort" clearable />
        </NFormItem>
      </NForm>
      <template #footer>
        <NSpace :size="16">
          <NButton @click="closeDrawer">{{ $t('common.cancel') }}</NButton>
          <NButton type="primary" @click="handleSubmit">{{ $t('common.confirm') }}</NButton>
        </NSpace>
      </template>
    </NDrawerContent>
  </NDrawer>
</template>

<style scoped></style>
