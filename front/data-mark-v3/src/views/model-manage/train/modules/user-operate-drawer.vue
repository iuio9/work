<script setup lang="ts">
import {computed, reactive, watch} from 'vue';
import {useFormRules, useNaiveForm} from '@/hooks/common/form';
import {$t} from '@/locales';
import {addExample, updateExample} from "@/service/api/dataManage";

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

const {formRef, validate, restoreValidation} = useNaiveForm();
const {defaultRequiredRule} = useFormRules();

const title = computed(() => {
  const titles: Record<NaiveUI.TableOperateType, string> = {
    add: $t('page.manage.user.addUser'),
    edit: $t('page.manage.user.editUser')
  };
  return titles[props.operateType];
});

const reqOptions = [
  { value: 0, label: "get"},
  { value: 1, label: "post"},
  { value: 2, label: "put"},
  { value: 3, label: "delete"},
]

interface Model {
  deptName: string,
  sort: string,
  [key: string]: any
}

const model: Model = reactive(createDefaultModel());

function createDefaultModel(): Model {
  return {
    algorithmName: null,
    url: null,
    requestType: null,
    params: null,
    algorithmDesc: null,
  };
}

type RuleKey = Extract<keyof Model, 'userName' | 'status' | 'password'>;

const rules: Record<RuleKey, App.Global.FormRule> = {
  algorithmName: defaultRequiredRule,
};

/** the enabled role options */

function handleInitModel() {
  Object.assign(model, props.rowData, createDefaultModel());
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
  if (props.operateType === 'edit') {
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
    const { id, ...rest } = model;
    res = await updateExample(rest);
  }
  if (props.operateType === 'add') {
    res = await addExample(model);
  }
  if (res.data >=1) {
    window.$message?.success(props.operateType === 'add' ? "新增成功" : $t('common.updateSuccess'));
    closeDrawer();
    emit('submitted');
  }
}

watch(visible, () => {
  if (visible.value) {
    handleInitModel();
    restoreValidation();
    // getRoleOptions();
  }
});

</script>

<template>
  <NDrawer v-model:show="visible" display-directive="show" :width="360">
    <NDrawerContent title="算法训练" :native-scrollbar="false" closable>
      <NForm ref="formRef" :model="model" :rules="rules">
        <NFormItem label="算法名称" path="algorithmName">
          <NInput v-model:value="model.algorithmName" placeholder="请输入算法名称"/>
        </NFormItem>
        <NFormItem label="算法请求类型" path="requestType">
          <NInput v-model:value="model.requestType" placeholder="请输入算法请求类型"/>
        </NFormItem>
        <NFormItem label="算法请求地址" path="url">
          <NInput v-model:value="model.url" placeholder="请输入算法业务类型"/>
        </NFormItem>
        <NFormItem label="算法描述" path="modelDesc">
          <NInput
            type="textarea"
            v-model:value="model.algorithmDesc" placeholder="请输入算法描述"/>
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
