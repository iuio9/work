<script setup lang="ts">
import {computed, reactive, watch} from 'vue';
import {useFormRules, useNaiveForm} from '@/hooks/common/form';
import {$t} from '@/locales';
import {addExample, updateExample, updateModel} from "@/service/api/dataManage";

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
const statusOptions = ref([
  { value: "训练中", label: "训练中"},
  { value: "成功", label: "成功"},
  { value: "失败", label: "失败"},
])

function createDefaultModel(): Model {
  return {
    modelName: null,
    trainStat: null,
    modelUrl: null,
    modelBizType: null,
    modelDesc: null,
  };
}

type RuleKey = Extract<keyof Model, 'userName' | 'status' | 'password'>;

const rules: Record<RuleKey, App.Global.FormRule> = {
  algorithmName: defaultRequiredRule,
};

/** the enabled role options */

function handleInitModel() {
  Object.assign(model, props.rowData);
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
    res = await updateModel(rest);
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
  }
});

</script>

<template>
  <NDrawer v-model:show="visible" display-directive="show" :width="360">
    <NDrawerContent title="模型训练" :native-scrollbar="false" closable>
      <NForm ref="formRef" :model="model" :rules="rules">
        <NFormItem label="模型名称" path="modelName">
          <NInput v-model:value="model.modelName" placeholder="请输入模型名称"/>
        </NFormItem>
        <NFormItem label="训练状态" path="trainStat">
          <!--<NInput v-model:value="model.trainStat" placeholder="请输入训练状态 (成功 || 失败)"/>-->
          <NRadioGroup v-model:value="model.trainStat">
            <NRadio v-for="item in statusOptions" :key="item.value" :value="item.value" :label="$t(item.label)"/>
          </NRadioGroup>
        </NFormItem>
        <NFormItem label="保存路径" path="modelUrl">
          <NInput v-model:value="model.modelUrl" placeholder="请输入保存路径"/>
        </NFormItem>
        <NFormItem label="模型业务类型" path="modelBizType">
          <NInput v-model:value="model.modelBizType" placeholder="请输入模型业务类型"/>
        </NFormItem>
        <NFormItem label="模型描述信息" path="modelDesc">
          <NInput
            type="textarea"
            v-model:value="model.modelDesc" placeholder="请输入模型描述信息"/>
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
