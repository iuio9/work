<script setup lang="ts">
import { computed, reactive, ref, watch } from 'vue';
import { useFormRules, useNaiveForm } from '@/hooks/common/form';
import { $t } from '@/locales';
import { fetchLabelGroupAdd, fetchLabelGroupEdit } from '@/service/api/tag';
import { enableStatusOptions } from '@/constants/business';

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
    add: '新增标签组',
    edit: '编辑标签组'
  };
  return titles[props.operateType];
});

type Model = Pick<
  Api.SystemManage.User,
  'userName' | 'userGender' | 'nickName' | 'userPhone' | 'userEmail' | 'userRoles' | 'status'
>;

const model: Model = reactive(createDefaultModel());

function createDefaultModel(): any {
  return {
    labelGroupName: null,
    englishLabelGroupName: null,
    labelGroupDesc: null,
    // sort: 0,
    // status: null
  };
}

type RuleKey = Extract<keyof Model, 'labelGroupName' | 'labelGroupName'>;

const rules: Record<RuleKey, App.Global.FormRule> = {
  labelGroupName: defaultRequiredRule,
  englishLabelGroupName: formRules.english,
};

/** the enabled role options */

function handleInitModel() {
  Object.assign(model, createDefaultModel());

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
    res = await fetchLabelGroupEdit(model);
  }
  if (props.operateType === 'add') {
    res = await fetchLabelGroupAdd(model);
  }
  if (res.data >= 1) {
    window.$message?.success(props.operateType === 'add' ? '新增成功' : $t('common.updateSuccess'));
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

const setOptions = ref<any>([]);
const useOptions = ref<any>([]);
const useRadios = ref<any>([
  { value: '1', label: '是' },
  { value: '0', label: '否' }
]);
</script>

<template>
  <NDrawer v-model:show="visible" display-directive="show" :width="360">
    <NDrawerContent :title="title" :native-scrollbar="false" closable>
      <NForm ref="formRef" :model="model" :rules="rules">
        <!--<NFormItem label="选择数据集" path="labelGroupName">
          <NCascader
            v-model:value="model.dataTypeId"
            clearable
            placeholder="请选择数据集"
            :options="setOptions"
            check-strategy="all"
          ></NCascader>
        </NFormItem>-->
        <NFormItem label="标签组名称" path="labelGroupName">
          <NInput v-model:value="model.labelGroupName" placeholder="请输入标签组名称" />
        </NFormItem>
        <NFormItem label="标签组名称(英文)" path="englishLabelGroupName">
          <NInput v-model:value="model.englishLabelGroupName" placeholder="请输入标签组名称（英文）" />
        </NFormItem>
        <!--<NFormItem label="排序号" path="labelGroupDesc">
          <NInputNumber v-model:value="model.sort" clearable />
        </NFormItem>
        <NFormItem label="是否使用" path="labelGroupDesc">
          <NRadioGroup v-model:value="model.status">
            <NRadio v-for="item in useRadios" :key="item.value" :value="item.value" :label="item.label" />
          </NRadioGroup>
        </NFormItem>-->
        <NFormItem label="标签组描述" path="labelGroupDesc">
          <NInput v-model:value="model.labelGroupDesc" placeholder="请输入标签组描述" />
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
