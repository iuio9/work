<script setup lang="ts">
import { computed, reactive, ref, watch } from 'vue';
import { useFormRules, useNaiveForm } from '@/hooks/common/form';
import {fetchDictAdd, fetchDictEdit, fetchGetAllRoles, fetchGetRoles, fetchUserAdd, fetchUserEdit} from '@/service/api';
import { $t } from '@/locales';
import { enableStatusOptions, userGenderOptions } from '@/constants/business';

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
const { defaultRequiredRule } = useFormRules();

const title = computed(() => {
  const titles: Record<NaiveUI.TableOperateType, string> = {
    add: "新增字典",
    edit: "编辑字典"
  };
  return titles[props.operateType];
});

interface Model {
  dictName: string,
  remark: string,
  status: any,
  [key: string]: any
} ;

const model: Model = reactive(createDefaultModel());

function createDefaultModel(): Model {
  return {
    dictName: "",
    remark: "",
    status: null
  };
}

type RuleKey = Extract<keyof Model, 'dictName'>;

const rules: Record<RuleKey, App.Global.FormRule> = {
  dictName: defaultRequiredRule,
};

function handleInitModel() {
  Object.assign(model, createDefaultModel());

  if (props.operateType === 'edit' && props.rowData) {
    Object.assign(model, props.rowData, {
      status:  `${props.rowData.status}`,
    });
  }
}

function closeDrawer() {
  visible.value = false;
}

async function handleSubmit() {
  await validate();
  // request
  const mapModel: any = Object.assign({}, model, {
    status: +model.status,
  });
  let res:any;
  if(props.operateType === 'edit') {
    res = await fetchDictEdit(mapModel);
  }
  if(props.operateType === 'add') {
    res = await fetchDictAdd(mapModel);
  }
  if(res.data >=1) {
    window.$message?.success(props.operateType === 'add' ? "新增成功":$t('common.updateSuccess'));
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
        <NFormItem label="字典名称" path="dictName">
          <NInput v-model:value="model.dictName" placeholder="请输入字典名称" />
        </NFormItem>
        <NFormItem label="状态" path="status">
          <NRadioGroup v-model:value="model.status">
            <NRadio v-for="item in enableStatusOptions" :key="item.value" :value="item.value" :label="$t(item.label)" />
          </NRadioGroup>
        </NFormItem>
        <NFormItem label="备注信息" path="nickName">
          <NInput
            type="textarea"
            v-model:value="model.remark" placeholder="请输入备注信息" />
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
