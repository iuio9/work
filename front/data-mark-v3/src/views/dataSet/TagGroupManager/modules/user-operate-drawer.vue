<script setup lang="ts">
import {computed, ref, watch} from 'vue';
import {useFormRules, useNaiveForm} from '@/hooks/common/form';
import {fetchGetAllRoles} from '@/service/api';
import {$t} from '@/locales';
import {enableStatusOptions, userGenderOptions} from '@/constants/business';
import {NAlert, NButton, NInput, NSelect, NSpace} from "naive-ui";

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
    add: "创建场景",
    edit: "修改场景"
  };
  return titles[props.operateType];
});

type Model = Pick<
  Api.SystemManage.User,
  'userName' | 'userGender' | 'nickName' | 'userPhone' | 'userEmail' | 'userRoles' | 'status'
>;

const model = ref(createDefaultModel());

function createDefaultModel(): Model {
  return {
    userName: '',
    userGender: null,
    nickName: '',
    userPhone: '',
    userEmail: '',
    userRoles: [],
    status: null,
    labelColor: null
  };
}

type RuleKey = Extract<keyof Model, 'userName' | 'status'>;

const rules: Record<RuleKey, App.Global.FormRule> = {
  userName: defaultRequiredRule,
  status: defaultRequiredRule
};

/** the enabled role options */
const tagOptions = ref([]);

async function getRoleOptions() {
  const {error, data} = await fetchGetAllRoles();

  if (!error) {
    const options = data.map(item => ({
      label: item.roleName,
      value: item.roleCode
    }));

    // the mock data does not have the roleCode, so fill it
    // if the real request, remove the following code
    const userRoleOptions = model.value.userRoles.map(item => ({
      label: item,
      value: item
    }));
    // end

    roleOptions.value = [...userRoleOptions, ...options];
  }
}

function handleInitModel() {
  model.value = createDefaultModel();

  if (props.operateType === 'edit' && props.rowData) {
    Object.assign(model.value, props.rowData);
  }
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
    handleInitModel();
    restoreValidation();
  }
});
</script>

<template>
  <NDrawer v-model:show="visible" display-directive="show" :width="360">
    <NDrawerContent :title="title" :native-scrollbar="false" closable>
      <NForm ref="formRef" :model="model" :rules="rules">
        <NFormItem label="场景名称" path="userName">
          <NSelect
            v-model:value="model.groupName"
            multiple
            :options="tagOptions"
            placeholder="请选择场景名称"
          />
        </NFormItem>
        <NFormItem span="24 m:24" label="标签名称" path="tagName">
          <NInput v-model:value="model.tagName" placeholder="请输入标签名称"/>
        </NFormItem>
        <NFormItem span="24 m:24" label="标签颜色" path="tagName">
          <n-color-picker v-model:value="model.labelColor" :show-alpha="false" />
        </NFormItem>
        <NFormItem span="24 m:24" label="排序号" path="sort">
          <n-input-number v-model:value="model.sort" clearable />
        </NFormItem>
        <NFormItem span="24 m:24" label="是否使用" path="isUse">
          <NSelect
            v-model:value="model.isUse"
            multiple
            :options="tagOptions"
            placeholder="是否使用"
          />
        </NFormItem>
      </NForm>
      <template #footer>
        <NSpace :size="16">
          <NButton @click="closeDrawer">{{ $t('common.cancel') }}</NButton>
          <NButton type="primary" @click="handleSubmit">{{ operateType === 'add' ? '创建' : '确认修改' }}</NButton>
        </NSpace>
      </template>
    </NDrawerContent>
  </NDrawer>
</template>

<style scoped></style>
