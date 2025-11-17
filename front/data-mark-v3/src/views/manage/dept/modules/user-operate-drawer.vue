<script setup lang="ts">
import {computed, reactive, ref, watch} from 'vue';
import {useFormRules, useNaiveForm} from '@/hooks/common/form';
import {fetchGetRoles, fetchUserAdd, fetchUserEdit} from '@/service/api';
import {$t} from '@/locales';
import {enableStatusOptions, userGenderOptions} from '@/constants/business';
import {fetchDeptAdd, fetchDeptEdit} from "@/service/api/dept";

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
const {defaultRequiredRule,formRules} = useFormRules();

const title = computed(() => {
  const titles: Record<NaiveUI.TableOperateType, string> = {
    add: "新增部门",
    edit: "编辑部门"
  };
  return titles[props.operateType];
});

interface Model {
  deptName: string,
  sort: string,
  [key: string]: any
}

const model: Model = reactive(createDefaultModel());

function createDefaultModel(): Model {
  return {
    deptName: "",
    sort: "",
    supt: "",
    email: "",
    telePhone: "",
    status: null
  };
}

type RuleKey = Extract<keyof Model, 'userName' | 'status' | 'password'>;


const rules: Record<RuleKey, App.Global.FormRule> = {
  deptName: defaultRequiredRule,
  sort: defaultRequiredRule,
  status: defaultRequiredRule,
  email: formRules.email,
  telePhone: formRules.phone,
};

/** the enabled role options */
const roleOptions = ref<CommonType.Option<string>[]>([]);

function handleInitModel() {
  Object.assign(model, createDefaultModel());

  if (props.operateType === 'edit' && props.rowData) {
    Object.assign(model, props.rowData, {
      status: `${props.rowData.status}`,
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
  let res: any;
  if (props.operateType === 'edit') {
    res = await fetchDeptEdit(mapModel);
  }
  if (props.operateType === 'add') {
    res = await fetchDeptAdd(mapModel);
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
    <NDrawerContent :title="title" :native-scrollbar="false" closable>
      <NForm ref="formRef" :model="model" :rules="rules">
        <NFormItem label="部门名称" path="deptName">
          <NInput v-model:value="model.deptName" placeholder="请输入部门名称"/>
        </NFormItem>
        <NFormItem label="显示排序" path="sort">
          <NInput v-model:value="model.sort" placeholder="请输入排序值"/>
        </NFormItem>
        <NFormItem label="负责人" path="supt">
          <NInput v-model:value="model.supt" placeholder="请输入负责人"/>
        </NFormItem>
        <NFormItem label="联系电话" path="telePhone">
          <NInput v-model:value="model.telePhone" placeholder="请输入联系电话"/>
        </NFormItem>
        <NFormItem label="邮箱" path="email">
          <NInput v-model:value="model.email" placeholder="请输入邮箱"/>
        </NFormItem>
        <NFormItem label="状态" path="status">
          <NRadioGroup v-model:value="model.status">
            <NRadio v-for="item in enableStatusOptions" :key="item.value" :value="item.value" :label="$t(item.label)"/>
          </NRadioGroup>
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
