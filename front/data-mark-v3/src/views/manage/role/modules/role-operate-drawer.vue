<script setup lang="ts">
import { computed, reactive, watch } from 'vue';
import { useBoolean } from '@sa/hooks';
import { useFormRules, useNaiveForm } from '@/hooks/common/form';
import { $t } from '@/locales';
import { enableStatusOptions } from '@/constants/business';
import MenuAuthModal from './menu-auth-modal.vue';
import ButtonAuthModal from './button-auth-modal.vue';
import { fetchRoleAdd, fetchRoleEdit } from "@/service/api";

defineOptions({
  name: 'RoleOperateDrawer'
});

interface Props {
  /** the type of operation */
  operateType: NaiveUI.TableOperateType;
  /** the edit row data */
  rowData?: Api.SystemManage.Role | null;
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
const { bool: menuAuthVisible, setTrue: openMenuAuthModal } = useBoolean();
const { bool: buttonAuthVisible, setTrue: openButtonAuthModal } = useBoolean();

const title = computed(() => {
  const titles: Record<NaiveUI.TableOperateType, string> = {
    add: "新增角色",
    edit: "编辑角色"
  };
  return titles[props.operateType];
});

type Model = Pick<Api.SystemManage.Role, 'roleName' | 'roleCode' | 'roleDesc' | 'status'>;

const model: Model = reactive(createDefaultModel());

function createDefaultModel(): Model {
  return {
    roleName: '',
    roleCode: '',
    roleDesc: '',
    status: null,
    mulStatusArray: null
  };
}

type RuleKey = Exclude<keyof Model, 'roleDesc'>;

const rules: Record<RuleKey, App.Global.FormRule> = {
  roleName: defaultRequiredRule,
  roleCode: defaultRequiredRule,
  status: defaultRequiredRule,
  mulStatusArray: defaultRequiredRule,
};

const roleId = computed(() => props.rowData?.id || -1);

const isEdit = computed(() => props.operateType === 'edit');

function handleInitModel() {
  Object.assign(model, createDefaultModel());

  if (props.operateType === 'edit' && props.rowData) {
    console.log('rowData: ', props.rowData);
    Object.assign(model, props.rowData, {
      status: `${props.rowData.status}`,
      // mulStatusArray: `${props.rowData.mulStatusArray}`,
      mulStatusArray: props.rowData.mulStatusArray.map(val => `${val}`),
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
    // mulStatusArray: +model.mulStatusArray,
    mulStatusArray: model.mulStatusArray
  });
  let res;
  if (props.operateType === 'edit') {
    res = await fetchRoleEdit(mapModel)
  };
  if (props.operateType === 'add') {
    res = await fetchRoleAdd(mapModel)
  };
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

const handleTreeSub = (params: any) => {
  console.log(params)
}
</script>

<template>
  <NDrawer v-model:show="visible" display-directive="show" :width="360">
    <NDrawerContent :title="title" :native-scrollbar="false" closable>
      <NForm ref="formRef" :model="model" :rules="rules">
        <NFormItem :label="$t('page.manage.role.roleName')" path="roleName">
          <NInput v-model:value="model.roleName" :placeholder="$t('page.manage.role.form.roleName')" />
        </NFormItem>
        <NFormItem :label="$t('page.manage.role.roleCode')" path="roleCode">
          <NInput v-model:value="model.roleCode" :placeholder="$t('page.manage.role.form.roleCode')" />
        </NFormItem>
        <NFormItem :label="$t('page.manage.role.roleStatus')" path="status">
          <NRadioGroup v-model:value="model.status">
            <NRadio v-for="item in enableStatusOptions" :key="item.value" :value="item.value" :label="$t(item.label)" />
          </NRadioGroup>
        </NFormItem>
        <NFormItem label="多人标注权限" path="mulStatusArray">
          <!-- <NRadioGroup v-model:value="model.mulStatusArray">
            <NRadio v-for="item in enableStatusOptions" :key="item.value" :value="item.value" :label="$t(item.label)"/>
          </NRadioGroup> -->
          <n-checkbox-group v-model:value="model.mulStatusArray">
            <n-checkbox label="发起任务" value="1"></n-checkbox>
            <n-checkbox label="标注员" value="2"></n-checkbox>
            <n-checkbox label="审核员" value="3"></n-checkbox>
            <n-checkbox label="无" value="4"></n-checkbox>
          </n-checkbox-group>
        </NFormItem>
        <NFormItem :label="$t('page.manage.role.roleDesc')" path="roleDesc">
          <NInput v-model:value="model.roleDesc" :placeholder="$t('page.manage.role.form.roleDesc')" />
        </NFormItem>
      </NForm>
      <NSpace v-if="isEdit">
        <NButton @click="openMenuAuthModal">{{ $t('page.manage.role.menuAuth') }}</NButton>
        <MenuAuthModal v-model:visible="menuAuthVisible" :role-id="roleId" />
        <NButton @click="openButtonAuthModal">{{ $t('page.manage.role.buttonAuth') }}</NButton>
        <ButtonAuthModal v-model:visible="buttonAuthVisible" :role-id="roleId" />
      </NSpace>
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
