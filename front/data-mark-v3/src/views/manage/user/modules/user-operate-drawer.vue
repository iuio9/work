<script setup lang="ts">
import { computed, reactive, ref, watch } from "vue";
import { useFormRules, useNaiveForm } from "@/hooks/common/form";
import { fetchGetRoles, fetchUserAdd, fetchUserEdit } from "@/service/api";
import { $t } from "@/locales";
import { enableStatusOptions, userGenderOptions } from "@/constants/business";
import { fetchGetDeptSelect } from "@/service/api/dept";

defineOptions({
  name: "UserOperateDrawer",
});

interface Props {
  /** the type of operation */
  operateType: NaiveUI.TableOperateType;
  /** the edit row data */
  rowData?: Api.SystemManage.User | null;
}

const props = defineProps<Props>();

interface Emits {
  (e: "submitted"): void;
}

const emit = defineEmits<Emits>();

const visible = defineModel<boolean>("visible", {
  default: false,
});

const { formRef, validate, restoreValidation } = useNaiveForm();
const { defaultRequiredRule, createConfirmUserNameRule } = useFormRules();

const title = computed(() => {
  const titles: Record<NaiveUI.TableOperateType, string> = {
    add: $t("page.manage.user.addUser"),
    edit: $t("page.manage.user.editUser"),
  };
  return titles[props.operateType];
});

type Model = Pick<
  Api.SystemManage.User,
  | "userName"
  | "userGender"
  | "nickName"
  | "userPhone"
  | "userEmail"
  | "userRoles"
  | "status"
  | "password"
> &
  "deptIds";

const model: Model = reactive(createDefaultModel());

function createDefaultModel(): Model {
  return {
    deptIds: "",
    userName: "",
    password: "",
    userGender: null,
    nickName: "",
    userPhone: "",
    userEmail: "",
    userRoles: [],
    status: null,
  };
}

type RuleKey = Extract<keyof Model, "userName" | "status" | "password">;

const rules: Record<RuleKey, App.Global.FormRule> = {
  deptIds: defaultRequiredRule,
  userName: createConfirmUserNameRule(model.userName),
  password: defaultRequiredRule,
  status: defaultRequiredRule,
  userRoles: defaultRequiredRule,
  nickName: defaultRequiredRule,
};

/** the enabled role options */
const roleOptions = ref<CommonType.Option<string>[]>([]);

async function getRoleOptions() {
  const { error, data } = await fetchGetRoles();

  if (!error) {
    const options = data.map((item) => ({
      label: item.roleName,
      value: item.id,
    }));

    // the mock data does not have the roleCode, so fill it
    // if the real request, remove the following code
    const userRoleOptions = model.userRoles.map((item) => ({
      label: item,
      value: item,
    }));
    // end
    roleOptions.value = [...options];
  }
}

// 部门列表
const deptSelects = ref<any>([]);
async function getDeptSelects() {
  const res = await fetchGetDeptSelect();
  deptSelects.value = res.data.map((val: any) => {
    return {
      label: val.deptName,
      value: val.id,
    };
  });
}

function handleInitModel() {
  Object.assign(model, createDefaultModel());

  if (props.operateType === "edit" && props.rowData) {
    Object.assign(model, props.rowData, {
      userRoles: Array.isArray(props.rowData.userRoles)
        ? props.rowData.userRoles
        : props.rowData.userRoles.split(",").map((val) => +val),
      deptIds: Array.isArray(props.rowData.deptIds)
        ? props.rowData.deptIds
        : props.rowData.deptIds.split(",").map((val) => +val),
      userGender: `${props.rowData.userGender}`,
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
    userRoles: model.userRoles.join(","),
    deptIds: model.deptIds.join(","),
    userGender: +model.userGender,
    status: +model.status,
  });
  let res: any;
  if (props.operateType === "edit") {
    res = await fetchUserEdit(mapModel);
  }
  if (props.operateType === "add") {
    res = await fetchUserAdd(mapModel);
  }
  if (res.data >= 1) {
    window.$message?.success(
      props.operateType === "add" ? "新增成功" : $t("common.updateSuccess"),
    );
    closeDrawer();
    emit("submitted");
  }
}

watch(visible, () => {
  if (visible.value) {
    handleInitModel();
    restoreValidation();
    getRoleOptions();
    getDeptSelects();
  }
});
</script>

<template>
  <NDrawer v-model:show="visible" display-directive="show" :width="360">
    <NDrawerContent :title="title" :native-scrollbar="false" closable>
      <NForm ref="formRef" :model="model" :rules="rules">
        <NFormItem label="部门" path="deptIds">
          <NSelect v-model:value="model.deptIds" multiple :options="deptSelects" placeholder="请选择部门" />
        </NFormItem>
        <NFormItem :label="$t('page.manage.user.userName')" path="userName">
          <NInput v-model:value="model.userName" :placeholder="$t('page.manage.user.form.userName')" />
        </NFormItem>
        <NFormItem :label="$t('page.manage.user.password')" path="password">
          <NInput :disabled="props.operateType === 'edit'" v-model:value="model.password"
            :placeholder="$t('page.manage.user.form.password')" type="password" show-password-on="click" />
        </NFormItem>
        <NFormItem :label="$t('page.manage.user.userGender')" path="userGender">
          <NRadioGroup v-model:value="model.userGender">
            <NRadio v-for="item in userGenderOptions" :key="item.value" :value="item.value" :label="$t(item.label)" />
          </NRadioGroup>
        </NFormItem>
        <NFormItem :label="$t('page.manage.user.nickName')" path="nickName">
          <NInput v-model:value="model.nickName" :placeholder="$t('page.manage.user.form.nickName')" />
        </NFormItem>
        <NFormItem :label="$t('page.manage.user.userPhone')" path="userPhone">
          <NInput v-model:value="model.userPhone" :placeholder="$t('page.manage.user.form.userPhone')" />
        </NFormItem>
        <NFormItem :label="$t('page.manage.user.userEmail')" path="email">
          <NInput v-model:value="model.userEmail" :placeholder="$t('page.manage.user.form.userEmail')" />
        </NFormItem>
        <NFormItem :label="$t('page.manage.user.userStatus')" path="status">
          <NRadioGroup v-model:value="model.status">
            <NRadio v-for="item in enableStatusOptions" :key="item.value" :value="item.value" :label="$t(item.label)" />
          </NRadioGroup>
        </NFormItem>
        <NFormItem :label="$t('page.manage.user.userRole')" path="userRoles">
          <NSelect v-model:value="model.userRoles" multiple :options="roleOptions"
            :placeholder="$t('page.manage.user.form.userRole')" />
        </NFormItem>
      </NForm>
      <template #footer>
        <NSpace :size="16">
          <NButton @click="closeDrawer">{{ $t("common.cancel") }}</NButton>
          <NButton type="primary" @click="handleSubmit">{{
            $t("common.confirm")
            }}</NButton>
        </NSpace>
      </template>
    </NDrawerContent>
  </NDrawer>
</template>

<style scoped></style>
