<script setup lang="ts">
import { computed } from "vue";
import type { VNode } from "vue";
import { useAuthStore } from "@/store/modules/auth";
import { useRouterPush } from "@/hooks/common/router";
import { useSvgIcon } from "@/hooks/common/icon";
import { $t } from "@/locales";
import { NModal, NForm, NFormItem, NInput, NSpace, NButton } from 'naive-ui';
import { fetchResetPassword } from "@/service/api/system-manage";
import { useFormRules, useNaiveForm } from '@/hooks/common/form';

defineOptions({
  name: "UserAvatar",
});

const authStore = useAuthStore();
const { routerPushByKey, toLogin } = useRouterPush();
const { SvgIconVNode } = useSvgIcon();

function loginOrRegister() {
  toLogin();
}

type DropdownKey = "user-center" | "logout";

type DropdownOption =
  | {
    key: DropdownKey;
    label: string;
    icon?: () => VNode;
  }
  | {
    type: "divider";
    key: string;
  };

const options = computed(() => {
  const opts: DropdownOption[] = [
    // {
    //   label: $t('common.userCenter'),
    //   key: 'user-center',
    //   icon: SvgIconVNode({ icon: 'ph:user-circle', fontSize: 18 })
    // },
    {
      label: "重置密码",
      key: 'reset',
      icon: SvgIconVNode({ localIcon: 'material-symbols--reset-tv', fontSize: 18 })
    },
    {
      type: "divider",
      key: "divider",
    },
    {
      label: $t("common.logout"),
      key: "logout",
      icon: SvgIconVNode({ icon: "ph:sign-out", fontSize: 18 }),
    },
  ];

  return opts;
});

function logout() {
  window.$dialog?.info({
    title: $t("common.tip"),
    content: $t("common.logoutConfirm"),
    positiveText: $t("common.confirm"),
    negativeText: $t("common.cancel"),
    onPositiveClick: () => {
      authStore.resetStore();
    },
  });
};

function resetPasswd() {
}


function handleDropdown(key: string) {
  if (key === "logout") {
    logout();
  } else if (key === 'reset') {
    openResetPasswordModal();
  } else {
    routerPushByKey(key);
  }

}

const showResetPasswordModal = ref(false);

const { formRef, validate } = useNaiveForm();

interface FormModel {
  userId: string;
  userName: string;
  password: string;
  confirmPassword: string;
}

const model: FormModel = reactive({
  userId: '',
  userName: '',
  password: '',
  confirmPassword: ''
});


type RuleRecord = Partial<Record<keyof FormModel, App.Global.FormRule[]>>;

const rules = computed<RuleRecord>(() => {
  const { formRules, createConfirmPwdRule } = useFormRules();

  return {
    phone: formRules.phone,
    password: formRules.pwd,
    confirmPassword: createConfirmPwdRule(model.password)
  };
});

function openResetPasswordModal() {
  model.userId = authStore.userInfo.id;
  model.userName = authStore.userInfo.nickName;
  showResetPasswordModal.value = true;
}

function closeResetPasswordModal() {
  showResetPasswordModal.value = false;
}

async function handleResetPasswordSubmit() {
  await validate();
  const { userId, password, confirmPassword } = model;
  const res = await fetchResetPassword({ userId, password, confirmPassword });
  if (res.data) {
    closeResetPasswordModal();
    window.$message?.success('重置密码成功！');
    authStore.resetStore();
  }
}

</script>

<template>
  <NButton v-if="!authStore.isLogin" quaternary @click="loginOrRegister">
    {{ $t("page.login.common.loginOrRegister") }}
  </NButton>
  <NDropdown v-else placement="bottom" trigger="click" :options="options" @select="handleDropdown">
    <div>
      <ButtonIcon>
        <SvgIcon local-icon="ph--user-circle" class="text-icon-large" />
        <span class="text-16px font-medium">{{ authStore.userInfo.nickName }}</span>
      </ButtonIcon>
    </div>
  </NDropdown>
  <NModal v-model:show="showResetPasswordModal" title="重置密码" :mask-closable="false" preset="card" class="w-600px">
    <NForm ref="formRef" :model="model" :rules="rules" size="large" :show-label="false">
      <!-- 表单项 -->
      <NFormItem>
        <NInput v-model:value="model.userName" disabled />
      </NFormItem>
      <NFormItem path="password">
        <NInput v-model:value="model.password" type="password" show-password-on="click"
          :placeholder="$t('page.login.common.passwordPlaceholder')" />
      </NFormItem>
      <NFormItem path="confirmPassword">
        <NInput v-model:value="model.confirmPassword" type="password" show-password-on="click"
          :placeholder="$t('page.login.common.confirmPasswordPlaceholder')" />
      </NFormItem>
      <NSpace vertical :size="18" class="w-full">
        <NButton type="primary" size="large" round block @click="handleResetPasswordSubmit">
          {{ $t('common.confirm') }}
        </NButton>
        <NButton size="large" round block @click="closeResetPasswordModal">
          {{ $t('common.cancel') }}
        </NButton>
      </NSpace>
    </NForm>
  </NModal>
</template>

<style scoped></style>
