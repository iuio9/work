<script setup lang="ts">
import { computed } from 'vue';
import { $t } from '@/locales';
import { useFormRules, useNaiveForm } from '@/hooks/common/form';
import { enableStatusOptions, userGenderOptions, yesOrNoOptions } from '@/constants/business';
import { translateOptions } from '@/utils/common';

defineOptions({
  name: 'UserSearch'
});

interface Emits {
  (e: 'reset'): void;
  (e: 'search'): void;
}

const emit = defineEmits<Emits>();

const { formRef, validate, restoreValidation } = useNaiveForm();

const model = defineModel<Api.SystemManage.UserSearchParams>('model', {
  required: true
});

type RuleKey = Extract<keyof Api.SystemManage.UserSearchParams, 'userEmail' | 'userPhone'>;

const rules = computed<Record<RuleKey, App.Global.FormRule>>(() => {
  const { patternRules } = useFormRules(); // inside computed to make locale reactive

  return {
    userEmail: patternRules.email,
    userPhone: patternRules.phone
  };
});

async function reset() {
  await restoreValidation();
  emit('reset');
}

async function search() {
  await validate();
  emit('search');
}
</script>

<template>
  <NCard :bordered="false" size="small" class="card-wrapper">
    <NCollapse>
      <NCollapseItem :title="$t('common.search')" name="user-search">
        <template #header-extra>
          <div class="w-auto flex items-center gap-24px">
            <div class="flex items-center justify-center">
              <span class="font-600">数据集名称：</span>
              <span>测试数据集</span>
            </div>
            <div class="flex items-center justify-center">
              <span class="font-600">数据集版本：</span>
              <span>V1</span>
            </div>
          </div>
        </template>
        <NForm ref="formRef" :model="model" :rules="rules" label-placement="left" :label-width="84">
          <NGrid responsive="screen" item-responsive>
            <NFormItemGi span="24 s:12 m:6" label="场景名称" path="tagGroupName" class="pr-24px">
              <NInput v-model:value="model.userName" placeholder="请输入场景名称" />
            </NFormItemGi>
            <NFormItemGi span="24 s:12 m:6" label="标签名称" path="tagName" class="pr-24px">
              <NInput v-model:value="model.nickName" placeholder="请输入标签名称" />
            </NFormItemGi>
            <NFormItemGi span="24 s:12 m:6" label="是否使用" path="userStatus" class="pr-24px">
              <NSelect
                v-model:value="model.status"
                placeholder="请选择"
                :options="translateOptions(yesOrNoOptions)"
                clearable
              />
            </NFormItemGi>
            <NFormItemGi span="24 m:6" class="pr-24px">
              <NSpace class="w-full" justify="end">
                <NButton @click="reset">
                  <template #icon>
                    <icon-ic-round-refresh class="text-icon" />
                  </template>
                  {{ $t('common.reset') }}
                </NButton>
                <NButton type="primary" ghost @click="search">
                  <template #icon>
                    <icon-ic-round-search class="text-icon" />
                  </template>
                  {{ $t('common.search') }}
                </NButton>
              </NSpace>
            </NFormItemGi>
          </NGrid>
        </NForm>
      </NCollapseItem>
    </NCollapse>
  </NCard>
</template>

<style scoped></style>
