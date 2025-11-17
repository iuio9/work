<script setup lang="ts">
import { computed } from 'vue';
import { $t } from '@/locales';
import { useFormRules, useNaiveForm } from '@/hooks/common/form';

defineOptions({
  name: 'UserSearch'
});

interface Emits {
  (e: 'reset'): void;
  (e: 'search'): void;
}

const emit = defineEmits<Emits>();

const { formRef, validate, restoreValidation } = useNaiveForm();

const model = defineModel<any>('model', { required: true });

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
  model.value.applyTimeArr = null
  emit('reset');
}

async function search() {
  await validate();
  emit('search');
}
</script>

<template>
  <NCard :title="$t('common.search')" :bordered="false" size="small" class="card-wrapper">
    <NForm ref="formRef" :model="model" :rules="rules" label-placement="left" :label-width="100">
      <NGrid responsive="screen" item-responsive>
        <!-- 申请单号 -->
        <NFormItemGi span="24 s:12 m:8" label="申请单号" path="applyForNum" class="pr-24px">
          <NInput v-model:value="model.applyForNum" placeholder="请输入申请单号" />
        </NFormItemGi>
        <NFormItemGi span="24 s:12 m:8" label="模型名称" path="modelName" class="pr-24px">
          <NInput v-model:value="model.modelName" placeholder="请输入模型名称" />
        </NFormItemGi>
        <NFormItemGi span="24 s:12 m:8" label="建设单位名称" path="buildUnitName" class="pr-24px">
          <NInput v-model:value="model.buildUnitName" placeholder="请输入建设单位名称" />
        </NFormItemGi>
        <NFormItemGi span="24 s:12 m:8" label="承建单位名称" path="btUnitName" class="pr-24px">
          <NInput v-model:value="model.btUnitName" placeholder="请输入承建单位名称" />
        </NFormItemGi>
        <NFormItemGi span="24 s:12 m:8" label="申请时间" path="applyTimeArr" class="pr-24px">
          <n-date-picker v-model:value="model.applyTimeArr" type="datetimerange" clearable class="!w-full"/>
        </NFormItemGi>
        <NFormItemGi span="24 m:8" class="pr-24px">
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
  </NCard>
</template>

<style scoped></style>
