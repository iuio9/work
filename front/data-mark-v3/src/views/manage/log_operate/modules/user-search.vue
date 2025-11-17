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

const model = defineModel<Api.SystemManage.UserSearchParams>('model', { required: true });

type RuleKey = Extract<keyof Api.SystemManage.UserSearchParams, 'userEmail' | 'userPhone'>;

const rules = computed<Record<RuleKey, App.Global.FormRule>>(() => {
  const { patternRules } = useFormRules(); // inside computed to make locale reactive

  return {
    userEmail: patternRules.email,
    userPhone: patternRules.phone
  };
});

const options: any = [{
  label: "成功",
  value: "0"
}, {
  label: "失败",
  value: "1"
}]
const operateTypes = [
  { value: "0", label: "其它"},
  { value: "1", label: "新增"},
  { value: "2", label: "修改"},
  { value: "3", label: "删除"},
]

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
  <NCard :title="$t('common.search')" :bordered="false" size="small" class="card-wrapper">
    <NForm ref="formRef" :model="model" :rules="rules" label-placement="left" :label-width="80">
      <NGrid responsive="screen" item-responsive>
        <NFormItemGi span="24 s:12 m:6" label="操作模块" path="title" class="pr-24px">
          <NInput v-model:value="model.title" placeholder="请输入操作模块" />
        </NFormItemGi>
        <NFormItemGi span="24 s:12 m:6" label="操作人员" path="dictName" class="pr-24px">
          <NInput v-model:value="model.operName" placeholder="请输入操作人员" />
        </NFormItemGi>
        <NFormItemGi span="24 s:12 m:6" label="类型" path="status" class="pr-24px">
          <NSelect
            v-model:value="model.businessType"
            placeholder="请选择类型"
            :options="operateTypes"
            clearable
          />
        </NFormItemGi>
        <NFormItemGi span="24 s:12 m:6" label="状态" path="status" class="pr-24px">
          <NSelect
            v-model:value="model.status"
            placeholder="请选择状态"
            :options="options"
            clearable
          />
        </NFormItemGi>
        <NFormItemGi span="24 s:12 m:12" label="操作时间" path="status" class="pr-24px">
          <n-date-picker v-model:value="model.timeArr" type="datetimerange" clearable />
        </NFormItemGi>
        <NFormItemGi span="24 s:12 m:12" class="pr-24px">
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
