<script setup lang="ts">
import { computed } from "vue";
import { $t } from "@/locales";
import { useFormRules, useNaiveForm } from "@/hooks/common/form";

defineOptions({
  name: "UserSearch",
});

interface Emits {
  (e: "reset"): void;
  (e: "search"): void;
}

const emit = defineEmits<Emits>();

const { formRef, validate, restoreValidation } = useNaiveForm();

const model = defineModel<Api.SystemManage.UserSearchParams>("model", {
  required: true,
});

const modelOptions = ref<any>([
  { value: "1", label: "模型1"},
  { value: "2", label: "模型2"},
  { value: "3", label: "模型3"},
])

type RuleKey = Extract<
  keyof Api.SystemManage.UserSearchParams,
  "userEmail" | "userPhone"
>;

const rules = computed<Record<RuleKey, App.Global.FormRule>>(() => {
  const { patternRules } = useFormRules(); // inside computed to make locale reactive

  return {
    userEmail: patternRules.email,
    userPhone: patternRules.phone,
  };
});

async function reset() {
  await restoreValidation();
  emit("reset");
}

async function search() {
  await validate();
  emit("search");
}
</script>

<template>
  <NCard
    :title="$t('common.search')"
    :bordered="false"
    size="small"
    class="card-wrapper"
  >
    <NForm
      ref="formRef"
      :model="model"
      :rules="rules"
      label-placement="left"
      :label-width="80"
    >
      <NGrid responsive="screen" item-responsive>
        <NFormItemGi
          span="24 s:12 m:6"
          label="任务名称"
          path="taskInputName"
          class="pr-24px"
        >
          <NInput
            v-model:value="model.taskInputName"
            placeholder="请输入任务名称"
          />
        </NFormItemGi>
        <NFormItemGi span="24 s:12 m:6" label="模型名称" path="userStatus" class="pr-24px">
          <NSelect
            v-model:value="model.modelName"
            placeholder="请选择模型名称"
            :options="modelOptions"
            clearable
          />
        </NFormItemGi>
        <NFormItemGi span="24 m:12" class="pr-24px">
          <NSpace class="w-full" justify="end">
            <NButton @click="reset">
              <template #icon>
                <icon-ic-round-refresh class="text-icon" />
              </template>
              {{ $t("common.reset") }}
            </NButton>
            <NButton type="primary" ghost @click="search">
              <template #icon>
                <icon-ic-round-search class="text-icon" />
              </template>
              {{ $t("common.search") }}
            </NButton>
          </NSpace>
        </NFormItemGi>
      </NGrid>
    </NForm>
  </NCard>
</template>

<style scoped></style>
