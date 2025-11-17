<script setup lang="ts">
import { computed } from "vue";
import { $t } from "@/locales";
import { useFormRules, useNaiveForm } from "@/hooks/common/form";
import { enableStatusOptions, userGenderOptions } from "@/constants/business";
import { translateOptions } from "@/utils/common";
import { getDataSetListNoPage } from "@/service/api/expansion";
import { NPopover } from "naive-ui";

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

const setOptions = ref<any>([]);

async function reset() {
  await restoreValidation();
  emit("reset");
}

async function search() {
  await validate();
  emit("search");
}

function renderLabel(option: { value?: string | number, label?: string }) {
  return [
    h("div", {
      class: "flex items-center"
    }, [
      h(NPopover, { trigger: "hover", placement: "top" }, {
        trigger: () => [
          h("span", { class: "truncate" }, `${option.label}`)
        ],
        default: () => [
          h("span", {}, `${option.label}`)
        ]
      })
    ])
  ]
}

// 数据集列表接口 noPage
async function getGroupList() {
  const recursionMapData = (data: any, label: any) => {
    const mapList = data.map((item: any, index: string | number) => {
      item.value = item.groupId || item.sonId;
      if (label) {
        // item.label = `${label} - ${item.groupName || `V${item.version}`}`;
        item.label = `${item.groupName || `V${item.version}`}`;
      } else {
        item.label = item.groupName || `V${item.version}`;
      }
      // item.label = item.groupName || `V${item.version}`;
      const children = item.dataSonResponseList || [];
      item.children = children.map((val: any) => {
        // 演示环境
        item.disabled = false;
        // val.disabled = val.count > 0 && val.progress == 100 ? false : true; // 正式环境
        val.disabled = false; // 演示环境
        return val;
      });
      if (item.children && item.children.length > 0) {
        recursionMapData(item.children, item.label);
      } else {
        delete item.children;
      }
      return item;
    });
    return mapList;
  };
  const res = await getDataSetListNoPage();
  const options = recursionMapData(res.data);
  setOptions.value = options;
}

onMounted(() => {
  getGroupList();
});
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
      :label-width="100"
    >
      <NGrid responsive="screen" item-responsive>
        <NFormItemGi
          span="24 s:12 m:6"
          label="数据集名称"
          path="userName"
          class="pr-24px"
        >
          <NCascader
            v-model:value="model.dataTypeId"
            clearable
            expand-trigger="hover"
            check-strategy="child"
            placeholder="请选择数据集"
            :options="setOptions"
            :render-label="renderLabel"
          ></NCascader>
        </NFormItemGi>
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
        <NFormItemGi
          span="24 s:12 m:6"
          label="模型名称"
          path="modelName"
          class="pr-24px"
        >
          <NInput
            v-model:value="model.modelName"
            placeholder="请输入模型名称"
          />
        </NFormItemGi>
        <NFormItemGi span="24 m:6" class="pr-24px">
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
