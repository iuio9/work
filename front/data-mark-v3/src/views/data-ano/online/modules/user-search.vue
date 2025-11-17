<script setup lang="ts">
import { computed } from "vue";
import { $t } from "@/locales";
import { useFormRules, useNaiveForm } from "@/hooks/common/form";
import { NCascader, NPopover } from "naive-ui";
import { getSelectDataSetDictList } from "@/service/api/dataManage";
import { getDataSetListNoPage } from "@/service/api/expansion";

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

async function reset() {
  await restoreValidation();
  emit("reset");
}

async function search() {
  await validate();
  emit("search");
}

// 级联筛选
const mapOptions = ref<any>([]);
const setOptions = ref<any>([]);
const recursionData = (data: any, label: any) => {
  // eslint-disable-next-line no-param-reassign
  data = data.map((item: any, index: string | number) => {
    if (item.children) {
      if (item.children.length > 0)
        recursionData(item.children, item.dictLabel);
      if (item.children.length === 0) delete item.children;
    }
    item.label = label ? `${item.dictLabel}` : item.dictLabel;
    item.value = item.id;
    return item;
  });
  return data;
};
async function getMapClassifyList() {
  const res = await getSelectDataSetDictList({ page: 1, limit: 10 });
  mapOptions.value = recursionData(res.data);
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

onMounted(async () => {
  getGroupList();
  await getMapClassifyList();
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
          path="deptName"
          class="pr-24px"
        >
          <NCascader
            v-model:value="model.sonId"
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
          label="数据类型"
          path="deptName"
          class="pr-24px"
        >
          <n-cascader
            v-model:value="model.dataTypeId"
            clearable
            placeholder="类型筛选"
            :options="mapOptions"
            check-strategy="all"
          >
          </n-cascader>
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
