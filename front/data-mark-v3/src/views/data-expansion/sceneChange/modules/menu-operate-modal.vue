<script setup lang="tsx">
import { computed, reactive, ref, watch } from "vue";
import type { SelectOption } from "naive-ui";
import { useFormRules, useNaiveForm } from "@/hooks/common/form";
import { $t } from "@/locales";
import {
  enableStatusOptions,
  enableStatusRecord1,
  menuIconTypeOptions,
  menuTypeOptions,
} from "@/constants/business";
import SvgIcon from "@/components/custom/svg-icon.vue";
import { getLocalIcons } from "@/utils/icon";
import { fetchGetAllRoles } from "@/service/api";
import {
  getLayoutAndPage,
  getPathParamFromRoutePath,
  getRoutePathByRouteName,
  getRoutePathWithParam,
  transformLayoutAndPageToComponent,
} from "./shared";
import { localStg } from "@/utils/storage";

defineOptions({
  name: "MenuOperateModal",
});

export type OperateType = NaiveUI.TableOperateType | "addChild";

interface Props {
  /** the type of operation */
  operateType: OperateType;
  /** the edit menu data or the parent menu data when adding a child menu */
  rowData?: Api.SystemManage.Menu | null;
  /** all pages */
  allPages: string[];
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
const { defaultRequiredRule } = useFormRules();

const title = computed(() => {
  const titles: Record<OperateType, string> = {
    add: $t("page.manage.menu.addMenu"),
    addChild: $t("page.manage.menu.addChildMenu"),
    edit: $t("page.manage.menu.editMenu"),
  };
  return titles[props.operateType];
});

type Model = Pick<
  Api.SystemManage.Menu,
  | "menuType"
  | "menuName"
  | "routeName"
  | "routePath"
  | "component"
  | "order"
  | "i18nKey"
  | "icon"
  | "iconType"
  | "status"
  | "parentId"
  | "keepAlive"
  | "constant"
  | "href"
  | "hideInMenu"
  | "activeMenu"
  | "multiTab"
  | "fixedIndexInTab"
> & {
  query: NonNullable<Api.SystemManage.Menu["query"]>;
  buttons: NonNullable<Api.SystemManage.Menu["buttons"]>;
  layout: string;
  page: string;
  pathParam: string;
};

const model: Model = reactive(createDefaultModel());

function createDefaultModel(): Model {
  return {
    taskName: null,
    taskProgress: null,
    taskStat: null,
    params: null,
  };
}

type RuleKey = Extract<
  keyof Model,
  "menuName" | "status" | "routeName" | "routePath"
>;

const rules: Record<RuleKey, App.Global.FormRule> = {
  menuName: defaultRequiredRule,
  status: defaultRequiredRule,
  routeName: defaultRequiredRule,
  routePath: defaultRequiredRule,
};

const disabledMenuType = computed(() => props.operateType === "edit");

const localIcons = getLocalIcons();
const localIconOptions = localIcons.map<SelectOption>((item) => ({
  label: () => (
    <div class="flex-y-center gap-16px">
      <SvgIcon localIcon={item} class="text-icon" />
      <span>{item}</span>
    </div>
  ),
  value: item,
}));

const showLayout = computed(() => model.parentId === 0);

const showPage = computed(() => model.menuType === "2");

const pageOptions = computed(() => {
  const allPages = [...props.allPages];

  if (model.routeName && !allPages.includes(model.routeName)) {
    allPages.unshift(model.routeName);
  }

  const opts: CommonType.Option[] = allPages.map((page) => ({
    label: page,
    value: page,
  }));

  return opts;
});

const layoutOptions: CommonType.Option[] = [
  {
    label: "base",
    value: "base",
  },
  {
    label: "blank",
    value: "blank",
  },
];

/** the enabled role options */
const roleOptions = ref<CommonType.Option<string>[]>([]);

async function getRoleOptions() {
  const { error, data } = await fetchGetAllRoles();

  if (!error) {
    const options = data.map((item) => ({
      label: item.roleName,
      value: item.roleCode,
    }));

    roleOptions.value = [...options];
  }
}

function handleInitModel() {
  Object.assign(model, createDefaultModel(), props.rowData);
}

function closeDrawer() {
  visible.value = false;
}

function handleUpdateRoutePathByRouteName() {
  if (model.routeName) {
    model.routePath = getRoutePathByRouteName(model.routeName);
  } else {
    model.routePath = "";
  }
}

function handleUpdateI18nKeyByRouteName() {
  if (model.routeName) {
    model.i18nKey = `route.${model.routeName}` as App.I18n.I18nKey;
  } else {
    model.i18nKey = null;
  }
}

function handleCreateButton() {
  const buttonItem: Api.SystemManage.MenuButton = {
    code: "",
    desc: "",
  };

  return buttonItem;
}

function getSubmitParams() {
  const { layout, page, pathParam, ...params } = model;

  const component = transformLayoutAndPageToComponent(layout, page);
  const routePath = getRoutePathWithParam(model.routePath, pathParam);

  params.component = component;
  params.routePath = routePath;

  return params;
}

async function handleSubmit() {
  await validate();

  const params = getSubmitParams();

  console.log("params: ", params);

  // request
  window.$message?.success($t("common.updateSuccess"));
  closeDrawer();
  emit("submitted");
}

watch(visible, () => {
  if (visible.value) {
    handleInitModel();
    restoreValidation();
    // getRoleOptions();
  }
});

watch(
  () => model.routeName,
  () => {
    handleUpdateRoutePathByRouteName();
    handleUpdateI18nKeyByRouteName();
  },
);

const tagMap: Record<Api.Common.EnableStatus, NaiveUI.ThemeColor> = {
  0: "success",
  1: "error",
};

const router = useRouter();
const navToMap = (datasetOutId: any) => {
  router.push({
    name: "data-manage_map",
    query: {
      datasetOutId,
    },
  });
};

const themeColor = localStg.get("themeColor") || "#646cff";
</script>

<template>
  <NModal v-model:show="visible" title="详情" preset="card" class="w-800px">
    <NScrollbar class="h-280px pr-20px">
      <NForm
        ref="formRef"
        :model="model"
        :rules="rules"
        label-placement="left"
        :label-width="100"
      >
        <NGrid responsive="screen" item-responsive class="ml-24px">
          <NFormItemGi
            span="24 m:24"
            label="创建时间:"
            path="createTime"
            class="h-42px"
          >
            {{ model.createTime }}
          </NFormItemGi>
          <NFormItemGi
            span="24 m:24"
            label="完成时间:"
            path="updateTime"
            class="h-42px"
          >
            {{ !!model.updateTime ? model.updateTime : "-" }}
          </NFormItemGi>
          <NFormItemGi
            span="24 m:24"
            label="任务名称:"
            path="taskName"
            class="h-42px"
          >
            {{ model.taskName ?? "-" }}
          </NFormItemGi>
          <NFormItemGi
            span="24 m:24"
            label="进度:"
            path="taskProgress"
            class="h-42px"
          >
            {{ model.taskProgress ?? "-" }}
          </NFormItemGi>
          <NFormItemGi
            span="24 m:24"
            label="任务状态:"
            path="taskStat"
            class="h-42px"
          >
            {{ model.taskStat ?? "-" }}
          </NFormItemGi>
          <NFormItemGi
            span="24 m:24"
            label="输出数据集:"
            path="taskStat"
            class="h-42px"
          >
            <!--{{ model.datasetOutId ?? "-"}}-->
            <span>{{ model.datasetOutId ?? "-" }}</span>
            <span
              class="ml-8px text-[12px] text-[#646cff] cursor-pointer"
              :style="{
                color: themeColor,
              }"
              v-show="!!model.datasetOutId"
              @click="navToMap(model.datasetOutId)"
              >点击查看</span
            >
          </NFormItemGi>
        </NGrid>
      </NForm>
    </NScrollbar>
  </NModal>
</template>

<style scoped></style>
