<script setup lang="ts">
import { computed, reactive, watch } from 'vue';
import { useFormRules, useNaiveForm } from '@/hooks/common/form';
import { $t } from '@/locales';
import { getLocalIcons } from '@/utils/icon';
import { addExample, updateExample } from "@/service/api/dataManage";
import {
  fetchGetDictList,
} from "@/service/api";
import { useTable, useTableOperate } from "@/hooks/common/table";
import { NButton } from 'naive-ui';

defineOptions({
  name: 'MenuOperateModal'
});

export type OperateType = NaiveUI.TableOperateType | 'addChild';

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
  (e: 'submitted'): void;
}

const emit = defineEmits<Emits>();

const visible = defineModel<boolean>('visible', {
  default: false
});

const { formRef, validate, restoreValidation } = useNaiveForm();
const { defaultRequiredRule } = useFormRules();

const title = computed(() => {
  const titles: Record<OperateType, string> = {
    add: "新增算法",
    addChild: $t('page.manage.menu.addChildMenu'),
    edit: "编辑算法"
  };
  return titles[props.operateType];
});

type Model = Pick<
  Api.SystemManage.Menu,
  | 'menuType'
  | 'menuName'
  | 'routeName'
  | 'routePath'
  | 'component'
  | 'order'
  | 'i18nKey'
  | 'icon'
  | 'iconType'
  | 'status'
  | 'parentId'
  | 'keepAlive'
  | 'constant'
  | 'href'
  | 'hideInMenu'
  | 'activeMenu'
  | 'multiTab'
  | 'fixedIndexInTab'
> & {
  query: NonNullable<Api.SystemManage.Menu['query']>;
  buttons: NonNullable<Api.SystemManage.Menu['buttons']>;
  layout: string;
  page: string;
  pathParam: string;
};

const model: Model = reactive(createDefaultModel());

function createDefaultModel(): Model {
  return {
    algorithmName: null,
    url: null,
    requestType: null,
    params: null,
    algorithmDesc: null,
    query: [],
    responseQuery: []
  };
}

type RuleKey = Extract<keyof Model, 'menuName' | 'status' | 'routeName' | 'routePath'>;

const rules: Record<RuleKey, App.Global.FormRule> = {
  algorithmName: defaultRequiredRule,
  url: defaultRequiredRule,
  requestType: defaultRequiredRule,
};

/** the enabled role options */
function handleInitModel() {
  Object.assign(model, createDefaultModel());

  if (!props.rowData) return;

  if (props.operateType === 'edit') {
    Object.assign(model, props.rowData);
  }

  if (!model.query) {
    model.query = [];
  }
  if (!model.responseQuery) {
    model.responseQuery = [];
  }
}

function closeDrawer() {
  visible.value = false;
}

function getSubmitParams() {
  const { layout, page, pathParam, ...params } = model;
  return params;
}

async function handleSubmit() {
  await validate();
  const submitParams = getSubmitParams();
  submitParams.paramsMap = submitParams.query.map(val => {
    return {
      ...val,
      type: val.type,
      serverKey: val.key,
      // label: val.value,
      // value: val.key,
    }
  });
  submitParams.params = JSON.stringify(submitParams.paramsMap);
  submitParams.responseParamsMap = submitParams.responseQuery.map(val => {
    return {
      ...val,
      type: val.type,
      serverKey: val.key,
      // label: val.value,
      // value: val.key,
    }
  });
  submitParams.responseParams = JSON.stringify(submitParams.responseParamsMap);
  const { index, query, responseQuery, ...rest } = submitParams;
  // request
  let res;
  if (props.operateType === "add") {
    res = await addExample(rest);
  };
  if (props.operateType === "edit") {
    res = await updateExample(rest);
  };
  if (res.data >= 0) {
    window.$message?.success($t('common.updateSuccess'));
    closeDrawer();
    emit('submitted');
  }
}

const handleIptCreate = () => {
  const rowData = {
    key: "", value: "", type: "",
    valuePlaceholder: "请输入参数Value", keyPlaceholder: "请输入参数Key",
    dictId: undefined, // 字典ID
    isRelDict: false // 是否关联字典
  };
  return rowData;
}
const handleCreate = (index: any) => {
  const rowData = {
    key: "", value: "", type: "",
    dictId: undefined, // 字典ID
    isRelDict: false // 是否关联字典
  };
  model.query.splice(index + 1, 0, rowData);
};
const handleRemove = (index: any) => {
  model.query.splice(index, 1);
};
const handleRel = (index: any) => {
  isShowDict.value = true;
  operateIdx.value = index;
};

const handleIptCreate1 = () => {
  const rowData = {
    key: "", value: "", type: "",
    valuePlaceholder: "请输入参数Value", keyPlaceholder: "请输入参数Key",
    dictId: undefined, // 字典ID
    isRelDict: false // 是否关联字典
  };
  return rowData;
}
const handleCreate1 = (index: any) => {
  const rowData = {
    key: "", value: "", type: "",
    dictId: undefined, // 字典ID
    isRelDict: false // 是否关联字典
  };
  model.responseQuery.splice(index + 1, 0, rowData);
};
const handleRemove1 = (index: any) => {
  model.responseQuery.splice(index, 1);
};
watch(visible, () => {
  if (visible.value) {
    handleInitModel();
    restoreValidation();
  }
});

// newCode
const isShowDict = ref<Boolean>(false);
const relModel = ref<any>({
  dictId: "",
})
const handleIptUpdate = (event, index) => {
  if (event === "select") {
    model.query[index].isRelDict = true;
  } else {
    model.query[index].isRelDict = false;
  }
}

const {
  columns,
  columnChecks,
  data,
  getData,
  getDataByPage,
  loading,
  mobilePagination,
  searchParams,
  resetSearchParams,
} = useTable({
  immediate: undefined,
  apiFn: fetchGetDictList,
  showTotal: true,
  apiParams: {
    current: 1,
    size: 10,
    dictName: "",
  },
  columns: () => [
    {
      type: "selection",
      align: "center",
      multiple: false,
    },
    {
      key: "id",
      title: "字典ID",
      align: "center",
      width: 60,
    },
    {
      key: "dictName",
      title: "字典名称",
      align: "center",
      width: 160,
      render: (row: any) => {
        return [
          h('div', {}, [
            h(NButton, {
              quaternary: true,
              type: 'info',
              onClick: () => navigateToDict(row)
            }, () => `${row.dictName}`)
          ])
        ]
      }
    },
    {
      key: "remark",
      title: "备注信息",
      align: "left",
    },
  ]
});

const {
  drawerVisible,
  operateType,
  editingData,
  handleAdd,
  handleEdit,
  checkedRowKeys,
  onBatchDeleted,
  onDeleted,
} = useTableOperate(data, getData);

const operateIdx = ref<string | number>(0);

const handleRelSubmit = () => {
  if (checkedRowKeys.value && checkedRowKeys.value.length === 1) {
    model.query[operateIdx.value].dictId = checkedRowKeys.value[0];
    model.query[operateIdx.value].isRelDict = true;
    isShowDict.value = false;
  }
};
</script>

<template>
  <NModal v-model:show="visible" :title="title" preset="card" class="w-900px">
    <NScrollbar class="h-480px pr-20px">
      <NForm ref="formRef" :model="model" :rules="rules" label-placement="left" :label-width="100">
        <NFormItem label="算法名称" path="algorithmName">
          <NInput v-model:value="model.algorithmName" placeholder="请输入算法名称" />
        </NFormItem>
        <NFormItem label="请求地址" path="url">
          <NInput v-model:value="model.url" placeholder="请输入算法业务类型" />
        </NFormItem>
        <NFormItem label="请求类型" path="requestType">
          <NInput v-model:value="model.requestType" placeholder="请输入算法请求类型（如：get | post | delete等）" />
        </NFormItem>
        <NFormItem span="24" label="请求配置">
          <NDynamicInput v-model:value="model.query" preset="pair" key-placeholder="请输入参数Key"
            value-placeholder="请输入参数Value" @create="handleIptCreate">
            <template #default="{ index }">
              <div class="w-[76%] flex items-start gap-8px">
                <NInput v-model:value="model.query[index].key" placeholder="请输入参数Key"></NInput>
                <NInput v-model:value="model.query[index].label" :placeholder="!!model.query[index].valuePlaceholder
                  ? model.query[index].valuePlaceholder : '请输入参数Value'" class="!w-120%"></NInput>
                <NInput class="!w-80%"  v-model:value="model.query[index].type" placeholder="请输入参数类型"
                  @update:value="handleIptUpdate($event, index)">
                </NInput>
              </div>
            </template>
            <template #action="{ index, value }">
              <NSpace class="ml-12px flex justify-around items-center">
                <NButton size="medium" @click="() => handleCreate(index)">
                  <icon-ic:round-plus class="text-icon" />
                </NButton>
                <NButton size="medium" @click="() => handleRemove(index)">
                  <icon-ic-round-remove class="text-icon" />
                </NButton>
                <n-tooltip trigger="hover">
                  <template #trigger>
                    <NButton size="medium" @click="() => handleRel(index)" :disabled="!model.query[index].isRelDict">
                      <SvgIcon localIcon="tdesign--relation" class="text-16px text-#000" />
                    </NButton>
                  </template>
                  关联字典表
                </n-tooltip>
              </NSpace>
            </template>
          </NDynamicInput>
        </NFormItem>
        <NFormItem span="24" label="响应配置">
          <NDynamicInput v-model:value="model.responseQuery" preset="pair" key-placeholder="请输入参数Key"
            value-placeholder="请输入参数Value" @create="handleIptCreate1">
            <template #default="{ index }">
              <div class="w-[76%] flex items-start gap-8px">
                <NInput v-model:value="model.responseQuery[index].key" placeholder="请输入参数Key"></NInput>
                <NInput class="!w-120%" v-model:value="model.responseQuery[index].label"
                  :placeholder="!!model.responseQuery[index].valuePlaceholder ? model.responseQuery[index].valuePlaceholder : '请输入参数Value'">
                </NInput>
                <NInput class="!w-80%" v-model:value="model.responseQuery[index].type" placeholder="请输入参数类型"></NInput>
              </div>
            </template>
            <template #action="{ index, value }">
              <NSpace class="ml-12px">
                <NButton size="medium" @click="() => handleCreate1(index)">
                  <icon-ic:round-plus class="text-icon" />
                </NButton>
                <NButton size="medium" @click="() => handleRemove1(index)">
                  <icon-ic-round-remove class="text-icon" />
                </NButton>
              </NSpace>
            </template>
          </NDynamicInput>
        </NFormItem>
        <NFormItem label="算法描述" path="modelDesc">
          <NInput type="textarea" v-model:value="model.algorithmDesc" placeholder="请输入算法描述" />
        </NFormItem>
      </NForm>
    </NScrollbar>
    <template #footer>
      <NSpace justify="end" :size="16">
        <NButton @click="closeDrawer">{{ $t('common.cancel') }}</NButton>
        <NButton type="primary" @click="handleSubmit">{{ $t('common.confirm') }}</NButton>
      </NSpace>
    </template>
  </NModal>
  <n-modal v-model:show="isShowDict" title="关联字典表ID" preset="card" class="w-800px">
    <NScrollbar class="h-auto pr-20px">
      <NDataTable v-model:checked-row-keys="checkedRowKeys" :columns="columns" :data="data" size="small"
        :loading="loading" remote :row-key="(row) => row.id" :pagination="mobilePagination" class="sm:h-full" />
    </NScrollbar>
    <template #footer>
      <NSpace justify="end" :size="16">
        <NButton @click="() => isShowDict = false">关闭</NButton>
        <NButton type="primary" @click="handleRelSubmit">{{ $t('common.confirm') }}</NButton>
      </NSpace>
    </template>
  </n-modal>
</template>

<style scoped></style>
