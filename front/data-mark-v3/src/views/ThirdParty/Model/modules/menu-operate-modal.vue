<script setup lang="ts">
import { computed, reactive, watch } from 'vue';
import { useFormRules, useNaiveForm } from '@/hooks/common/form';
import { $t } from '@/locales';
import { getLocalIcons } from '@/utils/icon';
import {addExample, updateExample} from "@/service/api/dataManage";

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
  console.log('submitParams: ', submitParams);
  const { index, query, responseQuery, ...rest } = submitParams;
  // request
  let res;
  if(props.operateType === "add") {
    res = await addExample(rest);
  };
  if(props.operateType === "edit") {
    res = await updateExample(rest);
  };
  if(res.data>=0) {
    window.$message?.success($t('common.updateSuccess'));
    closeDrawer();
    emit('submitted');
  }
}

const handleIptCreate = () => {
  const rowData = { key: "", value: "", type: "", valuePlaceholder: "请输入参数Value", keyPlaceholder: "请输入参数Key"};
  return rowData;
}
const handleCreate = (index: any) => {
  const rowData = { key: "", value: "", type: ""};
  model.query.splice(index + 1, 0, rowData);
};
const handleRemove = (index: any) => {
  model.query.splice(index, 1);
};

const handleIptCreate1 = () => {
  const rowData = { key: "", value: "", type: "", valuePlaceholder: "请输入参数Value", keyPlaceholder: "请输入参数Key"};
  return rowData;
}
const handleCreate1 = (index: any) => {
  const rowData = { key: "", value: "", type: ""};
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
</script>

<template>
  <NModal v-model:show="visible" title="模型配置" preset="card" class="w-800px">
    <NScrollbar class="h-auto pr-20px">
      <NForm ref="formRef" :model="model" :rules="rules" label-placement="left" :label-width="120">
        <NFormItem label="模型名称" path="algorithmName">
          <NInput v-model:value="model.algorithmName" placeholder="请输入模型名称" disabled/>
        </NFormItem>
        <NFormItem label="模型请求API" path="url">
          <NInput v-model:value="model.url" placeholder="请输入模型请求API"/>
        </NFormItem>
        <NFormItem label="模型API类型" path="requestType">
          <NInput v-model:value="model.requestType" placeholder="请输入模型请求API类型（如：get | post | delete等）"/>
        </NFormItem>
        <NFormItem span="24" label="参数配置">
          <NDynamicInput
            v-model:value="model.query"
            preset="pair"
            key-placeholder="请输入参数名称"
            value-placeholder="请输入参数默认值"
            @create="handleIptCreate"
          >
            <template #default="{ index }">
              <div class="w-[81%] flex items-start gap-16px">
                <NInput v-model:value="model.query[index].key" placeholder="请输入参数名称" type="textarea" rows="3"></NInput>
                <NInput
                  v-model:value="model.query[index].label"
                  :placeholder="!!model.query[index].valuePlaceholder
                  ? model.query[index].valuePlaceholder: '请输入参数默认值'"
                  type="textarea"
                  rows="3"
                ></NInput>
                <NInput v-model:value="model.query[index].type" placeholder="请输入参数描述" type="textarea" rows="3"></NInput>
              </div>
            </template>
            <template #action="{ index, value }">
              <NSpace class="ml-12px">
                <NButton size="medium" @click="() => handleCreate(index)">
                  <icon-ic:round-plus class="text-icon" />
                </NButton>
                <NButton size="medium" @click="() => handleRemove(index)">
                  <icon-ic-round-remove class="text-icon" />
                </NButton>
              </NSpace>
            </template>
          </NDynamicInput>
        </NFormItem>
      </NForm>
    </NScrollbar>
    <template #footer>
      <NSpace justify="end" :size="16">
        <NButton @click="closeDrawer">关闭窗口</NButton>
        <NButton type="primary" @click="handleSubmit">确认保存</NButton>
      </NSpace>
    </template>
  </NModal>
</template>

<style scoped></style>
