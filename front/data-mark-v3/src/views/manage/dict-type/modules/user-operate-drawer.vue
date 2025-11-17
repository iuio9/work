<script setup lang="ts">
import {computed, reactive, ref, watch} from 'vue';
import {useFormRules, useNaiveForm} from '@/hooks/common/form';
import {fetchTwoDictAdd, fetchTwoDictEdit} from '@/service/api';
import {$t} from '@/locales';
import {enableStatusOptions, userGenderOptions} from '@/constants/business';
import useDictStore from "@/store/modules/dict";

defineOptions({
  name: 'UserOperateDrawer'
});

interface Props {
  /** the type of operation */
  operateType: NaiveUI.TableOperateType;
  /** the edit row data */
  rowData?: Api.SystemManage.User | null;
}

const props = defineProps<Props>();

interface Emits {
  (e: 'submitted'): void;
}

const emit = defineEmits<Emits>();

const visible = defineModel<boolean>('visible', {
  default: false
});

const {formRef, validate, restoreValidation} = useNaiveForm();
const {defaultRequiredRule} = useFormRules();

const title = computed(() => {
  const titles: Record<NaiveUI.TableOperateType, string> = {
    add: "新增字典类型",
    edit: "编辑字典类型"
  };
  return titles[props.operateType];
});

interface Model {
  remark: string,
  status: any,
  [key: string]: any
};

const model: Model = reactive(createDefaultModel());
function createDefaultModel(): Model {
  return {
    remark: "",
    status: null,
    dictLabel: "",
    dictValue: "",
    dictSort: "",
  };
}

type RuleKey = Extract<keyof Model, 'dictName'>;

const rules: Record<RuleKey, App.Global.FormRule> = {
  dictLabel: defaultRequiredRule,
  dictValue: defaultRequiredRule,
  dictSort: defaultRequiredRule,
};

const route = useRoute();

function handleInitModel() {
  Object.assign(model, createDefaultModel());

  if (props.operateType === 'edit' && props.rowData) {
    Object.assign(model, props.rowData, {
      // status: `${props.rowData.status}`,
      dictSort: `${props.rowData.dictSort}`,
    });
  }
}

function closeDrawer() {
  visible.value = false;
}

async function handleSubmit() {
  await validate();
  // request
  let mapModel: any = Object.assign({}, model, {
    // status: +model.status,
    dictSort: +model.dictSort,
    typeId: props.rowData.typeId
  });
  let res: any;
  if (props.operateType === 'edit') {
    res = await fetchTwoDictEdit(mapModel);
  }
  if (props.operateType === 'add') {
    mapModel = Object.assign({}, mapModel, {
      parentId: !!props.rowData.parentId ? props.rowData.parentId : 0
    })
    res = await fetchTwoDictAdd(mapModel);
  }
  if(res.data >=1) {
    useDictStore().removeDict(props.rowData.typeId);
    window.$message?.success($t('common.updateSuccess'));
    closeDrawer();
    emit('submitted');
  }
}

watch(visible, () => {
  if (visible.value) {
    handleInitModel();
    restoreValidation();
  }
});
</script>

<template>
  <NDrawer v-model:show="visible" display-directive="show" :width="360">
    <NDrawerContent :title="title" :native-scrollbar="false" closable>
      <NForm ref="formRef" :model="model" :rules="rules">
        <NFormItem label="数据标签" path="dictLabel">
          <NInput v-model:value="model.dictLabel" placeholder="请输入数据标签"/>
        </NFormItem>
        <!-- <NFormItem label="数据键值" path="dictValue">
          <NInput v-model:value="model.dictValue" placeholder="请输入数据键值"/>
        </NFormItem> -->
        <!--显示排序-->
        <NFormItem label="显示排序" path="dictSort">
          <n-input-number v-model:value="model.dictSort" clearable/>
        </NFormItem>
        <!-- <NFormItem label="状态" path="status">
          <NRadioGroup v-model:value="model.status">
            <NRadio v-for="item in enableStatusOptions" :key="item.value" :value="item.value" :label="$t(item.label)"/>
          </NRadioGroup>
        </NFormItem> -->
        <NFormItem label="备注信息" path="nickName">
          <NInput
            type="textarea"
            v-model:value="model.remark" placeholder="请输入备注信息"/>
        </NFormItem>
      </NForm>
      <template #footer>
        <NSpace :size="16">
          <NButton @click="closeDrawer">{{ $t('common.cancel') }}</NButton>
          <NButton type="primary" @click="handleSubmit">{{ $t('common.confirm') }}</NButton>
        </NSpace>
      </template>
    </NDrawerContent>
  </NDrawer>
</template>

<style scoped></style>
