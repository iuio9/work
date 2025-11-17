<script setup lang="ts">
import { $t } from "@/locales";
import { LogInst, NCascader, VirtualListInst } from "naive-ui";
import { getDataSetListNoPage } from "@/service/api/expansion";
import { getModelList, trainStart, trainAssessStart } from "@/service/api/dataManage";
import { useFormRules, useNaiveForm } from "@/hooks/common/form";
import { computed } from "vue";

defineOptions({
  name: "MenuOperateModal",
});

export type OperateType = NaiveUI.TableOperateType | "addChild";

interface Props {
  taskId: string;
  trainType: string;
  items: any;
  sItems: any;
}

const props = defineProps<Props>();

interface Emits {
  (e: "submitted"): void;
}

const emit = defineEmits<Emits>();

const visible = defineModel<boolean>("visible", {
  default: false,
});

function closeDrawer() {
  visible.value = false;
}

watchEffect(async () => {
  if (visible.value) {
    restoreValidation();
    model.value = {
      taskInputName: null,
      datasetId: null,
      modelId: null,
    };
    // await getMapList();
    // await getModels(); // 获取模型列表
  }
})

const { formRef, validate, restoreValidation } = useNaiveForm();
const formRules = computed<any>(() => {
  const { defaultRequiredRule } = useFormRules();
  return {
    taskInputName: defaultRequiredRule,
    datasetId: defaultRequiredRule,
    modelId: defaultRequiredRule,
  };
});
const dataList = ref<any>([]);
const modelList = ref<any>([]);
const model = ref<any>({
  taskInputName: null,
  datasetId: null,
  modelId: null,
});

// 数据集列表接口
const getMapList = async () => {
  const recursionMapData = (data: any, label: any) => {
    const mapList = data.map((item: any, index: string | number) => {
      item.value = item.groupId || item.sonId;
      if (label) {
        item.label = `${label} - ${item.groupName || `V${item.version}`}`;
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
  dataList.value = options;
};
async function getModels() {
  const res = await getModelList({});
  const list = res.data || [];
  const options = list.map((item: any) => {
    return Object.assign({}, item, {
      label: item.modelName,
      value: item.modelUrl,
      // value: item.modelId,
    });
  });
  modelList.value = options;
}
async function handleSubmit() {
  // request
  const res = await trainAssessStart(model.value);
  if (res.data) {
    window.$message?.success?.("评估开始...");
    closeDrawer();
    // getDataByPage();
    emit('submitted');
  }
}

onMounted(async () => {
  await getMapList();
  await getModels(); // 获取模型列表
});
</script>

<template>
  <NModal v-model:show="visible" title="模型评估" preset="card" class="w-700px">
    <div class="h-auto pr-20px">
      <n-form ref="formRef" :rules="formRules" label-placement="left" label-width="auto"
        require-mark-placement="right-hanging" class="!w-100% h-full">
        <div class="w-full">
          <n-grid :cols="24" :x-gap="24" class="ml-24px">
            <n-form-item-gi :span="24" label="任务名称" path="taskInputName">
              <n-input v-model:value="model.taskInputName" clearable placeholder="请输入任务名称" />
            </n-form-item-gi>
          </n-grid>
        </div>
        <div class="w-full">
          <n-grid :cols="24" :x-gap="24" class="ml-24px">
            <!--dataImport-->
            <n-form-item-gi :span="24" label="数据集" path="datasetId">
              <n-cascader v-model:value="model.datasetId" clearable placeholder="请选择输入数据集" :options="dataList"
                check-strategy="child">
              </n-cascader>
            </n-form-item-gi>
          </n-grid>
        </div>
        <div class="w-full">
          <n-grid :cols="24" :x-gap="24" class="ml-24px">
            <!--dataImport-->
            <n-form-item-gi :span="24" label="训练模型" path="modelId">
              <n-select v-model:value="model.modelId" :options="modelList" />
            </n-form-item-gi>
          </n-grid>
        </div>
      </n-form>
    </div>
    <template #footer>
      <NSpace :size="16" class="!flex !justify-end">
        <NButton @click="closeDrawer">{{ $t("common.cancel") }}</NButton>
        <NButton type="primary" @click="handleSubmit">{{
          $t("common.confirm")
          }}</NButton>
      </NSpace>
    </template>
  </NModal>
</template>

<style scoped lang="scss">
:deep(.wrap_scrollMain) {
  .n-scrollbar-container {
    background: #1e1e1e !important;
  }
}
</style>
