<script setup lang="ts">
// 第三方库
import {
  NInputNumber,
  NPopover,
  NSelect,
  NCascader,
  NInput,
  useDialog,
  useMessage,
} from "naive-ui";
import { computed } from "vue";
import _ from "lodash";

// 本地资源
import inc_before from "@/assets/imgs/inc-before.png";
import inc_after from "@/assets/imgs/inc-after.jpg";
import arrow from "@/assets/svg-icon/arrow.svg";

// API 服务
import {
  getDataSetListNoPage,
  getExampleList,
  getTeamList,
  addManyMarkTask,
} from "@/service/api/expansion";
import {
  allocationNum
} from "@/service/api/ano";
import { getModelList, trainStart } from "@/service/api/model-manage";
import { selectDataSetLabel } from "@/service/api/ano";

// 自定义钩子
import { useFormRules, useNaiveForm } from "@/hooks/common/form";

// 自定义组件
import SvgIcon from "@/components/custom/svg-icon.vue";
import EvaluationMetrics from "./moduels/EvaluationMetrics.vue";
import TagConfig from "./moduels/TagConfig.vue";

// interface
interface FormObj {
  model: Model;
  rules: any;
}

interface Model {
  mapImport: string;

  [key: string]: any;
}

// const formRef = ref<FormInst | null>(null);
const { formRef, validate, restoreValidation } = useNaiveForm();
const formObj = reactive<FormObj>({
  model: {
    dataType: "string",
    anoType: "",
    areaVal: "",
    dataImport: "",
    tags: [],
    dataExport: "",
    increaseList: [
      {
        name: "Invert",
        beforeImg: inc_before,
        afterImg: inc_after,
        arrow,
        desc: "将图像转换为反色图像，能更好地根据轮廓识别图像",
        isCheck: false,
      },
    ],
    strategyVal: "",
  },
  rules: {},
});
// data
const taskRules = computed<Record<keyof Model, App.Global.FormRule[]>>(() => {
  const { defaultRequiredRule } = useFormRules();
  return {
    taskInputName: defaultRequiredRule,
    modelName: defaultRequiredRule,
  };
});

// ----------------newCode----------------
const router = useRouter();
const configList = ref<any>([
  {
    name: "基本信息",
    list: [
      {
        formName: "任务名称",
        type: "input",
        value: "",
        placeholder: "请输入任务名称",
        width: "40%",
        path: "taskInputName",
        serverKey: "taskName",
        isShow: true,
      },
      {
        formName: "选择数据集",
        type: "cascader",
        value: null,
        isMul: false,
        options: [],
        placeholder: "请选择数据集",
        width: "40%",
        serverKey: "sonId",
        isShow: true,
        btnText: "创建数据集",
        action: () => {
          router.push("/create-dataset");
        },
      },
      {
        formName: "设置标签",
        type: "setTag",
        value: "",
        width: "78%",
        path: "setTag",
        serverKey: "tagConfig",
        isShow: false,
        component: TagConfig,
      },
    ],
  },
  {
    name: "任务分配",
    list: [
      {
        formName: "选择标注团队",
        type: "select",
        value: null,
        isMul: true,
        options: [],
        placeholder: "选择标注团队",
        width: "30%",
        serverKey: "teamId",
        isShow: true,
        isDisabled: true,
      },
      {
        formName: "每人标注数量",
        type: "text",
        value: "24个 数据集需标注样本总数48个，每人分配样本数24个",
        width: "50%",
        serverKey: "sonId1",
        isShow: false,
      },
      {
        formName: "选择审核团队",
        type: "select",
        value: null,
        isMul: true,
        options: [],
        placeholder: "选择审核团队",
        width: "30%",
        serverKey: "auditTeamId",
        isShow: true,
        isDisabled: true,
      },
      // {
      //   formName: "每人审核数量",
      //   type: "text",
      //   value: "",
      //   width: "30%",
      //   serverKey: "auditTeamId1",
      //   isShow: false,
      // },
    ]
  }
]);
const modelList = ref<any>([]);

// methods
function handleRadioChange(e: any, row: any, formName: any) {
  if (formName === "选择基础模型") {
    // 评估对象配置
    const index = configList.value.findIndex(
      (item) => item.name === "评估对象配置"
    );
    if (index > -1) {
      const list = configList.value[index].list;
      const dataIndex = list.findIndex((item) => item.formName === "参数配置");
      if (dataIndex > -1) {
        list[dataIndex].query = row.modelParams
          ? JSON.parse(row.modelParams).map((val, idx) => {
            return {
              index: idx,
              ...val,
              key: val.serverKey, // val.value
              value: val.value ? val.value : null, // val.label
              type: val.type,
              valuePlaceholder: val.label,
            };
          })
          : [];
        modelList.value = list[dataIndex].query;
      }
    }
  }
  if (formName === "选择优化模块") {
  }
}

async function handleSelectChange(e: any, row: any) {
  if (row.formName === "选择标注团队") {
    // 选择数据集 value
    const basicInfoIndex = findConfigIndexByName("基本信息");
    if (basicInfoIndex === -1) {
      return;
    }
    const basicInfoList = configList.value[basicInfoIndex].list;
    const tagConfigIndex = findItemIndexByServerKey(basicInfoList, "sonId");
    if (tagConfigIndex === -1) {
      return;
    }
    const sonId = basicInfoList[tagConfigIndex].value;
    const params = {
      sonId, teamId: e
    }
    const res = await allocationNum(params);
    if (res.data) {
      const basicInfoIndex = findConfigIndexByName("任务分配");
      if (basicInfoIndex === -1) {
        return;
      }
      const basicInfoList = configList.value[basicInfoIndex].list;
      const tagConfigIndex = findItemIndexByServerKey(basicInfoList, "sonId1");
      if (tagConfigIndex === -1) {
        return;
      }
      basicInfoList[tagConfigIndex].value = res.data;
      basicInfoList[tagConfigIndex].isShow = true;
    }
  }
}

// newCode
const sonId = ref<string | null>(null);

function handleCascaderChange(e: string) {
  sonId.value = e;
  const basicInfoIndex = findConfigIndexByName("基本信息");
  if (basicInfoIndex === -1) {
    return;
  }
  const basicInfoList = configList.value[basicInfoIndex].list;
  const tagConfigIndex = findItemIndexByServerKey(basicInfoList, "tagConfig");
  if (tagConfigIndex === -1) {
    return;
  }
  basicInfoList[tagConfigIndex].isShow = true;
  // 选择标注团队
  const basicInfoIndex1 = findConfigIndexByName("任务分配");
  if (basicInfoIndex1 === -1) {
    return;
  }
  const basicInfoList1 = configList.value[basicInfoIndex1].list;
  const tagConfigIndex1 = findItemIndexByServerKey(basicInfoList1, "teamId");
  if (tagConfigIndex1 === -1) {
    return;
  }
  basicInfoList1[tagConfigIndex1].isShow = true;
  basicInfoList1[tagConfigIndex1].isDisabled = false;
  // 选择审核团队 serverKey = "auditTeamId"
  const basicInfoIndex2 = findConfigIndexByName("任务分配");
  if (basicInfoIndex2 === -1) {
    return;
  }
  const basicInfoList2 = configList.value[basicInfoIndex2].list;
  const tagConfigIndex2 = findItemIndexByServerKey(basicInfoList2, "auditTeamId");
  if (tagConfigIndex2 === -1) {
    return;
  }
  basicInfoList2[tagConfigIndex2].isShow = true;
  basicInfoList2[tagConfigIndex2].isDisabled = false;

  // --------------------------每人标注数量sonId1------------------------------
  const basicInfoIndex3 = findConfigIndexByName("任务分配");
  if (basicInfoIndex3 === -1) {
    return;
  }
  const basicInfoList3 = configList.value[basicInfoIndex3].list;
  console.log('basicInfoList3: ', basicInfoList3);
  const tagConfigIndex3 = findItemIndexByServerKey(basicInfoList3, "teamId");
  if (tagConfigIndex3 === -1) {
    return;
  }
  const teamId = basicInfoList3[tagConfigIndex3].value;
  if (teamId) {
    const params = {
      sonId: e, teamId: teamId
    }
    const res = allocationNum(params);
    if (res.data) {
      const basicInfoIndex = findConfigIndexByName("任务分配");
      if (basicInfoIndex === -1) {
        return;
      }
      const basicInfoList = configList.value[basicInfoIndex].list;
      const tagConfigIndex = findItemIndexByServerKey(basicInfoList, "sonId1");
      if (tagConfigIndex === -1) {
        return;
      }
      basicInfoList[tagConfigIndex].value = res.data;
      basicInfoList[tagConfigIndex].isShow = true;
    }
  }
}

function findConfigIndexByName(name: string): number {
  return configList.value.findIndex((item) => item.name === name);
}

function findItemIndexByServerKey(list: any[], serverKey: string): number {
  return list.findIndex((item) => item.serverKey === serverKey);
}

async function getGroupList(teamType: string, serverKey: string = 'teamId') {
  const res = await getTeamList({ teamType });
  if (res.data) {
    const basicInfoIndex = findConfigIndexByName("任务分配");
    if (basicInfoIndex === -1) {
      return;
    }

    const basicInfoList = configList.value[basicInfoIndex].list;

    const tagConfigIndex = findItemIndexByServerKey(basicInfoList, serverKey);
    if (tagConfigIndex === -1) {
      return;
    }

    // 显示 "tagConfig" 项
    basicInfoList[tagConfigIndex].options = res.data.map(item => {
      return {
        value: item.id,
        label: item.teamName,
      };
    });
  }
}

function extractServerKeyAndValue(data) {
  const result = {};
  data.forEach((item) => {
    item.list.forEach((listItem) => {
      result[listItem.serverKey] = listItem.value;
    });
  });
  return result;
}

async function handleOperate(sign: any) {
  if (sign === "submit") {
    await validate();
    const params = extractServerKeyAndValue(configList.value);
    const { sonId1, tagConfig, ...rest } = params;
    const res = await addManyMarkTask(rest);
    console.log('res: ', res);
    if (res.data == 1) {
      window.$message?.success?.("提交成功");
      router.back();
    }
  }
  if (sign === "back") {
    router.back();
  }
}

interface Item {
  groupId?: string | number;
  sonId?: string | number;
  groupName?: string;
  version?: number;
  dataSonResponseList?: Item[];
  isMany?: number;
  value?: string | number;
  label?: string;
  children?: Item[];
  disabled?: boolean;
}

interface ConfigItem {
  name: string;
  list: {
    serverKey: string;
    options?: Item[];
  }[];
}

const getMapList = async () => {
  try {
    // 发起异步请求获取数据集列表
    const res = await getDataSetListNoPage();

    // 递归处理数据
    const recursionMapData = (data: Item[], label?: string): Item[] => {
      return data.map((item: Item, index: number) => {
        // 设置 value
        item.value = item.groupId || item.sonId;

        // 设置 label
        item.label = label
          ? `${label} - ${item.groupName || `V${item.version}`}`
          : item.groupName || `V${item.version}`;

        // 处理子项
        const children = item.dataSonResponseList || [];
        item.children = children.map((val: Item) => {
          const isMany = val.isMany === 1;
          // 演示环境
          item.disabled = false;
          val.disabled = isMany;
          return val;
        });

        // 递归处理子项或删除空的 children
        if (item.children && item.children.length > 0) {
          item.children = recursionMapData(item.children, item.label);
        } else {
          delete item.children;
        }

        return item;
      });
    };

    // 查找 "基本信息" 配置项
    const basicInfoIndex = configList.value.findIndex((item) => item.name === "基本信息");
    if (basicInfoIndex === -1) {
      return;
    }

    // 查找 "sonId" 配置项
    const sonIdIndex = configList.value[basicInfoIndex].list.findIndex((item) => item.serverKey === "sonId");
    if (sonIdIndex === -1) {
      return;
    }

    // 更新 "sonId" 配置项的 options
    configList.value[basicInfoIndex].list[sonIdIndex].options = recursionMapData(res.data);
  } catch (error) {
    // 错误处理
    console.error('获取数据集列表时出错:', error);
  }
};

function renderLabel(option: { value?: string | number; label?: string }) {
  return [
    h(
      "div",
      {
        class: "flex items-center",
      },
      [
        h(
          NPopover,
          { trigger: "hover", placement: "top" },
          {
            trigger: () => [
              h("span", { class: "truncate" }, `${option.label}`),
            ],
            default: () => [h("span", {}, `${option.label}`)],
          }
        ),
      ]
    ),
  ];
}

onMounted(async () => {
  await getMapList();
  await getGroupList('1', 'teamId');
  await getGroupList('2', 'auditTeamId');
});
</script>

<template>
  <div class="wrap-container box-border h-full w-full flex flex-col items-start justify-start">
    <div class="box-border w-full flex-1 p-24px !overflow-y-auto">
      <NForm :ref="formRef" :rules="taskRules" label-placement="left" label-width="120px"
        require-mark-placement="right-hanging" class="h-full !w-100%">
        <div class="h-auto w-full flex flex-col items-center justify-start gap-12px !overflow-y-auto !pb-18px">
          <div v-for="(item, index) of configList" :key="index" class="h-auto w-full flex items-center">
            <NCard class="h-auto w-full" :title="item.name">
              <div class="h-auto w-full flex flex-col items-start">
                <div v-for="(val, idx) of item.list" :key="idx" :style="{ width: val.width ?? '100%' }"
                  class="flex justify-start align-center">
                  <!-- input -->
                  <NGrid v-if="val.type === 'input'" :cols="24" :x-gap="24" class="ml-24px">
                    <NFormItemGi :span="24" :label="val.formName" :path="val.path">
                      <NInput v-model:value="val.value" clearable :placeholder="val.placeholder" />
                    </NFormItemGi>
                  </NGrid>
                  <!-- textarea -->
                  <NGrid v-if="val.type === 'textarea'" :cols="24" :x-gap="24" class="ml-24px">
                    <NFormItemGi :span="24" :label="val.formName" :path="val.path">
                      <NInput v-model:value="val.value" type="textarea" clearable :placeholder="val.placeholder" />
                    </NFormItemGi>
                  </NGrid>
                  <!-- select -->
                  <NGrid v-if="val.type === 'select'" :cols="24" :x-gap="24" class="ml-24px">
                    <NFormItemGi v-if="val.isShow" :span="24" :label="val.formName" :path="val.path">
                      <NSelect v-model:value="val.value" :placeholder="val.placeholder" :multiple="val.isMultiple"
                        :disabled="val.isDisabled" :options="val.options"
                        @update:value="handleSelectChange($event, val)" />
                    </NFormItemGi>
                  </NGrid>
                  <!-- dynamicInput -->
                  <NGrid v-if="val.type === 'dynamicInput'" :cols="24" :x-gap="24" class="ml-24px">
                    <NFormItemGi :span="24" :label="val.formName" :path="val.path">
                      <NDataTable :bordered="false" :single-line="false" :columns="val.columns" :data="val.query" />
                    </NFormItemGi>
                  </NGrid>
                  <!-- text -->
                  <NGrid v-if="val.type === 'text'" :cols="24" :x-gap="24" class="ml-24px">
                    <NFormItemGi :span="24" :label="val.formName" :path="val.path" v-if="val.isShow">
                      <!-- {{ val.value }} -->
                      <pre style="{
                        'font-family': 'Segoe UI'
                      }">{{ val.value }}</pre>
                    </NFormItemGi>
                  </NGrid>
                  <!--quaternary-->
                  <NGrid v-if="val.type === 'quaternary'" :cols="24" :x-gap="24" class="ml-24px">
                    <NFormItemGi :span="24" :label="val.formName" :path="val.path">
                      <n-popover trigger="hover" placement="right">
                        <template #trigger>
                          <n-button quaternary type="info">
                            {{ val.value }}
                          </n-button>
                        </template>
                        <div>
                          <EvaluationMetrics />
                        </div>
                      </n-popover>
                    </NFormItemGi>
                  </NGrid>
                  <!-- radioGroup -->
                  <NGrid v-if="val.type === 'radioGroup'" :cols="24" :x-gap="24" class="ml-24px">
                    <NFormItemGi v-if="val.modelList.length > 0 && val.isShow" :span="24" :label="val.formName"
                      :path="val.path">
                      <NRadioGroup v-model:value="val.value" name="anoType" size="large">
                        <!-- 选中事件 -->
                        <NRadioButton v-for="(val1, idx1) of val.modelList" :key="idx1" :value="val1.value"
                          :label="val1.label" @change="
                            handleRadioChange($event, val1, val.formName)
                            "></NRadioButton>
                      </NRadioGroup>
                    </NFormItemGi>
                  </NGrid>
                  <!-- cascader -->
                  <NGrid v-if="val.type === 'cascader'" :cols="24" :x-gap="24" class="ml-24px">
                    <NFormItemGi :span="24" :label="val.formName" :path="val.path">
                      <NCascader v-model:value="val.value" :render-label="renderLabel" :multiple="val.isMul" cascade
                        check-strategy="child" :options="val.options"
                        @update:value="handleCascaderChange($event, val)" />
                    </NFormItemGi>
                  </NGrid>
                  <!--checkboxGroup-->
                  <NGrid v-if="val.type === 'checkboxGroup'" :cols="24" :x-gap="24" class="ml-24px">
                    <NFormItemGi :span="24" :label="val.formName" :path="val.path">
                      <NCheckboxGroup v-model:value="val.value">
                        <n-space item-style="display: flex;" align="center">
                          <n-checkbox v-for="item of val.options" :value="item.value" :key="item.value"
                            :label="item.label" />
                        </n-space>
                      </NCheckboxGroup>
                    </NFormItemGi>
                  </NGrid>
                  <!-- setTag -->
                  <NGrid v-if="val.type === 'setTag'" :cols="24" :x-gap="24" class="ml-24px">
                    <NFormItemGi :span="24" :label="val.formName" :path="val.path" v-if="val.isShow">
                      <component :is="val.component" v-model:sonId="sonId"></component>
                    </NFormItemGi>
                  </NGrid>
                </div>
              </div>
            </NCard>
          </div>
        </div>
      </NForm>
    </div>
    <div class="footer box-border w-full flex items-center justify-start gap-24px bg-[#fff] px-24px py-12px">
      <NButton type="info" class="w-88px" @click="handleOperate('submit')">提交
      </NButton>
      <NButton type="default" class="w-88px" @click="handleOperate('back')">返回
      </NButton>
    </div>
  </div>
</template>

<style scoped lang="scss">
:deep(.n-transfer-list--target) {
  display: none !important;
}

.wrap-container {
  padding: 0 !important;
}

:deep(.n-card__content) {
  height: 100%;
}

:deep(.n-input-wrapper) {
  width: 100% !important;
}
</style>
