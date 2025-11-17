<script setup lang="ts">
import {
  NInputNumber,
  NPopover,
  NSelect,
  useDialog,
  useMessage,
} from "naive-ui";
import {NCascader, NInput, NUpload} from "naive-ui";
import {computed} from "vue";
import _ from "lodash";
import inc_before from "@/assets/imgs/inc-before.png";
import inc_after from "@/assets/imgs/inc-after.jpg";
import arrow from "@/assets/svg-icon/arrow.svg";
import {
  getDataSetListNoPage,
  getExampleList,
} from "@/service/api/expansion";
import {useFormRules, useNaiveForm} from "@/hooks/common/form";
import {getModelList, trainStart} from "@/service/api/model-manage";
import {selectDataSetLabel} from "@/service/api/ano";
import SvgIcon from "@/components/custom/svg-icon.vue";
import EvaluationMetrics from "./moduels/EvaluationMetrics.vue"

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
const {formRef, validate, restoreValidation} = useNaiveForm();
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
  const {defaultRequiredRule} = useFormRules();
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
        formName: "说明",
        type: "quaternary",
        modelList: [],
        value: "评估指标-图表计算说明",
        width: "30%",
        serverKey: "modelVersion",
      },
      {
        formName: "任务名称",
        type: "input",
        value: "",
        placeholder: "请输入任务名称",
        width: "30%",
        path: "taskInputName",
        serverKey: "taskInputName",
        isShow: true,
      },
      {
        formName: "任务描述",
        type: "textarea",
        value: "",
        placeholder: "请输入任务描述",
        width: "30%",
        serverKey: "taskDesc",
        isShow: true,
      },
      {
        formName: "选择基础模型",
        type: "radioGroup",
        modelList: [],
        value: "0",
        width: "30%",
        serverKey: "modelId",
        isShow: false,
      },
      {
        formName: "选择优化模块",
        type: "radioGroup",
        modelList: [],
        value: "0",
        width: "30%",
        serverKey: "mode",
        isShow: false,
      },
    ],
  },
  {
    name: "评估对象配置",
    list: [
      {
        formName: "评估数据集",
        type: "cascader",
        value: null,
        options: [],
        placeholder: "请选择评估数据集",
        width: "30%",
        serverKey: "datasetId",
        isShow: true,
      },
      {
        formName: "待评估模型",
        type: "cascader",
        value: null,
        options: [
          {value: '1', label: '模型1'},
          {value: '2', label: '模型2'},
          {value: '3', label: '模型3'},
          {value: '4', label: '模型4'},
          {value: '5', label: '模型5'},
        ],
        placeholder: "请选择待评估模型",
        width: "30%",
        serverKey: "modelId",
        isShow: true,
      },
      // {
      //   formName: "参数配置",
      //   type: "dynamicInput",
      //   value: [
      //     { key: "", value: "", desc: "", label: "", valuePlaceholder: "" },
      //   ],
      //   placeholder: "",
      //   width: "98%",
      //   serverKey: "trainPrams",
      //   query: [],
      //   columns: [],
      //   isShow: true,
      // },
    ],
  },
  {
    name: "评估指标配置",
    list: [
      // 评估方法 type：checkbox
      {
        formName: "评估指标",
        type: "checkboxGroup",
        value: null,
        options: [
          { value: 'mPrecision', label: 'mPrecision' },
          { value: 'mAP', label: 'mAP' },
          { value: 'mAccuracy', label: 'mAccuracy' },
          { value: 'mRecall', label: 'mRecall' },
          { value: '漏检率', label: '漏检率' },
          { value: '虚警率', label: '虚警率' }
        ],
        width: "98%",
        serverKey: "evaluateMethods",
        isShow: true,
      },
      // 评估图表
      {
        formName: "评估图表",
        type: "checkboxGroup",
        value: null,
        options: [
          // P-R曲线图 混淆矩阵图
          { value: 'PRCurve', label: 'P-R曲线' },
          { value: 'ConfusionMatrix', label: '混淆矩阵图' },
        ],
        width: "98%",
        serverKey: "evaluateChartType",
        isShow: true,
      },
    ]
  }
]);

const prevTaskName = ref<any>(""); // 任务名称
const nextTaskName = ref<any>("");
const modelList = ref<any>([]);

// methods
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
  const index = configList.value.findIndex((item) => item.name === "数据配置");
  if (index > -1) {
    const list = configList.value[index].list;
    const dataIndex = list.findIndex((item) => item.serverKey === "datasetId");
    if (dataIndex > -1) {
      list[dataIndex].options = recursionMapData(res.data);
    }
  }
};

async function getModels() {
  const res = await getModelList({});
  const list = res.data || [];
  const options = list.map((item: any) => {
    return {...item, label: item.modelName, value: item.modelUrl};
  });
  configList.value[0].list[2].modelList = options;
  // 算法列表
  if (options.length > 0) {
    configList.value[0].list[2].value = options[0].modelId;
    const res1 = await getExampleList({modelId: options[0].modelId});

    configList.value[0].list[3].modelList = res1.data.map((item: any) => {
      return {...item, label: item.algorithmName, value: item.trainType};
    });
    // configList.value[0].list[3].value = configList.value[0].list[3].modelList[0].value;
    // ---------------------------------------------
    const res2 = await getExampleList({id: 12});
    configList.value[0].list[3].serverKey = res2.data[0].paramsMap[0].serverKey; // 选择优化模块绑定serverKey （后端动态变）
    configList.value[1].list[1].serverKey = res2.data[0].paramsMap[1].serverKey; // 数据集标签 serverKey
  }
  // 评估对象配置
  const modelParams = options[0].modelParams;
  const index = configList.value.findIndex((item) => item.name === "评估对象配置");
  if (index > -1) {
    const list = configList.value[index].list;
    const dataIndex = list.findIndex((item) => item.formName === "参数配置");
    if (dataIndex > -1) {
      // query
      list[dataIndex].query = modelParams
        ? JSON.parse(modelParams)
          .map((val, idx) => {
            return {
              index: idx,
              ...val,
              key: val.serverKey, // val.value
              value: val.value ? val.value : null, // val.label
              type: val.type,
              valuePlaceholder: val.label,
              disabled: Boolean(val.bindKey),
            };
          })
          .filter((val) => {
            return !val.isShow || !val.isShow;
          })
        : [];
      modelList.value = list[dataIndex].query;
      // columns
      list[dataIndex].columns = modelParams
        ? [
          {
            title: "超参数",
            key: "label",
            width: "200",
            render: (row: any) => {
              return [
                h(
                  "div",
                  {
                    class: "flex items-center",
                  },
                  [
                    h(
                      "span",
                      {
                        class: "text-[12px] text-[#151b26]",
                      },
                      `${row.label}`
                    ),
                    h(
                      NPopover,
                      {
                        trigger: "hover",
                        placement: "right",
                        style: {"max-width": "360px"},
                      },
                      {
                        trigger: () =>
                          h("span", {class: "block ml-4px"}, [
                            h(SvgIcon, {
                              icon: "fluent:info-24-regular",
                              class: "text-[14px]",
                              localIcon: "fluent--info-24-regular",
                            }),
                          ]),
                        default: () => h("div", {}, `${row.tooltip}`),
                      }
                    ),
                  ]
                ),
              ];
            },
          },
          {
            title: "数值",
            key: "value",
            width: "500",
            render: (row: any, index: any) => {
              if (row.type === "text") {
                return [
                  h("div", {}, [
                    h(NInput, {
                      style: {
                        display: row.sign ? "none" : "block",
                      },
                      class: "!w-full",
                      placeholder: row.valuePlaceholder,
                      value: row.value,
                      disabled: row.disabled,
                      onUpdateValue(v) {
                        modelList.value[index].value = v;
                        nextTaskName.value = v;
                      },
                    }),
                    h(NInputNumber, {
                      style: {
                        display: row.sign ? "block" : "none",
                      },
                      min: 0,
                      clearable: true,
                      step: row.step ? row.step : 1,
                      placeholder: row.valuePlaceholder,
                      value: row.value,
                      onUpdateValue(v) {
                        modelList.value[index].value = v;
                      },
                    }),
                  ]),
                ];
              }
              if (row.type === "select") {
                return [
                  h(NSelect, {
                    disabled: Boolean(row.disabled),
                    multiple: Boolean(row.isMulSelect),
                    placeholder: row.valuePlaceholder,
                    value: row.value,
                    onUpdateValue(v) {
                      modelList.value[index].value = v;
                    },
                    options: row.options,
                  }),
                ];
              }
            },
          },
          {
            title: "参数描述",
            key: "tooltip",
            minWidth: "300",
          },
        ]
        : [];

      // 任务名称 prev
      const nameInfo = list[dataIndex].query.find((val) => {
        return val.serverKey === "name";
      });
      prevTaskName.value = nameInfo.value;
    }
  }
}

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

function handleSelectChange(e: any, row: any) {
  if (row.formName === "选择数据集标签") {
    // configList 评估对象配置
    const trainCList = configList.value.find(
      (val) => val.name === "评估对象配置"
    ).list;
    // 参数配置
    const paramList = trainCList.find(
      (val) => val.formName === "参数配置"
    ).query;
    // 类型次数 + 类别名称
    const numInfoIdx = paramList.findIndex(
      (val) => val.bindKey && val.bindKey === "tagNum"
    );
    const nameInfoIdx = paramList.findIndex(
      (val) => val.bindKey && val.bindKey === "tags"
    );
    paramList[numInfoIdx].value = e.length;
    paramList[nameInfoIdx].value = e.join(",");
  }
}

async function handleCascaderChange(e: any, row: any) {
  configList.value[1].list[1].isShow = true; // 数据集标签 serverKey
  const res = await selectDataSetLabel({sonId: e});
  if (res.data) {
    // lodash 根据labelName去重
    const dataList = _.uniqBy(res.data, "labelName");
    configList.value[1].list[1].options = dataList.map((val) => {
      return {
        label: val.labelId,
        value: val.labelName,
      };
    });
  }
}

const dialog = useDialog();
const message = useMessage();

async function handleOperate(sign: any) {
  if (sign === "submit") {
    await validate();
    // 判断任务名称是否修改过
    console.log(nextTaskName.value);
    console.log(prevTaskName.value);

    if (!nextTaskName.value) {
      dialog.warning({
        title: "警告",
        content: "模型名称未修改，提交会覆盖原有模型,是否继续?",
        positiveText: "继续",
        negativeText: "取消",
        onPositiveClick: async () => {
          // message.success('继续');
          const params = {
            algorithmParam: {},
          };
          // configList.value 过滤评估对象配置
          const dataList1 = configList.value.filter(
            (item) => item.name === "评估对象配置"
          );
          const dataList2 = configList.value.filter(
            (item) => item.name !== "评估对象配置"
          );
          dataList2.forEach((val1: any) => {
            const list = val1.list;
            list.forEach((val2: any) => {
              if (val2.serverKey) {
                params[val2.serverKey] = val2.value;
              }
            });
          });
          dataList1.forEach((val1: any) => {
            const list = val1.list;
            list.forEach((val2: any) => {
              if (val2.serverKey) {
                params.algorithmParam[val2.serverKey] = JSON.stringify(
                  val2.query
                );
              }
            });
          });
          // 特殊处理 数据集标签 || 算法（目标检测等）
          const index = configList.value.findIndex(
            (item) => item.name === "基本信息"
          );
          if (index > -1) {
            const list = configList.value[index].list;
            const dataIndex = list.findIndex(
              (item) => item.formName === "选择优化模块"
            );
            if (dataIndex > -1) {
              const obj = list[dataIndex];
              params.algorithmParam[obj.serverKey] = obj.value;
              delete params[obj.serverKey];
            }
          }
          const index1 = configList.value.findIndex(
            (item) => item.name === "数据配置"
          );
          if (index1 > -1) {
            const list1 = configList.value[index1].list;

            const dataIndex1 = list1.findIndex(
              (item) => item.formName === "选择数据集标签"
            );
            if (dataIndex1 > -1) {
              const obj1 = list1[dataIndex1];
              params.algorithmParam[obj1.serverKey] = obj1.value;
              delete params.classes;
              if (
                Array.isArray(params.algorithmParam.classes) &&
                params.algorithmParam.classes.length > 0
              ) {
                params.algorithmParam.classes =
                  params.algorithmParam?.classes.join(",");
              }
              if (!params.taskInputName) {
                window.$message?.error?.("任务名称不能为空！");
                return;
              }
              const res = await trainStart(params);
              if (res.data) {
                setTimeout(() => {
                  window.$message?.success?.("任务训练开始...");
                  router.back();
                }, 1500);
              }
            }
          }
        },
        onNegativeClick: () => {
          // message.error('取消')
        },
      });
      return;
    }
    const params = {
      algorithmParam: {},
    };
    // configList.value 过滤评估对象配置
    const dataList1 = configList.value.filter(
      (item) => item.name === "评估对象配置"
    );
    const dataList2 = configList.value.filter(
      (item) => item.name !== "评估对象配置"
    );
    dataList2.forEach((val1: any) => {
      const list = val1.list;
      list.forEach((val2: any) => {
        if (val2.serverKey) {
          params[val2.serverKey] = val2.value;
        }
      });
    });
    dataList1.forEach((val1: any) => {
      const list = val1.list;
      list.forEach((val2: any) => {
        if (val2.serverKey) {
          params.algorithmParam[val2.serverKey] = JSON.stringify(val2.query);
        }
      });
    });
    // 特殊处理 数据集标签 || 算法（目标检测等）
    const index = configList.value.findIndex(
      (item) => item.name === "基本信息"
    );
    if (index > -1) {
      const list = configList.value[index].list;
      const dataIndex = list.findIndex(
        (item) => item.formName === "选择优化模块"
      );
      if (dataIndex > -1) {
        const obj = list[dataIndex];
        params.algorithmParam[obj.serverKey] = obj.value;
        delete params[obj.serverKey];
      }
    }
    const index1 = configList.value.findIndex(
      (item) => item.name === "数据配置"
    );
    if (index1 > -1) {
      const list1 = configList.value[index1].list;

      const dataIndex1 = list1.findIndex(
        (item) => item.formName === "选择数据集标签"
      );
      if (dataIndex1 > -1) {
        const obj1 = list1[dataIndex1];
        params.algorithmParam[obj1.serverKey] = obj1.value;
        delete params.classes;
        if (
          Array.isArray(params.algorithmParam.classes) &&
          params.algorithmParam.classes.length > 0
        ) {
          params.algorithmParam.classes =
            params.algorithmParam?.classes.join(",");
        }
        if (!params.taskInputName) {
          window.$message?.error?.("任务名称不能为空！");
          return;
        }
        const res = await trainStart(params);
        if (res.data) {
          // window.$message?.success?.("任务训练开始...");
          // router.back();
          setTimeout(() => {
            window.$message?.success?.("任务训练开始...");
            router.back();
          }, 1500);
        }
      }
    }
  }
  if (sign === "back") {
    router.back();
  }
}

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
          {trigger: "hover", placement: "top"},
          {
            trigger: () => [
              h("span", {class: "truncate"}, `${option.label}`),
            ],
            default: () => [h("span", {}, `${option.label}`)],
          }
        ),
      ]
    ),
  ];
}

// lifecycle
onMounted(async () => {
  await getMapList();
  await getModels(); // 获取模型列表
});

onBeforeUnmount(() => {
});
</script>

<template>
  <div
    class="wrap-container box-border h-full w-full flex flex-col items-start justify-start"
  >
    <div class="box-border w-full flex-1 p-24px !overflow-y-auto">
      <NForm
        :ref="formRef"
        :rules="taskRules"
        label-placement="left"
        label-width="auto"
        require-mark-placement="right-hanging"
        class="h-full !w-100%"
      >
        <div
          class="h-auto w-full flex flex-col items-center justify-start gap-12px !overflow-y-auto !pb-18px"
        >
          <div
            v-for="(item, index) of configList"
            :key="index"
            class="h-auto w-full flex items-center"
          >
            <NCard class="h-auto w-full" :title="item.name">
              <div class="h-auto w-full flex flex-col items-start">
                <div
                  v-for="(val, idx) of item.list"
                  :key="idx"
                  :style="{ width: val.width ?? '100%' }"
                >
                  <!-- input -->
                  <NGrid
                    v-if="val.type === 'input'"
                    :cols="24"
                    :x-gap="24"
                    class="ml-24px"
                  >
                    <NFormItemGi
                      :span="24"
                      :label="val.formName"
                      :path="val.path"
                    >
                      <NInput
                        v-model:value="val.value"
                        clearable
                        :placeholder="val.placeholder"
                      />
                    </NFormItemGi>
                  </NGrid>
                  <!-- textarea -->
                  <NGrid
                    v-if="val.type === 'textarea'"
                    :cols="24"
                    :x-gap="24"
                    class="ml-24px"
                  >
                    <NFormItemGi
                      :span="24"
                      :label="val.formName"
                      :path="val.path"
                    >
                      <NInput
                        v-model:value="val.value"
                        type="textarea"
                        clearable
                        :placeholder="val.placeholder"
                      />
                    </NFormItemGi>
                  </NGrid>
                  <!-- select -->
                  <NGrid
                    v-if="val.type === 'select'"
                    :cols="24"
                    :x-gap="24"
                    class="ml-24px"
                  >
                    <NFormItemGi
                      v-if="val.isShow"
                      :span="24"
                      :label="val.formName"
                      :path="val.path"
                    >
                      <NSelect
                        v-model:value="val.value"
                        :placeholder="val.placeholder"
                        :multiple="val.isMultiple"
                        :options="val.options"
                        :render-label="val.renderLabel"
                        @update:value="handleSelectChange($event, val)"
                      />
                    </NFormItemGi>
                  </NGrid>
                  <!-- dynamicInput -->
                  <NGrid
                    v-if="val.type === 'dynamicInput'"
                    :cols="24"
                    :x-gap="24"
                    class="ml-24px"
                  >
                    <NFormItemGi
                      :span="24"
                      :label="val.formName"
                      :path="val.path"
                    >
                      <!--
 <NDynamicInput v-model:value="val.query" preset="pair" key-placeholder="请输入参数Key"
                        value-placeholder="请输入参数Value" @create="handleIptCreate">
                        <template #default="{ index }">
                          <div class="w-[81%] flex items-start gap-16px">
                            <NInput v-model:value="val.query[index].key" placeholder="请输入参数Key"></NInput>
                            <NInput v-model:value="val.query[index].label" :placeholder="!!val.query[index].valuePlaceholder
                              ? val.query[index].valuePlaceholder
                              : '请输入参数Value'
                              "></NInput>
                            <NInput v-model:value="val.query[index].type" placeholder="请输入参数类型"></NInput>
                          </div>
                        </template>
<template #action="{ index, value }">
                          <NSpace class="ml-12px">
                            <NButton size="medium" @click="() => handleCreate(index, val)">
                              <icon-ic:round-plus class="text-icon" />
                            </NButton>
                            <NButton size="medium" @click="() => handleRemove(index, val)">
                              <icon-ic-round-remove class="text-icon" />
                            </NButton>
                          </NSpace>
                        </template>
</NDynamicInput>
-->
                      <NDataTable
                        :bordered="false"
                        :single-line="false"
                        :columns="val.columns"
                        :data="val.query"
                      />
                    </NFormItemGi>
                  </NGrid>
                  <!-- text -->
                  <NGrid
                    v-if="val.type === 'text'"
                    :cols="24"
                    :x-gap="24"
                    class="ml-24px"
                  >
                    <NFormItemGi
                      :span="24"
                      :label="val.formName"
                      :path="val.path"
                    >
                      {{ val.value }}
                    </NFormItemGi>
                  </NGrid>
                  <!--quaternary-->
                  <NGrid
                    v-if="val.type === 'quaternary'"
                    :cols="24"
                    :x-gap="24"
                    class="ml-24px"
                  >
                    <NFormItemGi
                      :span="24"
                      :label="val.formName"
                      :path="val.path"
                    >
                      <n-popover trigger="hover" placement="right">
                        <template #trigger>
                          <n-button quaternary type="info">
                            {{ val.value}}
                          </n-button>
                        </template>
                        <div>
                          <EvaluationMetrics/>
                        </div>
                      </n-popover>
                    </NFormItemGi>
                  </NGrid>
                  <!-- checkGroup -->
                  <!-- radioGroup -->
                  <NGrid
                    v-if="val.type === 'radioGroup'"
                    :cols="24"
                    :x-gap="24"
                    class="ml-24px"
                  >
                    <NFormItemGi
                      v-if="val.modelList.length > 0 && val.isShow"
                      :span="24"
                      :label="val.formName"
                      :path="val.path"
                    >
                      <NRadioGroup
                        v-model:value="val.value"
                        name="anoType"
                        size="large"
                      >
                        <!-- 选中事件 -->
                        <NRadioButton
                          v-for="(val1, idx1) of val.modelList"
                          :key="idx1"
                          :value="val1.value"
                          :label="val1.label"
                          @change="
                            handleRadioChange($event, val1, val.formName)
                          "
                        ></NRadioButton>
                      </NRadioGroup>
                    </NFormItemGi>
                  </NGrid>
                  <!-- cascader -->
                  <NGrid
                    v-if="val.type === 'cascader'"
                    :cols="24"
                    :x-gap="24"
                    class="ml-24px"
                  >
                    <NFormItemGi
                      :span="24"
                      :label="val.formName"
                      :path="val.path"
                    >
                      <NCascader
                        v-model:value="val.value"
                        clearable
                        :placeholder="val.placeholder"
                        :options="val.options"
                        check-strategy="child"
                        :show-path="true"
                        expand-trigger="hover"
                        :render-label="renderLabel"
                        @update:value="handleCascaderChange($event, val)"
                      />
                    </NFormItemGi>
                  </NGrid>
                  <!--checkboxGroup-->
                  <NGrid
                    v-if="val.type === 'checkboxGroup'"
                    :cols="24"
                    :x-gap="24"
                    class="ml-24px"
                  >
                    <NFormItemGi
                      :span="24"
                      :label="val.formName"
                      :path="val.path"
                    >
                      <NCheckboxGroup
                        v-model:value="val.value"
                      >
                        <n-space item-style="display: flex;" align="center">
                          <n-checkbox
                            v-for="item of val.options"
                            :value="item.value"
                            :key="item.value"
                            :label="item.label"
                          />
                        </n-space>
                      </NCheckboxGroup>
                    </NFormItemGi>
                  </NGrid>
                </div>
              </div>
            </NCard>
          </div>
        </div>
      </NForm>
    </div>
    <div
      class="footer box-border w-full flex items-center justify-start gap-24px bg-[#fff] px-24px py-12px"
    >
      <NButton type="info" class="w-88px" @click="handleOperate('submit')"
      >开始评估
      </NButton
      >
      <NButton type="default" class="w-88px" @click="handleOperate('back')"
      >返回
      </NButton
      >
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
