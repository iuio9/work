<script setup lang="ts">
import type { UploadFileInfo } from "naive-ui";
import {
  CascaderOption,
  FormInst,
  NInputNumber,
  NPopover,
  NSelect,
  useDialog,
  useMessage,
} from "naive-ui";
// import Flash16Regular from '@vicons/fluent/Flash16Regular'
import { NCascader, NInput, NUpload } from "naive-ui";
import { ArchiveOutline as ArchiveIcon } from "@vicons/ionicons5";
import { cloneDeep } from "lodash-es";
import { computed } from "vue";
import _ from "lodash";
import inc_before from "@/assets/imgs/inc-before.png";
import inc_after from "@/assets/imgs/inc-after.jpg";
import arrow from "@/assets/svg-icon/arrow.svg";
import {
  getDataSetListNoPage,
  getDictDataTree,
  getExampleList,
  submitTask,
} from "@/service/api/expansion";
import { useFormRules, useNaiveForm } from "@/hooks/common/form";
import { getModelList, trainStart } from "@/service/api/model-manage";
import { selectDataSetLabel } from "@/service/api/ano";
import SvgIcon from "@/components/custom/svg-icon.vue";

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
const taskConfig = ref<any>({
  options: [],
  mapImportOptions: [],
  mapExportOptions: [],
  checkStrategy: "all",
  checkStrategyIsChild: true,
  showPath: true,
  hoverTrigger: false,
  filterable: false,
  valueField: "dictValue",
  labelField: "dictLabel",
});
const taskModel = ref<any>({
  taskName: undefined,
  taskInputName: undefined,
  dataMap: {
    import: null,
    tags: [],
    export: null,
  },
  uploadUrl: "",
  jsonUrl: "",
  trainVal: null,
  errComposeVal: null,
  fileList: [],
  jsonList: [],
});
const taskRules = computed<Record<keyof Model, App.Global.FormRule[]>>(() => {
  const { defaultRequiredRule } = useFormRules();
  return {
    taskInputName: defaultRequiredRule,
    modelName: defaultRequiredRule,
  };
});

// ----------------newCode----------------
const router = useRouter();
const route = useRoute();
const fileAction = `${
  import.meta.env.VITE_SERVICE_BASE_URL
}/algorithm/task/readFileToJson`;
const uploadRef = ref(null);
const dynamicList = ref<any>([]);

const configList = ref<any>([
  {
    name: "基本信息",
    list: [
      {
        formName: "模型名称",
        type: "input",
        value: "",
        placeholder: "请输入模型名称",
        width: "45%",
        path: "modelName",
        serverKey: "modelName",
        isShow: true,
      },
      {
        formName: "模型类型",
        type: "text",
        value: "图像生成",
        width: "45%",
        serverKey: "modelType",
      },
      {
        formName: "模型描述",
        type: "textarea",
        value: "",
        placeholder: "请输入任务描述",
        width: "55%",
        serverKey: "modelDesc",
        isShow: true,
      },
    ],
  },
  {
    name: "模型配置",
    list: [
      // 模型物理路径
      {
        formName: "模型物理路径",
        type: "input",
        value: "",
        placeholder: "请输入模型物理路径",
        width: "45%",
        serverKey: "modelPath",
      },
      // 模型API类型 get post put
      {
        formName: "API类型",
        type: "select",
        value: "get",
        placeholder: "请选择API类型",
        width: "45%",
        serverKey: "modelMethod",
        options: [
          { label: "get", value: "get" },
          { label: "post", value: "post" },
          { label: "put", value: "put" },
        ],
        isShow: true,
      },
      // stream流类型  base64 二进制
      {
        formName: "数据输出类型",
        type: "select",
        value: "base64",
        placeholder: "请选择数据输出类型",
        width: "45%",
        serverKey: "streamType",
        options: [
          { label: "base64", value: "base64" },
          { label: "二进制", value: "binary" },
        ],
        isShow: true,
      },
      // 输入地址
      {
        formName: "输入地址",
        type: "input",
        value: "",
        placeholder: "请输入输入地址",
        width: "45%",
        serverKey: "inputUrl",
        isShow: true,
      },
      // 输出地址
      {
        formName: "输出地址",
        type: "input",
        value: "",
        placeholder: "请输入输出地址",
        width: "45%",
        serverKey: "outputUrl",
        isShow: true,
      },
    ],
  },
]);
const mulSelVals = ref<any>([]);

const prevTaskName = ref<any>(""); // 任务名称
const nextTaskName = ref<any>("");

// methods
const handleFinish = ({
  file,
  event,
}: {
  file: UploadFileInfo;
  event?: ProgressEvent;
}) => {
  const resData = JSON.parse((event?.target as XMLHttpRequest).response);
  if (resData.code == 200) {
    dynamicList.value = Array.isArray(resData.data) ? resData.data : [];
    window.$message?.success?.(
      resData.code == 200 ? "配置文件解析成功！" : "配置文件解析失败！"
    );
  } else {
    window.$message?.error?.("配置文件解析失败！");
  }
};
const handleFileChange = () => {
  if (uploadRef.value) {
    uploadRef.value.clearFiles();
  }
};
const handleCreate = (index: any, row: any) => {
  const rowData = { key: "", value: "", type: "" };
  row.query.splice(index + 1, 0, rowData);
};
const handleRemove = (index: any, row: any) => {
  row.query.splice(index, 1);
};
const handleIptCreate = () => {
  const rowData = {
    key: "",
    value: "",
    type: "",
    valuePlaceholder: "请输入参数Value",
    keyPlaceholder: "请输入参数Key",
  };
  return rowData;
};
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
  const index = configList.value.findIndex((item) => item.name === "数据配置");
  if (index > -1) {
    const list = configList.value[index].list;
    const dataIndex = list.findIndex((item) => item.serverKey === "datasetId");
    if (dataIndex > -1) {
      list[dataIndex].options = recursionMapData(res.data);
    }
  }
};
// --------------------------------

// 模型列表
const modelList = ref<any>([]);

async function getModels() {
  const res = await getModelList({});
  const list = res.data || [];
  const options = list.map((item: any) => {
    return { ...item, label: item.modelName, value: item.modelUrl };
  });
  configList.value[0].list[2].modelList = options;
  // 算法列表
  if (options.length > 0) {
    configList.value[0].list[2].value = options[0].modelId;
    const res1 = await getExampleList({ modelId: options[0].modelId });

    configList.value[0].list[3].modelList = res1.data.map((item: any) => {
      return { ...item, label: item.algorithmName, value: item.trainType };
    });
    // configList.value[0].list[3].value = configList.value[0].list[3].modelList[0].value;
    // ---------------------------------------------
    const res2 = await getExampleList({ id: 12 });
    configList.value[0].list[3].serverKey = res2.data[0].paramsMap[0].serverKey; // 选择优化模块绑定serverKey （后端动态变）
    configList.value[1].list[1].serverKey = res2.data[0].paramsMap[1].serverKey; // 数据集标签 serverKey
  }
  // 选择模型
  const modelParams = options[0].modelParams;
  const index = configList.value.findIndex((item) => item.name === "选择模型");
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
                          style: { "max-width": "360px" },
                        },
                        {
                          trigger: () =>
                            h("span", { class: "block ml-4px" }, [
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
    // 选择模型
    const index = configList.value.findIndex(
      (item) => item.name === "选择模型"
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
    // configList 选择模型
    const trainCList = configList.value.find(
      (val) => val.name === "选择模型"
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
  const res = await selectDataSetLabel({ sonId: e });
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
          // configList.value 过滤选择模型
          const dataList1 = configList.value.filter(
            (item) => item.name === "选择模型"
          );
          const dataList2 = configList.value.filter(
            (item) => item.name !== "选择模型"
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
    // configList.value 过滤选择模型
    const dataList1 = configList.value.filter(
      (item) => item.name === "选择模型"
    );
    const dataList2 = configList.value.filter(
      (item) => item.name !== "选择模型"
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

// lifecycle
onMounted(async () => {
  await getMapList();
  await getModels(); // 获取模型列表
});

onBeforeUnmount(() => {});
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
        >确定</NButton
      >
      <NButton type="default" class="w-88px" @click="handleOperate('back')"
        >取消</NButton
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
