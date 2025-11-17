<script setup lang="ts">
import {
  FormInst,
  CascaderOption,
  UploadFileInfo,
  useMessage,
  NInputNumber,
  NSelect,
  NPopover, useDialog,
} from "naive-ui";
// import Flash16Regular from '@vicons/fluent/Flash16Regular'
import { NCascader, NInput, NUpload } from "naive-ui";
import { ArchiveOutline as ArchiveIcon } from "@vicons/ionicons5";
import { cloneDeep } from "lodash-es";
import inc_before from "@/assets/imgs/inc-before.png";
import inc_after from "@/assets/imgs/inc-after.jpg";
import arrow from "@/assets/svg-icon/arrow.svg";
import {
  getDataSetListNoPage,
  getDictDataTree,
  getExampleList,
  submitTask,
} from "@/service/api/expansion";
import { computed } from "vue";
import { useFormRules, useNaiveForm } from "@/hooks/common/form";
import { getModelList, trainStart } from "@/service/api/model-manage";
import { selectDataSetLabel } from "@/service/api/ano";
import _ from "lodash";
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
const fileAction = `${import.meta.env.VITE_SERVICE_BASE_URL
  }/algorithm/task/readFileToJson`;
const uploadRef = ref(null);
const dynamicList = ref<any>([]);

const configList = ref<any>([
  {
    name: "基本信息",
    list: [
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
        formName: "描述信息",
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
        isShow: true,
      },
      // {
      //   formName: "基础模型版本",
      //   type: "text",
      //   modelList: [],
      //   value: "V1",
      //   width: "30%",
      //   serverKey: "modelVersion",
      // },
    ],
  },
  {
    name: "数据配置",
    list: [
      {
        formName: "选择数据集",
        type: "cascader",
        value: null,
        options: [],
        placeholder: "请选择数据集",
        width: "30%",
        serverKey: "datasetId",
        isShow: true,
      },
      {
        formName: "选择数据集标签",
        type: "select",
        isMultiple: true,
        value: null,
        options: [],
        placeholder: "请选择数据集",
        width: "30%",
        serverKey: "classes",
        isShow: false,
        renderLabel: (options: any) => {
          return [h("span", {}, `${options.value}`)];
        },
      },
    ],
  },
  {
    name: "训练配置",
    list: [
      {
        formName: "参数配置",
        type: "dynamicInput",
        value: [
          { key: "", value: "", desc: "", label: "", valuePlaceholder: "" },
        ],
        placeholder: "",
        width: "98%",
        serverKey: "trainPrams",
        query: [],
        columns: [],
        isShow: true,
      },
    ],
  },
  // {
  //   name: "发布模型",
  //   list: [
  //     {
  //       formName: "模型名称",
  //       type: "input",
  //       value: "",
  //       placeholder: "请输入模型名称",
  //       width: "30%",
  //       path: "modelName",
  //       serverKey: "modelName",
  //     },
  //     { formName: "模型版本", type: "text", value: "V1", width: "30%", serverKey: "modelVersion" },
  //     {
  //       formName: "版本描述",
  //       type: "textarea",
  //       value: "",
  //       placeholder: "请输入版本描述",
  //       width: "30%",
  //       serverKey: "modelDesc",
  //     },
  //   ],
  // },
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
    dynamicList.value = resData.data instanceof Array ? resData.data : [];
    window.$message?.success?.(
      resData.code == 200 ? "配置文件解析成功！" : "配置文件解析失败！",
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
    return Object.assign({}, item, {
      label: item.modelName,
      value: item.modelUrl,
    });
  });
  configList.value[0].list[2].modelList = options;
  // 算法列表
  if (options.length > 0) {
    configList.value[0].list[2].value = options[0].modelId;
    const res1 = await getExampleList({ modelId: options[0].modelId });

    configList.value[0].list[3].modelList = res1.data.map((item: any) => {
      return Object.assign({}, item, {
        label: item.algorithmName,
        value: item.trainType,
      });
    });
    // configList.value[0].list[3].value = configList.value[0].list[3].modelList[0].value;
    // ---------------------------------------------
    const res2 = await getExampleList({ id: 12 });
    configList.value[0].list[3].serverKey = res2.data[0].paramsMap[0].serverKey; // 选择优化模块绑定serverKey （后端动态变）
    configList.value[1].list[1].serverKey = res2.data[0].paramsMap[1].serverKey; // 数据集标签 serverKey
  }
  // 训练配置
  const modelParams = options[0].modelParams;
  const index = configList.value.findIndex((item) => item.name === "训练配置");
  if (index > -1) {
    const list = configList.value[index].list;
    const dataIndex = list.findIndex((item) => item.formName === "参数配置");
    if (dataIndex > -1) {
      // query
      list[dataIndex].query = modelParams
        ? JSON.parse(modelParams).map((val, idx) => {
          return {
            index: idx,
            ...val,
            key: val.serverKey, // val.value
            value: val.value ? val.value : null, // val.label
            type: val.type,
            valuePlaceholder: val.label,
            disabled: !!val.bindKey
          };
        }).filter(val => {
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
                      `${row.label}`,
                    ),
                    h(
                      NPopover,
                      { trigger: "hover", placement: "right", style: { 'max-width': '360px' } },
                      {
                        trigger: () =>
                          h("span", { class: "block ml-4px" }, [
                            h(SvgIcon, {
                              icon: "fluent:info-24-regular",
                              class: "text-[14px]",
                              localIcon: "fluent--info-24-regular",
                            }),
                          ]),
                        default: () =>
                          h("div", {}, `${row.tooltip}`),
                      },
                    ),
                  ],
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
                    disabled: row.disabled ? true : false,
                    multiple: row.isMulSelect ? true : false,
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
            minWidth: "300"
          }
        ]
        : [];

      // 任务名称 prev
      const nameInfo = list[dataIndex].query.find(val => {
        return val.serverKey === "name";
      });
      prevTaskName.value = nameInfo.value;
    }
  }
}

function handleRadioChange(e: any, row: any, formName: any) {
  if (formName === "选择基础模型") {
    // 训练配置
    const index = configList.value.findIndex(
      (item) => item.name === "训练配置",
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
    // configList 训练配置
    const trainCList = configList.value.find(val => val.name === "训练配置").list;
    // 参数配置
    const paramList = trainCList.find(val => val.formName === "参数配置").query;
    // 类型次数 + 类别名称
    const numInfoIdx = paramList.findIndex(val => val.bindKey && val.bindKey === "tagNum");
    const nameInfoIdx = paramList.findIndex(val => val.bindKey && val.bindKey === "tags");
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
        title: '警告',
        content: '模型名称未修改，提交会覆盖原有模型,是否继续?',
        positiveText: '继续',
        negativeText: '取消',
        onPositiveClick: async () => {
          // message.success('继续');
          let params = {
            algorithmParam: {},
          };
          // configList.value 过滤训练配置
          const dataList1 = configList.value.filter(
            (item) => item.name === "训练配置",
          );
          const dataList2 = configList.value.filter(
            (item) => item.name !== "训练配置",
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
            (item) => item.name === "基本信息",
          );
          if (index > -1) {
            const list = configList.value[index].list;
            const dataIndex = list.findIndex(
              (item) => item.formName === "选择优化模块",
            );
            if (dataIndex > -1) {
              const obj = list[dataIndex];
              params.algorithmParam[obj.serverKey] = obj.value;
              delete params[obj.serverKey];
            }
          }
          const index1 = configList.value.findIndex(
            (item) => item.name === "数据配置",
          );
          if (index1 > -1) {
            const list1 = configList.value[index1].list;

            const dataIndex1 = list1.findIndex(
              (item) => item.formName === "选择数据集标签",
            );
            if (dataIndex1 > -1) {
              const obj1 = list1[dataIndex1];
              params.algorithmParam[obj1.serverKey] = obj1.value;
              delete params.classes;
              if (
                params.algorithmParam.classes instanceof Array &&
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
        }
      });
      return;
    } else {
      let params = {
        algorithmParam: {},
      };
      // configList.value 过滤训练配置
      const dataList1 = configList.value.filter(
        (item) => item.name === "训练配置",
      );
      const dataList2 = configList.value.filter(
        (item) => item.name !== "训练配置",
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
        (item) => item.name === "基本信息",
      );
      if (index > -1) {
        const list = configList.value[index].list;
        const dataIndex = list.findIndex(
          (item) => item.formName === "选择优化模块",
        );
        if (dataIndex > -1) {
          const obj = list[dataIndex];
          params.algorithmParam[obj.serverKey] = obj.value;
          delete params[obj.serverKey];
        }
      }
      const index1 = configList.value.findIndex(
        (item) => item.name === "数据配置",
      );
      if (index1 > -1) {
        const list1 = configList.value[index1].list;

        const dataIndex1 = list1.findIndex(
          (item) => item.formName === "选择数据集标签",
        );
        if (dataIndex1 > -1) {
          const obj1 = list1[dataIndex1];
          params.algorithmParam[obj1.serverKey] = obj1.value;
          delete params.classes;
          if (
            params.algorithmParam.classes instanceof Array &&
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
  }
  if (sign === "back") {
    router.back();
  }
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

// lifecycle
onMounted(async () => {
  await getMapList();
  await getModels(); // 获取模型列表
});

onBeforeUnmount(() => { });
</script>

<template>
  <div class="w-full h-full box-border flex flex-col justify-start items-start wrap-container">
    <div class="w-full flex-1 box-border p-24px !overflow-y-auto">
      <n-form :ref="formRef" :rules="taskRules" label-placement="left" label-width="auto"
        require-mark-placement="right-hanging" class="!w-100% h-full">
        <div class="w-full h-auto flex flex-col justify-start items-center gap-12px !overflow-y-auto !pb-18px">
          <div class="w-full h-auto flex items-center" v-for="(item, index) of configList" :key="index">
            <n-card class="w-full h-auto" :title="item.name">
              <div class="w-full h-auto flex flex-col items-start">
                <div :style="{ width: val.width ?? '100%' }" v-for="(val, idx) of item.list" :key="idx">
                  <!-- input -->
                  <n-grid :cols="24" :x-gap="24" class="ml-24px" v-if="val.type === 'input'">
                    <n-form-item-gi :span="24" :label="val.formName" :path="val.path">
                      <n-input v-model:value="val.value" clearable :placeholder="val.placeholder" />
                    </n-form-item-gi>
                  </n-grid>
                  <!-- textarea -->
                  <n-grid :cols="24" :x-gap="24" class="ml-24px" v-if="val.type === 'textarea'">
                    <n-form-item-gi :span="24" :label="val.formName" :path="val.path">
                      <n-input type="textarea" v-model:value="val.value" clearable :placeholder="val.placeholder" />
                    </n-form-item-gi>
                  </n-grid>
                  <!-- select -->
                  <n-grid :cols="24" :x-gap="24" class="ml-24px" v-if="val.type === 'select'">
                    <n-form-item-gi :span="24" :label="val.formName" :path="val.path" v-if="val.isShow">
                      <n-select @update:value="handleSelectChange($event, val)" v-model:value="val.value"
                        placeholder="请选择数据集标签" :multiple="val.isMultiple" :options="val.options"
                        :render-label="val.renderLabel" />
                    </n-form-item-gi>
                  </n-grid>
                  <!-- dynamicInput -->
                  <n-grid :cols="24" :x-gap="24" class="ml-24px" v-if="val.type === 'dynamicInput'">
                    <n-form-item-gi :span="24" :label="val.formName" :path="val.path">
                      <!-- <NDynamicInput v-model:value="val.query" preset="pair" key-placeholder="请输入参数Key"
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
</NDynamicInput> -->
                      <n-data-table :bordered="false" :single-line="false" :columns="val.columns" :data="val.query" />
                    </n-form-item-gi>
                  </n-grid>
                  <!-- text -->
                  <n-grid :cols="24" :x-gap="24" class="ml-24px" v-if="val.type === 'text'">
                    <n-form-item-gi :span="24" :label="val.formName" :path="val.path">
                      {{ val.value }}
                    </n-form-item-gi>
                  </n-grid>
                  <!-- radioGroup -->
                  <n-grid :cols="24" :x-gap="24" class="ml-24px" v-if="val.type === 'radioGroup'">
                    <n-form-item-gi :span="24" :label="val.formName" :path="val.path"
                      v-if="val.modelList.length > 0 && val.isShow">
                      <n-radio-group v-model:value="val.value" name="anoType" size="large">
                        <!-- 选中事件 -->
                        <n-radio-button v-for="(val1, idx1) of val.modelList" :key="idx1" :value="val1.value"
                          :label="val1.label" @change="
                            handleRadioChange($event, val1, val.formName)
                            "></n-radio-button>
                      </n-radio-group>
                    </n-form-item-gi>
                  </n-grid>
                  <!-- cascader -->
                  <n-grid :cols="24" :x-gap="24" class="ml-24px" v-if="val.type === 'cascader'">
                    <n-form-item-gi :span="24" :label="val.formName" :path="val.path">
                      <n-cascader @update:value="handleCascaderChange($event, val)" v-model:value="val.value" clearable
                        :placeholder="val.placeholder" :options="val.options" check-strategy="child" :show-path="true"
                        expand-trigger="hover" :render-label="renderLabel" />
                    </n-form-item-gi>
                  </n-grid>
                </div>
              </div>
            </n-card>
          </div>
        </div>
      </n-form>
    </div>
    <div class="footer w-full box-border flex justify-start items-center gap-24px px-24px py-12px bg-[#fff]">
      <n-button type="info" @click="handleOperate('submit')" class="w-88px">
        开始训练
      </n-button>
      <n-button type="default" @click="handleOperate('back')" class="w-88px">
        返回
      </n-button>
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
