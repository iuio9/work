<script setup lang="ts">
import {
  NInputNumber,
  NPopover,
  NSelect,
  NCascader,
  NInput,
  useDialog,
  useMessage,
  SelectOption,
} from "naive-ui";
import { computed } from "vue";
import _ from "lodash";
import inc_before from "@/assets/imgs/inc-before.png";
import inc_after from "@/assets/imgs/inc-after.jpg";
import arrow from "@/assets/svg-icon/arrow.svg";
import {
  getExampleList,
} from "@/service/api/expansion";
import { useFormRules, useNaiveForm } from "@/hooks/common/form";
import { getModelList, trainStart } from "@/service/api/model-manage";
import { selectDataSetLabel } from "@/service/api/ano";
import SvgIcon from "@/components/custom/svg-icon.vue";
import EvaluationMetrics from "./moduels/EvaluationMetrics.vue"

import { getModelTypeList, getDataSetListNoPage, createAssessTask, selectIncrementIdAssessTask, assessModelDetail, assessModelEdit, getModelDebugInfo, getAlgorithmList, assessTaskDetail, deleteFile, assessDeleteFile, editAssessTask } from "@/service/api/third";
import { render } from "nprogress";
import { useTable, useTableOperate } from "@/hooks/common/table";

import BindIndicatorModal from "./moduels/BindIndicatorModal.vue";
import { nanoid } from '~/packages/utils/src';
import { useBoolean } from '~/packages/hooks/src';

interface Model {
  mapImport: string;

  [key: string]: any;
}

const { formRef, validate, restoreValidation } = useNaiveForm();
// data
const taskRules = computed<Record<keyof Model, App.Global.FormRule[]>>(() => {
  const { defaultRequiredRule } = useFormRules();
  return {
    taskName: defaultRequiredRule,
    modelName: defaultRequiredRule,
  };
});

// ----------------newCode----------------
const router = useRouter();
const route = useRoute();

const prevTaskName = ref<any>(""); // 任务名称
const nextTaskName = ref<any>("");
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
                      clearable: true,
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

const dialog = useDialog();

let timer: number | null = null;
let timer1: number | null = null;

async function handleOperate(sign: any) {
  if (sign === "submit") {
    await validate();
    if (!nextTaskName.value) {
      dialog.warning({
        title: "警告",
        content: "模型名称未修改，提交会覆盖原有模型,是否继续?",
        positiveText: "继续",
        negativeText: "取消",
        onPositiveClick: async () => {
          // message.success('继续');
          const params: any = {
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
                if (val2.serverKey === 'modelClass') {
                  params.modelClass = JSON.stringify(val2.value);
                } else {
                  params[val2.serverKey] = val2.value;
                }
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
              if (!params.taskName) {
                window.$message?.error?.("任务名称不能为空！");
                return;
              }
              const res = await trainStart(params);
              if (res.data) {
                timer = setTimeout(() => {
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
    const params: any = {
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
          if (val2.serverKey === 'modelClass') {
            console.log('val1: ', val1);
            console.log('val2: ', val2);
            params.modelClass = JSON.stringify(val2.value);
          } else {
            params[val2.serverKey] = val2.value;
          }
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
        if (!params.taskName) {
          window.$message?.error?.("任务名称不能为空！");
          return;
        }
        const res = await trainStart(params);
        if (res.data) {
          // window.$message?.success?.("任务训练开始...");
          // router.back();
          timer1 = setTimeout(() => {
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

// ----------------------------------newCode-----------------------------------//
// 定义数据集节点类型
interface DatasetNode {
  groupId?: string;
  sonId?: string;
  groupName?: string;
  version?: string;
  dataSonResponseList?: DatasetNode[];
  value: string | number;
  label: string;
  children?: DatasetNode[];
  disabled: boolean;
}

// 定义配置项类型
interface ConfigSection {
  name: string;
  isShow: boolean;
  list: ConfigField[];
}

interface ConfigField {
  formName: string;
  type: string;
  serverKey: string;
  options?: DatasetNode[];
  [key: string]: any;
}

// 定义任务类型选项
enum TaskType {
  TEST = '1', // 测试
  EVALUATION = '2' // 评估
}

// 定义配置字段类型
interface ConfigField {
  serverKey: string;
  isShow: boolean;
}

// 定义行数据类型
interface RadioRow {
  label: string;
  value: string;
}

interface ModelOption {
  label: string;
  value: string | number;
}

// 定义表单字段类型
type FormFieldType =
  | 'input'
  | 'textarea'
  | 'radioGroup'
  | 'cascader'
  | 'checkboxGroup'
  | 'upload'
  | 'quaternary';

// 定义基础表单字段接口
interface BaseFormField {
  formName: string;
  type: FormFieldType;
  value: any;
  width?: string;
  serverKey: string;
  isShow: boolean;
  placeholder?: string;
  path?: string;
}

// 单选组字段
interface RadioGroupField extends BaseFormField {
  type: 'radioGroup';
  modelList: { label: string; value: string | number }[];
}

// 级联选择字段
interface CascaderField extends BaseFormField {
  type: 'cascader';
  options: any[];
}

// 复选框组字段
interface CheckboxGroupField extends BaseFormField {
  type: 'checkboxGroup';
  options: { label: string; value: string | number }[];
}

// 上传字段
interface UploadField extends BaseFormField {
  type: 'upload';
}

// 输入字段
interface InputField extends BaseFormField {
  type: 'input' | 'textarea';
}

// 四元按钮字段
interface QuaternaryField extends BaseFormField {
  type: 'quaternary';
}

// 联合所有字段类型
type FormField =
  | InputField
  | RadioGroupField
  | CascaderField
  | CheckboxGroupField
  | UploadField
  | QuaternaryField;

// 配置部分接口
interface ConfigSection {
  name: string;
  isShow: boolean;
  list: any[];
}

// codeMap
const codeMap = ref(new Map<string | number, string | number>())

// 指标列表
const metricList = ref<any>([
  { label: '平均精度 (mPrecision)', prop: 'mPrecision', value: "1", key: 0 },
  { label: '平均召回率 (mRecall)', prop: 'mRecall', value: "1", key: 1 },
  { label: '均值平均精度 (mAP@0.5)', prop: 'mAP@0.5', value: "1", key: 2 },
  { label: '漏检率 (MissRate)', prop: 'MissRate', value: "1", key: 3 },
  { label: '虚警率 (FalseAlarmRate)', prop: 'FalseAlarmRate', value: "1", key: 4 },
  { label: '平均正确率 (mAccuracy)', prop: 'mAccuracy', value: "1", key: 5 }
])

const labelList = ref<any>([]);

// 配置列表
const configList = ref<any>([
  {
    name: "基本信息",
    isShow: true,
    list: [
      {
        formName: "任务名称",
        type: "input",
        value: "",
        placeholder: "请输入任务名称",
        width: "30%",
        path: "taskName",
        serverKey: "taskName",
        isShow: true,
      },
      {
        formName: "任务类型",
        type: "radioGroup",
        modelList: [
          {
            label: "分类任务",
            value: "1",
          },
          {
            label: "目标检测",
            value: "2",
          },
        ],
        value: "1",
        width: "30%",
        serverKey: "taskType",
        isShow: true,
      },
      {
        formName: "选择评估模型",
        type: "cascader",
        value: null,
        options: [],
        placeholder: "请选择评估模型",
        width: "30%",
        serverKey: "modelId",
        isShow: false,
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
        formName: "任务版本",
        type: "input",
        value: "",
        placeholder: "请输入任务版本",
        width: "30%",
        serverKey: "taskVersion",
        isShow: true,
      },
      {
        formName: "版本描述",
        type: "textarea",
        value: "",
        placeholder: "请输入版本描述",
        width: "30%",
        serverKey: "versionDesc",
        isShow: true,
      }
    ],
  },
  {
    name: "评估对象配置",
    isShow: true,
    list: [
      {
        formName: "评估数据集",
        type: "cascader",
        value: null,
        options: [],
        placeholder: "请选择评估数据集",
        width: "30%",
        serverKey: "sonId",
        isShow: true,
      },
      {
        formName: "标签映射",
        type: "table1",
        value: [],
        columns: [
          {
            type: 'selection'
          },
          {
            title: '模型识别类别编码',
            key: 'code',
          },
          {
            title: '模型识别类别标签',
            key: 'algorithm',
          },
          {
            title: '模型识别类别说明',
            key: 'chineseInfo',
          },
          {
            title: '评估数据集标签',
            key: 'labelName',
            render(row: any) {
              return h(NSelect, {
                value: row.labelName,
                filterable: true,
                options: tagList.value,
                clearable: true,
                renderLabel: (option: any) => {
                  return [
                    h('div', {
                      class: 'w-full flex justify-start gap-8px',
                    }, [
                      h(NPopover, { trigger: "hover", placement: "top" }, {
                        trigger: () => [
                          h(
                            "span",
                            { class: 'truncate' },
                            `${option.label}`
                          ),
                        ],
                        default: () => [
                          h("div", { class: "w-full truncate" }, `${option.label}`)
                        ]
                      })
                    ])
                  ]
                },
                'onUpdate:value': (val: any) => {
                  row.labelName = val;
                }
              })
            }
          },
        ],
        algorithmList: [],
        width: "80%",
        serverKey: "labelMap",
        isShow: false,
      },
      {
        formName: "模型接口地址",
        type: "input",
        value: "",
        placeholder: "请输入模型接口地址",
        width: "45%",
        serverKey: "modelAddress",
        isShow: true,
      },
      {
        formName: "模型传输方式",
        type: "cascader",
        value: null,
        options: [
          { value: '2', label: 'get' },
          { value: '1', label: 'post' },
        ],
        placeholder: "请选择模型传输方式",
        width: "45%",
        serverKey: "requestType",
        isShow: true,
      },
      {
        formName: "上传模型参数文件",
        type: "upload",
        value: "",
        placeholder: "请上传文件",
        width: "45%",
        serverKey: "modelParamsFile",
        isShow: true,
        fileList: [],
        beforeUpload: (options: any) => {
          const { file } = options;
          const isJson = file.type === 'application/json';
          const isExcel = file.type === 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet';
          if (!isJson && !isExcel) {
            window.$message?.error?.('只能上传json和excel文件');
            return false;
          }
        }
      },
      {
        formName: "替换参数名",
        type: "input",
        value: "",
        placeholder: "请输入替换参数名",
        width: "45%",
        serverKey: "modelFileName",
        isShow: true,
      },
      {
        formName: "评估描述",
        type: "textarea",
        value: "",
        placeholder: "请输入评估描述",
        width: "45%",
        serverKey: "assessDesc",
        isShow: true,
      },
    ],
  },
  // {
  //   name: "评估指标配置",
  //   isShow: false,
  //   list: [
  //     {
  //       formName: "评估指标",
  //       type: "table",
  //       value: [
  //         { label: '平均精度 (mPrecision)', prop: 'mPrecision', value: "1", key: 0 },
  //         { label: '平均召回率 (mRecall)', prop: 'mRecall', value: "1", key: 1 },
  //         { label: '均值平均精度 (mAP@0.5)', prop: 'mAP@0.5', value: "1", key: 2 },
  //         { label: '漏检率 (MissRate)', prop: 'MissRate', value: "1", key: 3 },
  //         { label: '虚警率 (FalseAlarmRate)', prop: 'FalseAlarmRate', value: "1", key: 4 },
  //         { label: '平均正确率 (mAccuracy)', prop: 'mAccuracy', value: "1", key: 5 }
  //       ],
  //       columns: [
  //         {
  //           type: 'selection'
  //         },
  //         {
  //           title: '指标名称',
  //           key: 'label',
  //         },
  //         {
  //           title: '指标标识',
  //           key: 'prop',
  //         },
  //         {
  //           title: '指标值',
  //           key: 'value',
  //           render: (row, index) => {
  //             return h(NInput, {
  //               clearable: true,
  //               value: row.value,
  //               onUpdateValue: (v) => {
  //                 row.value = v // 直接通过index更新数据
  //               }
  //             })
  //           }
  //         }
  //       ],
  //       checkedRowKeys: [],
  //       defaultCheckedRowKeys: [],
  //       width: "80%",
  //       serverKey: "assessTarget",
  //       isShow: true
  //     },
  //     {
  //       formName: "评估图表",
  //       type: "checkboxGroup",
  //       value: ["PRCurve", "ConfusionMatrix"],
  //       options: [
  //         { value: 'PRCurve', label: 'P-R曲线' },
  //         { value: 'ConfusionMatrix', label: '混淆矩阵图' },
  //       ],
  //       width: "98%",
  //       serverKey: "assessChart",
  //       isShow: true,
  //     },
  //   ]
  // }
  {
    name: '模型识别配置',
    list: [
      {
        formName: '模型识别类别',
        type: 'dynamicModalClass',
        value: [],
        serverKey: 'modelClass',
        isShow: true,
        width: "60%"
      },
      {
        formName: "评估图表",
        type: "checkboxGroup",
        value: ["PRCurve", "ConfusionMatrix", "ROC"],
        options: [
          { value: 'PRCurve', label: 'P-R曲线', key: 'PRCurve' },
          { value: 'ConfusionMatrix', label: '混淆矩阵图', key: 'ConfusionMatrix' },
          { value: 'ROC', label: 'ROC曲线', key: 'ROC' },
        ],
        width: "98%",
        serverKey: "assessChart",
        isShow: true,
      },
    ],
    isShow: true
  }
]);
const tagList = ref<any>([]);

const fetchDatasetOptions = async () => {
  // 处理数据集树形结构
  const processDatasetTree = (items: DatasetNode[], parentLabel?: string): DatasetNode[] => {
    return items.map(item => {
      const value = item.groupId || item.sonId;
      const label = parentLabel
        ? `${parentLabel} - ${item.groupName || `V${item.version}`}`
        : item.groupName || `V${item.version}`;

      // 处理子节点
      const children = item.dataSonResponseList || [];
      const processedChildren = children.map(child => ({
        ...child,
        value: child.groupId || child.sonId,
        label: child.groupName || `V${child.version}`
      }));

      // 递归处理子节点
      const hasChildren = processedChildren.length > 0;
      return {
        ...item,
        value,
        label,
        children: hasChildren ? processDatasetTree(processedChildren, label) : undefined,
        disabled: item.isMany == 1,
      };
    });
  };

  try {
    // 获取数据集列表
    const { data } = await getDataSetListNoPage();

    // 类型断言确保数据结构匹配
    const datasetItems = data as DatasetNode[];

    // 更新配置项中的数据集选项
    const configSectionIndex = configList.value.findIndex((section: ConfigSection) =>
      section.name === "评估对象配置"
    );

    if (configSectionIndex !== -1) {
      const datasetFieldIndex = configList.value[configSectionIndex].list.findIndex(
        (field: ConfigField) => field.serverKey === "sonId"
      );

      if (datasetFieldIndex !== -1) {
        configList.value[configSectionIndex].list[datasetFieldIndex].options = processDatasetTree(datasetItems);
      }
    }
  } catch (error) {
    console.log('获取数据集列表失败:', error);
  }
};

const handleRadioChange = async (e: Event, row: RadioRow, formName: string) => {
  const { value } = row;

  // 任务类型变更逻辑
  if (formName === "任务类型") {
    // 更新评估指标配置可见性
    // configList.value[2].isShow = row.value == 2;

    // 更新评估对象配置中特定字段的可见性
    const evaluationFields = configList.value[1].list;
    const fieldsToToggle = ['fileName'];

    evaluationFields.forEach((field: ConfigField) => {
      if (fieldsToToggle.includes(field.serverKey)) {
        field.isShow = isTestTask;
      }
    });

    // 配置模型config
    loadAndUpdateModelOptions(value);
  }
};

/**
 * 根据模型方式加载模型列表并更新配置
 * @param modelWay 模型方式值
 * @param resetFirst 是否先重置（默认true）
 */
const loadAndUpdateModelOptions = async (modelWay: string, resetFirst = true) => {
  try {
    // 预处理：清空现有选项并隐藏字段
    if (resetFirst) {
      const basicInfoSectionIndex = configList.value.findIndex(
        (section: ConfigSection) => section.name === "基本信息"
      );

      if (basicInfoSectionIndex !== -1) {
        const modelFieldIndex = configList.value[basicInfoSectionIndex].list.findIndex(
          (field: ConfigField) => field.serverKey === "modelId"
        );

        if (modelFieldIndex !== -1) {
          const modelField = configList.value[basicInfoSectionIndex].list[modelFieldIndex];
          modelField.isShow = false;
          modelField.options = [];
          modelField.value = ""; // 清空选中值
        }
      }
    }

    // 获取模型类型列表
    const { data } = await getModelTypeList({ modelWay });
    if (data && data.length >= 0) {
      // 转换为选项格式
      const modelOptions: ModelOption[] = data.map(item => ({
        label: item.modelName,
        value: item.id
      }));

      // 更新配置项中的模型选项
      const basicInfoSectionIndex = configList.value.findIndex(
        (section: ConfigSection) => section.name === "基本信息"
      );

      if (basicInfoSectionIndex !== -1) {
        const modelFieldIndex = configList.value[basicInfoSectionIndex].list.findIndex(
          (field: ConfigField) => field.serverKey === "modelId"
        );

        if (modelFieldIndex !== -1) {
          // 更新模型字段配置
          const modelField = configList.value[basicInfoSectionIndex].list[modelFieldIndex];
          modelField.isShow = true;
          modelField.options = modelOptions;

          // 如果选项不为空，默认选择第一个
          if (modelOptions.length > 0) {
            // modelField.value = modelOptions[0].value;
            modelField.value = null;
          } else {
            modelField.value = null; // 确保字段值为空
          }
        }
      }
    }
  } catch (error) {
    console.log('加载模型列表失败:', error);

    // 错误时确保字段隐藏
    const basicInfoSectionIndex = configList.value.findIndex(
      (section: ConfigSection) => section.name === "基本信息"
    );

    if (basicInfoSectionIndex !== -1) {
      const modelFieldIndex = configList.value[basicInfoSectionIndex].list.findIndex(
        (field: ConfigField) => field.serverKey === "modelId"
      );

      if (modelFieldIndex !== -1) {
        configList.value[basicInfoSectionIndex].list[modelFieldIndex].isShow = false;
      }
    }
  }
};

function filterHiddenItems(configList) {
  return configList
    .filter(section => section.isShow !== false) // 过滤顶级隐藏分组
    .map(section => ({
      ...section,
      list: section.list
        .filter(item => item.isShow !== false) // 过滤隐藏的表单项
        .map(item => {
          // 递归处理嵌套结构（如果有）
          if (item.children && Array.isArray(item.children)) {
            return {
              ...item,
              children: filterHiddenItems(item.children)
            };
          }
          return item;
        })
    }));
}

const handleDefine = async () => {
  const formData = new FormData();

  const dataList = filterHiddenItems(configList.value);

  dataList.forEach(section => {
    section.list.forEach(item => {
      const { serverKey, value, checkedRowKeys } = item;

      // 特殊处理：modelParamsFile 需提取文件列表中的文件名
      if (serverKey === 'modelParamsFile') {
        const file = item.fileList[0]?.file;
        if (file) {
          formData.append(serverKey, file);
        }
      } else if (serverKey === 'labelMap') {
        if (value) {
          // formData.append(serverKey, JSON.stringify(labelList.value));
          const labelMap = algorithmData.value.map((item: any) => {
            const labelObj = (tagList.value).find((tag: any) => tag.labelId === item.labelName);
            return {
              ...item,
              ...labelObj,
              mapLabel: item.code,
            };
          });
          formData.append(serverKey, JSON.stringify(labelMap));
        }
      } else if (serverKey === 'modelClass') {
        if (value) {
          formData.append(serverKey, JSON.stringify(value));
        }
      } else if (serverKey === 'assessTarget') {
        formData.append(serverKey, checkedRowKeys.join(','));
        formData.append('assessTargetMap', JSON.stringify(metricList.value));
      } else {
        // 其他字段直接处理 value
        if (Array.isArray(value)) {
          formData.append(serverKey, value.join(","));
        } else if (value !== undefined && value !== null) {
          // 非数组类型：直接添加
          formData.append(serverKey, value);
        }
      }
    });
  });

  if (route.query.sign === 'create') {
    const res = await createAssessTask(formData);
    if (res.data) {
      window.$message?.success?.('评估任务创建成功');
      router.back();
    }
  }

  if (route.query.sign === 'edit') {
    const id = route.query.id;
    formData.append('id', id);
    const res = await editAssessTask(formData);
    if (res.data) {
      window.$message?.success?.('评估任务编辑成功');
      router.back();
    }
  }
}

// 选择评估模型
const handleCasUpdate = async (event, row) => {
  if (row.serverKey === 'modelId') {
    modelId.value = row.value;

    // prev clean
    const [debugInfo, algorithmListRes] = await Promise.all([
      getModelDebugInfo({ modelId: row.value }),
      getAlgorithmList({ modelId: row.value, page: 1, limit: 10 })
    ]);
    if (!_.isEqual(algorithmListRes.data, algorithmList.value)) {
      algorithmList.value = algorithmListRes.data || [];
      codeMap.value.clear();
      labelList.value = labelList.value.map(val => {
        return {
          ...val,
          mapLabel: undefined
        }
      })
    }
    debugInfo.data && updateConfigByServerKey(configList.value, debugInfo.data);
  }

  if (row.serverKey === 'sonId') {
    // const res = await selectDataSetLabel({
    //   sonId: row.value
    // });
    // if (res.data) {
    //   const basicInfoSectionIndex = configList.value.findIndex(
    //     (section: ConfigSection) => section.name === "评估对象配置"
    //   );

    //   if (basicInfoSectionIndex !== -1) {
    //     const modelFieldIndex = configList.value[basicInfoSectionIndex].list.findIndex(
    //       (field: ConfigField) => field.serverKey === "labelMap"
    //     );

    //     if (modelFieldIndex !== -1) {
    //       configList.value[basicInfoSectionIndex].list[modelFieldIndex].isShow = true;
    //       configList.value[basicInfoSectionIndex].list[modelFieldIndex].value = res.data.map(item => {
    //         return {
    //           ...item,
    //           labelMap: undefined
    //         }
    //       });
    //       labelList.value = res.data.map(item => {
    //         return {
    //           ...item,
    //           labelMap: undefined
    //         }
    //       });
    //     }
    //   }
    // }

    // newCode
    const basicInfoSectionIndex = configList.value.findIndex(
      (section: ConfigSection) => section.name === "基本信息"
    );
    const modelIdIndex = configList.value[basicInfoSectionIndex].list.findIndex(
      (field: ConfigField) => field.serverKey === "modelId"
    );
    const modelId = configList.value[basicInfoSectionIndex].list[modelIdIndex].value;
    const datasetId = row.value; // 评估数据集
    if (modelId && datasetId) {
      const datasetSectionIndex = configList.value.findIndex(
        (section: ConfigSection) => section.name === "评估对象配置"
      );
      const labelMapIndex = configList.value[datasetSectionIndex].list.findIndex(
        (field: ConfigField) => field.serverKey === "labelMap"
      );
      configList.value[datasetSectionIndex].list[labelMapIndex].isShow = true;
      updateSearchParams({
        current: 1,
        size: 10,
        modelId
      });
      await getData();
      await setTagList(row.value);
    }
  }
}

async function setTagList(sonId: string) {
  const res = await selectDataSetLabel({
    sonId
  });
  if (res.data) {
    tagList.value = res.data.map(tagItem => {
      return {
        ...tagItem,
        label: `${tagItem.labelName}-${tagItem.englishLabelName}`,
        value: tagItem.labelId,
      }
    });
  }
}

onMounted(async () => {
  // 配置评估数据集对象
  await fetchDatasetOptions();
  await loadAndUpdateModelOptions('1');

  const { query } = route;
  if (query.id) {
    await nextTick();
    assessTaskDetail({
      id: query.id
    }).then(async res => {
      const { data } = res;
      if (data) {
        if (data.modelId) {
          // const [debugInfo, algorithmListRes] = await Promise.all([
          //   getModelDebugInfo({ modelId: data.modelId }),
          //   getAlgorithmList({ modelId: data.modelId, page: 1, limit: 10 })
          // ]);
          // algorithmList.value = algorithmListRes.data || [];
          // debugInfo.data && updateConfigByServerKey(configList.value, debugInfo.data);

          // newCode
          modelId.value = data.modelId;
          updateConfigItemByServerKey(configList.value, 'taskType', {
            value: (data.taskType).toString()
          });
          await setCasOptions('modelId', configList.value);
          updateConfigItemByServerKey(configList.value, 'modelId', {
            value: data.modelId
          })
        }

        if (data.modelId && data.sonId) {
          updateSearchParams({
            current: 1,
            size: 10,
            modelId: data.modelId
          });
          await getData();
          await setTagList(data.sonId);
          updateConfigItemByServerKey(configList.value, 'labelMap', {
            isShow: true
          });
          const labelMap = data.labelMap ? JSON.parse(data.labelMap) : [];
          labelMap.forEach((item: any) => {
            const algoIdx = (algorithmData.value).findIndex((algoItem: any) => algoItem.code === item.code);
            if (algoIdx !== -1) {
              algorithmData.value[algoIdx].labelName = item.labelName;
            }
          })
        }

        if (data.modelParamsPath) {
          let fileName = getFileNameWithExtension(data.modelParamsPath);
          let fileList = [
            { id: data.id, name: fileName, status: 'finished' }
          ];
          const basicInfoSectionIndex = configList.value.findIndex(
            (section: ConfigSection) => section.name === "评估对象配置"
          );
          if (basicInfoSectionIndex !== -1) {
            const modelParamsFileIndex = configList.value[basicInfoSectionIndex].list.findIndex(
              (field: ConfigField) => field.serverKey === "modelParamsFile"
            );
            if (modelParamsFileIndex !== -1) {
              configList.value[basicInfoSectionIndex].list[modelParamsFileIndex].fileList = fileList;
            }
          }
        }

        configList.value = updateConfigByServerKey1(configList.value, data);
      }
    })
  }
})

onUnmounted(() => {
  if (timer) {
    clearTimeout(timer);
  }
  if (timer1) {
    clearTimeout(timer1);
  }
});

const algorithmList = ref<any>([]);
const isAlgorithmModal = ref<Boolean>(false);

const handleAlgorithmClick = async (event, value) => {
  // const res = await getAlgorithmList({
  //   modelId: value
  // });
  // if (res.data) {
  //   algorithmList.value = res.data;
  //   isAlgorithmModal.value = true;
  // }
  const params = {
    current: 1,
    size: 10,
    modelId: value
  };
  updateSearchParams(params);
  await getData();
  isAlgorithmModal.value = true;
}

const updateConfigByServerKey = (configList, editData) => {
  configList.forEach(group =>
    group.list.forEach(item => {
      if (editData.hasOwnProperty(item.serverKey)) {
        // 特殊处理 requestType，转为 number 类型
        if (item.serverKey === 'requestType') {
          item.value = `${editData[item.serverKey]}`;
        } else if (item.serverKey === 'modelId') {
          item.value = +editData[item.serverKey];
        } else if (item.serverKey === 'taskType') {
          item.value = `${editData[item.serverKey]}`;
          if (editData[item.serverKey] == 2) {
            configList[2].isShow = true;
          }
        } else if (item.serverKey === 'assessTarget') {
          item.checkedRowKeys = editData[item.serverKey].split(",");
          item.value = JSON.parse(editData['assessTargetMap']);
        } else {
          item.value = editData[item.serverKey];
        }
      }
    })
  );
  return configList;
};

const updateConfigByServerKey1 = (configList, editData) => {
  // 深拷贝原数组，避免直接修改源数据（保持immutable）
  return configList.map(group => {
    // 处理taskType时需要修改的特定组索引
    const shouldShowThirdGroup = group?.list.some(
      item => item.serverKey === 'taskType' && editData[item.serverKey] == 2
    );

    // 映射处理每个item
    const updatedList = group?.list.map(item => {
      // 如果editData中没有对应的serverKey，直接返回原item
      if (!Object.prototype.hasOwnProperty.call(editData, item.serverKey)) {
        return item;
      }

      // 根据不同的serverKey进行处理
      switch (item.serverKey) {
        case 'requestType':
        case 'taskType':
          // 统一处理需要转为字符串的类型
          return { ...item, value: `${editData[item.serverKey]}` };

        case 'assessTarget':
          metricList.value = editData['assessTargetMap'] ? JSON.parse(editData['assessTargetMap']) : item.value;
          return {
            ...item,
            checkedRowKeys: editData['assessTarget'].split(',').map(val => +val),
            value: editData['assessTargetMap'] ? JSON.parse(editData['assessTargetMap']) : item.value
          };

        case 'labelMap':
          labelList.value = editData['labelMap'] ? JSON.parse(editData['labelMap']) : item.value;
          return {
            ...item,
            value: editData['labelMap'] ? JSON.parse(editData['labelMap']) : []
          };

        case 'assessChart':
          return {
            ...item,
            value: editData['assessChart'] ? editData['assessChart'].split(",") : item.value
          };
        case 'modelId':
          return {
            ...item,
            value: editData['modelId']
          };

        case 'modelClass':
          return {
            ...item,
            value: editData['modelClass'] ? JSON.parse(editData['modelClass']) : []
          };

        default:
          return { ...item, value: editData[item.serverKey] };
      }
    });

    // 处理第三组的显示状态（仅当当前是第三组且需要显示时）
    if (configList.indexOf(group) === 2 && shouldShowThirdGroup) {
      return { ...group, list: updatedList, isShow: true };
    }

    return { ...group, list: updatedList };
  });
};

/**
 * 从文件路径中获取文件名（包含后缀）
 * @param {string} filePath - 文件的完整路径
 * @returns {string} 文件名+后缀
 */
function getFileNameWithExtension(filePath) {
  // 处理不同操作系统的路径分隔符
  const separators = ['/', '\\'];
  let fileName = filePath;

  // 找到最后一个路径分隔符的位置
  separators.forEach(sep => {
    const index = fileName.lastIndexOf(sep);
    if (index !== -1) {
      fileName = fileName.substring(index + 1);
    }
  });

  return fileName;
}

async function handleRemove({ file }: any, rowData: any) {
  if (rowData.serverKey === 'modelParamsFile') {
    if (file.status === 'finished') {
      // res
      const res = await assessDeleteFile({
        serverKey: rowData?.serverKey,
        id: file.id,
      });
      if (res.data >= 1) {
        window.$message?.success?.("文件删除成功！");
        rowData.fileList = rowData.fileList.filter((item: any) => item.id !== file.id);
      }
    } else {
      rowData.fileList = rowData.fileList.filter((item: any) => item.id !== file.id);
    }
  }
}


// newCode
const {
  columns: algorithmColumns,
  columnChecks: algorithmColumnChecks,
  data: algorithmData,
  getData,
  loading,
  mobilePagination,
  updateSearchParams,
} = useTable({
  apiFn: getAlgorithmList,
  showTotal: true,
  apiParams: reactive({ current: 1, size: 10, modelId: '' }), // 初始参数,
  columns: () => [
    {
      title: '模型识别类别编码',
      key: 'code',
    },
    {
      title: '模型识别类别标签',
      key: 'algorithm',
    },
    {
      title: '模型识别类别说明',
      key: 'chineseInfo',
    },
    {
      title: '评估数据集标签',
      key: 'labelName',
      render(row: any) {
        return h(NSelect, {
          value: row.labelName,
          filterable: true,
          options: tagList.value,
          clearable: true,
          renderLabel: (option: any) => {
            return [
              h('div', {
                class: 'w-full flex justify-start gap-8px',
              }, [
                h(NPopover, { trigger: "hover", placement: "top" }, {
                  trigger: () => [
                    h(
                      "span",
                      { class: 'truncate' },
                      `${option.label}`
                    ),
                  ],
                  default: () => [
                    h("div", { class: "w-full truncate" }, `${option.label}`)
                  ]
                })
              ])
            ]
          },
        })
      }
    },
  ],
  immediate: false,
});

const {
  checkedRowKeys: algorithmCheckedRowKeys,
} = useTableOperate(algorithmData, getData);

// newCode
const modelId = ref<string | null>(null);
const classModel = ref<any>({}); // 模型识别类别
const { bool: indicatorDrawerVisible, setTrue: openIndicatorDrawer } = useBoolean();

function bindIndicator(item: any) {
  classModel.value = item;
  openIndicatorDrawer();
}

const handleCreate = () => ({
  className: undefined,
  classId: nanoid(),
  value: undefined,
  gridCheckedRowKeys: [],
  commonCheckedRowKeys: [],
  modelGridData: [
    { label: '召回率/发现率/检出率', prop: 'recall', value: "1", key: 0 },
    { label: '误检比', prop: 'falseAlarmRate', value: "1", key: 1 },
    { label: '误报率/误检率', prop: 'falseAlarmRate1', value: "1", key: 2 },
    { label: '平均精度AP', prop: 'ap', value: "1", key: 3 },
    { label: 'F1-分数', prop: 'f1', value: "1", key: 4 },
    { label: '识别时间', prop: 'time', value: "1", key: 5 },
    { label: 'IOU平均值', prop: 'iou', value: "1", key: 6 }
  ],
  modelCommonData: [
    { label: '平均精度 (mPrecision)', prop: 'mPrecision', value: "1", key: 0 },
    { label: '平均召回率 (mRecall)', prop: 'mRecall', value: "1", key: 1 },
    { label: '均值平均精度 (mAP@0.5)', prop: 'mAP@0.5', value: "1", key: 2 },
    { label: '漏检率 (MissRate)', prop: 'MissRate', value: "1", key: 3 },
    { label: '虚警率 (FalseAlarmRate)', prop: 'FalseAlarmRate', value: "1", key: 4 },
    { label: '平均正确率 (mAccuracy)', prop: 'mAccuracy', value: "1", key: 5 }
  ]
})

function handleAfterLeave(params: any) {
  const { classId, modelCommonData, modelGridData } = params;
  const modelTypeSectionIndex = configList.value.findIndex(
    (section: any) => section.name === "模型识别配置"
  );
  const modelTypeIndex = (configList.value[modelTypeSectionIndex].list).findIndex((item: any) => item.serverKey === 'modelClass');
  if (modelTypeIndex !== -1) {
    const classIndex = (configList.value[modelTypeSectionIndex].list[modelTypeIndex].value).findIndex((item: any) => item.classId === classId);
    if (classIndex !== -1) {
      configList.value[modelTypeSectionIndex].list[modelTypeIndex].value[classIndex].modelGridData = modelGridData;
      configList.value[modelTypeSectionIndex].list[modelTypeIndex].value[classIndex].modelCommonData = modelCommonData;
      configList.value[modelTypeSectionIndex].list[modelTypeIndex].value[classIndex].gridCheckedRowKeys = params.gridCheckedRowKeys;
      configList.value[modelTypeSectionIndex].list[modelTypeIndex].value[classIndex].commonCheckedRowKeys = params.commonCheckedRowKeys;
    }
  }
}

/**
 * 根据serverKey查找配置项并修改其属性值
 * @param configList 配置列表
 * @param serverKey 要查找的serverKey
 * @param props 要更新的属性键值对
 * @returns 是否找到并更新成功
 */
function updateConfigItemByServerKey(
  configList: any[],
  serverKey: string,
  props: Record<string, any>
): boolean {
  // 遍历所有配置组
  for (const group of configList) {
    // 遍历组内的配置项
    for (const item of group.list) {
      // 找到匹配serverKey的配置项
      if (item.serverKey === serverKey) {
        // 更新属性值
        Object.assign(item, props);
        return true; // 找到并更新成功，返回true
      }
    }
  }
  return false; // 未找到对应配置项，返回false
}

/**
 * 根据serverKey查找并返回对应的配置项
 * @param configList 配置列表
 * @param serverKey 要查找的serverKey
 * @returns 找到的配置项或undefined
 */
function getConfigItemByServerKey(
  configList: any[],
  serverKey: string
): any | undefined {
  // 遍历所有配置组
  for (const group of configList) {
    // 遍历组内的配置项
    for (const item of group.list) {
      // 找到匹配serverKey的配置项
      if (item.serverKey === serverKey) {
        return item; // 返回找到的配置项
      }
    }
  }
  return undefined; // 未找到对应配置项，返回undefined
}

async function setCasOptions(serverKey: string = 'modelId', configList: any) {
  // 评估模型
  if (serverKey === 'modelId') {
    const taskType = getConfigItemByServerKey(configList, 'taskType').value; // 任务类型
    const res = await getModelTypeList({ modelWay: taskType });
    if (res.data) {
      updateConfigItemByServerKey(configList, 'modelId', {
        options: (res.data).map((item: any) => ({
          label: item.modelName,
          value: item.id
        }))
      })
    }
  }
}

</script>

<template>
  <div class="wrap-container box-border h-full w-full flex flex-col items-start justify-start">
    <div class="box-border w-full flex-1 p-24px !overflow-y-auto">
      <NForm ref="formRef" :rules="taskRules" label-placement="left" label-width="140px"
        require-mark-placement="right-hanging" class="h-full !w-100%">
        <div class="h-auto w-full flex flex-col items-center justify-start gap-12px !overflow-y-auto !pb-18px">
          <div v-for="(item, index) of configList" :key="index" class="h-auto w-full flex items-center">
            <NCard class="h-auto w-full" :title="item.name" v-show="item.isShow">
              <div class="h-auto w-full flex" v-for="(val, idx) of item.list" :key="idx"
                :style="{ width: val.width ?? '100%' }">
                <div class="flex w-full" v-show="val.isShow">
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
                        :options="val.options" :render-label="val.renderLabel"
                        @update:value="handleSelectChange($event, val)" />
                    </NFormItemGi>
                  </NGrid>
                  <!-- dynamicInput -->
                  <NGrid v-if="val.type === 'dynamicInput'" :cols="24" :x-gap="24" class="ml-24px">
                    <NFormItemGi :span="24" :label="val.formName" :path="val.path">
                      <n-dynamic-input v-model:value="val.value" preset="pair" key-placeholder="请输入参数键"
                        value-placeholder="请输入参数值" />
                    </NFormItemGi>
                  </NGrid>
                  <!-- dynamicModalClass -->
                  <NGrid v-if="val.type === 'dynamicModalClass'" :cols="24" :x-gap="24" class="ml-24px">
                    <NFormItemGi :span="24" :label="val.formName" :path="val.path">
                      <NDynamicInput v-model:value="val.value" class="w-full" :on-create="handleCreate">
                        <template #default="{ value: item }">
                          <div class="w-full flex-center gap-8px">
                            <NInput class="w-1/2" :value="item.className"
                              @update:value="(name) => item.className = name" placeholder="请输入模型识别类别" />
                            <NButton class="w-160px" type="primary" @click="bindIndicator(item)">
                              关联指标
                            </NButton>
                          </div>
                        </template>

                        <!-- 创建按钮默认文本插槽 -->
                        <template #create-button-default>
                          添加模型识别类别
                        </template>
                      </NDynamicInput>
                    </NFormItemGi>
                  </NGrid>
                  <!-- text -->
                  <NGrid v-if="val.type === 'text'" :cols="24" :x-gap="24" class="ml-24px">
                    <NFormItemGi :span="24" :label="val.formName" :path="val.path">
                      {{ val.value }}
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
                      <!-- @update:value='handleCasUpdate' -->
                      <NCascader v-model:value="val.value" clearable :placeholder="val.placeholder"
                        :options="val.options" check-strategy="child" :show-path="true" expand-trigger="hover"
                        :render-label="renderLabel" @update:value='handleCasUpdate($event, val)'
                        :disabled="val.serverKey === 'sonId' ? !modelId : false" />
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
                  <!-- upload -->
                  <NGrid v-if="val.type === 'upload'" :cols="24" :x-gap="24" class="ml-24px">
                    <NFormItemGi :span="24" :label="val.formName" :path="val.path">
                      <div class="w-full flex-col justify-start">
                        <n-upload v-model:file-list="val.fileList" @before-upload="val.beforeUpload" :max="1"
                          @remove="handleRemove($event, val)">
                          <n-button>上传文件</n-button>
                        </n-upload>
                        <div>注：上传的文件主要作为测试评估参数</div>
                      </div>
                    </NFormItemGi>
                  </NGrid>
                  <!-- table -->
                  <NGrid v-if="val.type === 'table'" :cols="24" :x-gap="24" class="ml-24px">
                    <NFormItemGi :span="24" :label="val.formName" :path="val.path">
                      <div class="w-full flex-col justify-start">
                        <NDataTable :columns="val.columns" :data="metricList" :bordered="true" :pagination="false"
                          v-model:checked-row-keys="val.checkedRowKeys"
                          v-model:default-checked-row-keys="val.defaultCheckedRowKeys"
                          @update:checked-row-keys="(keys) => val.checkedRowKeys = keys" />
                      </div>
                    </NFormItemGi>
                  </NGrid>
                  <NGrid v-if="val.type === 'table1'" :cols="24" :x-gap="24" class="ml-24px">
                    <NFormItemGi :span="24" :label="val.formName" :path="val.path">
                      <div class="w-full flex-col justify-start">
                        <!-- <NDataTable :columns="val.columns" :data="labelList" :bordered="true" :pagination="false" /> -->
                        <NDataTable v-model:checked-row-keys="algorithmCheckedRowKeys" :columns="val.columns"
                          :data="algorithmData" :loading="loading" remote :row-key="(row) => row.code"
                          :pagination="mobilePagination" class="sm:h-full" />
                      </div>
                    </NFormItemGi>
                  </NGrid>
                  <!-- ----------------suffix------------- -->
                  <n-button quaternary type="primary" v-show="val.serverKey === 'modelId'" class="ml-16px"
                    :disabled="!val.value" @click="handleAlgorithmClick($event, val.value)">
                    查看模型算法编码
                  </n-button>
                </div>
              </div>
            </NCard>
          </div>
        </div>
      </NForm>
    </div>
    <div class="footer box-border w-full flex items-center justify-start gap-24px bg-[#fff] px-24px py-12px">
      <NButton type="info" class="w-88px" @click="handleDefine">完成任务
      </NButton>
      <NButton type="default" class="w-88px" @click="handleOperate('back')">返回
      </NButton>
    </div>
    <NModal v-model:show="isAlgorithmModal">
      <NCard style="width: 900px" title="模型算法编码" :bordered="false" size="huge" role="dialog" aria-modal="true">
        <div class="content">
          <NDataTable v-model:checked-row-keys="algorithmCheckedRowKeys" :columns="algorithmColumns"
            :data="algorithmData" :loading="loading" remote :row-key="(row) => row.code" :pagination="mobilePagination"
            class="sm:h-full" />
        </div>
        <template #footer>
          <NSpace justify="end" :size="16">
            <NButton type="primary" @click="() => (isAlgorithmModal = false)">我知道了</NButton>
          </NSpace>
        </template>
      </NCard>
    </NModal>

    <BindIndicatorModal v-model:visible="indicatorDrawerVisible" v-model:classId="classModel.classId"
      v-model:modelGridData="classModel.modelGridData" v-model:modelCommonData="classModel.modelCommonData"
      v-model:gridCheckedRowKeys="classModel.gridCheckedRowKeys"
      v-model:commonCheckedRowKeys="classModel.commonCheckedRowKeys" @afterLeave="handleAfterLeave" />
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
