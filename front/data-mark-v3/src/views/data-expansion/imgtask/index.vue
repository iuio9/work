<script setup lang="ts">
import type { FormInst, CascaderOption } from "naive-ui";
// import Flash16Regular from '@vicons/fluent/Flash16Regular'
import { NButton, NCascader, NImage, NInput, NPopover, NUpload } from "naive-ui";
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
import { getToken } from "@/store/modules/auth/shared";
import { computed } from "vue";
import { useFormRules, useNaiveForm } from "@/hooks/common/form";
import { getModelList, imgSubmitTask, selectDataSetLabelPage } from "@/service/api/dataManage";

import { useTable, useTableOperate } from "@/hooks/common/table";
import { $t } from "@/locales";

// interface
interface FormObj {
  model: Model;
  rules: any;
}

interface Model {
  mapImport: string;
}

// const formRef = ref<FormInst | null>(null);
const { formRef, validate, restoreValidation } = useNaiveForm();
const formObj = reactive<FormObj>({
  model: {
    dataEnhanceType: "string",
    anoType: "",
    datasetEnhanceArea: "",
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
  dataEnhanceType: "0",
  dataEnhanceMarkType: "0",
  datasetEnhanceArea: "0",
});
const taskRules = computed<Record<keyof Model, App.Global.FormRule[]>>(() => {
  const { defaultRequiredRule } = useFormRules();
  return {
    dataImport: defaultRequiredRule,
    taskInputName: defaultRequiredRule,
  };
});
const modelId = ref<string>("4");
const algorithmId = ref<string>("2");
const isShowExport = ref<Boolean>(false);
const isAccordion = ref<Boolean>(true);
const router = useRouter();
const route = useRoute();
const trainList = ref<any>([]);
const isShowPar = ref<Boolean>(false);
const dynamicList = ref<any>([]); // 动态数据 ==》 用于生成动态表单

// file-upload
const fileAction = `${import.meta.env.VITE_SERVICE_BASE_URL
  }/algorithm/file/uploadFile`;
const headers = reactive<any>({
  Authorization: `Bearer ${getToken()}`,
  // 'Content-Type': 'multipart/form-data'
});

// watch + computed
const isCheckTrain = computed(() => {
  return trainList.value.length == 1 && trainList.value[0].isCheck;
});

// methods
const recursionData = (data: any, label: any) => {
  // eslint-disable-next-line no-param-reassign
  data = data.map((item: any, index: string | number) => {
    if (item.children) {
      if (item.children.length > 0)
        recursionData(item.children, item.dictLabel);
      if (item.children.length === 0) delete item.children;
    }
    item.label = label ? `${item.remark}` : item.remark;
    item.value = item.dictLabel;
    return item;
  });
  return data;
};

// taskType datalist
const getTaskList = async () => {
  const params = {
    typeId: "7",
  };
  const res = await getDictDataTree(params);
  taskConfig.value.options = recursionData(res.data);
};

// 区域生成
const getAreaList = async () => {
  const params = {
    typeId: "10",
  };
  const res = await getDictDataTree(params);
  const options =
    res.data && res.data instanceof Array ? recursionData(res.data) : [];
  return options && options instanceof Array ? options : [];
};

// 缺陷生成
const getFlawList = async () => {
  const params = {
    typeId: "12",
  };
  const res = await getDictDataTree(params);
  const options =
    res.data && res.data instanceof Array ? recursionData(res.data) : [];
  return options && options instanceof Array ? options : [];
};

// 自动标注检测类型
const getAutoList = async () => {
  // const params = {
  //   typeId: "14", // 原13
  // };
  // const res = await getDictDataTree(params);
  // const options =
  //   res.data && res.data instanceof Array ? recursionData(res.data) : [];
  // return options && options instanceof Array ? options : [];

  const params = {};
  const res = await getModelList(params);

  return res.data instanceof Array
    ? res.data.map((val) => {
      return {
        value: val.modelName,
        label: val.modelUrl,
      };
    })
    : [];
};

// 自动标注检测物类型
const getAutoObjList = async () => {
  const params = {
    typeId: "14",
  };
  const res = await getDictDataTree(params);
  const options =
    res.data && res.data instanceof Array ? recursionData(res.data) : [];
  return options && options instanceof Array ? options : [];
};

// map datalist
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
        // 正式环境
        // if (val.count > 0 && val.progress == 100) {
        //   item.disabled = false;
        // } else {
        //   item.disabled = true;
        // }
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
  taskConfig.value.mapImportOptions = recursionMapData(res.data);
};
const getMapExportList = (dataList: any[], value: any) => {
  const recursionMapData = (data: any[]) => {
    const mapList = data.map((item: any, index: string | number) => {
      item.children = item.children.map((val: any) => {
        if (val.sonId == value) {
          item.disabled = true;
          val.disabled = true;
        } else {
          item.disabled = false;
          val.disabled = false;
        }
        return val;
      });
      return item;
    });
    return mapList;
  };
  taskConfig.value.mapExportOptions = recursionMapData(dataList);
};

// train dataList
const getTrainList = async () => {
  const res = await getExampleList({
    modelId: modelId.value,
    id: algorithmId.value,
  });
  trainList.value = res.data.map(({ id, ...rest }) => {
    // 判断是否需要选中该项
    const isCheck = rowQuery.value.id || res.data.length === 1;
    return {
      id,
      ...rest,
      isCheck,
      beforeImg: inc_before,
      afterImg: inc_after,
      arrow
    };
  });
  imgOperatorModel.value.dataEnhanceLst = trainList.value.reduce((acc, val) => {
    acc[val.id] = {};
    return acc;
  }, {});

  // model train 模型训练跳转
  const isCheckTrain =
    trainList.value.length == 1 && trainList.value[0].isCheck;
  if (isCheckTrain) {
    const index = 0;
    dynamicList.value = trainList.value[index].paramsMap.map((val) => {
      val.key = val.serverKey ?? val.value;
      val.value = val.value ? val.value : null;
      return val;
    });
  }
  if (rowQuery.value.id) {
    const index = trainList.value.findIndex(
      (val) => val.id == rowQuery.value.id,
    );
    dynamicList.value = trainList.value[index].paramsMap.map((val) => {
      val.key = val.serverKey ?? val.value;
      val.value = val.value ? val.value : null;
      return val;
    });
  }
  const newVal = trainList.value;
  if (newVal instanceof Array && newVal.length === 1) {
    // 自动标注
    if (newVal[0].modelId === "3") {
      const index = 0;
      const autoOptions = await getAutoList();
      dynamicList.value[index].options = autoOptions.map((val) => {
        return {
          value: val.label,
          label: val.value,
        };
      });
    }
  }
};
const handleUpdateValue = (value: string, option: CascaderOption) => {
  // export dataList
  const mapImportOptions = cloneDeep(taskConfig.value.mapImportOptions);
  isShowExport.value = Boolean(value);
  getMapExportList(mapImportOptions, value);
};

const handleClickCheck = async (index: any, row: any) => {
  if (rowQuery.value.id) {
    return;
  }
  trainList.value = trainList.value.map((item) => {
    item.isCheck = false;
    return item;
  });
  trainList.value[index].isCheck = true;
  isShowPar.value = true;
  dynamicList.value = trainList.value[index].paramsMap.map((val) => {
    val.key = val.serverKey ?? val.value;
    val.value = val.value ? val.value : null;
    val.id = row?.id;
    return val;
  });

  // 场景转化
  if (trainList.value[index].modelId == 5) {
    // 根据serverKey查找dynamicList.value的索引
    const index = dynamicList.value.findIndex(
      (val) => val.serverKey === "text_prompt1",
    );
    const areaOptions = await getAreaList();
    dynamicList.value[index].options = areaOptions.map((val) => {
      return {
        value: val.value,
        label: val.label,
      };
    });
    const index1 = dynamicList.value.findIndex(
      (val) => val.serverKey === "text_prompt2",
    );
    dynamicList.value[index1].disabled = dynamicList.value[index].value
      ? ""
      : 1;
  }
  // 异常区域
  if (trainList.value[index].modelId == 4) {
    // 缺陷生成
    if (row.id == 4) {
      const options = await getFlawList();
      dynamicList.value[0].options = options.map((val) => {
        return {
          value: val.value,
          label: val.label,
        };
      });
    }
    // 异常区域生成
    if (row.id == 5) {
      const index = dynamicList.value.findIndex(
        (val) => val.serverKey === "text_prompt1",
      );
      const areaOptions = await getAreaList();
      dynamicList.value[index].options = areaOptions.map((val) => {
        return {
          value: val.value,
          label: val.label,
        };
      });
      const index1 = dynamicList.value.findIndex(
        (val) => val.serverKey === "text_prompt2",
      );
      dynamicList.value[index1].disabled = dynamicList.value[index].value
        ? ""
        : 1;
    }
  }
  // 自动标注
  if (trainList.value[index].modelId == 3) {
    const index = 0;
    const autoOptions = await getAutoList();
    dynamicList.value[index].options = autoOptions.map((val) => {
      return {
        value: val.label,
        label: val.value,
      };
    });
  }
  // 图像算子
  if (trainList.value[index].modelId == 9) {
    console.log("图像算子");
  }
};
const handleRemove = async (options: any) => {
  const { file } = options;
  for (let i = 0; i < taskModel.value.fileList.length; i++) {
    const item = taskModel.value.fileList[i];
    if (item.id === file.id) {
      taskModel.value.fileList.splice(i, 1);
      const delList = taskModel.value.imgList.splice(i, 1);
      // eslint-disable-next-line no-await-in-loop
      await fileUploadDel({
        ids: delList.map((item) => item.id),
      });
      return false;
    }
  }
};
const beforeUpload = (options: any) => {
  taskModel.value.fileList = [];
  const { file } = options;
  const isLtSize = file.file.size / 1024 / 1024 < 10;
  if (!isLtSize) {
    window.$message?.error("上传图片大小不能超过 10 MB!");
    return false;
  }
  return true;
};

const beforeUpload1 = (options: any) => {
  taskModel.value.jsonList = [];
  const { file } = options;
  const fileSuffix = file.name.substring(file.name.lastIndexOf(".") + 1);
  const whiteList = ["json"];
  if (!whiteList.includes(fileSuffix)) {
    window.$message?.error("上传文件只能是 json 格式");
    return false;
  }
  const isLtSize = file.file.size / 1024 / 1024 < 10;
  if (!isLtSize) {
    window.$message?.error("上传图片大小不能超过 10 MB!");
    return false;
  }
  return true;
};
const handleUploadSuccess = (options: any) => {
  const res = JSON.parse(options.event.currentTarget.response);
  if (res.code == 200) {
    taskModel.value.uploadUrl = res.data;
  }
};
const handleUploadSuccess1 = (options: any) => {
  const res = JSON.parse(options.event.currentTarget.response);
  if (res.code == 200) {
    taskModel.value.jsonUrl = res.data;
  }
};

function isBooleanCol(item: any) {
  let bool: any;
  bool = item.isShow === undefined || item.isShow === true;
  return bool;
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

const rowQuery = ref({});
// lifecycle
onMounted(async () => {
  const query = JSON.parse(localStorage.getItem("row"));
  rowQuery.value = query;
  // ---------------------------
  // taskModel.value.taskName = query.name;
  modelId.value = query.modelId;
  algorithmId.value = query.id;
  await getTaskList();
  await getMapList();
  await getTrainList();
});

onBeforeUnmount(() => {
  localStorage.removeItem("row");
});

// newCode
const anoList = ref<any>([
  { value: "0", label: "图像分类" },
  { value: "1", label: "物体检测(支持常规矩形边界框)" },
  { value: "2", label: "图像分割" },
]);

const areaList = ref<any>([
  { value: "0", label: "图片全局增强" },
  { value: "1", label: "全局以及标注框局部增强" },
]);

const isShowArea = ref<Boolean>(false);

const tagData = ref<any>([
]);
const tagColumns = ref<any>([
  {
    type: 'selection',
  },
  {
    title: '全选',
    dataIndex: 'labelName',
    key: 'labelName',
  }
]);
const tagLoading = ref<any>(false);
const tagMobilePagination = ref<any>({});

const imgOperatorModel = ref<any>({
  taskInputName: null,
  dataEnhanceType: "0",
  dataEnhanceMarkType: null,
  datasetId: null,
  datasetTags: [],
  dataEnhanceLst: [],
  dataEnhanceTactics: null,
  checkList: [],
});

const {
  data,
  getData,
  loading,
  mobilePagination,
  updateSearchParams
} = useTable({
  apiFn: selectDataSetLabelPage,
  showTotal: true,
  apiParams: {
    current: 1,
    size: 10,
    sonId: imgOperatorModel.value.datasetId
  },
  columns: () => [
    {
      type: 'selection',
      align: 'center',
      width: 48,
      fixed: 'left'
    },
    {
      key: 'labelName',
      title: "全选",
      align: 'center'
    },
  ],
});
const {
  checkedRowKeys,
} = useTableOperate(data, getData);

// 算子任务表格
const imgColumns = ref<any>([
  {
    type: 'selection',
    fixed: 'left'
  },
  {
    title: '算子任务名称',
    key: 'algorithmName',
  },
  {
    title: '增强前',
    key: 'beforeUrl',
    render: row => {
      return [
        h('div', [
          h(NImage, {
            width: '66',
            src: row.beforeUrl,
            lazy: true
          })
        ])
      ]
    }
  },
  {
    title: '增强后',
    key: 'afterUrl',
    render: row => {
      return [
        h('div', [
          h(NImage, {
            width: '66',
            src: row.afterUrl,
            lazy: true
          })
        ])
      ]
    }
  },
  {
    key: 'operate',
    width: 300,
    title: $t('common.operate'),
    align: 'center',
    render: row => {
      return [
        h(
          NButton,
          {
            type: 'primary',
            ghost: true,
            size: 'small',
            style: { marginRight: '10px' },
            onClick: () => setupParameters(row)
          },
          '参数配置'
        ),
      ];
    },
    fixed: "right"
  }
]);
const paramShowModal = ref<Boolean>(false);
const imgRow = ref<any>({});


const handleAnoRadio = (val: string | number) => {
  const value = `${val}`;
  if (value === '0') {
    isShowArea.value = false;
  }
  if (value === '1') {
    areaList.value = [
      { value: "0", label: "图片全局增强" },
      { value: "1", label: "全局以及标注框局部增强" },
    ]
    isShowArea.value = true;
  }
  if (value === '2') {
    areaList.value = [
      { value: "0", label: "图片全局增强" },
    ]
    isShowArea.value = true;
  }
}

const handleCascaderUpdate = async (value: string) => {
  const params = {
    current: 1,
    size: 10,
    sonId: imgOperatorModel.value.datasetId
  };
  updateSearchParams(params);
  await getData();
  tagData.value = data.value;
  tagLoading.value = loading;
  tagMobilePagination.value = mobilePagination;
  imgOperatorModel.value.datasetTags = checkedRowKeys;

}

const handleUpValue = (value: string | number, sign: string, row: any) => {
  let processedValue: any;
  switch (sign) {
    case 'input':
      processedValue = value;
      break;
    case 'inputNumber':
      processedValue = parseInt(value.toString());
      break;
    case 'select':
      processedValue = value;
      break;
    default:
      return;
  }
  imgOperatorModel.value.dataEnhanceLst[row.id] = Object.assign({}, imgOperatorModel.value.dataEnhanceLst[row.id], {
    [row.serverKey]: processedValue
  });
};

const handleOperate = async (sign: string) => {
  try {
    switch (sign) {
      case "submit":
        await validate();
        const params = { ...imgOperatorModel.value };
        params.dataEnhanceLst = Object.entries(imgOperatorModel.value.dataEnhanceLst).map(([algorithmId, value]) => ({
          algorithmId,
          ...value
        })).filter(item => (imgOperatorModel.value.checkList).includes(Number(item.algorithmId)));;
        delete params.checkList;
        const res = await imgSubmitTask(params);
        if (res.data) {
          window.$message?.success(res.data);
          router.back();
        }
        break;
      case "back":
        router.back();
        break;
      case "define":
        paramShowModal.value = false;
        break;
      case "cancel":
        const paramsMap = (imgRow.value)?.paramsMap instanceof Array ? imgRow.value?.paramsMap : [];
        dynamicList.value = paramsMap.map((val) => {
          val.key = val.serverKey ?? val.value;
          val.value = null;
          val.id = imgRow.value?.id;
          return val;
        });
        paramShowModal.value = false;
        break;
      default:
        break;
    }
  } catch (error) {
    console.error('操作过程中出现错误:', error);
    window.$message?.error('操作失败，请稍后重试');
  }
};

const setupParameters = (row: any) => {
  imgRow.value = row;
  const paramsMap = row?.paramsMap instanceof Array ? row?.paramsMap : [];
  dynamicList.value = paramsMap.map((val) => {
    val.key = val.serverKey ?? val.value;
    val.value = val.value ? val.value : null;
    val.id = row?.id;
    return val;
  });
  paramShowModal.value = true;
}

</script>

<template>
  <div class="w-full h-full p-24px box-border flex flex-col justify-start items-start wrap-container">
    <div class="w-full flex-1 box-border p-24px">
      <n-card class="w-full h-full overflow-y-auto">
        <n-form ref="formRef" :model="imgOperatorModel" :rules="imgOperatorRules" label-placement="left"
          label-width="auto" require-mark-placement="right-hanging" class="w-100% h-full">
          <!-- 请输入任务名称 taskInputName -->
          <div class="w-30%" v-if="true">
            <div class="w-full flex items-center mb-16px">
              <span class="block w-3px h-14px bg-[#000] mr-8px"></span>
              <span class="block font-[500] text-[#000]">请输入任务名称</span>
            </div>
            <n-grid :cols="24" :x-gap="24" class="ml-24px">
              <n-form-item-gi :span="24" label="任务名称" path="taskName">
                <n-input v-model:value="imgOperatorModel.taskInputName" clearable placeholder="请输入任务名称" />
              </n-form-item-gi>
            </n-grid>
          </div>
          <!-- 请选择标注类型 dataEnhanceType -->
          <div class="w-30%" v-if="false">
            <div class="w-full flex items-center mb-16px">
              <span class="block w-3px h-14px bg-[#000] mr-8px"></span>
              <span class="block font-[500] text-[#000]">请选择标注类型</span>
            </div>
            <n-grid :cols="24" :x-gap="24" class="ml-24px">
              <n-form-item-gi :span="24" label="数据类型" path="dataEnhanceType">
                <n-radio :checked="imgOperatorModel.dataEnhanceType === '0'" value="0" name="basic-demo"
                  @change="handleChange">
                  图像类
                </n-radio>
              </n-form-item-gi>
              <n-form-item-gi :span="24" label="标注类型" path="dataEnhanceMarkType">
                <n-radio-group v-model:value="imgOperatorModel.dataEnhanceMarkType" name="radiogroup"
                  @update:value="handleAnoRadio">
                  <n-space>
                    <n-radio v-for="item in anoList" :key="item.value" :value="item.value">
                      {{ item.label }}
                    </n-radio>
                  </n-space>
                </n-radio-group>
              </n-form-item-gi>
              <n-form-item-gi :span="24" label="增强区域" path="datasetEnhanceArea" v-if="isShowArea">
                <n-radio-group v-model:value="imgOperatorModel.datasetEnhanceArea" name="radiogroup">
                  <n-space>
                    <n-radio v-for="item in areaList" :key="item.value" :value="item.value">
                      {{ item.label }}
                    </n-radio>
                  </n-space>
                </n-radio-group>
              </n-form-item-gi>
            </n-grid>
          </div>
          <!-- 请选择数据集 -->
          <div class="w-30%">
            <div class="w-full flex items-center mb-16px">
              <span class="block w-3px h-14px bg-[#000] mr-8px"></span>
              <span class="block font-[500] text-[#000]">请选择数据集</span>
            </div>
            <n-grid :cols="24" :x-gap="24" class="ml-24px">
              <n-form-item-gi :span="24" label="数据输入" path="datasetId">
                <n-cascader v-model:value="imgOperatorModel.datasetId" clearable placeholder="请选择输入数据集"
                  :options="taskConfig.mapImportOptions" :show-path="true" expand-trigger="hover"
                  :render-label="renderLabel" check-strategy="child" @update:value="handleCascaderUpdate">
                </n-cascader>
              </n-form-item-gi>
              <n-form-item-gi :span="24" label="选择标签" path="" v-if="false">
                <!-- <n-data-table :columns="tagColumns" :data="tagData" :pagination="false" /> -->
                <NDataTable v-model:checked-row-keys="imgOperatorModel.datasetTags" :columns="tagColumns"
                  :data="tagData" remote :row-key="(row) => row.labelId" :pagination="tagMobilePagination"
                  class="sm:h-full" />
              </n-form-item-gi>
            </n-grid>
          </div>
          <!-- 配置算子任务参数 -->
          <!-- <div class="w-full h-auto">
            <div class="w-full flex items-center mb-16px">
              <span class="block w-3px h-14px bg-[#000] mr-8px"></span>
              <span class="block font-[500] text-[#000]">{{
                rowQuery.id ? "训练模型" : "配置算子任务参数"
              }}</span>
            </div>
            <div class="w-full h-auto max-h-320px overflow-y-auto border-1 border-[#eee] box-border">
              <div v-for="(item, index) of trainList" :key="index">
                <div class="w-full flex justify-start items-start border-b-1 border-b-[#eee] p-12px box-border">
                  <n-checkbox v-model:checked="item.isCheck" @click="handleClickCheck(index, item)"
                    :disabled="rowQuery.id || isCheckTrain">
                  </n-checkbox>
                  <n-collapse class="ml-8px mt-[-2px]" :accordion="isAccordion">
                    <n-collapse-item :title="item.algorithmName" name="1">
                      <div class="content w-520px h-232px border-1 border-[#eee] px-24px py-10px box-border">
                        <div>{{ item.desc }}</div>
                        <div class="flex justify-start items-center mt-8px">
                          <img :src="item.beforeUrl" alt="" class="w-200px h-158px object-cover" />
                          <img :src="item.arrow" class="mx-24px" />
                          <img :src="item.afterUrl" alt="" class="w-200px h-158px object-cover" />
                        </div>
                        <div class="flex justify-start items-center">
                          <div class="w-200px flex justify-center items-center">
                            增强前
                          </div>
                          <div class="mx-24px"></div>
                          <div class="w-200px flex justify-center items-center">
                            增强后
                          </div>
                        </div>
                      </div>
                      <template #header-extra>
                        <div class="flex justify-center items-center gap-14px">
                          <span>效果展示</span>
                        </div>
                      </template>
</n-collapse-item>
</n-collapse>
</div>
</div>
</div>
</div>
<div class="w-50% my-24px" v-for="(item, index) of dynamicList" :key="item.type" v-show="isBooleanCol(item)">
  <div class="w-full flex items-center mb-16px">
    <span class="block w-3px h-14px bg-[#000] mr-8px"></span>
    <span class="block font-[500] text-[#000]">{{ item.label }}</span>
  </div>
  <div class="wrap_col">
    <div class="" v-if="item.type === 'image'">
      <n-upload directory-dnd name="file" :action="fileAction" :headers="headers" @before-upload="beforeUpload"
        @finish="handleUploadSuccess" v-model:file-list="taskModel.fileList">
        <n-upload-dragger>
          <div style="margin-bottom: 12px">
            <n-icon size="48" :depth="3">
              <ArchiveIcon />
            </n-icon>
          </div>
          <n-text style="font-size: 16px">
            {{ item.label }}
          </n-text>
        </n-upload-dragger>
      </n-upload>
    </div>
    <div class="" v-if="item.type === 'json'">
      <n-upload name="file" :action="fileAction" :headers="headers" @before-upload="beforeUpload1"
        @finish="handleUploadSuccess1" v-model:file-list="taskModel.jsonList">
        <n-button>{{ item.label }}</n-button>
      </n-upload>
    </div>
    <div class="" v-if="item.type === 'text'">
      <n-input v-model:value="item.value" type="textarea" :placeholder="item.label" v-show="!item.sign"
        @update:value="handleUpValue($event, 'input', item)" />
      <n-input-number :min="0" v-model:value="item.value" clearable :placeholder="item.label"
        :step="item.step ? item.step : 1" v-show="item.sign"
        @update:value="handleUpValue($event, 'inputNumber', item)" />
    </div>
    <div class="" v-if="item.type === 'select'">
      <n-select v-model:value="item.value" :options="item.options" :placeholder="item.label"
        :disabled="item.disabled ? true : false" :multiple="item.isMulSelect ? true : false"
        @update:value="handleUpValue($event, 'select', item)" />
    </div>
  </div>
</div> -->

          <!-- 请选择算子任务 -->
          <div class="w-80% mt-24px" v-if="true">
            <div class="w-full flex flex-wrap items-center mb-16px">
              <span class="block w-3px h-14px bg-[#000] mr-8px"></span>
              <span class="block font-[500] text-[#000]">请选择算子任务</span>
            </div>
            <n-grid :cols="24" :x-gap="24" class="ml-24px">
              <n-form-item-gi :span="24" label="" path="taskList">
                <!-- <n-checkbox-group v-model:value="imgOperatorModel.checkList">
                  <n-space item-style="display: flex;">
                    <n-checkbox v-for="item in trainList" :key="item.value" :value="item.id"
                      :label="item.algorithmName" />
                  </n-space>
                </n-checkbox-group> -->
                <NDataTable v-model:checked-row-keys="imgOperatorModel.checkList" :columns="imgColumns"
                  :data="trainList" remote :row-key="(row) => row.id" class="sm:h-full" />
              </n-form-item-gi>
            </n-grid>
          </div>
          <!-- 请选择算子处理策略 -->
          <div class="w-45% mt-24px" v-if="true">
            <div class="w-full flex items-center mb-16px">
              <span class="block w-3px h-14px bg-[#000] mr-8px"></span>
              <span class="block font-[500] text-[#000]">请选择算子处理策略</span>
            </div>
            <n-grid :cols="24" :x-gap="24" class="ml-24px">
              <n-form-item-gi :span="24" label="处理策略" path="dataEnhanceTactics">
                <n-radio-group v-model:value="imgOperatorModel.dataEnhanceTactics" name="dataEnhanceTactics">
                  <n-space>
                    <n-radio value="0"> 串行叠加 </n-radio>
                    <n-radio value="1"> 并行遍历 </n-radio>
                  </n-space>
                </n-radio-group>
              </n-form-item-gi>
            </n-grid>
          </div>
        </n-form>
      </n-card>
    </div>
    <div class="footer w-full box-border flex justify-start items-center gap-24px px-24px py-12px bg-[#fff]">
      <n-button type="info" @click="handleOperate('submit')" class="w-88px">
        提交
      </n-button>
      <n-button type="default" @click="handleOperate('back')" class="w-88px">
        返回
      </n-button>
    </div>
    <!-- 参数配置modal -->
    <n-modal v-model:show="paramShowModal">
      <n-card style="width: 600px" title="参数配置" :bordered="false" size="huge" role="dialog" aria-modal="true">
        <div class="w-full" v-for="(item, index) of dynamicList" :key="item.type" v-show="isBooleanCol(item)">
          <div class="w-full flex items-center mt-12px mb-8px">
            <span class="block w-3px h-14px bg-[#000] mr-8px"></span>
            <span class="block font-[500] text-[#000]">{{ item.label }}</span>
          </div>
          <div class="wrap_col">
            <div class="" v-if="item.type === 'image'">
              <n-upload directory-dnd name="file" :action="fileAction" :headers="headers" @before-upload="beforeUpload"
                @finish="handleUploadSuccess" v-model:file-list="taskModel.fileList">
                <n-upload-dragger>
                  <div style="margin-bottom: 12px">
                    <n-icon size="48" :depth="3">
                      <ArchiveIcon />
                    </n-icon>
                  </div>
                  <n-text style="font-size: 16px">
                    {{ item.label }}
                  </n-text>
                </n-upload-dragger>
              </n-upload>
            </div>
            <div class="" v-if="item.type === 'json'">
              <n-upload name="file" :action="fileAction" :headers="headers" @before-upload="beforeUpload1"
                @finish="handleUploadSuccess1" v-model:file-list="taskModel.jsonList">
                <n-button>{{ item.label }}</n-button>
              </n-upload>
            </div>
            <div class="" v-if="item.type === 'text'">
              <n-input v-model:value="item.value" type="textarea" :placeholder="item.label" v-show="!item.sign"
                @update:value="handleUpValue($event, 'input', item)" />
              <n-input-number :min="0" v-model:value="item.value" clearable :placeholder="item.label"
                :step="item.step ? item.step : 1" v-show="item.sign"
                @update:value="handleUpValue($event, 'inputNumber', item)" />
            </div>
            <div class="" v-if="item.type === 'select'">
              <n-select v-model:value="item.value" :options="item.options" :placeholder="item.label"
                :disabled="item.disabled ? true : false" :multiple="item.isMulSelect ? true : false"
                @update:value="handleUpValue($event, 'select', item)" />
            </div>
          </div>
        </div>
        <template #footer>
          <NSpace justify="end" :size="16">
            <NButton @click="() => handleOperate('cancel')">{{ $t('common.cancel') }}</NButton>
            <NButton type="primary" @click="handleOperate('define')">确定</NButton>
          </NSpace>
        </template>
      </n-card>
    </n-modal>
  </div>
</template>

<style scoped lang="scss">
:deep(.n-transfer-list--target) {
  display: none !important;
}

.wrap-container {
  padding: 0 !important;
}
</style>
