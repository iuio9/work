<script setup lang="ts">
import type { FormInst, CascaderOption } from "naive-ui";
// import Flash16Regular from '@vicons/fluent/Flash16Regular'
import { NCascader, NInput, NPopover, NUpload } from "naive-ui";
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
import { getModelList } from "@/service/api/dataManage";

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
  trainList.value = res.data.map((val) => {
    val.isCheck = rowQuery.value.id || res.data.length == 1 ? true : false;
    val.beforeImg = inc_before;
    val.afterImg = inc_after;
    val.arrow = arrow;
    return val;
  });
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
      // const index1 = 1;
      // dynamicList.value[index1].disabled = dynamicList.value[index].value
      //   ? ""
      //   : 1;
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
    // const index1 = 1;
    // dynamicList.value[index1].disabled = dynamicList.value[index].value
    //   ? ""
    //   : 1;
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

const handleOperate = async (sign: string) => {
  // 提交
  if (sign === "submit") {
    await validate();
    const params = {};
    const algorithmParam = {};
    params.datasetId = taskModel.value.dataMap.import;
    params.datasetOutId = taskModel.value.dataMap.export;
    dynamicList.value.forEach((item) => {
      console.log(item);

      algorithmParam[item.key] = item.value;
      if (item.type === "image") {
        algorithmParam[item.key] = taskModel.value.uploadUrl;
      }
      if (item.type === "json") {
        algorithmParam[item.key] = taskModel.value.jsonUrl;
      }
    });
    params.algorithmId = trainList.value.filter((item) => {
      return item.isCheck;
    })[0].id;
    params.modelId = Number(modelId.value);
    const concatPar = Object.assign({}, params, {
      algorithmParam,
      taskInputName: taskModel.value.taskInputName, // 任务名称
    });
    const res = await submitTask(concatPar);
    if (res.data) {
      window.$message?.success(res.data);
      router.back();
    }
  }
  if (sign === "back") {
    router.back();
  }
};

const handleSelectUpdate = async (e: any, row: any) => {
  if (row.serverKey === "text_prompt1" && e) {
    const idx = dynamicList.value.findIndex(
      (val) => val.serverKey === "text_prompt2",
    );
    const areaOptions = await getAreaList();
    console.log(areaOptions);
    // 根据e查找areaOptions的索引
    const index = areaOptions.findIndex((val) => val.value === e);
    dynamicList.value[idx].value = null;
    dynamicList.value[idx].options =
      areaOptions[index].children instanceof Array
        ? areaOptions[index].children
        : [];
    dynamicList.value[idx].disabled = "";
  }

  if (
    row.serverKey === "classes_string" &&
    e instanceof Array &&
    e.length > 0
  ) {
    const idx = 1;
    const autoObjOptions = await getAutoObjList();
    // const index = autoObjOptions.findIndex((val) => val.value === e);
    dynamicList.value[idx].value = null;
    dynamicList.value[idx].options = autoObjOptions;
    dynamicList.value[idx].disabled = "";
  }

  if (
    row.serverKey === "classes_string" &&
    e instanceof Array &&
    e.length == 0
  ) {
    const idx = 1;
    const autoObjOptions: never[] = [];
    // const index = autoObjOptions.findIndex((val) => val.value === e);
    dynamicList.value[idx].value = null;
    dynamicList.value[idx].options = autoObjOptions;
    dynamicList.value[idx].disabled = "1";
  }

  if (row.serverKey === "mode") {
    dynamicList.value[1].disabled = "";
  }
};

function isBooleanCol(item: any) {
  let bool: any;
  // item.isShow等于true或者undefined时 bool等于true；item.isShow等于false时，bool等于false
  bool = item.isShow === undefined || item.isShow === true;
  return bool;

  // let bool: any;
  // if(item.isShow) {
  //   bool = true;
  // } else {
  //   if(item.isShow === "false") {
  //     bool = false;
  //   } else {
  //     bool = true;
  //   }
  // }
  // return bool;
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
</script>

<template>
  <div class="w-full h-full p-24px box-border flex flex-col justify-start items-start wrap-container">
    <div class="w-full flex-1 box-border p-24px">
      <n-card class="w-full h-full overflow-y-auto">
        <n-form ref="formRef" :model="taskModel" :rules="taskRules" label-placement="left" label-width="auto"
          require-mark-placement="right-hanging" class="w-100% h-full">
          <!-- <div class="w-30%" v-if="false">
            <div class="w-full flex items-center mb-16px">
              <span class="block w-3px h-14px bg-[#000] mr-8px"></span>
              <span class="block font-[500] text-[#000]">请选择任务类型</span>
            </div>
            <n-grid :cols="24" :x-gap="24" class="ml-24px">
              <n-form-item-gi :span="24" label="任务类型" path="taskType">
                <n-cascader
                  v-model:value="taskModel.taskVal"
                  clearable
                  placeholder="请选择任务类型"
                  :options="taskConfig.options"
                  check-strategy="child"
                />
              </n-form-item-gi>
            </n-grid>
          </div> -->
          <!-- 任务类型 || 任务名称 -->
          <div class="w-30%">
            <div class="w-full flex items-center mb-16px">
              <span class="block w-3px h-14px bg-[#000] mr-8px"></span>
              <span class="block font-[500] text-[#000]">请输入任务名称</span>
            </div>
            <n-grid :cols="24" :x-gap="24" class="ml-24px">
              <n-form-item-gi :span="24" label="任务名称" path="taskName">
                <n-input v-model:value="taskModel.taskInputName" clearable placeholder="请输入任务名称" />
              </n-form-item-gi>
            </n-grid>
          </div>
          <div class="w-30%">
            <div class="w-full flex items-center mb-16px">
              <span class="block w-3px h-14px bg-[#000] mr-8px"></span>
              <span class="block font-[500] text-[#000]">请选择数据集</span>
            </div>
            <n-grid :cols="24" :x-gap="24" class="ml-24px">
              <!--dataImport-->
              <n-form-item-gi :span="24" label="数据输入" path="">
                <!-- @update:value="handleUpdateValue" -->
                <!-- expand-trigger="hover" -->
                <n-cascader v-model:value="taskModel.dataMap.import" clearable placeholder="请选择输入数据集"
                  :options="taskConfig.mapImportOptions" :show-path="true" expand-trigger="hover" :render-label="renderLabel"
                  check-strategy="child">
                </n-cascader>
              </n-form-item-gi>
              <!--<n-form-item-gi
                :span="24"
                label="选择标签"
                path="dataImport"
              >
                <n-transfer v-model:value="formObj.model.tags" :options="options"/>
              </n-form-item-gi>
              <n-form-item-gi
                :span="24"
                label="无标注数据"
                path="anoType"
              >
                <n-radio-group v-model:value="formObj.model.anoType" name="anoType">
                  <n-space>
                    <n-radio value="Radio 1">
                      增强
                    </n-radio>
                    <n-radio value="Radio 2">
                      不增强
                    </n-radio>
                  </n-space>
                </n-radio-group>
              </n-form-item-gi>-->
              <!--<n-form-item-gi
                :span="24"
                label="数据输出"
                path="dataImport"
                v-if="isShowExport"
              >
                <n-cascader
                  v-model:value="taskModel.dataMap.export"
                  clearable
                  placeholder="请选择输出数据集"
                  :options="taskConfig.mapExportOptions"
                  check-strategy="child"
                />
              </n-form-item-gi>-->
            </n-grid>
          </div>
          <div class="w-full h-auto">
            <div class="w-full flex items-center mb-16px">
              <span class="block w-3px h-14px bg-[#000] mr-8px"></span>
              <span class="block font-[500] text-[#000]">{{
                rowQuery.id ? "训练模型" : "请选择训练模型"
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
                      <template #header-extra> 效果展示 </template>
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
                <n-upload directory-dnd name="file" :action="fileAction" :headers="headers"
                  @before-upload="beforeUpload" @finish="handleUploadSuccess" v-model:file-list="taskModel.fileList">
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
                <n-input v-model:value="item.value" type="textarea" :placeholder="item.label" v-show="!item.sign" />
                <n-input-number :min="0" v-model:value="item.value" clearable :placeholder="item.label"
                  :step="item.step ? item.step : 1" v-show="item.sign" />
              </div>
              <div class="" v-if="item.type === 'select'">
                <n-select v-model:value="item.value" :options="item.options" :placeholder="item.label"
                  :disabled="item.disabled ? true : false" :multiple="item.isMulSelect ? true : false"
                  @update:value="handleSelectUpdate($event, item)" />
              </div>
            </div>
          </div>
          <div class="w-45% mt-24px" v-if="false">
            <div class="w-full flex items-center mb-16px">
              <span class="block w-3px h-14px bg-[#000] mr-8px"></span>
              <span class="block font-[500] text-[#000]">请选择算子处理策略</span>
            </div>
            <n-grid :cols="24" :x-gap="24" class="ml-24px">
              <n-form-item-gi :span="24" label="处理策略" path="anoType">
                <n-radio-group v-model:value="formObj.model.anoType" name="anoType">
                  <n-space>
                    <n-radio value="Radio 1"> 串行叠加 </n-radio>
                    <n-radio value="Radio 2"> 并行遍历 </n-radio>
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
