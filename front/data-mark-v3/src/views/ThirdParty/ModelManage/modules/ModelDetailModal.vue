<script setup lang="tsx">
import { ref, reactive, computed, watch } from "vue";
import { useRouter } from "vue-router";
import { NModal, NScrollbar, NForm, NGrid, NFormItemGi } from "naive-ui";
import { fetchModelDetails } from "@/service/api/third";


import VueJsonPretty from 'vue-json-pretty'
import 'vue-json-pretty/lib/styles.css'

import { nanoid } from '~/packages/utils/src';
import { useBoolean } from "~/packages/hooks/src";

import BindIndicatorModal from "./BindIndicatorModal.vue";

interface FormFieldConfig {
  formName?: string;
  type: 'input' | 'textarea' | 'select' | 'dynamicInput' | 'text' | 'radioGroup' | 'cascader' | 'upload' | 'datetime' | 'customUpload' | 'checkbox' | 'table' | 'dynamicModalClass';
  value: unknown;
  placeholder?: string;
  width?: string;
  serverKey: string;
  isShow?: boolean;
  isMultiple?: boolean;
  options?: any[];
  renderLabel?: (options: any) => any;
  columns?: any[];
  query?: any[];
  modelList?: { value: string | number; label: string, [key: string]: any }[];
  checkboxList?: { key: string; value: number | string; formName: string, [key: string]: any }[];
  accept?: string;
  buttonText?: string;
  [prop: string]: any;
}

defineOptions({
  name: "ModelDetailModal",
});

const visible = defineModel<boolean>("visible", {
  default: false,
});

const applyId = defineModel<string>("applyId", {
  default: "",
});

const router = useRouter();

interface Props {
  formData?: any;
}

interface FormGroupConfig {
  tab?: string;
  name: string;
  list: FormFieldConfig[];
  isShow?: boolean;
}

const props = withDefaults(defineProps<Props>(), {
  formData: {}
});

const emit = defineEmits<{
  (e: "submitted"): void;
  (e: "preview", params: any): void;
}>();

// 展示数据模型
const model = reactive<Record<string, any>>({
  createTime: "",
  updateTime: "",
  taskName: "",
  taskProgress: "",
  taskStat: "",
  datasetOutId: "",
});

function handleInitModel() {
  if (props.rowData) {
    Object.assign(model, props.rowData);
  }
}

// 关闭弹窗
function closeDrawer() {
  visible.value = false;
}

const basicInfoList = [
  {
    formName: "模型名称",
    type: "input",
    value: "人脸识别模型V2.0",
    placeholder: "请输入模型名称",
    width: "30%",
    serverKey: "modelName",
    isShow: true
  },
  {
    formName: "模型来源",
    type: "input",
    value: "AI研发部-2025001项目",
    placeholder: "数字化项目应提供项目编号及项目名称",
    width: "30%",
    serverKey: "modelSource",
    isShow: true
  },
  {
    formName: "模型封装方式",
    type: "input",
    value: "Docker 镜像",
    placeholder: "Docker 镜像/模型文件",
    width: "30%",
    serverKey: "modelPackaging",
    isShow: true
  },
  {
    formName: "模型部署位置",
    type: "input",
    value: "生产环境-华东区-服务器集群A",
    placeholder: "",
    width: "30%",
    serverKey: "modelDeploymentLocation",
    isShow: true
  },
  {
    formName: "业务单位",
    type: "input",
    value: "安全管理部",
    placeholder: "",
    width: "30%",
    serverKey: "businessUnit",
    isShow: true
  },
  {
    formName: "业务单位负责人/联系方式",
    type: "input",
    value: "张三/13800138000",
    placeholder: "",
    width: "30%",
    serverKey: "businessUnitContact",
    isShow: true
  },
  {
    formName: "开发单位",
    type: "input",
    value: "AI技术有限公司",
    placeholder: "",
    width: "30%",
    serverKey: "developmentUnit",
    isShow: true
  },
  {
    formName: "开发单位负责人/联系方式",
    type: "input",
    value: "李四/13900139000",
    placeholder: "",
    width: "30%",
    serverKey: "developmentUnitContact",
    isShow: true
  },
  {
    formName: "登记日期",
    type: "datetime",
    value: "2025-05-01",
    placeholder: "",
    width: "30%",
    serverKey: "registrationDate",
    isShow: true
  }
];

// 表单配置数据 - 准备工作
const preparationWorkList = [
  {
    formName: "模型文件名",
    type: "input",
    value: "face_recognition_model_v2.0.h5",
    placeholder: "",
    width: "30%",
    serverKey: "modelFileName",
    isShow: true
  },
  {
    formName: "模型文件大小",
    type: "input",
    value: "2.4G",
    placeholder: "例: 7.8G",
    width: "30%",
    serverKey: "modelFileSize",
    isShow: true
  },
  {
    formName: "模型API接口说明",
    type: "upload",
    value: undefined,
    placeholder: "附表记录",
    width: "30%",
    serverKey: "modelApiDescription",
    isShow: true
  },
  {
    formName: "模型对外暴露端口",
    type: "input",
    value: "25001",
    placeholder: "例: 25001",
    width: "30%",
    serverKey: "modelExposedPort",
    isShow: true
  },
  {
    formName: "模型cuda版本",
    type: "input",
    value: "12.1",
    placeholder: "例: 12.1",
    width: "30%",
    serverKey: "modelCudaVersion",
    isShow: true
  },
  {
    formName: "模型驱动版本",
    type: "input",
    value: "nv 驱动 530.30.02",
    placeholder: "例: nv 驱动 530.30.02",
    width: "30%",
    serverKey: "modelDriverVersion",
    isShow: true
  },
  {
    formName: "模型调用例",
    type: "input",
    value: "POST /api/v2/face/recognition",
    placeholder: "附表记录",
    width: "30%",
    serverKey: "modelCallExample",
    isShow: true
  },
  {
    formName: "模型功能",
    type: "input",
    value: "基于深度学习的人脸识别和分析，支持特征提取、人脸比对等功能",
    placeholder: "概述模型功能",
    width: "30%",
    serverKey: "modelFunction",
    isShow: true
  },
  {
    formName: "模型检测场景",
    type: "input",
    value: "人脸检测、口罩佩戴检测、活体检测、表情识别",
    placeholder: "详细罗列检测场景，例: 人脸检测、口罩佩戴检测...",
    width: "30%",
    serverKey: "modelDetectionScene",
    isShow: true
  },
  {
    formName: "训练样本",
    type: "input",
    value: "CASIA-WebFace数据集，标注文件: annotations_v2.csv",
    placeholder: "提供训练样本集及标注文件",
    width: "30%",
    serverKey: "trainingSamples",
    isShow: true
  }
];

// 表单配置数据 - 测试指标
const testIndicatorsList = [
  {
    formName: "准确率",
    type: "input",
    value: "≥0.98",
    placeholder: "例: 需达到 大于等于0.93",
    width: "30%",
    serverKey: "accuracy",
    isShow: true
  },
  {
    formName: "精确率",
    type: "input",
    value: "≥0.97",
    placeholder: "例: 需达到 大于等于0.91",
    width: "30%",
    serverKey: "precision",
    isShow: true
  },
  {
    formName: "召回率",
    type: "input",
    value: "≥0.96",
    placeholder: "例: 需满足 小于0.02",
    width: "30%",
    serverKey: "recall",
    isShow: true
  },
  {
    formName: "F1-分数",
    type: "input",
    value: "≥0.97",
    placeholder: "例: 需满足 大于等于 0.87",
    width: "30%",
    serverKey: "f1Score",
    isShow: true
  },
  {
    formName: "IoU",
    type: "input",
    value: "≥0.85",
    placeholder: "例: 不能低于0.3",
    width: "30%",
    serverKey: "iou",
    isShow: true
  },
  {
    formName: "均方误差",
    type: "input",
    value: "≤0.02",
    placeholder: "例: 需达到 大于等于0.91",
    width: "30%",
    serverKey: "meanSquaredError",
    isShow: true
  },
  {
    formName: "场景覆盖",
    type: "input",
    value: "室内场景、室外场景、不同光照条件、不同年龄段、不同种族",
    placeholder: "填写需要满足设计要求的场景列表及详细内容",
    width: "30%",
    serverKey: "scenarioCoverage",
    isShow: true
  }
];

const attachmentList: FormFieldConfig[] = [
  {
    formName: "上传附件",
    type: "upload",
    value: "上传模型文件",
    placeholder: "",
    width: "60%",
    serverKey: "modelFile",
    isShow: true
  },
]

// 计算当前标签页的表单字段
const activeFields = computed(() => {
  switch (activeTab.value) {
    case '基本信息':
      return basicInfoList;
    case '准备工作':
      return preparationWorkList;
    case '测试指标':
      return testIndicatorsList;
    case '附件':
      return attachmentList;
    default:
      return [{}];
  }
});

// 网格列数
const gridCols = ref(3);

// 根据字段类型获取对应的Naive UI组件
const getComponent = (type) => {
  switch (type) {
    case 'input':
      return 'n-input';
    case 'select':
      return 'n-select';
    case 'datetime':
      return 'n-date-picker';
    case 'upload':
      return 'n-upload';
    default:
      return 'n-input';
  }
};

// ---------------------------------------------------------
const configList = ref<FormGroupConfig[]>([
  {
    name: "基本信息",
    list: [
      {
        formName: "模型名称",
        type: "input",
        value: "",
        placeholder: "请输入模型的具体名称（如：输电线路巡检模型）",
        width: "30%",
        serverKey: "modelName",
        isShow: true
      },
      {
        formName: "模型来源",
        type: "input",
        value: "",
        placeholder: "格式：项目编号+项目名称（如：TJ2023-001 天津电网智能巡检项目）",
        width: "30%",
        serverKey: "modelSource",
        isShow: true
      },
      {
        formName: "测试需求简述",
        type: "textarea",
        value: "",
        placeholder: "请输入测试需求的简述",
        width: "30%",
        serverKey: "testDemandDesc",
        isShow: true
      },
      {
        formName: "模型类型",
        type: "input",
        value: "",
        placeholder: "如：深度学习/机器学习/传统算法",
        width: "30%",
        serverKey: "modelType",
        isShow: true
      },
      {
        formName: "模型功能",
        type: "input",
        value: "",
        placeholder: "简要描述模型实现的核心功能",
        width: "30%",
        serverKey: "modelFunction",
        isShow: true
      },
      {
        formName: "建设单位-单位名称",
        type: "input",
        value: "",
        placeholder: "请输入建设单位-单位名称",
        width: "30%",
        serverKey: "buildUnitName",
        isShow: true
      },
      {
        formName: "建设单位-单位地址",
        type: "textarea",
        value: "",
        placeholder: "格式：省/市/区+详细地址",
        width: "60%",
        serverKey: "buildUnitAddress",
        isShow: true
      },
      {
        formName: "建设单位-联系人",
        type: "input",
        value: "",
        placeholder: "请输入建设单位-联系人",
        width: "30%",
        serverKey: "buildUnitLeader",
        isShow: true
      },
      {
        formName: "建设单位-联系电话",
        type: "input",
        value: "",
        placeholder: "请输入建设单位-联系电话",
        width: "30%",
        serverKey: "buildUnitContact",
        isShow: true
      },
      {
        formName: "承建单位-单位名称",
        type: "input",
        value: "",
        placeholder: "请输入承建单位-单位名称",
        width: "30%",
        serverKey: "btUnitName",
        isShow: true
      },
      {
        formName: "承建单位-单位地址",
        type: "textarea",
        value: "",
        placeholder: "格式：省/市/区+详细地址",
        width: "60%",
        serverKey: "btUnitAddress",
        isShow: true
      },
      {
        formName: "承建单位-联系人",
        type: "input",
        value: "",
        placeholder: "请输入承建单位-联系人",
        width: "30%",
        serverKey: "btUnitLeader",
        isShow: true
      },
      {
        formName: "承建单位-联系电话",
        type: "input",
        value: "",
        placeholder: "请输入承建单位-联系电话",
        width: "30%",
        serverKey: "btUnitContact",
        isShow: true
      }
    ],
    isShow: true
  },
  {
    name: "准备工作",
    list: [
      {
        formName: "模型文件名",
        type: "input",
        value: "",
        placeholder: "请输入模型文件的名称，如model.pth",
        width: "30%",
        serverKey: "modelFileName",
        isShow: true
      },
      {
        formName: "模型封装方式",
        type: "input",
        value: "",
        placeholder: "请输入模型封装方式，如Docker镜像",
        width: "30%",
        serverKey: "modelEncapWay",
        isShow: true
      },
      {
        formName: "模型文件大小",
        type: "input",
        value: "",
        placeholder: "例: 7.8G",
        width: "30%",
        serverKey: "modelFileSize",
        isShow: true
      },
      {
        formName: "MD5",
        type: "input",
        value: "",
        placeholder: "请输入MD5校验值",
        width: "30%",
        serverKey: "modelMd5Value",
        isShow: true
      },
      {
        formName: "SHA256校验",
        type: "input",
        value: "",
        placeholder: "请输入SHA256校验值",
        width: "30%",
        serverKey: "modelHashValue",
        isShow: true
      },
      {
        formName: "模型部署位置",
        type: "input",
        value: "",
        placeholder: "请输入模型部署的位置，如服务器地址",
        width: "30%",
        serverKey: "modelDeployAddr",
        isShow: true
      },
      {
        formName: "模型对外暴露端口",
        type: "input",
        value: "",
        placeholder: "例: 25001",
        width: "30%",
        serverKey: "modelPort",
        isShow: true
      },
      {
        formName: "GPU架构",
        type: "input",
        value: "",
        placeholder: "例: 12.1",
        width: "30%",
        serverKey: "modelCudaVersion",
        isShow: true
      },
      {
        formName: "驱动版本",
        type: "input",
        value: "",
        placeholder: "例: nv 驱动 530.30.02",
        width: "30%",
        serverKey: "modelDriveVersion",
        isShow: true
      },
      {
        formName: "模型API接口说明/模型调用例",
        type: "upload",
        value: "",
        placeholder: "附表记录",
        width: "50%",
        serverKey: "modelInterfaceDesc",
        isShow: true,
        fileList: [],
        accept: ".jpg,.jpeg,.png,.gif,.bmp,.JPG,.JPEG,.PBG,.GIF,.BMP"
      },
      {
        formName: "模型算法编码",
        type: "upload",
        value: "",
        placeholder: "附表记录",
        width: "50%",
        serverKey: "modelAlgorithmCode",
        isShow: true,
        fileList: [],
        accept: '.xlsx,.xls'
      },
      {
        formName: "训练样本/测试用例",
        type: "upload",
        value: "",
        placeholder: "附表记录",
        width: "50%",
        serverKey: "trainSample",
        isShow: true,
        fileList: [],
        accept: '.zip,.rar'
      },
      {
        formName: "训练样本/测试用例说明",
        type: "upload",
        value: "",
        placeholder: "附表记录",
        width: "50%",
        serverKey: "testCase",
        isShow: true,
        fileList: [],
        accept: '.xlsx,.xls'
      },
      // 上传模型训练代码
      {
        formName: "模型训练代码",
        type: "upload",
        value: "",
        placeholder: "附表记录",
        width: "50%",
        serverKey: "modelTrainCode",
        isShow: true,
        fileList: [],
      },
    ],
    isShow: true
  },
  {
    name: "申请类型",
    list: [
      {
        formName: "申请类型",
        type: "radioGroup",
        value: "1",
        width: "30%",
        serverKey: "applyForType",
        isShow: true,
        modelList: [
          { value: "1", label: "线下申请" },
          // { value: "2", label: "线上申请" }
        ]
      },
      {
        formName: "模型方式",
        type: "radioGroup",
        value: "1",
        width: "30%",
        serverKey: "modelWay",
        isShow: true,
        modelList: [
          { value: "1", label: "分类任务" },
          { value: "2", label: "目标检测" }
        ],
        onUpdateValue: (val) => {
          console.log(val);
        }
      }
    ],
    isShow: true
  },
  {
    name: '模型识别配置',
    list: [
      {
        formName: '模型识别类别',
        type: 'dynamicModalClass',
        value: [],
        serverKey: 'modelClass',
        isShow: true,
        width: "60%",
        span: '24 m:24'
      },
      {
        formName: "评估图表",
        type: "checkbox",
        value: null,
        checkboxList: [
          { value: '0', formName: 'P-R曲线', key: 'PRCurve' },
          { value: '1', formName: '混淆矩阵图', key: 'ConfusionMatrix' },
          { value: '2', formName: 'ROC曲线', key: 'ROC' },

        ],
        width: "98%",
        serverKey: "assessChart",
        isShow: true,
      },
    ],
    isShow: true
  }
]);

const activeTab = ref<string>('基本信息');
const detailData = ref<any>({});

const getActiveFields = computed(() => {
  const group = configList.value.find(group => group.name === activeTab.value)
  return group ? group.list.filter(item => item.isShow) : []
})

watch(() => applyId.value, (newVlue) => {
  if (newVlue) {
    getApplyDetail(newVlue)
  }
}, {
  immediate: true
})

const getApplyDetail = async (id: string) => {
  const res = await fetchModelDetails({
    id
  });
  if (res.data) {
    detailData.value = res.data;
    detailData.value.modelClass = JSON.parse(detailData.value.modelClass);

    // assessChart 评估图表
    detailData.value.assessChart = detailData.value.assessChart.split(',');
  }
}

const getCheckName = (value: string, checkboxList: any) => {
  const item = checkboxList.find(item => `${item.value}` == `${value}`);
  return item ? item.formName : '';
}

const getFileExtension = (url: string): string => {
  try {
    const pathname = new URL(url).pathname;
    const basename = pathname.split('/').pop() || '';
    const lastDotIndex = basename.lastIndexOf('.');

    return lastDotIndex > 0
      ? basename.substring(lastDotIndex + 1)
      : '';
  } catch (error) {
    console.error('Invalid URL:', error);
    return '';
  }
};


const preview = (url: string) => {
  emit("preview", {
    previewPath: new URL(url),
    previewFileSuffix: getFileExtension(url)
  });
}

const downloadFile = (url: string) => {
  const fileExtension = getFileExtension(url);
  const a = document.createElement('a');
  a.href = url;
  a.download = `模型训练代码.${fileExtension}`; // 指定下载的文件名
  a.style.display = 'none';
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
}


// newCode
const serverKeyMap = {
  'assessChart': [
    { value: '0', formName: 'P-R曲线', key: 'PRCurve' },
    { value: '1', formName: '混淆矩阵图', key: 'ConfusionMatrix' },
    { value: '2', formName: 'ROC曲线', key: 'ROC' },
  ]
}

// 创建按钮文本
const createButtonText = '添加模型识别类别'

const { bool: indicatorDrawerVisible, setTrue: openIndicatorDrawer } = useBoolean();

const bindIndicator = (item: any) => {
  classModel.value = item;
  openIndicatorDrawer();
}

const classModel = ref<any>({}); // 模型识别类别
</script>

<template>
  <NModal v-model:show="visible" title="详情" preset="card" class="w-1200px">
    <NScrollbar class="h-auto max-h-600px min-h-300px pr-20px">
      <n-tabs type="line" animated v-model:value="activeTab">
        <n-tab-pane v-for="group in configList" :key="group.name" :name="group.name" :title="group.name"></n-tab-pane>
      </n-tabs>

      <NForm ref="formRef" :model="model" label-placement="left">
        <NGrid responsive="screen" item-responsive x-gap="24">
          <NFormItemGi v-for="field in getActiveFields" :key="field.serverKey" :path="field.serverKey"
            :label="field.formName + ':'" class="h-auto" :span="!!field.span ? field.span : '24 m:12'">
            <div v-if="field.serverKey === 'testIndic'">
              <n-space>
                <n-tag v-for="item in detailData.testIndic" :key="item.key"
                  :type="detailData[field.serverKey].includes(item.key) ? 'success' : 'info'">
                  {{ getCheckName(item, field.checkboxList) }}
                </n-tag>
              </n-space>
            </div>
            <n-button quaternary type="primary"
              v-else-if="field.serverKey === 'modelInterfaceDesc' || field.serverKey === 'modelCase' || field.serverKey === 'modelAlgorithmCode' || field.serverKey === 'testCase'"
              @click="preview(detailData[field.serverKey])" :disabled="!detailData[field.serverKey]">
              点击查看
            </n-button>
            <div v-else-if="field.serverKey === 'modelWay'">
              {{ detailData[field.serverKey] == '1' ? '分类任务' : '目标检测' }}
            </div>
            <div v-else-if="field.serverKey === 'applyForType'">
              {{ detailData[field.serverKey] == '1' ? '线下申请' : '线上申请' }}
            </div>
            <!-- trainSample -->
            <div v-else-if="field.serverKey === 'trainSample'">
              {{ detailData['groupNameAndVersion'] }}
            </div>
            <!-- modelTrainCode 点击下载 -->
            <n-button quaternary type="primary" v-else-if="field.serverKey === 'modelTrainCode'"
              @click="downloadFile(detailData[field.serverKey])">
              点击下载
            </n-button>
            <!-- netWorkLog -->
            <div v-else-if="field.serverKey === 'netWorkLog'" class="w-full">
              <n-virtual-list ref="virtualListInst" class="!h-full !text-[#7d8799]" :item-size="64"
                :items="detailData['netWorkLog']">
                <template #default="{ item, index }">
                  <div :key="index" class="item box-border px-8px" style="height: 64px">
                    <div>{{ `时间: ${item.timestamp}, 模型名称: ${item.modelName}, 状态: ${item.status}, ` }}</div>
                  </div>
                </template>
              </n-virtual-list>
            </div>
            <!-- jsonLog -->
            <div v-else-if="field.serverKey === 'jsonLog'" class="w-full h-300px overflow-auto">
              <VueJsonPretty path="res" :data="detailData['jsonLog']" :show-length="true" />
            </div>
            <!-- modelClass -->
            <div v-else-if="field.serverKey === 'modelClass'" class="w-full h-auto flex-col">
              <NDynamicInput v-model:value="detailData['modelClass']" class="w-full" disabled
                :create-button-default="createButtonText">
                <template #default="{ value: item }">
                  <div class="w-full flex-center gap-8px">
                    <NInput class="w-1/2" v-model:value="item.className" placeholder="请输入模型识别类别" disabled />
                    <NButton class="w-160px" type="primary" @click="() => bindIndicator(item)">
                      查看关联指标
                    </NButton>
                  </div>
                </template>
              </NDynamicInput>
            </div>
            <!-- assessChart -->
            <div v-else-if="field.serverKey === 'assessChart'">
              <n-space>
                <n-tag v-for="item in detailData[field.serverKey]" :key="item" type="success">
                  {{ getCheckName(item, serverKeyMap[field.serverKey]) }}
                </n-tag>
              </n-space>
            </div>
            <div v-else class="line-clamp-3">{{ detailData[field.serverKey] }}</div>
          </NFormItemGi>
        </NGrid>
      </NForm>
    </NScrollbar>

    <BindIndicatorModal v-model:visible="indicatorDrawerVisible" v-model:classId="classModel.classId"
      v-model:modelGridData="classModel.modelGridData" v-model:modelCommonData="classModel.modelCommonData"
      v-model:gridCheckedRowKeys="classModel.gridCheckedRowKeys"
      v-model:commonCheckedRowKeys="classModel.commonCheckedRowKeys" />
  </NModal>
</template>

<style scoped></style>
