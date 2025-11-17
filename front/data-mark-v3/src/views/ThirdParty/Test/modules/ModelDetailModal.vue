<script setup lang="tsx">
import { ref, reactive, computed, watch } from "vue";
import { useRouter } from "vue-router";
import { NModal, NScrollbar, NForm, NGrid, NFormItemGi } from "naive-ui";

defineOptions({
  name: "ModelDetailModal",
});

const visible = defineModel<boolean>("visible", {
  default: false,
});

const router = useRouter();

interface Props {
  formData?: any;
}

interface FormGroupConfig {
  tab?: string;
  name: string;
  list: FormFieldConfig[];
}

const props = withDefaults(defineProps<Props>(), {
  formData: {}
});

const emit = defineEmits<{
  (e: "submitted"): void;
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

const configList = ref<FormGroupConfig[]>([
  {
    tab: "basic",
    name: "基本信息",
    list: basicInfoList
  },
  {
    tab: "preparation",
    name: "准备工作",
    list: preparationWorkList
  },
  {
    tab: "indicators",
    name: "测试指标",
    list: testIndicatorsList
  },
  {
    tab: "attachment",
    name: "附件",
    list: attachmentList
  }
]);

// 当前激活的标签页
const activeTab = ref('基本信息');

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
</script>

<template>
  <NModal v-model:show="visible" title="详情" preset="card" class="w-800px">
    <NScrollbar class="h-300px pr-20px">
      <n-tabs type="line" animated default-value="基本信息" v-model:value="activeTab">
        <n-tab-pane name="基本信息" title="basic"></n-tab-pane>
        <n-tab-pane name="准备工作" title="basic"></n-tab-pane>
        <n-tab-pane name="测试指标" title="basic"></n-tab-pane>
        <n-tab-pane name="附件" title="basic"></n-tab-pane>
      </n-tabs>
      <NForm ref="formRef" :model="model" :rules="rules" label-placement="left">
        <NGrid responsive="screen" item-responsive x-gap="24">
          <NFormItemGi v-for="field in activeFields" :key="field.serverKey" :path="field.serverKey"
            :label="field.formName + ':'" class="h-48px" span="24 m:12">
            {{ field.value }}
          </NFormItemGi>
        </NGrid>
      </NForm>
    </NScrollbar>
  </NModal>
</template>

<style scoped></style>
