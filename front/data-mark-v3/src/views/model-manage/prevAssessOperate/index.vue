<template>
  <div class="w-full h-full box-border flex flex-col justify-start items-start wrap-container">
    <div class="w-full flex-1 box-border p-24px overflow-y-auto">
      <n-form ref="formRef" :rules="rules" label-placement="left" label-width="auto"
        require-mark-placement="right-hanging" class="w-full h-full">
        <div class="w-full h-auto flex flex-col justify-start items-center gap-12px overflow-y-auto pb-18px">
          <div class="w-full h-auto flex items-center" v-for="(item, index) of configList" :key="index">
            <n-card class="w-full h-auto" :title="item.name">
              <div class="w-full h-auto flex flex-col items-start">
                <div :style="{ width: val.width || '100%' }" v-for="(val, idx) of item.list" :key="idx">
                  <n-grid :cols="24" :x-gap="24" class="ml-24px">
                    <n-form-item-gi :span="24" :label="val.formName" :path="val.serverKey">
                      <!-- 使用 v-if 避免渲染未定义的组件 -->
                      <component :is="getFormComponent(val.type)" v-if="getFormComponent(val.type)"
                        v-model:value="val.value" :placeholder="val.placeholder" :options="val.options"
                        :modelList="val.modelList" :isMultiple="val.isMultiple" :columns="val.columns" :data="val.query"
                        @update:value="handleFieldChange(val, $event)" />
                    </n-form-item-gi>
                  </n-grid>
                </div>
              </div>
            </n-card>
          </div>
        </div>
      </n-form>
    </div>
    <div class="footer w-full box-border flex justify-start items-center gap-24px px-24px py-12px bg-white">
      <n-button type="info" @click="handleOperate('submit')" class="w-88px">
        确定提交
      </n-button>
      <n-button type="info" @click="handleOperate('submit')" class="w-88px">
        审核通过
      </n-button>
      <n-button type="default" @click="handleOperate('back')" class="w-88px">
        返回页面
      </n-button>
    </div>
  </div>
</template>

<script setup lang="ts">
import { h, ref, computed } from 'vue';
import { NForm, NCard, NGrid, NFormItemGi, NInput, NSelect, NDataTable, NRadioGroup, NRadioButton, NCascader, NButton, NUpload } from 'naive-ui';
import { useRouter, useRoute } from 'vue-router';
import { useFormRules, useNaiveForm } from '@/hooks/common/form';

// 表单字段配置接口
interface FormFieldConfig {
  formName: string;
  type: 'input' | 'textarea' | 'select' | 'dynamicInput' | 'text' | 'radioGroup' | 'cascader' | 'upload' | 'datetime';
  value: string | number | undefined;
  placeholder?: string;
  width?: string;
  serverKey: string;
  isShow?: boolean;
  isMultiple?: boolean;
  options?: any[];
  renderLabel?: (options: any) => any;
  columns?: any[];
  query?: any[];
  modelList?: { value: string | number; label: string }[];
}

// 表单分组配置接口
interface FormGroupConfig {
  name: string;
  list: FormFieldConfig[];
}

// 生成基本信息部分数据
const basicInfoList: FormFieldConfig[] = [
  {
    formName: "模型名称",
    type: "input",
    value: undefined,
    placeholder: "请输入模型名称",
    width: "30%",
    serverKey: "modelName",
    isShow: true
  },
  {
    formName: "模型来源",
    type: "input",
    value: undefined,
    placeholder: "数字化项目应提供项目编号及项目名称",
    width: "30%",
    serverKey: "modelSource",
    isShow: true
  },
  {
    formName: "模型封装方式",
    type: "input",
    value: undefined,
    placeholder: "Docker 镜像/模型文件",
    width: "30%",
    serverKey: "modelPackaging",
    isShow: true
  },
  {
    formName: "模型部署位置",
    type: "input",
    value: undefined,
    placeholder: "",
    width: "30%",
    serverKey: "modelDeploymentLocation",
    isShow: true
  },
  {
    formName: "业务单位",
    type: "input",
    value: undefined,
    placeholder: "",
    width: "30%",
    serverKey: "businessUnit",
    isShow: true
  },
  {
    formName: "业务单位负责人/联系方式",
    type: "input",
    value: undefined,
    placeholder: "",
    width: "30%",
    serverKey: "businessUnitContact",
    isShow: true
  },
  {
    formName: "开发单位",
    type: "input",
    value: undefined,
    placeholder: "",
    width: "30%",
    serverKey: "developmentUnit",
    isShow: true
  },
  {
    formName: "开发单位负责人/联系方式",
    type: "input",
    value: undefined,
    placeholder: "",
    width: "30%",
    serverKey: "developmentUnitContact",
    isShow: true
  },
  {
    formName: "登记日期",
    type: "input",
    value: undefined,
    placeholder: "",
    width: "30%",
    serverKey: "registrationDate",
    isShow: true
  }
];

// 生成准备工作部分数据
const preparationWorkList: FormFieldConfig[] = [
  {
    formName: "模型文件名",
    type: "input",
    value: undefined,
    placeholder: "",
    width: "30%",
    serverKey: "modelFileName",
    isShow: true
  },
  {
    formName: "模型文件大小",
    type: "input",
    value: undefined,
    placeholder: "例: 7.8G",
    width: "30%",
    serverKey: "modelFileSize",
    isShow: true
  },
  {
    formName: "模型API接口说明",
    type: "input",
    value: undefined,
    placeholder: "附表记录",
    width: "30%",
    serverKey: "modelApiDescription",
    isShow: true
  },
  {
    formName: "模型对外暴露端口",
    type: "input",
    value: undefined,
    placeholder: "例: 25001",
    width: "30%",
    serverKey: "modelExposedPort",
    isShow: true
  },
  {
    formName: "模型cuda版本",
    type: "input",
    value: undefined,
    placeholder: "例: 12.1",
    width: "30%",
    serverKey: "modelCudaVersion",
    isShow: true
  },
  {
    formName: "模型驱动版本",
    type: "input",
    value: undefined,
    placeholder: "例: nv 驱动 530.30.02",
    width: "30%",
    serverKey: "modelDriverVersion",
    isShow: true
  },
  {
    formName: "模型调用例",
    type: "input",
    value: undefined,
    placeholder: "附表记录",
    width: "30%",
    serverKey: "modelCallExample",
    isShow: true
  },
  {
    formName: "模型功能",
    type: "input",
    value: undefined,
    placeholder: "概述模型功能",
    width: "30%",
    serverKey: "modelFunction",
    isShow: true
  },
  {
    formName: "模型检测场景",
    type: "input",
    value: undefined,
    placeholder: "详细罗列检测场景，例: 人脸检测、口罩佩戴检测...",
    width: "30%",
    serverKey: "modelDetectionScene",
    isShow: true
  },
  {
    formName: "训练样本",
    type: "input",
    value: undefined,
    placeholder: "提供训练样本集及标注文件",
    width: "30%",
    serverKey: "trainingSamples",
    isShow: true
  }
];

// 生成测试指标部分数据
const testIndicatorsList: FormFieldConfig[] = [
  {
    formName: "准确率",
    type: "input",
    value: undefined,
    placeholder: "例: 需达到 大于等于0.93",
    width: "30%",
    serverKey: "accuracy",
    isShow: true
  },
  {
    formName: "精确率",
    type: "input",
    value: undefined,
    placeholder: "例: 需达到 大于等于0.91",
    width: "30%",
    serverKey: "precision",
    isShow: true
  },
  {
    formName: "召回率",
    type: "input",
    value: undefined,
    placeholder: "例: 需满足 小于0.02",
    width: "30%",
    serverKey: "recall",
    isShow: true
  },
  {
    formName: "F1-分数",
    type: "input",
    value: undefined,
    placeholder: "例: 需满足 大于等于 0.87",
    width: "30%",
    serverKey: "f1Score",
    isShow: true
  },
  {
    formName: "IoU",
    type: "input",
    value: undefined,
    placeholder: "例: 不能低于0.3",
    width: "30%",
    serverKey: "iou",
    isShow: true
  },
  {
    formName: "均方误差",
    type: "input",
    value: undefined,
    placeholder: "例: 需达到 大于等于0.91",
    width: "30%",
    serverKey: "meanSquaredError",
    isShow: true
  },
  {
    formName: "场景覆盖",
    type: "input",
    value: undefined,
    placeholder: "填写需要满足设计要求的场景列表及详细内容",
    width: "30%",
    serverKey: "scenarioCoverage",
    isShow: true
  }
];

// 附件
const attachmentList: FormFieldConfig[] = [
  {
    formName: "上传附件",
    type: "upload",
    value: undefined,
    placeholder: "",
    width: "60%",
    serverKey: "modelFile",
    isShow: true
  },
]

const configList = ref<FormGroupConfig[]>([
  {
    name: "基本信息",
    list: basicInfoList
  },
  {
    name: "准备工作",
    list: preparationWorkList
  },
  {
    name: "测试指标",
    list: testIndicatorsList
  },
  {
    name: "附件",
    list: attachmentList
  }
]);

const { formRef, validate } = useNaiveForm();
const rules = computed<Record<string, any[]>>(() => {
  const { formRules } = useFormRules();
  return {};
});

const router = useRouter();
const route = useRoute();

const handleOperate = (type: 'submit' | 'back') => {
  if (type === 'submit') {
    validate().then(() => {
      console.log('表单验证通过，准备提交');
      // 这里添加实际的提交逻辑
    }).catch((error) => {
      console.log('表单验证失败', error);
    });
  }
  if (type === 'back') {
    router.back();
  }
};

// 处理表单字段值的变化
const handleFieldChange = (field: FormFieldConfig, value: any) => {
  field.value = value;
};

// 定义文件列表
const fileList = ref([
  {
    id: 'url-test',
    name: '附件1:模型API接口说明',
    url: 'https://www.mocky.io/v2/5e4bafc63100007100d8b70f',
    status: 'finished'
  },
  {
    id: 'text-message',
    name: '附表2 模型调用例',
    status: 'error'
  },
]);

// 处理文件上传变化事件
const handleUploadChange = (file: any, fileList: any) => {
  console.log('文件上传变化:', file, fileList);
};

// 处理文件移除事件
const handleRemove = (file: any) => {
  console.log('文件移除:', file);
};

// 处理文件列表更新事件
const handleFileListChange = (newFileList: any) => {
  fileList.value = newFileList;
  console.log('文件列表更新:', newFileList);
};

// 根据表单字段类型返回对应的组件
const getFormComponent = (type: FormFieldConfig['type']) => {
  switch (type) {
    case 'input':
      return NInput;
    case 'textarea':
      return h(NInput, { type: 'textarea' });
    case 'select':
      return NSelect;
    case 'dynamicInput':
      return NDataTable;
    case 'text':
      return {
        setup() {
          return () => h('span', field.value);
        }
      };
    case 'radioGroup':
      return {
        setup(props) {
          return () => h(NRadioGroup, {
            'v-model:value': props.value,
            name: 'anoType',
            size: 'large'
          }, props.modelList.map((item: any) =>
            h(NRadioButton, {
              value: item.value,
              label: item.label,
              onChange: (e: any) => handleFieldChange(field, e)
            })
          ));
        }
      };
    case 'cascader':
      return NCascader;
    case 'upload':
      return h(NUpload, {
        'v-model:file-list': fileList.value,
        action: 'https://www.mocky.io/v2/5e4bafc63100007100d8b70f',
        onChange: handleUploadChange,
        onRemove: handleRemove,
        'onUpdate:file-list': handleFileListChange
      }, [
        h(NButton, {}, '上传文件')
      ]);
    default:
      return null;
  }
};
</script>

<style scoped>
</style>
