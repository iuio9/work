<template>
  <n-modal v-model:show="visible" preset="card" title="模型调试" class="w-80% h-800px" :close-on-esc="false"
    :closable="false">
    <n-form ref="formRef" :rules="rules" label-placement="left" label-width="auto"
      require-mark-placement="right-hanging" class="w-full h-full">
      <div class="w-full h-full flex justify-between items-start gap-24px overflow-y-auto pb-18px">
        <div class="w-1/2 h-full flex items-center" v-for="(item, index) of configList" :key="index">
          <n-card class="w-full h-640px overflow-y-auto" :title="item.name">
            <div class="w-full h-full flex flex-col items-start">
              <div :style="{ width: val.width || '100%' }" v-for="(val, idx) of item.list" :key="idx">
                <n-grid :cols="24" :x-gap="24" class="ml-24px">
                  <n-form-item-gi :span="24" :label="val.formName" :path="val.serverKey" v-if="val.type !== 'log'">
                    <template #label>
                      <!-- 上传请求参数模板 || 上传调试文件 -->
                      <div v-if="val.formName === '上传请求参数模版'" class="flex justify-end items-center gap-2px">
                        <span>{{ val.formName }}</span>
                        <n-popover trigger="hover" placement="top">
                          <template #trigger>
                            <div class="operate-btn" @click="handlePass('2')">
                              <SvgIcon local-icon="memory--tooltip-end-help" class="text-22px" />
                            </div>
                          </template>
                          用于上传接口参数的数据模板文件，系统解析后将其中数据作为接口调用参数发送.
                        </n-popover>
                      </div>
                      <div v-else-if="val.formName === '上传调试文件'" class="flex justify-end items-center gap-2px">
                        <span>{{ val.formName }}</span>
                        <n-popover trigger="hover" placement="top">
                          <template #trigger>
                            <div class="operate-btn" @click="handlePass('2')">
                              <SvgIcon local-icon="memory--tooltip-end-help" class="text-22px" />
                            </div>
                          </template>
                          系统将解析其中数据并作为参数发送至接口 。
                        </n-popover>
                      </div>
                      <div v-else>
                        {{ val.formName }}
                      </div>
                    </template>
                    <component :is="getFormComponent(val.type, val.value, val)"
                      v-if="getFormComponent(val.type, val.value, val)" v-model:value="val.value"
                      :placeholder="val.placeholder" :options="val.options" :modelList="val.modelList"
                      :logData="val.logData" :isMultiple="val.isMultiple" :columns="val.columns" :data="val.query"
                      :items="netWorkList" @update:value="handleFieldChange(val, $event)" />
                    <div class="w-300px" v-show="val.fileTooltip">{{ val.fileTooltip }}</div>
                  </n-form-item-gi>
                  <n-form-item-gi :span="24" :label="val.formName" :path="val.serverKey" v-else>
                    <div class="json_pretty_container">
                      <VueJsonPretty path="res" :data="val.logData" :show-length="true" />
                    </div>
                  </n-form-item-gi>
                </n-grid>
              </div>
            </div>
            <template #footer>
              <n-space justify="end" space="[16px]">
                <n-button v-for="val of item.buttons" :key="val" :type="val.type" @click="handleOperate(val.text)">{{
                  val.text }}</n-button>
              </n-space>
            </template>
          </n-card>
        </div>
      </div>
    </n-form>
    <template #footer>
      <n-space justify="end" space="[16px]">
        <n-button @click="closeModal" class="bg-gray-200 text-gray-800 hover:bg-gray-300">取消</n-button>
      </n-space>
    </template>
  </n-modal>
  <UploadOperateDrawer ref="uploadRef" v-model:visible="drawerVisible" v-model:isUpSuccess="isUpSuccess"
    v-model:markStatus="model.markStatus" v-model:importMode="model.importMode" @submitted="uploadSubmit" />
</template>

<script setup lang="ts">
import { h, VNode } from 'vue';
import {
  NInput,
  NSelect,
  NRadioGroup,
  NRadio,
  NCascader,
  NDatePicker,
  NButton,
  NSpin,
  NDynamicInput,
  NCheckboxGroup,
  NCheckbox,
  NUpload,
  type UploadFileInfo,
  NVirtualList,
  useDialog
} from 'naive-ui';

import VueJsonPretty from 'vue-json-pretty'
import 'vue-json-pretty/lib/styles.css'

import { getToken } from '@/store/modules/auth/shared';
import { useFormRules, useNaiveForm } from '@/hooks/common/form';
import { useBoolean } from '~/packages/hooks/src';

import UploadOperateDrawer from './upload-operate-drawer.vue';
import { debugModel, getParamName, isSavaDebugLogGet, savaDebugLog, savaDebugLogGet } from '@/service/api/third';

// 类型定义
type FormComponentType = 'input' | 'textarea' | 'select' | 'text' | 'radioGroup' | 'cascader'
  | 'datetime' | 'upload' | 'customUpload' | 'dynamicInput' | 'checkbox' | 'log' | 'netWork';

interface RowData {
  modelList?: { value: string; label: string }[];
  fileList?: UploadFileInfo[];
  serverKey?: string;
  buttonText?: string;
  checkboxList?: { value: string; formName: string }[];
  logData?: any;
}

interface UploadOptions {
  action: string;
  headers: Record<string, string>;
  onFinish: (file: UploadFileInfo) => void;
  onRemove: (file: UploadFileInfo) => void;
  onBeforeUpload: (options: any) => boolean;
  buttonText: string;
}

interface FormField {
  value: unknown;
  [key: string]: unknown;
}

// 表单分组配置接口
interface FormGroupConfig {
  name: string;
  buttons?: {
    text: string;
    type: 'primary' | 'default';
    onClick?: () => void;
  }[];
  list: FormField[];
}


defineOptions({ name: 'ModelApplyTestModal' });

// 组件props
const visible = defineModel<boolean>('visible', { default: false });

const modelId = defineModel<string>('modelId', { default: '' });

const dialog = useDialog()

// 关闭模态框
const closeModal = async () => {
  const res = await isSavaDebugLogGet({
    id: modelId.value
  })
  if (res.data) {
    visible.value = false;
  } else {
    dialog.warning({
      title: '警告',
      content: '当前结果未保存，请确定是否要保存？',
      positiveText: '保存',
      negativeText: '取消',
      draggable: true,
      onPositiveClick: async () => {
        const res1 = await savaDebugLog({
          ...saveInfo.value,
          modelBaseId: modelId.value
        });
        if (res1.data) {
          visible.value = false;
          window.$message?.success?.('保存成功！');
        }
      },
      onNegativeClick: () => {
        visible.value = false;
      }
    })
  }
};

const { formRef, validate } = useNaiveForm();
const rules = computed<Record<string, any[]>>(() => {
  const { formRules } = useFormRules();
  return {};
});

// file upload
const { bool: drawerVisible, setTrue: openDrawer } = useBoolean();
const isUpSuccess = ref<boolean>(false);
const model = ref<any | null>({
  fileList: [],
  imgList: [],
  uploadList: [],
  importMode: "0-0"
});

const configList = ref<FormGroupConfig[]>([
  {
    name: '模型调试',
    list: [
      // 模型地址
      {
        formName: '模型地址',
        serverKey: 'modelAddress',
        type: 'input',
        value: '',
        placeholder: '请输入模型地址',
        required: true,
        width: '88%'
      },
      // 接口请求方式
      {
        formName: "接口请求方式",
        type: "radioGroup",
        value: "",
        width: '88%',
        serverKey: "requestType",
        isShow: true,
        modelList: [
          { value: "1", label: "POST" },
          { value: "2", label: "GET" },
          { value: "3", label: "PUT" },
        ]
      },
      // 提交方式
      {
        formName: "提交方式",
        type: "radioGroup",
        value: "",
        width: '88%',
        serverKey: "applyForType",
        isShow: true,
        modelList: [
          { value: "1", label: "Json" },
          { value: "2", label: "Excel(FormData)" }
        ]
      },
      {
        formName: '上传请求参数模版',
        serverKey: 'modelFile',
        type: 'upload',
        value: undefined,
        buttonText: '上传请求参数模版',
        // json excel
        accept: ".json,.xlsx,.xls",
        fileList: [],
        required: true,
        width: '88%'
      },
      // 输入参数
      {
        formName: '测试文件参数名',
        serverKey: 'paramName',
        type: 'input',
        value: undefined,
        options: [],
        placeholder: '请输入参数名',
        required: true,
        width: '88%'
      },
      {
        formName: '上传调试文件',
        serverKey: 'debugFile',
        type: 'upload',
        value: undefined,
        buttonText: '上传调试文件',
        // 提示
        fileTooltip: '注：文件大小控制在10M内',
        accept: ".zip,.rar,.7z",
        fileList: [],
        required: true,
        width: '10%',
        beforeUpload: (options: any) => {
          console.log('options: ', options);
          const { file } = options;
          const isLtSize = file.file.size / 1024 / 1024 < 10;
          if (!isLtSize) {
            window.$message?.error("上传文件大小不能超过 10 MB!");
            return false;
          }
          return true;
        }
      },
    ],
    buttons: [
      {
        text: '开始调试',
        type: 'primary',
      },
      {
        text: '清空',
        type: 'primary',
      },
      {
        text: '保存测试结果',
        type: 'primary',
      }
    ]
  },
  // 日志输出
  {
    name: '日志输出',
    list: [
      {
        formName: '',
        serverKey: 'netWork',
        type: 'netWork',
        value: undefined,
        logData: null,
        width: '94%',
        required: true,
      },
      {
        formName: '',
        serverKey: 'logOutput',
        type: 'log',
        value: undefined,
        logData: null,
        required: true,
        width: '88%',
      }
    ]
  }
]);

const netWorkList = ref<any>([])
const virtualListInst = ref(null); // 创建 ref 对象

const saveInfo = ref<any>({});

// 根据表单字段类型返回对应的组件
const getFormComponent = (
  type: FormComponentType,
  value: unknown,
  rowData: RowData
): VNode | null => {
  const commonProps = {
    'onUpdate:value': (val: unknown) => handleFieldChange(rowData as FormField, val)
  };

  switch (type) {
    case 'input':
      return h(NInput, { ...commonProps, modelValue: value });

    case 'textarea':
      return h(NInput, { ...commonProps, modelValue: value, type: 'textarea' });

    case 'select':
      return h(NSelect, { ...commonProps, modelValue: value });

    case 'text':
      return h('span', value ?? '无');

    case 'radioGroup':
      return h(
        NRadioGroup,
        { ...commonProps, modelValue: value, name: 'anoType', size: 'large' },
        {
          default: () => rowData.modelList?.map(item =>
            h(NRadio, { key: item.value, value: item.value, label: item.label })
          )
        }
      );

    case 'cascader':
      return h(NCascader, { ...commonProps, modelValue: value });

    case 'datetime':
      return h(NDatePicker, {
        ...commonProps,
        modelValue: value,
        type: 'datetime',
        clearable: true,
        class: '!w-full'
      });

    case 'upload': {
      const action = `${import.meta.env.VITE_SERVICE_BASE_URL}/upload`;
      const uploadType = rowData.serverKey === 'modelInterfaceDesc' ? '1' : '2';
      return createUploadComponent(rowData.fileList || [], {
        action,
        headers: {
          Authorization: `Bearer ${getToken()}`,
          type: uploadType
        },
        onFinish: (file) => handleUploadFinish(file, rowData),
        onRemove: (file) => handleRemove(file, rowData),
        onBeforeUpload: rowData?.beforeUpload,
        buttonText: '上传文件'
      });
    }

    case 'customUpload':
      return () => [
        h(NButton, {
          type: 'primary',
          onClick: () => openDrawer()
        }, rowData.buttonText || '上传文件'),
        isUpSuccess.value && h('div', {
          class: 'ml-16px flex items-center justify-start'
        }, [
          h(NSpin, { size: 'small' }),
          h('div', { class: 'ml-8px text-14px' }, '文件异步上传中... 请稍等!!!')
        ]),
        h('div', {
          class: 'ml-16px flex items-center justify-start',
          style: { display: model.value.imgList.length !== 0 ? '' : 'none' }
        }, [
          h('span', '已上传'),
          h('span', `${model.value.imgList.length}个文件`)
        ])
      ]

    case 'dynamicInput':
      return h(NDynamicInput, {
        ...commonProps,
        modelValue: value as Array<{ formName: string; key: string; value: unknown }>,
        class: "w-full",
        onCreate: () => ({ formName: "", key: "", value: undefined })
      }, {
        default: ({ value: item }) => h('div', { class: "w-full flex-center gap-8px" }, [
          h(NInput, {
            modelValue: item.formName,
            'onUpdate:modelValue': (val: string) => item.formName = val,
            placeholder: "请输入指标名称"
          }),
          h(NInput, {
            modelValue: item.key,
            'onUpdate:modelValue': (val: string) => item.key = val,
            placeholder: "请输入指标键名"
          }),
          h(NInput, {
            modelValue: item.value,
            'onUpdate:modelValue': (val: unknown) => item.value = val,
            placeholder: "请输入指标值"
          })
        ]),
        'create-button-default': () => '添加指标'
      });

    case 'checkbox':
      return h(NCheckboxGroup, {
        ...commonProps,
        modelValue: value as string[],
      }, {
        default: () => rowData.checkboxList?.map(item =>
          h(NCheckbox, { key: item.value, value: item.value, label: item.formName })
        )
      });

    case 'log':
      return h(VueJsonPretty, {
        path: 'res',
        data: rowData.logData,
        'show-length': true
      });

    case 'netWork':
      return h('div', {
        class: 'w-full h-200px overflow-y-auto bg-#000 py-8px box-border',
      }, [
        h(
          NVirtualList,
          {
            ref: virtualListInst,
            class: '!h-full !text-[#7d8799]',
            itemSize: 64,
            items: netWorkList.value
          },
          {
            default: ({ item, index }) => h(
              'div',
              {
                key: index,
                class: 'item box-border px-8px',
                style: 'height: 64px'
              },
              [
                h('div', `耗时: ${item.timestamp || 'N/A'}, 状态: ${item.status || 'N/A'}`),
              ]
            )
          }
        )
      ])

    default:
      return null;
  }
};


// 封装通用上传组件生成逻辑
const createUploadComponent = (
  fileList: UploadFileInfo[],
  options: UploadOptions
): VNode => {
  return h(NUpload, {
    'v-model:file-list': fileList,
    ...options,
    max: 1,
  }, { default: () => h(NButton, {}, options.buttonText) });
};

// 处理文件上传完成事件
const handleUploadFinish = (file: UploadFileInfo, rowData: RowData) => {
  if (!file.event?.currentTarget?.response) return;

  const res = JSON.parse(file.event.currentTarget.response);
  if (res.code === 200) {
    file.url = res.data;
    file.status = 'finished';
    rowData.fileList?.push(file);

    // getParamName().then(res => {
    //   configList.value[0].list.forEach(item => {
    //     if (item.serverKey === 'paramName') {
    //       item.options = res.data;
    //     }
    //   })
    // })

  }
};

// 处理文件移除事件
const handleRemove = (file: UploadFileInfo, rowData: RowData) => {
  if (!rowData.fileList) return;

  rowData.fileList = rowData.fileList.filter(item => item.id !== file.id);
};

const beforeUpload = (options: any) => {
  const { file } = options;
  const isLtSize = file.file.size / 1024 / 1024 < 10;
  if (!isLtSize) {
    window.$message?.error("上传图片大小不能超过 10 MB!");
    return false;
  }
  return true;
};

// 处理表单字段值变化
const handleFieldChange = (field: FormField, value: unknown) => {
  field.value = value;
};

async function* createLogGenerator(logsArray, delay = 1000) {
  // 直接遍历数组并按指定间隔生成数据
  for (const log of logsArray) {
    yield log; // 直接返回原始数组元素
    await new Promise(resolve => setTimeout(resolve, delay)); // 控制流式间隔
  }
}

const handleOperate = async (text: string) => {
  if (text === '开始调试') {
    const params = extractFormValues(configList.value[0].list);
    const res = await debugModel(params);
    if (res.data) {
      const { netData, interfaceData, modelDebugLogEntity } = res.data;
      saveInfo.value = modelDebugLogEntity;
      configList.value.forEach(item => {
        item.list.forEach(subItem => {
          if (subItem.serverKey === 'netWork') {
            subItem.logData = netData;
          }
          if (subItem.serverKey === 'logOutput') {
            subItem.logData = interfaceData;
          }
        })
      });

      netWorkList.value = [netData];
    }

  }
  if (text === '清空') {
    configList.value[0].list.forEach(item => {
      item.value = '';
      if (item.fileList) {
        item.fileList = [];
      }
    })
    configList.value[1].list.forEach(item => {
      item.logData = null;
    })
    saveInfo.value = {};

  }
  if (text === '保存测试结果') {
    const res = await savaDebugLog({
      ...saveInfo.value,
      modelBaseId: modelId.value
    });
    if (res.data) {
      window.$message?.success('保存成功');
    }
  }
}

interface FileFormData {
  append(key: string, value: File): void;
}


function extractFormValues(formConfig: FormField[]): FormData {
  const formData = new FormData();

  formConfig.forEach(item => {
    const { serverKey, value, fileList } = item;

    // 特殊处理 modelFile 和 debugFile
    if (serverKey === 'modelFile' || serverKey === 'debugFile') {
      if (fileList && fileList.length > 0) {
        formData.append(serverKey, fileList[0].file.file);
      }
    } else {
      // 处理普通字段，将值转换为字符串
      formData.append(serverKey, value !== undefined && value !== null ? String(value) : '');
    }
  });

  return formData;
}
</script>

<style scoped lang="scss"></style>
