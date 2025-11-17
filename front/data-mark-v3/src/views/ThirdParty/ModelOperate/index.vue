<template>
  <div class="w-full h-full box-border flex flex-col justify-start items-start wrap-container">
    <div class="w-full flex-1 box-border p-24px overflow-y-auto">
      <n-form ref="formRef" :rules="rules" label-placement="left" label-width="auto"
        require-mark-placement="right-hanging" class="w-full h-full">
        <div class="w-full h-auto flex flex-col justify-start items-center gap-12px overflow-y-auto pb-18px">
          <div class="w-full h-auto flex items-center" v-for="(item, index) of configList" :key="index">
            <n-card class="w-full h-auto" :title="item.name" v-show="item.isShow">
              <div class="w-full h-auto flex flex-col items-start">
                <div :style="{ width: val.width || '100%' }" v-for="(val, idx) of item.list" :key="idx">
                  <n-grid :cols="24" :x-gap="24" class="ml-24px">
                    <n-form-item-gi :span="24" :label="val.formName" :path="val.serverKey" class="">
                      <template #label>
                        <div class="w-full h-auto flex justify-end items-center">
                          <span>{{ val.formName }}</span>
                          <n-popover trigger="hover" v-if="val.tooltip">
                            <template #trigger>
                              <div>
                                <SvgIcon local-icon="ic--round-help" class="w-16px h-16px ml-4px" />
                              </div>
                            </template>
                            <span>{{ val.tooltip }}</span>
                          </n-popover>
                        </div>
                      </template>
                      <component :is="getFormComponent(val.type, val.value, val)"
                        v-if="getFormComponent(val.type, val.value, val)" v-model:value="val.value"
                        :placeholder="val.placeholder" :options="val.options" :modelList="val.modelList"
                        :isMultiple="val.isMultiple" :columns="val.columns" :data="val.query"
                        v-model:fileList="val.fileList" :tableData="val.tableData"
                        v-model:checkedRowKeys="val.checkedRowKeys"
                        v-model:defaultCheckedRowKeys="val.defaultCheckedRowKeys" />
                      <n-button type="primary" quaternary @click="handleDownloadTemplate(val.serverKey)"
                        v-show="val.serverKey === 'modelAlgorithmCode' || val.serverKey === 'testCase'"
                        class="ml-24px mb-10px">
                        下载模板
                      </n-button>
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
      <div v-for="(item, index) of applicationButtons" :key="index">
        <n-popconfirm v-if="item.confirmText" :title="item.confirmText"
          @positive-click="handleApplicationAction(item.action, route.query)">
          <template #trigger>
            <n-button type="primary" v-hasPermi="item.permission">
              {{ item.text }}
            </n-button>
          </template>
          {{ item.confirmText }}
        </n-popconfirm>

        <n-button v-else type="primary" v-hasPermi="item.permission"
          @click="handleApplicationAction(item.action, route.query)">
          {{ item.text }}
        </n-button>
      </div>
      <!-- ---------------------------------------------------- -->
      <n-button type="info" @click="handleOperate('submit')" class="w-88px">
        保存
      </n-button>
      <n-button type="default" @click="handleOperate('back')" class="w-88px">
        取消
      </n-button>
    </div>

    <UploadOperateDrawer ref="uploadRef" v-model:visible="drawerVisible" v-model:isUpSuccess="isUpSuccess"
      v-model:markStatus="model.markStatus" v-model:importMode="model.importMode" @submitted="uploadSubmit" />

    <ModelDetailModal v-model:visible="visible" v-model:apply-id="rowData.id" @preview="handlePreview" />
    <!-- <ModelTestModal v-model:visible="testVisible" /> -->
    <ModelApplyTestModal v-model:visible="testVisible" v-model:modelId="rowData.id" />
    <ModelApplyModal v-model:visible="uploadVisible" v-model:id="rowData.id" />

    <!-- preview file -->
    <FilePreviewModal v-model:visible="fileVisible" v-model:previewPath="previewPath"
      v-model:previewFileSuffix="previewFileSuffix" />

    <BindIndicatorModal v-model:visible="indicatorDrawerVisible" v-model:classId="classModel.classId"
      v-model:modelGridData="classModel.modelGridData" v-model:modelCommonData="classModel.modelCommonData"
      v-model:gridCheckedRowKeys="classModel.gridCheckedRowKeys"
      v-model:commonCheckedRowKeys="classModel.commonCheckedRowKeys" @afterLeave="handleAfterLeave" />
  </div>
</template>

<script setup lang="ts">
import { h, ref, computed } from 'vue';
import {
  NForm, NCard, NGrid, NFormItemGi, NInput, NSelect, NDataTable, NRadioGroup, NRadio,
  NRadioButton, NCascader, NButton, NUpload, NDatePicker, NDynamicInput, NCheckboxGroup, NCheckbox,
  NSpin,
  useDialog, NPopover
} from 'naive-ui';
import { useRouter, useRoute } from 'vue-router';
import { useFormRules, useNaiveForm } from '@/hooks/common/form';
import { getToken } from '@/store/modules/auth/shared';

import UploadOperateDrawer from './modules/upload-operate-drawer.vue';
import { useBoolean } from '~/packages/hooks/src';
import {
  addModelEvaluation, delModelEvaluation, editModelEvaluation, fetchModelDetails, submitApproveModelEvaluation, deleteFile
} from '@/service/api/third';
import { downloadByData } from '@/utils/common';

import axios from 'axios';

import ModelDetailModal from "./modules/ModelDetailModal.vue";
import ModelApplyModal from "./modules/ModelApplyModal.vue";
import FilePreviewModal from "./modules/FilePreviewModal.vue";
import ModelApplyTestModal from "./modules/ModelApplyTestModal.vue";
import BindIndicatorModal from "./modules/BindIndicatorModal.vue";
import { downloadFile } from '@/utils/util';

import { isEqual } from "lodash";
import { nanoid } from '~/packages/utils/src';

// 表单字段配置接口
interface FormFieldConfig {
  formName?: string;
  type: 'input' | 'textarea' | 'select' | 'dynamicInput' | 'text' | 'radioGroup' | 'cascader' | 'upload' | 'datetime' | 'customUpload' | 'checkbox' | 'table' | 'dynamicModalClass';
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
  modelList?: { value: string | number; label: string, [key: string]: any }[];
  checkboxList?: { key: string; value: number; formName: string, [key: string]: any }[];
  accept?: string;
  buttonText?: string;
  [prop: string]: any;
}

// 表单分组配置接口
interface FormGroupConfig {
  name: string;
  list: FormFieldConfig[];
  isShow: boolean;
}

const checkedRowKeys = ref<string[]>([])

// 生成表单配置数据
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
      // 测试需求简述
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
      // {
      //   formName: "IOU值",
      //   type: "input",
      //   value: "",
      //   placeholder: "请输入IOU值",
      //   width: "30%",
      //   serverKey: "iouValue",
      //   isShow: true
      // },
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
        serverKey: "modelInterfaceDesc", // modelInterfaceDesc || modelCase
        isShow: true,
        fileList: [],
        accept: ".jpg,.jpeg,.png,.gif,.bmp,.JPG,.JPEG,.PBG,.GIF,.BMP",
        // tooltip 提示上传文件类型
        tooltip: "请上传jpg、jpeg、png、gif、bmp格式的文件",
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
        accept: '.xlsx,.xls',
        // tooltip 提示上传文件类型
        tooltip: "请上传xlsx、xls格式的文件",
      },
      {
        formName: "训练样本/测试用例",
        type: "customUpload",
        value: "",
        placeholder: "附表记录",
        width: "50%",
        serverKey: "trainSample",
        isShow: true,
        fileList: [],
        accept: '.zip,.rar',
        // tooltip 提示上传文件类型
        tooltip: "请上传zip格式的文件",
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
        accept: '.xlsx,.xls',
        // tooltip 提示上传文件类型
        tooltip: "请上传xlsx、xls格式的文件",
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
  // {
  //   name: "测试指标",
  //   list: [
  //     // {
  //     //   formName: "国网企标",
  //     //   type: "checkbox",
  //     //   value: null,
  //     //   checkboxList: [
  //     //     { formName: '召回率/发现率/检出率', value: "1", key: "recall" },
  //     //     { formName: '误检比', value: "2", key: "falseAlarmRate" },
  //     //     { formName: '误报率/误检率', value: "3", key: "falseAlarmRate" },
  //     //     { formName: '平均精度AP', value: "4", key: "ap" },
  //     //     { formName: 'F1-分数', value: "5", key: "f1" },
  //     //     { formName: '识别时间', value: "6", key: "time" },
  //     //     { formName: 'IOU平均值', value: "6", key: "iou" }
  //     //   ],
  //     //   width: "80%",
  //     //   serverKey: "testIndicGrid",
  //     //   isShow: true
  //     // },
  //     // {
  //     //   formName: "通用指标",
  //     //   type: "checkbox",
  //     //   value: null,
  //     //   checkboxList: [
  //     //     { formName: '平均精度 (mPrecision)', value: "1", key: "mPrecision" },
  //     //     { formName: '平均召回率 (mRecall)', value: "2", key: "mRecall" },
  //     //     { formName: '均值平均精度 (mAP@0.5)', value: "3", key: "mAP@0.5" },
  //     //     { formName: '漏检率 (MissRate)', value: "4", key: "MissRate" },
  //     //     { formName: '虚警率 (FalseAlarmRate)', value: "5", key: "FalseAlarmRate" },
  //     //     { formName: '平均正确率 (mAccuracy)', value: "6", key: "mAccuracy" }
  //     //   ],
  //     //   width: "80%",
  //     //   serverKey: "testIndic",
  //     //   isShow: true
  //     // },
  //     // ------------------------table---------------------
  //     // 国网企标
  //     {
  //       formName: "国网企标",
  //       type: "table",
  //       value: [
  //         { label: '召回率/发现率/检出率', prop: 'recall', value: "1", key: 0 },
  //         { label: '误检比', prop: 'falseAlarmRate', value: "1", key: 1 },
  //         { label: '误报率/误检率', prop: 'falseAlarmRate', value: "1", key: 2 },
  //         { label: '平均精度AP', prop: 'ap', value: "1", key: 3 },
  //         { label: 'F1-分数', prop: 'f1', value: "1", key: 4 },
  //         { label: '识别时间', prop: 'time', value: "1", key: 5 },
  //         { label: 'IOU平均值', prop: 'iou', value: "1", key: 6 }
  //       ],
  //       columns: [
  //         {
  //           type: 'selection'
  //         },
  //         {
  //           title: '指标名称',
  //           key: 'label',
  //           width: '30%'
  //         },
  //         {
  //           title: '指标标识',
  //           key: 'prop',
  //           width: '30%'
  //         },
  //         {
  //           title: '指标值',
  //           key: 'value',
  //           render: (row, index) => {
  //             return h(NInput, {
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
  //       width: "60%",
  //       serverKey: "testIndicGridMap",
  //       isShow: true
  //     },
  //     {
  //       formName: "通用指标",
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
  //           width: '30%'
  //         },
  //         {
  //           title: '指标标识',
  //           key: 'prop',
  //           width: '30%'
  //         },
  //         {
  //           title: '指标值',
  //           key: 'value',
  //           render: (row, index) => {
  //             return h(NInput, {
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
  //       width: "60%",
  //       serverKey: "testIndicMap",
  //       isShow: true
  //     },
  //   ],
  //   isShow: true
  // },
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

const { formRef, validate } = useNaiveForm();
const rules = computed<Record<string, any[]>>(() => {
  const { formRules } = useFormRules();
  return {};
});

const router = useRouter();
const route = useRoute();

// file upload
const { bool: drawerVisible, setTrue: openDrawer } = useBoolean();
const { bool: indicatorDrawerVisible, setTrue: openIndicatorDrawer } = useBoolean();
const isUpSuccess = ref<boolean>(false);
const model = ref<any | null>({
  fileList: [],
  imgList: [],
  uploadList: [],
  importMode: "0-1"
});

const editId = ref<string | null>(null);
const editData = ref<any | null>(null);

const uploadSubmit = (data: any) => {
  model.value.imgList = data.fileList;
}

const generateSubmitParams = (configList) => {
  return configList.value.flatMap(group =>
    group.list.filter(item => item.isShow).map(item => {
      const { serverKey, value } = item;
      // 处理特殊类型字段（如多选框/单选框）
      if (item.type === 'checkbox') {
        const value = item.value instanceof Array ? item.value.filter(val => val) : [];
        // 多选框：返回选中项的 key 数组（假设 value 是选中项的 value 数组）
        return { [serverKey]: value?.map(checked => item.checkboxList[checked].value) || [] };
      } else if (item.type === 'radioGroup') {
        // 单选框：返回选中项的 value（假设 modelList 中 value 对应后端需要的参数值）
        return { [serverKey]: item.value };
      } else if (item.type === 'upload' || item.type === 'customUpload') {
        // 上传组件：假设需要文件对象（根据实际情况调整，可能需 .raw 或其他属性）
        return { [serverKey]: item?.fileList || undefined }; // 示例取第一个文件的原始对象
      } else if (item.type === 'table') {
        if (serverKey === 'testIndicMap') {
          return {
            [serverKey]: JSON.stringify(item.value),
            testIndic: item.checkedRowKeys,
          };
        }
        if (serverKey === 'testIndicGridMap') {
          return {
            [serverKey]: JSON.stringify(item.value),
            testIndicGrid: item.checkedRowKeys,
          };
        }
      } else if (item.type === 'dynamicModalClass') {
        return {
          [serverKey]: JSON.stringify(item.value),
        };
      } else {
        // 普通输入框/文本域：直接返回 value
        return { [serverKey]: value };
      }
    })
  ).reduce((acc, curr) => ({ ...acc, ...curr }), {});
};

const handleOperate = async (type: 'submit' | 'back') => {
  if (type === 'submit') {
    const params = generateSubmitParams(configList);
    // 判断并处理 modelInterfaceDesc
    if (Array.isArray(params.modelInterfaceDesc) && params.modelInterfaceDesc.length > 0) {
      params.modelInterfaceDesc = params.modelInterfaceDesc[0].url;
    }
    // 判断并处理 modelCase
    if (Array.isArray(params.modelCase) && params.modelCase.length > 0) {
      params.modelCase = params.modelCase[0].url;
    }
    // 判断并处理 trainSample
    if (Array.isArray(params.trainSample) && params.trainSample.length > 0) {
      params.trainSample = params.trainSample[0].url;
    }
    // 判断并处理 testCase
    if (Array.isArray(params.testCase) && params.testCase.length > 0) {
      params.testCase = params.testCase[0].url;
    }
    // 判断并处理 modelAlgorithmCode
    if (Array.isArray(params.modelAlgorithmCode) && params.modelAlgorithmCode.length > 0) {
      params.modelAlgorithmCode = params.modelAlgorithmCode[0].url;
    }
    // modelTrainCode
    if (Array.isArray(params.modelTrainCode) && params.modelTrainCode.length > 0) {
      params.modelTrainCode = params.modelTrainCode[0].url;
    }

    params.testIndic = params.testIndic?.join(',');
    params.testIndicGrid = params.testIndicGrid?.join(',');
    params.assessChart = params.assessChart?.join(',');
    // 判断并处理model.value.imgList
    if (Array.isArray(model.value.imgList) && model.value.imgList.length > 0) {
      params.fileId = model.value.imgList.map(val => val.id).join(',');
    }
    if (params.modelCase instanceof Array && params.modelCase.length == 0) {
      delete params.modelCase;
    }
    if (params.modelInterfaceDesc instanceof Array && params.modelInterfaceDesc.length == 0) {
      delete params.modelInterfaceDesc;
    }
    if (params.trainSample instanceof Array && params.trainSample.length == 0) {
      delete params.trainSample;
    }
    if (params.testCase instanceof Array && params.testCase.length == 0) {
      delete params.testCase;
    }
    if (params.modelAlgorithmCode instanceof Array && params.modelAlgorithmCode.length == 0) {
      delete params.modelAlgorithmCode;
    }
    if (params.modelTrainCode instanceof Array && params.modelTrainCode.length == 0) {
      delete params.modelTrainCode;
    }

    // if(!params.modelInterfaceDesc) {
    //   window.$message?.error?.("请上传模型API接口说明");
    //   return;
    // }

    // 判断路由参数是否包含 id
    if (route.query.sign === 'edit') {
      params.id = route.query.id || editId.value;
      const res = await editModelEvaluation(params);
      if (res.data >= 1) {
        window.$message?.success?.("模型申请编辑成功！");
        // router.back();

        const params = generateSubmitParams(configList);
        defaultParams.value = params;
      }
    }
    if (route.query.sign === 'create') {
      const res = await addModelEvaluation(params);
      if (res.data) {
        window.$message?.success?.("模型申请成功！");
        editId.value = res.data?.id;
        editData.value = res.data;
        applicationButtons.value = getApplicationButtons1();

        const params = generateSubmitParams(configList);
        defaultParams.value = params;
      }
    }
  }
  if (type === 'back') {
    // 预处理，等待prevDispose执行完成
    const shouldContinue = await prevDispose();
    if (!shouldContinue) return; // 用户取消操作，直接返回
    router.back();
  }
};

// 处理表单字段值的变化
const handleFieldChange = (field: FormFieldConfig, value: any) => {
  // field.value = value;
  if (field.serverKey === 'modelWay') {
    // 隐藏测试指标
    // configList.value[3].isShow = value === '2';
  }
};


// 处理文件上传变化事件
const handleUploadFinish = (file: any, rowData: any) => {
  const res = JSON.parse(file.event.currentTarget.response);
  if (res.code === 200) {
    file.url = res.data;
    file.status = 'finished';
    // rowData.fileList.push(file);

    // 判断rowData.fileList长度
    if (rowData.fileList.length === 0) {
      rowData.fileList.push(file);
    } else {
      // const config = configList.value
      //   .flatMap(group => group.list)
      //   .find(item => item.serverKey === rowData.serverKey);

      // if (config && config.fileList?.length) {
      //   const index = config.fileList.length - 1;
      //   config.fileList.splice(index, 1, { ...config.fileList[index], url: res.data });
      //   console.log(configList.value);
      // }
      updateFileListByServerKey(rowData.serverKey, [
        {
          ...file,
          name: file.file.name,
        }
      ]);
      console.log(configList.value);
    }
  }
};

// 处理文件移除事件
const handleRemove = async ({ file }: any, rowData: any) => {
  // 定义允许的serverKey集合，用于快速校验
  const allowedServerKeys = [
    'modelInterfaceDesc',
    'modelCase',
    'trainSample',
    'testCase',
    'modelAlgorithmCode',
    'modelTrainCode'
  ];

  // 如果不是目标serverKey，直接返回
  if (!allowedServerKeys.includes(rowData.serverKey)) {
    return;
  }

  // 处理文件已完成的情况（需要调用接口）
  if (file.status === 'finished') {
    try {
      const res = await deleteFile({
        serverKey: rowData.serverKey,
        modelId: editId.value || route.query.id
      });

      if (res.data >= 1) {
        window.$message?.success?.("文件删除成功！");
        // 过滤掉已删除的文件
        rowData.fileList = rowData.fileList.filter(item => item.id !== file.id);
      }
    } catch (error) {
      console.error('文件删除失败:', error);
      window.$message?.error?.("文件删除失败，请重试！");
    }
  } else {
    // 未完成的文件直接从列表中移除
    rowData.fileList = rowData.fileList.filter(item => item.id !== file.id);
  }
};

// 根据表单字段类型返回对应的组件
const getFormComponent = (type: FormFieldConfig['type'], value: unknown, rowData: any) => {
  const commonProps = {
    'v-model:value': value
  };

  switch (type) {
    case 'input':
      return h(NInput);

    case 'textarea':
      return h(NInput, { type: 'textarea' });

    case 'select':
      return h(NSelect);

    case 'text':
      return h('span', value);

    case 'radioGroup':

      return h(NRadioGroup, {
        ...commonProps,
        name: 'anoType',
        size: 'large',
        onUpdateValue: (val: any) => handleFieldChange(rowData, val)
      }, rowData.modelList.map((item: any) =>
        h(NRadio, {
          value: item.value,
          label: item.label,
          // onChange: (e: any) => handleFieldChange(rowData, e)
        })
      ));

    case 'cascader':
      return h(NCascader);

    case 'datetime':
      return h(NDatePicker, {
        ...commonProps,
        type: 'datetime',
        clearable: true,
        class: '!w-full'
      });

    case 'upload':
      const action = `${import.meta.env.VITE_SERVICE_BASE_URL}/upload`;
      const type = rowData?.serverKey === 'modelInterfaceDesc' ? 1 : 2;
      return createUploadComponent(rowData.fileList, {
        action,
        headers: {
          Authorization: `Bearer ${getToken()}`,
          type
        },
        onFinish: ($event) => handleUploadFinish($event, rowData),
        onRemove: ($event) => handleRemove($event, rowData),
        buttonText: '上传文件',
        accept: rowData.accept
      });

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
        class: "w-full",
        onCreate: () => ({ formName: "", key: "", value: undefined })
      }, {
        default: ({ value: item }) => h('div', { class: "w-full flex-center gap-8px" }, [
          h(NSelect, {
            'v-model:value': item.value,
            options: rowData.options
          }),
          h(NInput, {
            modelValue: item.value,
            'onUpdate:modelValue': (val: string | number) => item.value = val,
            placeholder: "请输入指标值"
          })
        ]),
        'create-button-default': () => '添加指标'
      });

    case 'dynamicModalClass':
      return h(NDynamicInput, {
        ...commonProps,
        class: "w-full",
        onCreate: () => ({
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
      }, {
        default: ({ value: item }) => h('div', { class: "w-full flex-center gap-8px" }, [
          h(NInput, {
            class: 'w-1/2',
            value: item.className,
            'on-update:value': (name: string | number) => item.className = name,
            placeholder: "请输入模型识别类别"
          }),
          h(NButton, {
            class: 'w-160px',
            type: "primary",
            onClick: () => bindIndicator(item)
          }, '关联指标'),
        ]),
        'create-button-default': () => '添加模型识别类别'
      })

    case 'table':
      return () => h(NDataTable, {
        columns: rowData.columns,
        data: rowData.value,
        bordered: true,
        pagination: false,
        'checked-row-keys': rowData.checkedRowKeys,
        'default-checked-row-keys': rowData.defaultCheckedRowKeys,
        'on-update:checked-row-keys': (keys) => {
          rowData.checkedRowKeys = keys
        }
      });

    case 'checkbox':
      return h(NCheckboxGroup, commonProps, () =>
        rowData.checkboxList.map((item: any) =>
          h(NCheckbox, { value: item.value, label: item.formName })
        )
      );

    default:
      return null;
  }
};

// 封装通用上传组件生成逻辑
const createUploadComponent = (fileList, options) => {
  return h(NUpload, {
    'v-model:file-list': fileList,
    max: '1',
    ...options
  }, [h(NButton, {}, options.buttonText)]);
};

function extractFileNames(filePaths) {
  return filePaths.map(path => {
    // 处理不同操作系统的路径分隔符
    const parts = path.split(/[\\/]/);
    return parts[parts.length - 1];
  });
}

function combinePathAndFilenames(paths, filenames) {
  // 处理两个数组长度不一致的情况，以较短的数组长度为准
  const minLength = Math.min(paths.length, filenames.length);

  return Array.from({ length: minLength }, (_, i) => {
    // 确保路径以路径分隔符结尾
    let formattedPath = paths[i];
    if (!formattedPath.endsWith('/') && !formattedPath.endsWith('\\')) {
      formattedPath += '/';
    }

    // 构建完整路径
    const fullPath = formattedPath + filenames[i];

    return {
      path: paths[i],
      filename: filenames[i],
      fullPath: fullPath
    };
  });
}

function updateConfigListByServerKey(data: Record<string, any>, configList: FormGroupConfig[]) {
  // 遍历 configList 中的每个分组
  configList.forEach(group => {
    // 遍历分组中的每个表单项
    group.list.forEach(item => {
      // 如果存在 serverKey 且数据对象中包含该键
      if (item.serverKey && data.hasOwnProperty(item.serverKey)) {
        // 处理不同表单类型的特殊值
        switch (item.type) {
          // 多选框处理（testIndic 字段）
          case 'checkbox':
            if (item.serverKey === 'testIndic') {
              const selectedValues = (data[item.serverKey] as string).split(',').map(Number);
              item.value = item.checkboxList.map(option =>
                selectedValues.includes(option.value) ? option.value : null
              );
            }
            if (item.serverKey === 'assessChart') {
              const selectedValues = (data[item.serverKey] as string).split(',');
              item.value = item.checkboxList.map(option =>
                selectedValues.includes(option.value) ? option.value : null
              );
            }
            break;
          // 单选组处理（radioGroup）
          case 'radioGroup':
            item.value = String(data[item.serverKey]); // 确保值为字符串类型
            break;
          // 上传组件处理（保留 fileList，仅更新 value）
          case 'upload':
            if (data[item.serverKey]) {
              const pathList = data[item.serverKey].split(",");
              const nameList = extractFileNames(pathList);
              const fileObjects = combinePathAndFilenames(pathList, nameList);
              item.fileList = fileObjects.map(item => ({
                id: item.fullPath,
                name: item.filename,
                status: 'finished'
              })) || [];
            }
            break;
          case 'customUpload':
            if (data['groupNameAndVersion']) {
              const fileList = [
                {
                  id: data['groupNameAndVersion'],
                  name: data['groupNameAndVersion'],
                  status: 'finished'
                }
              ];
              model.value.imgList = fileList;
            }
            break;
          case 'table':
            if (item.serverKey === 'testIndicMap') {
              item.checkedRowKeys = item.defaultCheckedRowKeys = data['testIndic'].split(',').map(Number);
              item.value = JSON.parse(data['testIndicMap']);
            }
            if (item.serverKey === 'testIndicGridMap') {
              item.checkedRowKeys = item.defaultCheckedRowKeys = data['testIndicGrid'].split(',').map(Number);
              item.value = JSON.parse(data['testIndicGridMap']);
            }
            break;
          case 'dynamicModalClass':
            item.value = JSON.parse(data[item.serverKey]) ?? [];
            console.log('item.value: ', item.value);
            break;
          // 其他类型直接赋值
          default:
            item.value = data[item.serverKey] ?? '';
        }
      }

      if (item.serverKey === 'modelWay') {
        // const isShow = data[item.serverKey] == 2;
        // configList[3].isShow = isShow;
        // if (isShow) {
        //   configList[3].list[0].checkedRowKeys = data["testIndic"].split(',');
        // }
      }
    });
  });
}

function updateFileListByServerKey(serverKey: string, newFileList: any[]): boolean {
  for (const group of configList.value) {
    for (let i = 0; i < group.list.length; i++) {
      const item = group.list[i];
      if (item.serverKey === serverKey && 'fileList' in item) {
        // 创建新对象并替换原有item
        group.list[i] = {
          ...item,
          fileList: newFileList
        };
        return true;
      }
    }
  }

  // 未找到匹配的项
  console.warn(`未找到serverKey为"${serverKey}"且包含fileList属性的表单项`);
  return false;
}

onMounted(() => {
  const { query } = route;
  if (query.id) {
    fetchModelDetails({
      id: query.id
    }).then(async res => {
      const { data } = res;
      if (data) {
        updateConfigListByServerKey(data, configList.value);
        // const processedParams = processSubmitParams();
        // console.log('processedParams: ', processedParams);

        const params = generateSubmitParams(configList);
        defaultParams.value = params;
      }
    })
  } else {
    const params = generateSubmitParams(configList);
    defaultParams.value = params;
  }
})

// 申请状态枚举（数值类型）
enum ApplicationStatus {
  Draft = 1,          // 草稿
  UnderReview = 2,    // 审批中
  Approved = 3,       // 审批通过
  Rejected = 4,       // 审批打回
  Completed = 5,       // 已完成
  Testing = 6,
}

// 申请操作类型
enum ApplicationAction {
  UploadReport = 'uploadReport',  // 上传报告
  ViewDetails = 'viewDetails',    // 查看详情
  Edit = 'edit',                  // 编辑
  Export = 'export',               // 导出文件
  Commit = 'commit',               // 提交审核
  GenerateFile = 'generateFile',
  Test = 'test',
  Delete = 'delete',
}

// 申请操作按钮配置类型
type ApplicationButtonConfig = {
  text: string;
  type?: 'primary' | 'success' | 'warning' | 'error';
  action: ApplicationAction;
  confirmText?: string;
};

const applicationButtons = ref<any>([]);
const isExport = ref<Boolean>(false);
const { bool: visible, setTrue: openModal } = useBoolean();
const { bool: testVisible, setTrue: openTestModal } = useBoolean();
const { bool: uploadVisible, setTrue: openUploadModal } = useBoolean();
const { bool: fileVisible, setTrue: openFileModal } = useBoolean();
const rowData = ref<any>({});
const previewPath = ref<string | null>(null);
const previewFileSuffix = ref<string | null>(null);

const getApplicationButtons = (): ApplicationButtonConfig[] => {
  // 通过Vue Router获取query参数
  const route = useRoute();
  const { applyForStatus, applyForType } = route.query as any;

  // 基础按钮配置
  const baseButtons = [
    // { text: '查看详情', action: ApplicationAction.ViewDetails, permission: 'thirdparty:mul:details' },
  ];

  // 模型调试按钮
  const testButton = {
    text: '模型调试',
    type: 'primary',
    action: ApplicationAction.Test,
    permission: "thirdparty:mul:modelDebugging"
  };

  // 提交审核按钮
  const commitButton = {
    text: '提交审核',
    type: 'primary',
    action: ApplicationAction.Commit,
    confirmText: '确定提交审核吗？',
    permission: "thirdparty:mul:submitForReview"
  };

  // 编辑按钮
  const editButton = {
    text: '编辑',
    type: 'primary',
    action: ApplicationAction.Edit,
    permission: "thirdparty:mul:edit"
  };

  // 生成/上传附件按钮 (仅applyForType=1时显示)
  const fileButtons = applyForType === '1' ? [
    { text: '生成数据附件', type: 'success', action: ApplicationAction.GenerateFile, isExport: isExport.value, permission: 'thirdparty:mul:generateAttachments' },
    { text: '上传数据附件', type: 'success', action: ApplicationAction.UploadReport, permission: 'thirdparty:mul:uploadAttachments' },
  ] : [];

  // 根据状态返回按钮配置
  switch (+applyForStatus) {
    case ApplicationStatus.Draft:
      return [testButton, commitButton, ...baseButtons, ...fileButtons];
    case ApplicationStatus.UnderReview:
      return [testButton, ...baseButtons];

    case ApplicationStatus.Approved:
      return [...baseButtons];

    case ApplicationStatus.Rejected:
      return [testButton, commitButton, ...baseButtons, ...fileButtons];

    case ApplicationStatus.Completed:
      return [...baseButtons];

    default:
      return [];
  }
};

const getApplicationButtons1 = (): ApplicationButtonConfig[] => {
  // 通过Vue Router获取query参数
  const route = useRoute();
  const { applyForStatus, applyForType } = editData.value as any;

  // 基础按钮配置
  const baseButtons = [
    // { text: '查看详情', action: ApplicationAction.ViewDetails, permission: 'thirdparty:mul:details' },
  ];

  // 模型调试按钮
  const testButton = {
    text: '模型调试',
    type: 'primary',
    action: ApplicationAction.Test,
    permission: "thirdparty:mul:modelDebugging"
  };

  // 提交审核按钮
  const commitButton = {
    text: '提交审核',
    type: 'primary',
    action: ApplicationAction.Commit,
    confirmText: '确定提交审核吗？',
    permission: "thirdparty:mul:submitForReview"
  };

  // 编辑按钮
  const editButton = {
    text: '编辑',
    type: 'primary',
    action: ApplicationAction.Edit,
    permission: "thirdparty:mul:edit"
  };

  // 生成/上传附件按钮 (仅applyForType=1时显示)
  const fileButtons = `${applyForType}` === '1' ? [
    { text: '生成数据附件', type: 'success', action: ApplicationAction.GenerateFile, isExport: isExport.value, permission: 'thirdparty:mul:generateAttachments' },
    { text: '上传数据附件', type: 'success', action: ApplicationAction.UploadReport, permission: 'thirdparty:mul:uploadAttachments' },
  ] : [];

  // 根据状态返回按钮配置
  switch (+applyForStatus) {
    case ApplicationStatus.Draft:
      return [testButton, commitButton, ...baseButtons, ...fileButtons];
    case ApplicationStatus.UnderReview:
      return [testButton, ...baseButtons];

    case ApplicationStatus.Approved:
      return [...baseButtons];

    case ApplicationStatus.Rejected:
      return [testButton, commitButton, ...baseButtons, ...fileButtons];

    case ApplicationStatus.Completed:
      return [...baseButtons];

    default:
      return [];
  }
};


function downloadPost(config: any) {
  return new Promise((resolve, reject) => {
    axios({
      url: config.url, // 请求地址
      method: 'post',
      data: config.data, // 参数
      responseType: 'blob' // 表明返回服务器返回的数据类型
    })
      .then(res => {
        resolve(res);
      })
      .catch(err => {
        reject(err);
      });
  });
}

const dialog = useDialog()
const prevDispose = () => {
  return new Promise((resolve) => {
    const curParams = generateSubmitParams(configList);
    if (isEqual(defaultParams.value, curParams)) {
      // 参数未修改，直接resolve继续执行
      resolve(true);
      return;
    }

    // 参数已修改，显示确认对话框
    dialog.warning({
      title: '警告',
      content: '当前表单内容已修改，是否保存？',
      positiveText: '保存',
      negativeText: '取消',
      draggable: true,
      onPositiveClick: async () => {
        await handleOperate('submit');
        resolve(true); // 保存后继续执行后续逻辑
      },
      onNegativeClick: () => {
        resolve(true); // 取消则不执行后续逻辑
      }
    });
  });
};

const handleApplicationAction = async (action: ApplicationAction, row: any) => {
  // 预处理，等待prevDispose执行完成
  const shouldContinue = await prevDispose();
  if (!shouldContinue) return; // 用户取消操作，直接返回

  // rowData.value = row;
  if (row && row.id) {
    rowData.value = row;
  } else {
    rowData.value = editData.value;
  }
  switch (action) {
    case ApplicationAction.UploadReport:
      // 上传报告逻辑
      openUploadModal();
      break;
    case ApplicationAction.GenerateFile:
      // 生成数据报告
      isExport.value = true;
      const baseUrl = import.meta.env.VITE_SERVICE_BASE_URL;
      const config = {
        url: `${baseUrl}/model/evaluation/generatePad?id=${rowData.value?.id}`,
        data: {
          id: rowData.value?.id,
        }
      };
      const fileName = `数据报告${rowData.value?.id}.docx`;
      const response: any = await downloadPost(config);
      if (response.data) {
        isExport.value = false;
        await downloadByData(response.data, fileName);
      }
      break;
    case ApplicationAction.ViewDetails:
      // 查看详情逻辑
      rowData.value = row;
      openModal();
      break;
    case ApplicationAction.Edit:
      // 编辑逻辑
      router.push({
        name: "thirdparty_modeloperate",
        query: {
          // id: row.id,
          id: rowData.value?.id,
          sign: "edit",
          applyForStatus: rowData.value?.applyForStatus,
          applyForType: rowData.value?.applyForType,
        }
      })
      break;
    case ApplicationAction.Export:
      // 导出文件逻辑
      console.log('导出文件操作', row);
      break;
    case ApplicationAction.Commit:
      // 提交审核逻辑
      console.log('提交审核操作', row);
      const res = await submitApproveModelEvaluation({
        id: rowData.value?.id,
      });
      if (res.data) {
        window.$message?.success?.("提交成功！");
        router.back();
      }
      break;
    case ApplicationAction.Test:
      // 测试逻辑
      openTestModal();
      break;
    case ApplicationAction.Delete:
      // 删除逻辑
      const res1 = await delModelEvaluation({
        id: rowData.value?.id,
      })
      if (res1.data) {
        window.$message?.success?.("删除成功！");
      }
      break;
  }
};

const handlePreview = (params: any) => {
  const { previewFileSuffix: suffix, previewPath: filePath } = params;
  previewFileSuffix.value = suffix;
  previewPath.value = filePath;
  openFileModal();
};

onMounted(() => {
  applicationButtons.value = getApplicationButtons();
})

const handleDownloadTemplate = async (serverKey: string) => {
  const baseUrl = import.meta.env.VITE_SERVICE_BASE_URL;
  const url = new URL('/api/downloadTemplate', baseUrl).toString();

  const headers = {
    Authorization: `Bearer ${getToken()}`,
    serverKey,
  };

  try {
    await downloadFile({
      url,
      params: {
        serverKey,
      },
      headers
    });
  } catch (error) {
    throw error;
  }
}


const defaultParams = ref<any>({}); // 默认
const operateParams = ref<any>({}); // 编辑后

function processSubmitParams() {
  // 深度克隆参数对象，避免修改原始数据
  const processedParams = generateSubmitParams(configList);

  // 处理数组类型的属性，提取第一个元素的url
  ['modelInterfaceDesc', 'modelCase', 'trainSample', 'testCase', 'modelAlgorithmCode', 'modelTrainCode'].forEach(key => {
    if (Array.isArray(processedParams[key]) && processedParams[key].length > 0) {
      processedParams[key] = processedParams[key][0].url;
    } else if (Array.isArray(processedParams[key]) && processedParams[key].length === 0) {
      delete processedParams[key];
    }
  });

  // 处理 testIndic，将数组转为逗号分隔的字符串
  if (processedParams.testIndic) {
    processedParams.testIndic = processedParams.testIndic.join(',');
  }
  // 处理 assessChart，将数组转为逗号分隔的字符串
  if (processedParams.assessChart) {
    processedParams.assessChart = processedParams.assessChart.join(',');
  }

  // 处理 model.value.imgList，提取id并转为逗号分隔的字符串
  if (model && model.value && Array.isArray(model.value.imgList) && model.value.imgList.length > 0) {
    processedParams.fileId = model.value.imgList.map(val => val.id).join(',');
  }

  return processedParams;
}


// newCode
const classModel = ref<any>({}); // 模型识别类别

function bindIndicator(item: any) {
  classModel.value = item;
  openIndicatorDrawer();
}

function handleAfterLeave(params: any) {
  const { classId, modelCommonData, modelGridData } = params;
  const modelTypeSectionIndex = configList.value.findIndex(
    (section: any) => section.name === "模型识别类别"
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
</script>

<style scoped></style>
