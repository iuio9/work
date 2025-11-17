<script setup lang="ts">
import { NButton, NCascader, NInput, NPopover, NSelect } from 'naive-ui';
import { useAppStore } from '@/store/modules/app';
import aType from '@/assets/imgs/aType.png';
import { useTable, useTableOperate } from '@/hooks/common/table';
import { getTaskPage } from '@/service/api/dataManage';
import RadarChart from './modules/radar-chart.vue';
import ConfusionMatrixChart from './modules/ConfusionMatrixChart.vue';
import PRChart from './modules/PRChart.vue';
import CustomImageViewer from './modules/CustomImageViewer.vue';

const appStore = useAppStore();
const gap = computed(() => (appStore.isMobile ? 0 : 16));
const tabState = ref<string>('1');

const taskOptions = [
  {
    name: '基本信息',
    list: [
      {
        formName: '',
        type: 'grid',
        value: '',
        width: '100%',
        path: '',
        serverKey: '',
        isShow: true,
        dataList: [
          { name: '任务名称', span: '8 s:8 m:8', value: '' },
          { name: '任务描述', span: '8 s:8 m:8', value: '' },
          { name: '任务版本', span: '8 s:8 m:8', value: '' },
          { name: '版本描述', span: '8 s:8 m:8', value: '' },
        ]
      }
    ]
  },
  // 评估对象
  {
    name: '评估对象',
    list: [
      // {
      //   formName: "",
      //   type: "grid",
      //   value: "",
      //   width: "100%",
      //   path: "",
      //   serverKey: "",
      //   isShow: true,
      //   dataList: [
      //     { name: "评估数据集", span: '8 s:8 m:8', value: ''},
      //     { name: "待评估模型", span: '8 s:8 m:8', value: ''},
      //     { name: "参数配置", span: '8 s:8 m:8', value: ''},
      //     { name: "创建时间", span: '8 s:8 m:8', value: ''},
      //     { name: "任务ID", span: '8 s:8 m:8', value: ''},
      //   ]
      // },
      {
        formName: '',
        type: 'gridTable',
        value: '',
        width: '100%',
        path: '',
        serverKey: '',
        isShow: true,
        columns: [
          {
            title: '版本',
            key: 'version'
          },
          {
            title: '数据集ID',
            key: 'datasetId'
          },
          {
            title: '数据集类型',
            key: 'datasetType'
          },
          {
            title: '数据量',
            key: 'dataAmount'
          },
          {
            title: '最近导入状态',
            key: 'latestImportStatus'
          },
          {
            title: '标注类型',
            key: 'annotationType'
          },
          {
            title: '标注状态',
            key: 'annotationStatus'
          },
          {
            title: '创建时间',
            key: 'createTime'
          }
        ],
        dataList: [
          {
            version: 'v1.0',
            datasetId: 'DS001',
            datasetType: 'Image',
            dataAmount: 1000,
            latestImportStatus: 'Success',
            annotationType: 'Bounding Box',
            annotationStatus: 'Completed',
            createTime: '2024-08-02 17:32:05'
          }
        ]
      },
      {
        formName: '',
        type: 'table',
        data: [
          { modelName: '模型名称001', status: '已完成', setName: '推理数据集001', modleDesc: '描述信息001' },
          { modelName: '模型名称002', status: '已完成', setName: '推理数据集002', modleDesc: '描述信息002' }
        ],
        placeholder: '',
        width: '98%',
        serverKey: 'trainPrams',
        query: [],
        columns: [
          { title: '模型名称', key: 'modelName' },
          {
            title: '状态',
            key: 'status',
            render: (row: any) => {
              return [
                h(
                  NButton,
                  {
                    type: 'primary',
                    ghost: true,
                    size: 'small',
                    style: { marginRight: '10px' }
                  },
                  `${row.status}`
                )
              ];
            }
          },
          { title: '推理数据集名称', key: 'setName', ellipsis: { tooltip: true } },
          { title: '模型描述', key: 'modelDesc', ellipsis: { tooltip: true } },
          { title: '操作', key: 'operate' }
        ],
        isShow: true
      }
    ]
  },
  // 评估方法
  {
    name: '评估指标',
    list: [
      {
        formName: '',
        type: 'grid',
        value: '',
        width: '100%',
        path: '',
        serverKey: '',
        isShow: true,
        dataList: [
          { name: '评估指标', span: '8 s:8 m:8', value: 'mPrecision, mAP' },
          { name: '评估图表', span: '8 s:8 m:8', value: 'P-R曲线, 混淆矩阵图' }
        ]
      }
    ]
  }
];

// 自动规则打分指标数据
const autoRuleData = [
  {
    indicatorName: '平均精度(mPrecision)',
    value: 0.94,
    score: '优秀（可根据具体规则设定打分标准，比如大于0.9为优秀）'
  },
  {
    indicatorName: '平均召回率(mRecall)',
    value: 0.914,
    score: '良好（假设大于0.85为良好）'
  },
  {
    indicatorName: '均值平均精度(mAP@0.5)',
    value: 0.943,
    score: '优秀'
  },
  {
    indicatorName: '漏检率(MissRate)',
    value: 0.086,
    score: '较低（假设小于0.1为较低）'
  },
  {
    indicatorName: '虚警率(FalseAlarmRate)',
    value: 0.06,
    score: '较低'
  },
  {
    indicatorName: '平均正确率(mAccuracy)',
    value: 0.869,
    score: '良好'
  }
];

// 自动规则打分指标列配置
const autoRuleColumns = [
  {
    title: '指标名称',
    key: 'indicatorName'
  },
  {
    title: '指标数值',
    key: 'value'
  },
  {
    title: '打分',
    key: 'score'
  }
];

// 定义评估日志的数组对象
const evaluationLogs = [
  {
    timestamp: '2023-10-05T14:30:00Z',
    status: 'success',
    message: '评估任务已成功启动。',
    details: {
      taskName: '评估任务1',
      taskId: 'task-123'
    }
  },
  {
    timestamp: '2023-10-05T14:35:00Z',
    status: 'warning',
    message: '评估过程中检测到潜在问题。',
    details: {
      issueType: 'data inconsistency',
      affectedRecords: 10
    }
  },
  {
    timestamp: '2023-10-05T14:40:00Z',
    status: 'error',
    message: '评估任务因数据异常而失败。',
    details: {
      errorType: 'invalid data',
      errorMessage: '数据格式不正确'
    }
  },
  {
    timestamp: '2023-10-05T14:45:00Z',
    status: 'info',
    message: '评估任务重新启动。',
    details: {
      restartReason: '手动重启'
    }
  },
  {
    timestamp: '2023-10-05T14:50:00Z',
    status: 'success',
    message: '评估任务已完成。',
    details: {
      finalStatus: 'completed',
      completionTime: '2023-10-05T14:50:00Z'
    }
  }
];
const ruleShowModal = ref<Boolean>(false);
const exportData = ref<any>({});
const ruleList = ref<any>([
  { label: '准确率', value: 1 },
  { label: 'F1分数', value: 2 },
  { label: 'ROUGE-1', value: 3 },
  { label: 'ROUGE-2', value: 4 },
  { label: 'ROUGE-L', value: 5 },
  { label: 'BLEU-4', value: 6 },
  { label: '裁判员模型打分中位数', value: 7 },
  { label: '裁判员模型打分平均值', value: 8 },
  { label: '裁判员模型打分标准差', value: 9 }
]);
const modleList = ref<any>([
  { label: '评估模型1', value: 1 },
  { label: '评估模型2', value: 2 }
]);

const imageData = [
  {
    original: aType,
    effect: aType,
    title: 'Image 1',
    description: 'Description for Image 1'
  },
  {
    original: aType,
    effect: aType,
    title: 'Image 2',
    description: 'Description for Image 2'
  }
];

const isOperateModal = ref<Boolean>(false);
const operateList = ref<any[]>([
  {
    taskId: 346,
    taskStatus: '已完成',
    exportLocation: '本地',
    exportContent: 'Prompt、System(入设信息)',
    createTime: '2024-08-02 17:32:05',
    finishTime: '2024-08-02 17:32:06'
  }
]);
const OperateCols = ref<any>([
  {
    title: '任务ID',
    key: 'taskId'
  },
  {
    title: '任务状态',
    key: 'taskStatus'
  },
  {
    title: '导出位置',
    key: 'exportLocation'
  },
  {
    title: '导出内容',
    key: 'exportContent'
  },
  {
    title: '创建时间',
    key: 'createTime'
  },
  {
    title: '完成时间',
    key: 'finishTime'
  },
  {
    title: '操作',
    key: 'action',
    render: row => {
      return h('n-space', null, [
        h(
          'n-button',
          {
            type: 'primary',
            size: 'small',
            onClick: () => handleDownload(row)
          },
          '下载'
        )
      ]);
    }
  }
]);

const visible = ref<Boolean>(false);
const currentImage = ref<any>({});

const handleTabChange = (e: any) => {
  tabState.value = e;
};

const handleExportConfig = () => {
  ruleShowModal.value = true;
};

const {
  columns,
  data,
  getData,
  loading,
} = useTable({
  sign: 'id',
  apiFn: getTaskPage,
  showTotal: true,
  apiParams: {
    isTrain: 1,
    current: 1,
    size: 10,
    algorithmName: null,
    taskInputName: null,
    modelName: null,
    recordType: 1
  },
  columns: () => [
    {
      title: '原始数据集图片',
      key: 'response',
      ellipsis: {
        tooltip: true
      },
      render: (row: any) => {
        return [
          h("div", {
            class: "flex items-center cursor-pointer",
            onClick: () => handleImgView(row)
          }, [
            h(NPopover, { trigger: "hover", placement: "top" }, {
              default: () => [
                h("img", {
                  class: "object-cover w-300px h-300px",
                  src: aType,
                  alt: "原始数据集图片"
                })
              ],
              trigger: () => [
                h("span", {}, `原始数据集图片`)
              ]
            })
          ])
        ];
      }
    },
    {
      title: '评估图片',
      key: 'completion',
      ellipsis: {
        tooltip: true
      },
      render: (row: any) => {
        return [
          h("div", {
            class: "flex items-center cursor-pointer",
            onClick: () => handleImgView(row)
          }, [
            h(NPopover, { trigger: "hover", placement: "top" }, {
              default: () => [
                h("img", {
                  class: "object-cover w-300px h-300px",
                  src: aType,
                  alt: "评估结果图片"
                })
              ],
              trigger: () => [
                h("span", {}, `评估结果图片`)
              ]
            })
          ])
        ];
      }
    },
    {
      title: '已打标签',
      key: 'tags',
      ellipsis: {
        tooltip: true
      },
      render: (row: any) => {
        return [
          h('div', { class: 'flex items-center gap-12px' }, [
            h(
              NButton,
              {
                type: 'primary',
                ghost: true,
                size: 'small'
              },
              '标签1'
            ),
            h(
              NButton,
              {
                type: 'primary',
                ghost: true,
                size: 'small'
              },
              '标签2'
            )
          ])
        ];
      }
    },
    {
      title: 'BLEU-4',
      key: 'BLEU_4',
      ellipsis: {
        tooltip: true
      }
    },
    {
      title: 'ROUGE-1',
      key: 'ROUGE_1',
      ellipsis: {
        tooltip: true
      }
    },
    {
      title: 'ROUGE-2',
      key: 'ROUGE_2',
      ellipsis: {
        tooltip: true
      }
    },
    {
      title: 'ROUGE-L',
      key: 'ROUGE_L',
      ellipsis: {
        tooltip: true
      }
    }
  ]
});
const {
  checkedRowKeys,
} = useTableOperate(data, getData);
const tableData = ref([
  {
    modelName: 'ERNIE-3.5-8K-0701',
    inferenceSetName: '自动规则 080201_t0bcf6',
    systemInfo: '你是一个人工智能助手。',
    prompt: '生成一个关于人工智能的标题吧。',
    response: '《未来之智能新篇章：人工智能的发展与挑战》',
    completion: '"人工智能：重塑未来生活与工作的革命性力量"',
    BLEU_4: '5.06 %',
    ROUGE_1: '43.48 %',
    ROUGE_2: '-',
    ROUGE_L: '16.67 %'
  },
  {
    modelName: 'ERNIE-3.5-8K-0701',
    inferenceSetName: '自动规则 080201_t0bcf6',
    systemInfo: '你是一个人工智能助手。',
    prompt: '生成一个关于人工智能的标题吧。',
    response: '《未来之智能新篇章：人工智能的发展与挑战》',
    completion: '"人工智能：重塑未来生活与工作的革命性力量"',
    BLEU_4: '5.06 %',
    ROUGE_1: '43.48 %',
    ROUGE_2: '-',
    ROUGE_L: '16.67 %'
  },
  {
    modelName: 'ERNIE-3.5-8K-0701',
    inferenceSetName: '自动规则 080201_t0bcf6',
    systemInfo: '你是一个人工智能助手。',
    prompt: '生成一个关于人工智能的标题吧。',
    response: '《未来之智能新篇章：人工智能的发展与挑战》',
    completion: '"人工智能：重塑未来生活与工作的革命性力量"',
    BLEU_4: '5.06 %',
    ROUGE_1: '43.48 %',
    ROUGE_2: '-',
    ROUGE_L: '16.67 %'
  }
]);

const tagOptions = ref<any>([
  { value: "1", label: "标签1"},
  { value: "2", label: "标签2"},
  { value: "3", label: "标签3"},
  { value: "4", label: "标签4"},
]);
const sortOptions = ref<any>([
  // 正序 倒序
  { value: "ascend", label: "正序" },
  { value: "descend", label: "倒序" },
]);
const indictorOptions = ref<any>(
  [
  { value: 'mPrecision', label: 'mPrecision' },
    { value: 'mAP', label: 'mAP' },
    { value: 'mAccuracy', label: 'mAccuracy' },
    { value: 'mRecall', label: 'mRecall' },
    { value: '漏检率', label: '漏检率' },
    { value: '虚警率', label: '虚警率' }
  ]
);

function handleOperate(type: 'operate' | 'export') {
  if (type === 'operate') {
    isOperateModal.value = true;
  } else if (type === 'export') {
    handleExportConfig();
  }
}

function handleImgView(row: any) {
  currentImage.value = {
    original: aType,
    effect: aType,
    title: 'Image 1',
    description: 'Description for Image 1'
  };
  visible.value = true;
}

function handleImgClose() {
  visible.value = false;
}
</script>

<template>
  <div class="wrap_modelInfo h-full w-full flex-col items-center justify-start !p-0">
    <!--tabs-->
    <div class="item_tabs h-auto w-full bg-[#fff]">
      <NTabs type="line" animated class="wrap_tabs ml-14px w-auto" @update:value="handleTabChange($event)">
        <!-- <NTabPane name="0" tab="任务配置"></NTabPane> -->
        <NTabPane name="1" tab="评估报告"></NTabPane>
        <!-- <NTabPane name="2" tab="评估详情"></NTabPane> -->
        <!-- <NTabPane name="3" tab="任务日志"></NTabPane> -->
      </NTabs>
    </div>
    <!--content-->
    <div v-if="tabState === '0'" class="content_modelInfo box-border h-full w-full flex-grow-1 overflow-y-auto p-16px">
      <div class="h-auto w-full flex flex-col items-center justify-start gap-12px !overflow-y-auto !pb-18px">
        <div v-for="(item, index) of taskOptions" :key="index" class="h-auto w-full flex items-center">
          <NCard class="h-auto w-full" :title="item.name">
            <div class="h-auto w-full flex flex-col items-start">
              <div v-for="(val, idx) of item.list" :key="idx" :style="{ width: val.width ?? '100%' }">
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
                <NGrid v-if="val.type === 'dynamicInput'" :cols="24" :x-gap="24" class="ml-24px">
                  <NFormItemGi :span="24" :label="val.formName" :path="val.path">
                    <NDataTable :bordered="false" :single-line="false" :columns="val.columns" :data="val.query" />
                  </NFormItemGi>
                </NGrid>
                <!-- text -->
                <NGrid v-if="val.type === 'text'" :cols="24" :x-gap="24" class="ml-24px">
                  <NFormItemGi :span="24" :label="val.formName" :path="val.path">
                    {{ val.value }}
                  </NFormItemGi>
                </NGrid>
                <!-- radioGroup -->
                <NGrid v-if="val.type === 'radioGroup'" :cols="24" :x-gap="24" class="ml-24px">
                  <NFormItemGi
                    v-if="val.modelList.length > 0 && val.isShow"
                    :span="24"
                    :label="val.formName"
                    :path="val.path"
                  >
                    <NRadioGroup v-model:value="val.value" name="anoType" size="large">
                      <!-- 选中事件 -->
                      <NRadioButton
                        v-for="(val1, idx1) of val.modelList"
                        :key="idx1"
                        :value="val1.value"
                        :label="val1.label"
                        @change="handleRadioChange($event, val1, val.formName)"
                      ></NRadioButton>
                    </NRadioGroup>
                  </NFormItemGi>
                </NGrid>
                <!-- cascader -->
                <NGrid v-if="val.type === 'cascader'" :cols="24" :x-gap="24" class="ml-24px">
                  <NFormItemGi :span="24" :label="val.formName" :path="val.path">
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
                <!--checkboxGroup-->
                <NGrid v-if="val.type === 'checkboxGroup'" :cols="24" :x-gap="24" class="ml-24px">
                  <NFormItemGi :span="24" :label="val.formName" :path="val.path">
                    <NCheckboxGroup v-model:value="val.value">
                      <NSpace item-style="display: flex;" align="center">
                        <NCheckbox
                          v-for="item of val.options"
                          :key="item.value"
                          :value="item.value"
                          :label="item.label"
                        />
                      </NSpace>
                    </NCheckboxGroup>
                  </NFormItemGi>
                </NGrid>
                <!--grid-->
                <NGrid v-if="val.type === 'grid'" :cols="24" :x-gap="24" class="ml-24px">
                  <NGi v-for="(val1, idx1) of val.dataList" :key="idx1" :span="val1.span">
                    <div class="w-full flex items-center justify-start leading-34px">
                      <span class="after:content-[':'] mr-4px">{{ val1.name }}</span>
                      <span class="">{{ val1.value }}</span>
                    </div>
                  </NGi>
                </NGrid>
                <!--table-->
                <NGrid v-if="val.type === 'table'" :cols="24" :x-gap="24" class="ml-24px -mt-24px">
                  <NFormItemGi :span="24" :label="val.formName" :path="val.path">
                    <NDataTable :bordered="false" :single-line="false" :columns="val.columns" :data="val.data" />
                  </NFormItemGi>
                </NGrid>
                <!--gridTable-->
                <NGrid v-if="val.type === 'gridTable'" :cols="24" :x-gap="24" class="ml-24px -mt-24px">
                  <NFormItemGi :span="24" :label="val.formName" :path="val.path">
                    <NDataTable size="small" :columns="val.columns" :data="val.dataList" :bordered="false" />
                  </NFormItemGi>
                </NGrid>
                <!-- excel -->
              </div>
            </div>
          </NCard>
        </div>
      </div>
    </div>
    <!--评估报告-->
    <div v-if="tabState === '1'" class="content_modelInfo box-border h-full w-full flex-grow-1 overflow-y-auto p-16px">
      <NGrid :x-gap="gap" :y-gap="16" responsive="screen" item-responsive>
        <NGi span="24 s:12 m:12">
          <NCard :bordered="false" class="card-wrapper">
            <RadarChart />
          </NCard>
        </NGi>
        <NGi span="24 s:24 m:12">
          <NCard :bordered="false" class="card-wrapper">
            <NDataTable
              :columns="autoRuleColumns"
              :data="autoRuleData"
              remote
              :row-key="row => row.id"
              :pagination="false"
              class="sm:h-full"
            />
          </NCard>
        </NGi>
      </NGrid>
      <div class="mt-24px"></div>
      <NGrid :x-gap="gap" :y-gap="16" responsive="screen" item-responsive>
        <NGi span="24 s:24 m:12">
          <NCard :bordered="false" class="card-wrapper">
            <ConfusionMatrixChart />
          </NCard>
        </NGi>
        <NGi span="24 s:24 m:12">
          <NCard :bordered="false" class="card-wrapper">
            <PRChart />
          </NCard>
        </NGi>
      </NGrid>
    </div>
    <!--评估详情-->
    <div v-if="tabState === '2'" class="content_modelInfo box-border h-full w-full flex-grow-1 overflow-y-auto p-16px">
      <NCard :bordered="false" class="h-full w-full card-wrapper !p-0px">
        <div class="h-auto w-full flex flex-col items-start">
          <div class="mb-14px h-auto w-full flex items-center justify-between">
            <div class="l w-600px flex justify-start gap-6px">
              <!--<NButton type="primary" @click="handleOperate('export')">导出</NButton>-->
              <NSelect v-model="exportData.ruleId" :options="tagOptions" placeholder="请选择标签类型" />
              <NSelect v-model="exportData.indictor" :options="indictorOptions" placeholder="请选择排序规则" />
              <NSelect v-model="exportData.sort" :options="sortOptions" placeholder="请选择排序方法" />
            </div>
            <div class="r min-w-200px flex justify-end">
              <NButton type="" @click="handleOperate('operate')">操作记录</NButton>
            </div>
          </div>
          <NDataTable
            v-model:checked-row-keys="checkedRowKeys"
            :columns="columns"
            :data="tableData"
            size="small"
            :scroll-x="962"
            :loading="loading"
            remote
            :row-key="row => row.modelId"
            class="sm:h-full"
          />
        </div>
      </NCard>
      <CustomImageViewer :currentImage="currentImage" :visible="visible" @close="handleImgClose"/>
    </div>
    <!--任务日志-->
    <div v-if="tabState === '3'" class="content_modelInfo box-border h-full w-full flex-grow-1 overflow-y-auto p-16px">
      <div class="mb-14px h-auto w-full flex items-center justify-between">
        <div class="l w-300px flex justify-start gap-6px">
          <NSelect v-model="exportData.ruleId" :options="modleList" placeholder="请选择评估模型" />
        </div>
      </div>
      <NGrid :x-gap="gap" :y-gap="16" responsive="screen" item-responsive class="h-full">
        <NGi span="24 s:24 m:12">
          <NCard :bordered="false" class="h-full w-full card-wrapper !relative" title="评估日志">
            <!--下载-->
            <div class="absolute right-24px top-8px flex justify-end">
              <NButton type="primary" :loading="loading" @click="handleOperate('download')">下载</NButton>
            </div>
            <div class="h-full border-1 border-[#eee] bg-[#ffffff] pr-20px -mt-12px">
              <NVirtualList
                ref="virtualListInst"
                class="font-550 -mt-14px !h-full !text-[#000000]"
                :item-size="32"
                :items="evaluationLogs"
              >
                <template #default="{ item, index }">
                  <div :key="index" class="item box-border px-8px" style="height: 32px">
                    <pre class="px-0px">
                      <div>{{ item.message }}</div>
                    </pre>
                  </div>
                </template>
              </NVirtualList>
            </div>
          </NCard>
        </NGi>
        <NGi span="24 s:24 m:12">
          <NCard :bordered="false" class="h-full w-full card-wrapper !relative" title="系统日志">
            <!--下载-->
            <div class="absolute right-24px top-8px flex justify-end">
              <NButton type="primary" :loading="loading" @click="handleOperate('download')">下载</NButton>
            </div>
            <!--日志列表-->
            <div class="h-full border-1 border-[#eee] bg-[#ffffff] pr-20px -mt-12px">
              <NVirtualList
                ref="virtualListInst"
                class="font-550 -mt-14px !h-full !text-[#000000]"
                :item-size="32"
                :items="evaluationLogs"
              >
                <template #default="{ item, index }">
                  <div :key="index" class="item box-border px-8px" style="height: 32px">
                    <pre class="px-0px">
                      <div>{{ item.message }}</div>
                    </pre>
                  </div>
                </template>
              </NVirtualList>
            </div>
          </NCard>
        </NGi>
      </NGrid>
    </div>
    <!--modal -->
    <NModal v-model:show="ruleShowModal">
      <NCard style="width: 600px" title="导出评估结果" :bordered="false" size="huge" role="dialog" aria-modal="true">
        <div class="wrap_content">
          <!--select-->
          <NSelect v-model="exportData.ruleId" :options="ruleList" placeholder="请选择所属规则" />
        </div>
        <template #footer>
          <NSpace justify="end" :size="16">
            <NButton type="primary" @click="handleDefine()">确认导出</NButton>
            <NButton @click="() => (ruleShowModal = false)">取消返回</NButton>
          </NSpace>
        </template>
      </NCard>
    </NModal>
    <NModal v-model:show="isOperateModal">
      <NCard style="width: 900px" title="操作记录" :bordered="false" size="huge" role="dialog" aria-modal="true">
        <div class="content">
          <NDataTable :columns="OperateCols" :data="operateList" :bordered="false" />
        </div>
        <template #footer>
          <NSpace justify="end" :size="16">
            <NButton type="primary" @click="() => (isOperateModal = false)">我知道了</NButton>
          </NSpace>
        </template>
      </NCard>
    </NModal>
  </div>
</template>

<style lang="scss" scoped>
:deep(.wrap_tabs) {
  .n-tabs-nav-scroll-content {
    border: none !important;
  }

  .n-tabs-pane-wrapper {
    display: none !important;
  }

  .n-tabs-bar {
    height: 3px !important;
  }
}

:deep(.con_card) {
  .n-card__content {
    padding: 0 !important;
  }
}
</style>
