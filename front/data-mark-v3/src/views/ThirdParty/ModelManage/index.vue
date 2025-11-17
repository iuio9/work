<script setup lang="tsx">
import { NButton, NPopconfirm, NProgress, NSpace } from "naive-ui";
import { $t } from "@/locales";
import { useAppStore } from "@/store/modules/app";
import { useTable, useTableOperate } from "@/hooks/common/table";
import UserOperateDrawer from "./modules/user-operate-drawer.vue";
import UserSearch from "./modules/user-search.vue";
import { fetchModelEvaluationList, submitApproveModelEvaluation, generatePadModelEvaluation, delModelEvaluation } from "@/service/api/third";
import { useBoolean } from "~/packages/hooks";

import ModelDetailModal from "./modules/ModelDetailModal.vue";
import ModelTestModal from "./modules/ModelTestModal.vue";
import ModelApplyModal from "./modules/ModelApplyModal.vue";
import FilePreviewModal from "./modules/FilePreviewModal.vue";
import ModelApplyTestModal from "./modules/ModelApplyTestModal.vue";
import { downloadByData } from "@/utils/common";

import { getCurrentInstance, resolveDirective, withDirectives } from 'vue';

import axios from 'axios';

const appStore = useAppStore();

const handleDelete = async (row) => {
  const res = await delTask({ taskId: row.taskId });
  if (res.data) {
    window.$message?.success?.("删除成功！");
    getDataByPage();
  }
};

const mockData = [
  {
    modelName: '模型A',
    modelUnit: '单位A',
    businessUnit: '业务部A',
    developmentUnit: '开发部A',
    registrationDate: '2023-01-01',
    status: 1,
    applyType: 0,
    modelStatus: '草稿'
  },
  {
    modelName: '模型B',
    modelUnit: '单位B',
    businessUnit: '业务部B',
    developmentUnit: '开发部B',
    registrationDate: '2023-02-01',
    status: 2,
    applyType: 1,
    modelStatus: '审批中'
  },
  {
    modelName: '模型C',
    modelUnit: '单位C',
    businessUnit: '业务部C',
    developmentUnit: '开发部C',
    registrationDate: '2023-03-01',
    status: 3,
    applyType: 1,
    modelStatus: '审批通过'
  },
  {
    modelName: '模型D',
    modelUnit: '单位D',
    businessUnit: '业务部D',
    developmentUnit: '开发部D',
    registrationDate: '2023-03-01',
    status: 4,
    applyType: 1,
    modelStatus: '审批打回'
  },
]

const {
  columns,
  columnChecks,
  data,
  getData,
  getDataByPage,
  loading,
  mobilePagination,
  searchParams,
  resetSearchParams,
} = useTable({
  apiFn: fetchModelEvaluationList,
  showTotal: true,
  apiParams: {
    current: 1,
    size: 10,
    modelName: null,
    buildUnitName: null,
    btUnitName: null,
    applyTimeArr: null
  },
  columns: () => [
    {
      title: '序号',
      key: 'index',
      width: 60
    },
    {
      title: '模型名称',
      key: 'modelName'
    },
    {
      title: '模型类型',
      key: 'modelType'
    },
    {
      title: '建设单位名称',
      key: 'buildUnitName'
    },
    {
      title: '承建单位名称',
      key: 'btUnitName'
    },
    {
      title: '申请类型',
      key: 'applyForType',
      render(row: any) {
        return mapApplyTypeToText(row.applyForType);
      }
    },
    {
      title: '申请日期',
      key: 'applyForDate'
    },
    {
      title: '申请状态',
      key: 'applyForStatus',
      width: 100,
      render(row: any) {
        return mapStatusToText(row.applyForStatus);
      }
    },
    {
      title: '操作',
      key: 'operation',
      width: 320,
      render(row) {
        const buttons = getApplicationButtons(row);
        const buttonElements = buttons.map((button, index) => {
          if (button.confirmText) {
            return createConfirmButton(
              button.text,
              button.confirmText,
              () => handleApplicationAction(button.action, row),
              button.permission
            );
          } else {
            return createButton(
              {
                type: button.type,
                onClick: () => handleApplicationAction(button.action, row)
              },
              button.text,
              button.permission
            );
          }
        });

        // return h(NSpace, { size: 4 }, buttonElements);
        return h('div', { class: 'flex items-center gap-2 flex-wrap' }, buttonElements);
      }
    }
  ]
});

const {
  drawerVisible,
  operateType,
  editingData,
  checkedRowKeys,
} = useTableOperate(data, getData);

const router = useRouter();
const { bool: visible, setTrue: openModal } = useBoolean();
const { bool: testVisible, setTrue: openTestModal } = useBoolean();
const { bool: uploadVisible, setTrue: openUploadModal } = useBoolean();
const { bool: fileVisible, setTrue: openFileModal } = useBoolean();
const detailData = ref<any>({});


enum ActionType {
  View = 'view',
  Edit = 'edit',
  SubmitForApproval = 'submitForApproval',
  ExportFile = 'exportFile',
  Test = 'test',
  ApplyUpload = 'applyUpload',
}

const handleAction = (actionType: string, row: any) => {
  switch (actionType) {
    case 'view':
      openModal();
      break;
    case 'edit':
      router.push({
        name: "thirdparty_modeloperate",
        query: {
          id: row.id,
          sign: "edit"
        }
      })
      break;
    case 'create':
      router.push({
        name: "thirdparty_modeloperate",
        query: {
          sign: "create",
        }
      })
      break;
    case 'submitForApproval':
      // 实现提交审批逻辑
      console.log('提交审批操作', row);
      break;
    case 'exportFile':
      // 实现导出文件逻辑
      console.log('导出文件操作', row);
      break;
    case 'test':
      openTestModal();
      break;
    case 'applyUpload':
      openUploadModal();
      break;
    default:
      console.log('未知操作类型');
  }
};

// ----------------------------------------------------
const rowData = ref<any>({});

const previewPath = ref<string | null>(null);
const previewFileSuffix = ref<string | null>(null);

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

/**
 * 创建普通按钮
 */
const createButton = (props: any, content: string, permission?: string) => {
  const authDir = resolveDirective('hasPermi')
  return withDirectives(
    h(NButton, {
      ...props,
      size: 'small',
      type: 'primary',
      ghost: false,
    }, content),
    [
      [
        authDir,
        permission
      ]
    ]
  )
};

/**
 * 创建确认按钮
 */
const createConfirmButton = (
  text: string,
  confirmText: string,
  onClick: (event: Event) => void,
  permission?: string
) => {
  return h(
    NPopconfirm,
    {
      onPositiveClick: (event: Event) => {
        event.stopPropagation();
        onClick(event);
      },
    },
    {
      trigger: () => createButton({ type: 'error' }, text, permission),
      default: () => h("span", {}, confirmText),
    },
  );
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

const isExport = ref<Boolean>(false);

// 申请操作处理器
const handleApplicationAction = async (action: ApplicationAction, row: any) => {
  rowData.value = row;
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
        url: `${baseUrl}/model/evaluation/generatePad?id=${row.id}`,
        data: {
          id: row.id,
        }
      };
      const fileName = `数据报告${row.id}.pdf`;
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
          id: row.id,
          sign: "edit",
          applyForStatus: row.applyForStatus,
          applyForType: row.applyForType,
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
        id: row.id,
      });
      if (res.data) {
        window.$message?.success?.("提交成功！");
        getDataByPage();
      }
      break;
    case ApplicationAction.Test:
      // 测试逻辑
      openTestModal();
      break;
    case ApplicationAction.Delete:
      // 删除逻辑
      const res1 = await delModelEvaluation({
        id: row.id
      })
      if (res1.data) {
        window.$message?.success?.("删除成功！");
        getDataByPage();
      }
      break;
  }
};

// 根据申请状态获取操作按钮配置
const getApplicationButtons = (row: any): ApplicationButtonConfig[] => {
  const { applyForStatus, applyForType } = row;

  // 基础按钮配置
  const baseButtons = [
    { text: '查看详情', action: ApplicationAction.ViewDetails, permission: 'thirdparty:mul:details' },
  ];

  // 删除
  const deleteButton = {
    text: '删除',
    type: 'error',
    action: ApplicationAction.Delete,
    confirmText: '确定删除吗？',
    permission: 'thirdparty:mul:delete'
  }

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

  // 导出文件按钮
  const exportButton = {
    text: '导出文件',
    type: 'primary',
    action: ApplicationAction.Export,
    permission: "thirdparty:mul:exportFile"
  };

  // 生成/上传附件按钮 (仅applyForType=1时显示)
  const fileButtons = applyForType === 1 ? [
    { text: '生成数据附件', type: 'success', action: ApplicationAction.GenerateFile, isExport: isExport.value, permission: 'thirdparty:mul:generateAttachments' },
    { text: '上传数据附件', type: 'success', action: ApplicationAction.UploadReport, permission: 'thirdparty:mul:uploadAttachments' },
  ] : [];

  // 根据状态返回按钮配置
  switch (applyForStatus) {
    case ApplicationStatus.Draft:
      // return [testButton, commitButton, editButton, ...baseButtons, ...fileButtons, deleteButton];
      return [editButton, ...baseButtons, deleteButton];

    case ApplicationStatus.UnderReview:
      // return [testButton, ...baseButtons, deleteButton];
      return [...baseButtons, deleteButton];

    case ApplicationStatus.Approved:
      return [...baseButtons, deleteButton];

    case ApplicationStatus.Rejected:
      // return [testButton, commitButton, editButton, ...baseButtons, ...fileButtons, deleteButton];
      return [editButton, ...baseButtons, deleteButton];

    case ApplicationStatus.Completed:
      return [...baseButtons, deleteButton];

    default:
      return [];
  }
};

// 将申请状态码映射为文本显示
function mapStatusToText(status: number): string {
  switch (status) {
    case ApplicationStatus.Draft:
      return '草稿';
    case ApplicationStatus.UnderReview:
      return '审批中';
    case ApplicationStatus.Approved:
      return '审批通过';
    case ApplicationStatus.Rejected:
      return '审批打回';
    case ApplicationStatus.Completed:
      return '已完成';
    default:
      return '未知状态';
  }
}

// 将申请类型码映射为文本显示
function mapApplyTypeToText(type: number): string {
  switch (type) {
    case 1:
      return '线下申请';
    case 2:
      return '线上申请';
  }
}

// 申请表格列定义
const applicationColumns = [
  {
    title: '序号',
    key: 'index',
    width: 50
  },
  {
    title: '模型名称',
    key: 'modelName'
  },
  {
    title: '模型方式',
    key: 'modelWayStr'
  },
  {
    title: '模型单位',
    key: 'modelUnit'
  },
  {
    title: '申请类型',
    key: 'applicationType'
  },
  {
    title: '业务单位',
    key: 'businessUnit'
  },
  {
    title: '开发单位',
    key: 'developmentUnit'
  },
  {
    title: '登记日期',
    key: '登记日期'
  },
  {
    title: '状态',
    key: 'status',
    width: 100,
    render(row) {
      return mapStatusToText(row.status);
    }
  },
  {
    title: '操作',
    key: 'operation',
    width: 220,
    render(row) {
      const buttons = getApplicationButtons(row);
      const buttonElements = buttons.map((button: any, index) => {
        if (button.confirmText) {
          return createConfirmButton(
            button.text,
            button.confirmText,
            () => handleApplicationAction(button.action, row),
            button.permission
          );
        } else {
          return createButton(
            {
              type: button.type,
              onClick: () => handleApplicationAction(button.action, row),
              loading: button?.isExport ?? false,
            },
            button.text,
            button.permission
          );
        }
      });

      return h(NSpace, { size: 4 }, buttonElements);
    }
  }
];

// 模拟申请数据
const applicationData = ref([
  {
    "id": 17,
    "applyForNum": "20250601001",
    "modelName": "输电线路覆冰检测模型",
    "buildUnitName": "国网湖南省电力有限公司",
    "btUnitName": "腾讯科技（深圳）有限公司",
    "modelType": "CNN+注意力机制",
    "applyForType": 1,          // 模型新建
    "applyForStatus": 1,        // Draft（草稿）
    "applyForDate": "2025-06-01",
    "nickName": "工程师_李四"
  },
  {
    "id": 16,
    "applyForNum": "20250601002",
    "modelName": "变电站缺陷识别模型",
    "buildUnitName": "国网山东省电力有限公司",
    "btUnitName": "旷视科技有限公司",
    "modelType": "图像识别（YOLOv8）",
    "applyForType": 2,          // 模型优化
    "applyForStatus": 2,        // UnderReview（审批中）
    "applyForDate": "2025-06-01",
    "nickName": "审核员_王芳"
  },
  {
    "id": 15,
    "applyForNum": "20250601003",
    "modelName": "配网台区线损预测模型",
    "buildUnitName": "国网四川省电力有限公司",
    "btUnitName": "华为云技术有限公司",
    "modelType": "LSTM时间序列模型",
    "applyForType": 1,          // 模型新建
    "applyForStatus": 3,        // Approved（审批通过）
    "applyForDate": "2025-06-01",
    "nickName": "管理员"
  },
  {
    "id": 24,
    "applyForNum": "20250601004",
    "modelName": "电力设备异常声音检测模型",
    "buildUnitName": "国网湖北省电力有限公司",
    "btUnitName": "科大讯飞股份有限公司",
    "modelType": "音频识别（MFCC+CNN）",
    "applyForType": 2,          // 模型优化
    "applyForStatus": 4,        // Rejected（审批打回）
    "applyForDate": "2025-06-01",
    "nickName": "技术主管_陈工"
  },
  {
    "id": 25,
    "applyForNum": "20250601005",
    "modelName": "新能源场站功率预测模型",
    "buildUnitName": "国网内蒙古电力（集团）有限公司",
    "btUnitName": "百度在线网络技术（北京）有限公司",
    "modelType": "Transformer时序模型",
    "applyForType": 1,          // 模型新建
    "applyForStatus": 5,        // Completed（已完成）
    "applyForDate": "2025-05-30",
    "nickName": "系统自动审批"
  }
]);

// 当前组件示例
const instance = getCurrentInstance();


const handlePreview = (params: any) => {
  const { previewFileSuffix: suffix, previewPath: filePath } = params;
  previewFileSuffix.value = suffix;
  previewPath.value = filePath;
  openFileModal();
};
</script>

<template>
  <div class="min-h-500px flex-col-stretch gap-16px overflow-hidden lt-sm:overflow-auto">
    <UserSearch v-model:model="searchParams" @reset="resetSearchParams" @search="getDataByPage" />
    <NCard title="测试评估申请" :bordered="false" size="small" class="sm:flex-1-hidden card-wrapper">
      <template #header-extra>
        <TableHeaderOperation v-model:columns="columnChecks" :disabled-delete="checkedRowKeys.length === 0"
          :loading="loading" :is-add="false" :is-del="false" @refresh="getData">
          <template #prefix>
            <NButton size="small" class="-mr-24px" @click="handleAction('create')" type="primary"
              v-hasPermi="'thirdparty:mul:modelApply'">
              模型申请
            </NButton>
          </template>
        </TableHeaderOperation>
      </template>

      <NDataTable v-model:checked-row-keys="checkedRowKeys" :columns="columns" :data="data"
        :flex-height="!appStore.isMobile" :scroll-x="962" :loading="loading" remote :row-key="(row) => row.taskId"
        :pagination="mobilePagination" class="sm:h-full" />

      <UserOperateDrawer v-model:visible="drawerVisible" :operate-type="operateType" :row-data="editingData"
        @submitted="getDataByPage" />

      <!-- 查看详情 -->
      <ModelDetailModal v-model:visible="visible" v-model:apply-id="rowData.id" @preview="handlePreview" />
      <ModelApplyTestModal v-model:visible="testVisible" v-model:modelId="rowData.id" />
      <ModelApplyModal v-model:visible="uploadVisible" v-model:id="rowData.id" />

      <!-- preview file -->
      <FilePreviewModal v-model:visible="fileVisible" v-model:previewPath="previewPath"
        v-model:previewFileSuffix="previewFileSuffix" />
    </NCard>
  </div>
</template>

<style scoped></style>
