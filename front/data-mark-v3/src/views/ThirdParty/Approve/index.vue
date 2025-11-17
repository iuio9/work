<script setup lang="tsx">
import { NButton, NPopconfirm, NProgress, NSpace } from "naive-ui";
import { $t } from "@/locales";
import { useAppStore } from "@/store/modules/app";
import { useTable, useTableOperate } from "@/hooks/common/table";
import UserOperateDrawer from "./modules/user-operate-drawer.vue";
import UserSearch from "./modules/user-search.vue";
import { delTask, getTaskDetail, getTaskPage } from "@/service/api/expansion";
import { useBoolean } from "~/packages/hooks";

import ModelDetailModal from "./modules/ModelDetailModal.vue";
import FilePreviewModal from "./modules/FilePreviewModal.vue";

import { fetchModelApproveList, passModelApprove, notPassModelApprove, oneClickDebugging, delModelEvaluation } from "@/service/api/third";
import { resolveDirective, withDirectives } from "vue";

const appStore = useAppStore();

const handleDelete = async (row) => {
  const res = await delTask({ taskId: row.taskId });
  if (res.data) {
    window.$message?.success?.("删除成功！");
    getDataByPage();
  }
};

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
  apiFn: fetchModelApproveList,
  showTotal: true,
  apiParams: {
    current: 1,
    size: 10,
    modelName: null,
    modelUnit: null,
    businessUnit: null,
    taskTimeArr: null,
    applyForNum: null,
    buildUnitName: null,
    btUnitName: null
  },
  columns: () => [
    {
      title: '序号',
      key: 'index',
      width: 60
    },
    {
      title: '申请单号',
      key: 'applyForNum'
    },
    {
      title: '申请人',
      key: 'nickName'
    },
    {
      title: '模型方式',
      key: 'modelWayStr'
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
      width: 300,
      render(row) {
        const buttons = getApprovalButtons(row);
        const buttonElements = buttons.map((button, index) => {
          if (button.confirmText) {
            return createConfirmButton(
              button.text,
              button.confirmText,
              () => handleApprovalAction(button.action, row),
              button.permission
            );
          } else {
            return createButton(
              {
                type: button.type,
                onClick: () => handleApprovalAction(button.action, row)
              },
              button.text,
              button.permission
            );
          }
        });

        return h(NSpace, { size: 4 }, buttonElements);
      }
    }
  ]
});

const {
  drawerVisible,
  operateType,
  editingData,
  handleAdd,
  handleEdit,
  checkedRowKeys,
  onBatchDeleted,
  onDeleted
  // closeDrawer
} = useTableOperate(data, getData);

const router = useRouter();
const { bool: visible, setTrue: openModal } = useBoolean();
const detailData = ref<any>({});

// 将申请类型码映射为文本显示
function mapApplyTypeToText(type: number): string {
  switch (type) {
    case 1:
      return '线下申请';
    case 2:
      return '线上申请';
  }
}


enum ActionType {
  View = 'view',
  Edit = 'edit',
  SubmitForApproval = 'submitForApproval',
  ExportFile = 'exportFile',
  Test = 'test'
}

// const createButton = (text: string, action: ActionType, row: any) => {
//   return h(NButton, {
//     type: 'primary',
//     ghost: true,
//     size: 'small',
//     onClick: () => handleAction(action, row)
//   }, () => text);
// };

// const createConfrimButton = (text: string, action: ActionType, row: any, config: any) => {
//   return h(NPopconfirm, {
//     onPositiveClick: (event) => handleOperate(event, row, 'delete'),
//     // 'positive-text': '审批通过',
//     // 'negative-text': '审批退回'
//     'positive-text': config.positiveText,
//     'negative-text': config.negativeText
//   }, {
//     // default: () => h("span", {}, '是否通过该模型的审批？'),
//     default: () => h("span", {}, config.defaultText),
//     trigger: () => h(NButton, {
//       type: "primary",
//       ghost: true,
//       size: "small",
//     }, () => text)
//   })
// }

// 定义状态和按钮生成函数的映射
type ButtonGenerator = (row: any) => ReturnType<typeof h>[];

// --------------------------------------------------------------
// 使用数值的枚举
// 审批状态枚举（数值类型）
enum ApprovalStatus {
  Pending = 1,      // 待审批
  Approved = 2,     // 已通过
  Rejected = 3,     // 已退回
  // Processing = 4,   // 处理中
  // Completed = 5     // 已完成
}

// 审批操作类型
enum ApprovalAction {
  View = 'view',        // 查看详情
  Approve = 'approve',  // 审批通过
  Reject = 'reject',    // 审批退回
  Export = 'export',     // 导出文件
  Test = 'test',        // 测试
  Delete = 'delete'  // 删除
}

// 审批操作按钮配置类型
type ApprovalButtonConfig = {
  text: string;
  type?: 'primary' | 'success' | 'warning' | 'error';
  action: ApprovalAction;
  confirmText?: string;
  permission?: string;
};

/**
 * 创建普通按钮
 */
const createButton = (props: any, content: string, permission?: string) => {
  const authDir = resolveDirective('hasPermi');
  return withDirectives(
    h(NButton, {
      ...props,
      size: 'small',
      type: 'primary',
      ghost: false,
    }, content),
    [[authDir, permission]]
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

// 审批操作处理器
const handleApprovalAction = async (action: ApprovalAction, row: any) => {
  console.log(`执行审批操作: ${action}`, row);

  switch (action) {
    case ApprovalAction.View:
      // 查看详情逻辑
      rowData.value = row;
      openModal();
      break;
    case ApprovalAction.Approve:
      // 审批通过逻辑
      const res = await passModelApprove({
        id: row.id
      });
      if (res.data) {
        window.$message?.success?.("审批通过成功！");
        getDataByPage();
      }
      break;
    case ApprovalAction.Reject:
      // 审批退回逻辑
      const res1 = await notPassModelApprove({
        id: row.id
      })
      if (res1.data) {
        window.$message?.success?.("审批退回成功！");
        getDataByPage();
      }
      break;
    case ApprovalAction.Export:
      // 导出文件逻辑
      break;
    case ApprovalAction.Test:
      // 测试逻辑
      const res2 = await oneClickDebugging({
        id: row.id
      });
      if (res2.data) {
        window.$message?.success?.(`${res2.data}`);
        getDataByPage();
      }
      break;
    case ApprovalAction.Delete:
      const res3 = await delModelEvaluation({
        id: row.id
      })
      if (res3.data) {
        window.$message?.success?.("删除成功！");
        getDataByPage();
      }
      break;
  }
};

// 根据审批状态获取操作按钮配置
const getApprovalButtons = (row: any): ApprovalButtonConfig[] => {
  switch (row.applyForStatus) {
    case ApprovalStatus.Pending:
      return [
        {
          text: '查看详情',
          action: ApprovalAction.View,
          permission: "thirdparty:approve:detail"
        },
        {
          text: '审批通过',
          type: 'success',
          action: ApprovalAction.Approve,
          confirmText: '确定要通过该审批吗？',
          permission: "thirdparty:approve:approve"
        },
        {
          text: '审批退回',
          type: 'error',
          action: ApprovalAction.Reject,
          confirmText: '确定要退回该审批吗？',
          permission: "thirdparty:approve:reject"
        },
        // 一键测试
        {
          text: '一键测试',
          type: 'primary',
          action: ApprovalAction.Test,
          permission: "thirdparty:approve:test"
        },
        // 删除
        {
          text: '删除',
          type: 'primary',
          action: ApprovalAction.Delete,
          confirmText: '确定要删除该审批吗？',
          permission: "thirdparty:approve:delete"
        },
      ];
    case ApprovalStatus.Approved:
    case ApprovalStatus.Completed:
      return [
        {
          text: '查看详情',
          action: ApprovalAction.View,
          permission: "thirdparty:approve:detail"
        },
        {
          text: '导出文件',
          type: 'primary',
          action: ApprovalAction.Export,
          permission: "thirdparty:approve:export"
        },
        // 一键测试
        {
          text: '一键测试',
          type: 'primary',
          action: ApprovalAction.Test,
          permission: "thirdparty:approve:test"
        },
        // 删除
        {
          text: '删除',
          type: 'primary',
          action: ApprovalAction.Delete,
          confirmText: '确定要删除该审批吗？',
          permission: "thirdparty:approve:delete"
        },
      ];
    case ApprovalStatus.Rejected:
      return [
        {
          text: '查看详情',
          action: ApprovalAction.View,
          permission: "thirdparty:approve:detail"
        },
        // 一键测试
        {
          text: '一键测试',
          type: 'primary',
          action: ApprovalAction.Test,
          permission: "thirdparty:approve:test"
        },
        // 删除
        {
          text: '删除',
          type: 'primary',
          action: ApprovalAction.Delete,
          confirmText: '确定要删除该审批吗？',
          permission: "thirdparty:approve:delete"
        },
      ];
    case ApprovalStatus.Processing:
      return [
        {
          text: '查看详情',
          action: ApprovalAction.View,
          permission: "thirdparty:approve:detail"
        },
        // 一键测试
        {
          text: '一键测试',
          type: 'primary',
          action: ApprovalAction.Test,
          permission: "thirdparty:approve:test"
        },
        // 删除
        {
          text: '删除',
          type: 'primary',
          action: ApprovalAction.Delete,
          confirmText: '确定要删除该审批吗？',
          permission: "thirdparty:approve:delete"
        },
      ];
    default:
      return [];
  }
};

// 将审批状态码映射为文本显示
function mapStatusToText(status: number): string {
  switch (status) {
    case ApprovalStatus.Pending:
      return '待审批';
    case ApprovalStatus.Approved:
      return '已通过';
    case ApprovalStatus.Rejected:
      return '已退回';
    case ApprovalStatus.Processing:
      return '处理中';
    case ApprovalStatus.Completed:
      return '已完成';
    default:
      return '未知状态';
  }
};

const mockData = ref<any>([
  {
    "index": 1,
    "applyForNum": "APL-20250001",
    "nickName": "Li Hua",
    "modelName": "Image Classification Model V1",
    "modelType": "Image Classification",
    "buildUnitName": "Tech Pioneer Co., Ltd.",
    "btUnitName": "Innovative AI Studio",
    "applyForType": 1,
    "applyForDate": "2025-05-10",
    "applyForStatus": 1
  },
  {
    "index": 2,
    "applyForNum": "APL-20250002",
    "nickName": "Wang Qiang",
    "modelName": "Object Detection Model V2",
    "modelType": "Object Detection",
    "buildUnitName": "Smart Vision Research Institute",
    "btUnitName": "Advanced Algorithm Team",
    "applyForType": 2,
    "applyForDate": "2025-05-12",
    "applyForStatus": 1
  },
  {
    "index": 3,
    "applyForNum": "APL-20250003",
    "nickName": "Chen Li",
    "modelName": "Semantic Segmentation Model V3",
    "modelType": "Semantic Segmentation",
    "buildUnitName": "Intelligent Image Lab",
    "btUnitName": "Excellence Technology Co.",
    "applyForType": 1,
    "applyForDate": "2025-05-15",
    "applyForStatus": 2
  },
  {
    "index": 4,
    "applyForNum": "APL-20250004",
    "nickName": "Zhao Gang",
    "modelName": "Instance Segmentation Model V4",
    "modelType": "Instance Segmentation",
    "buildUnitName": "Vision Innovation Center",
    "btUnitName": "Top-Tech Enterprise",
    "applyForType": 2,
    "applyForDate": "2025-05-18",
    "applyForStatus": 2
  },
  {
    "index": 5,
    "applyForNum": "APL-20250005",
    "nickName": "Yang Fang",
    "modelName": "OCR Model V5",
    "modelType": "OCR Recognition",
    "buildUnitName": "Text Recognition Institute",
    "btUnitName": "Efficient Algorithm Studio",
    "applyForType": 1,
    "applyForDate": "2025-05-20",
    "applyForStatus": 3
  },
])

const rowData = ref<any>({});
const previewPath = ref<string | null>(null);
const previewFileSuffix = ref<string | null>(null);

const { bool: fileVisible, setTrue: openFileModal } = useBoolean();

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
    <NCard title="测试评估审批" :bordered="false" size="small" class="sm:flex-1-hidden card-wrapper">
      <template #header-extra>
        <TableHeaderOperation v-model:columns="columnChecks" :disabled-delete="checkedRowKeys.length === 0"
          :loading="loading" :is-add="false" :is-del="false" @refresh="getData">
          <template #prefix>
            <!-- <NButton size="small" @click="handleAction('apply')" type="primary" class="-mr-24px">
              模型申请
            </NButton> -->
            <!-- <NButton type="primary" ghost size="small" class="-mr-24px" @click="handleModelTest">
              <template #icon>
                <svg-icon local-icon="ThirdParty_Test" class="text-[24px]"></svg-icon>
              </template>
申请测试
</NButton> -->
          </template>
        </TableHeaderOperation>
      </template>

      <NDataTable v-model:checked-row-keys="checkedRowKeys" :columns="columns" :data="data"
        :flex-height="!appStore.isMobile" :scroll-x="962" :loading="loading" remote :row-key="(row) => row.id"
        :pagination="mobilePagination" class="sm:h-full" />

      <UserOperateDrawer v-model:visible="drawerVisible" :operate-type="operateType" :row-data="editingData"
        @submitted="getDataByPage" />

      <!-- <ModelDetailModal v-model:visible="visible" /> -->
      <ModelDetailModal v-model:visible="visible" v-model:apply-id="rowData.id" @preview="handlePreview" />
      <!-- preview file -->
      <FilePreviewModal v-model:visible="fileVisible" v-model:previewPath="previewPath"
        v-model:previewFileSuffix="previewFileSuffix" />
    </NCard>
  </div>
</template>

<style scoped></style>
