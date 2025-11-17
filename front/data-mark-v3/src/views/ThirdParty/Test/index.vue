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
import ModelTestModal from "./modules/ModelTestModal.vue";

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
  apiFn: getTaskPage,
  showTotal: true,
  apiParams: {
    current: 1,
    size: 10,
    modelName: null,
    modelUnit: null,
    businessUnit: null,
    taskTimeArr: null
  },
  columns: () => [
    {
      type: 'selection'
    },
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
      title: '模型单位',
      key: 'modelUnit'
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
      title: '测试状态',
      key: 'testStatus',
      render(row) {
        return mapStatusToText(row.testStatus);
      }
    },
    {
      title: '操作',
      key: 'operation',
      width: 300, // 增加宽度以确保按钮不换行
      render(row) {
        const buttons = getActionButtons(row);
        const buttonElements = buttons.map((button, index) => {
          if (button.confirmText) {
            return createConfirmButton(
              button.text,
              button.confirmText,
              () => handleAction(button.action, row)
            );
          } else {
            return createButton(
              {
                type: button.type,
                onClick: () => handleAction(button.action, row)
              },
              button.text
            );
          }
        });

        // 使用 NSpace 组件设置按钮间距
        return h(NSpace, { size: 4 }, buttonElements);
      }
    },
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
const detailData = ref<any>({});


enum ActionType {
  View = 'view',
  Edit = 'edit',
  SubmitForApproval = 'submitForApproval',
  ExportFile = 'exportFile',
  Test = 'test',
  Repulse = 'repulse'
}

// const createButton = (text: string, action: ActionType, row: any) => {
//   return h(NButton, {
//     type: 'primary',
//     ghost: true,
//     size: 'small',
//     onClick: () => handleAction(action, row)
//   }, () => text);
// };


const createConfrimButton = (text: string, action: ActionType, row: any, config: any) => {
  return h(NPopconfirm, {
    onPositiveClick: (event) => handleOperate(event, row, 'delete'),
    // 'positive-text': '审批通过',
    // 'negative-text': '审批退回'
    'positive-text': config.positiveText,
    'negative-text': config.negativeText
  }, {
    // default: () => h("span", {}, '是否通过该模型的审批？'),
    default: () => h("span", {}, config.defaultText),
    trigger: () => h(NButton, {
      type: "primary",
      ghost: true,
      size: "small",
    }, () => text)
  })
}

// 定义状态和按钮生成函数的映射
type ButtonGenerator = (row: any) => ReturnType<typeof h>[];

const config = {
  positiveText: '确认打回',
  negativeText: '取消',
  defaultText: '是否打回当前模型申请？'
};
const statusButtonMapping: Record<number, ButtonGenerator> = {
  // 1: (row) => [
  //   createButton('详情', ActionType.View, row),
  //   createButton('编辑', ActionType.Edit, row),
  //   createButton('提交审批', ActionType.SubmitForApproval, row)
  // ],
  // 2: (row) => [
  //   createButton('详情', ActionType.View, row)
  // ],
  // 3: (row) => [
  //   createButton('详情', ActionType.View, row),
  //   createButton('导出文件', ActionType.ExportFile, row)
  // ]

  1: (row) => [
    createButton('详情', ActionType.View, row),
    createButton('导出文件', ActionType.ExportFile, row),
    createButton('模型测试', ActionType.Test, row),
  ],
  2: (row) => [
    createButton('详情', ActionType.View, row),
    createButton('导出文件', ActionType.ExportFile, row),
    createButton('模型测试', ActionType.Test, row),
  ],
  3: (row) => [
    createButton('详情', ActionType.View, row),
    createButton('申请打回', ActionType.Repulse, row),
    createButton('导出文件', ActionType.ExportFile, row),
    createButton('模型测试', ActionType.Test, row),
  ]
};

const handleAction = (actionType: string, row: any) => {
  switch (actionType) {
    case 'view':
      openModal();
      break;
    case 'edit':
      router.push({
        name: "thirdparty_modeloperate"
      })
      break;
    case 'create':
      router.push({
        name: "thirdparty_modeloperate"
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
    default:
      console.log('未知操作类型');
  }
};

const handleModelTest = () => {
  openTestModal();
}

// --------------------------
// 使用数值的枚举
enum TestStatus {
  InProgress = 1,  // 进行中
  Paused = 2,      // 已暂停
  Completed = 3,   // 已完成
  Failed = 4,      // 测试失败
  Waiting = 5      // 等待测试
}

// 操作类型
enum ActionType {
  View = 'view',
  Edit = 'edit',
  Start = 'start',
  Pause = 'pause',
  Continue = 'continue',
  Terminate = 'terminate',
  Retest = 'retest',
  ViewReport = 'viewReport',
  Delete = 'delete'
}

// 操作按钮配置类型
type ButtonConfig = {
  text: string;
  type?: 'primary' | 'success' | 'warning' | 'error';
  action: ActionType;
  confirmText?: string;
};

/**
 * 创建普通按钮
 */
const createButton = (props: any, content: string) => {
  return h(NButton, {
    ...props,
    size: 'small',
    type: 'primary',
    ghost: false,
  }, content);
};

/**
 * 创建确认按钮
 */
const createConfirmButton = (
  text: string,
  confirmText: string,
  onClick: (event: Event) => void
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
      trigger: () => createButton({ type: 'error' }, text),
      default: () => h("span", {}, confirmText),
    },
  );
};

// 操作处理器
const handleOperateAction = (action: ActionType, row: any) => {
  console.log(`执行操作: ${action}`, row);

  switch (action) {
    case ActionType.View:
      // 查看详情逻辑
      break;
    case ActionType.Edit:
      // 编辑逻辑
      break;
    case ActionType.Start:
      // 开始测试逻辑
      break;
    case ActionType.Pause:
      // 暂停测试逻辑
      break;
    case ActionType.Continue:
      // 继续测试逻辑
      break;
    case ActionType.Terminate:
      // 终止测试逻辑
      break;
    case ActionType.Retest:
      // 重新测试逻辑
      break;
    case ActionType.ViewReport:
      // 查看报告逻辑
      break;
    case ActionType.Delete:
      // 删除逻辑
      break;
  }
};

// 根据状态获取操作按钮配置
const getActionButtons = (row: any): ButtonConfig[] => {
  switch (row.testStatus) {
    case TestStatus.InProgress:
      return [
        {
          text: '暂停',
          type: 'warning',
          action: ActionType.Pause,
          confirmText: '确定要暂停当前测试吗？'
        },
        {
          text: '查看详情',
          action: ActionType.View
        }
      ];
    case TestStatus.Paused:
      return [
        {
          text: '继续',
          type: 'success',
          action: ActionType.Continue
        },
        {
          text: '终止',
          type: 'error',
          action: ActionType.Terminate,
          confirmText: '确定要终止测试吗？'
        },
        // {
        //   text: '编辑',
        //   action: ActionType.Edit
        // }
      ];
    case TestStatus.Completed:
      return [
        {
          text: '查看报告',
          type: 'primary',
          action: ActionType.ViewReport
        },
        {
          text: '重新测试',
          type: 'success',
          action: ActionType.Retest,
          confirmText: '确定要重新执行测试吗？'
        },
        {
          text: '删除',
          type: 'error',
          action: ActionType.Delete,
          confirmText: '确定要删除该测试记录吗？'
        }
      ];
    case TestStatus.Failed:
      return [
        {
          text: '查看详情',
          action: ActionType.View
        },
        {
          text: '重新测试',
          type: 'success',
          action: ActionType.Retest,
          confirmText: '确定要重新执行测试吗？'
        }
      ];
    case TestStatus.Waiting:
      return [
        {
          text: '开始测试',
          type: 'success',
          action: ActionType.Start
        },
        // {
        //   text: '编辑',
        //   action: ActionType.Edit
        // }
      ];
    default:
      return [];
  }
};

// 将状态码映射为文本显示
function mapStatusToText(status: number): string {
  switch (status) {
    case TestStatus.InProgress:
      return '进行中';
    case TestStatus.Paused:
      return '已暂停';
    case TestStatus.Completed:
      return '已完成';
    case TestStatus.Failed:
      return '测试失败';
    case TestStatus.Waiting:
      return '等待测试';
    default:
      return '未知状态';
  }
}

// 模拟数据（直接使用数值）
const tableData = ref([
  {
    index: 1,
    modelName: '模型A',
    modelUnit: '单位A',
    businessUnit: '业务部门A',
    developmentUnit: '开发团队A',
    '登记日期': '2025-05-01',
    testStatus: 1 // 对应 TestStatus.InProgress
  },
  {
    index: 2,
    modelName: '模型B',
    modelUnit: '单位B',
    businessUnit: '业务部门B',
    developmentUnit: '开发团队B',
    '登记日期': '2025-05-05',
    testStatus: 2 // 对应 TestStatus.Paused
  },
  {
    index: 3,
    modelName: '模型C',
    modelUnit: '单位C',
    businessUnit: '业务部门C',
    developmentUnit: '开发团队C',
    '登记日期': '2025-05-10',
    testStatus: 3 // 对应 TestStatus.Completed
  },
  {
    index: 4,
    modelName: '模型D',
    modelUnit: '单位D',
    businessUnit: '业务部门D',
    developmentUnit: '开发团队D',
    '登记日期': '2025-05-12',
    testStatus: 4 // 对应 TestStatus.Failed
  },
  {
    index: 5,
    modelName: '模型E',
    modelUnit: '单位E',
    businessUnit: '业务部门E',
    developmentUnit: '开发团队E',
    '登记日期': '2025-05-13',
    testStatus: 5 // 对应 TestStatus.Waiting
  }
]);
</script>

<template>
  <div class="min-h-500px flex-col-stretch gap-16px overflow-hidden lt-sm:overflow-auto">
    <UserSearch v-model:model="searchParams" @reset="resetSearchParams" @search="getDataByPage" />
    <NCard title="模型列表" :bordered="false" size="small" class="sm:flex-1-hidden card-wrapper">
      <template #header-extra>
        <TableHeaderOperation v-model:columns="columnChecks" :disabled-delete="checkedRowKeys.length === 0"
          :loading="loading" :is-add="false" :is-del="false" @refresh="getData">
          <template #prefix>
            <!-- <NButton size="small" @click="handleAction('create')" type="primary">
              模型申请
            </NButton> -->
            <NButton type="primary" ghost size="small" class="-mr-24px" @click="handleModelTest">
              <template #icon>
                <svg-icon local-icon="ThirdParty_Test" class="text-[24px]"></svg-icon>
              </template>
              一键测试
            </NButton>
          </template>
        </TableHeaderOperation>
      </template>

      <NDataTable v-model:checked-row-keys="checkedRowKeys" :columns="columns" :data="tableData"
        :flex-height="!appStore.isMobile" :scroll-x="962" :loading="loading" remote :row-key="(row) => row.taskId"
        :pagination="mobilePagination" class="sm:h-full" />

      <UserOperateDrawer v-model:visible="drawerVisible" :operate-type="operateType" :row-data="editingData"
        @submitted="getDataByPage" />

      <ModelDetailModal v-model:visible="visible" />
      <ModelTestModal v-model:visible="testVisible" />
    </NCard>
  </div>
</template>

<style scoped></style>
