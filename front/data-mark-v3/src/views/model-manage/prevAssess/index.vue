<script setup lang="tsx">
import { NButton, NPopconfirm, NProgress } from "naive-ui";
import { $t } from "@/locales";
import { useAppStore } from "@/store/modules/app";
import { useTable, useTableOperate } from "@/hooks/common/table";
import UserOperateDrawer from "./modules/user-operate-drawer.vue";
import UserSearch from "./modules/user-search.vue";
import { delTask, getTaskDetail, getTaskPage } from "@/service/api/expansion";
import { useBoolean } from "~/packages/hooks";
import MenuOperateModal from "@/views/data-expansion/errArea/modules/menu-operate-modal.vue";

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
    status:  1,
    modelStatus: '草稿'
  },
  {
    modelName: '模型B',
    modelUnit: '单位B',
    businessUnit: '业务部B',
    developmentUnit: '开发部B',
    registrationDate: '2023-02-01',
    status: 2,
    modelStatus: '审批中'
  },
  {
    modelName: '模型C',
    modelUnit: '单位C',
    businessUnit: '业务部C',
    developmentUnit: '开发部C',
    registrationDate: '2023-03-01',
    status: 3,
    modelStatus: '已审批'
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
      title: '序号',
      key: 'index',
      render(row, index) {
        return index + 1;
      }
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
      key: 'registrationDate'
    },
    {
      title: '状态',
      key: 'modelStatus'
    },
    {
      title: '操作',
      key: 'actions',
      render: (row: any) => {
        const buttons = statusButtonMapping[row.status] ? statusButtonMapping[row.status](row) : [];
        return h('div', { class: 'flex gap-8px' }, buttons);
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
const detailData = ref<any>({});


enum ActionType {
  View = 'view',
  Edit = 'edit',
  SubmitForApproval = 'submitForApproval',
  ExportFile = 'exportFile'
}

const createButton = (text: string, action: ActionType, row: any) => {
  return h(NButton, {
    type: 'primary',
    ghost: true,
    size: 'small',
    onClick: () => handleAction(action, row)
  }, () => text);
};

// 定义状态和按钮生成函数的映射
type ButtonGenerator = (row: any) => ReturnType<typeof h>[];
const statusButtonMapping: Record<number, ButtonGenerator> = {
  1: (row) => [
    createButton('查看', ActionType.View, row),
    createButton('编辑', ActionType.Edit, row),
    createButton('提交审批', ActionType.SubmitForApproval, row)
  ],
  2: (row) => [
    createButton('查看', ActionType.View, row)
  ],
  3: (row) => [
    createButton('查看', ActionType.View, row),
    createButton('导出文件', ActionType.ExportFile, row)
  ]
};

const handleAction = (actionType: string, row: any) => {
  switch (actionType) {
    case 'view':
      // 实现查看逻辑
      console.log('查看操作', row);
      router.push({
        name: "model-manage_prevassessoperate"
      })
      break;
    case 'edit':
      // 实现编辑逻辑
      console.log('编辑操作', row);
      router.push({
        name: "model-manage_prevassessoperate"
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
    default:
      console.log('未知操作类型');
  }
};
</script>

<template>
  <div class="min-h-500px flex-col-stretch gap-16px overflow-hidden lt-sm:overflow-auto">
    <UserSearch v-model:model="searchParams" @reset="resetSearchParams" @search="getDataByPage" />
    <NCard title="预评模块" :bordered="false" size="small" class="sm:flex-1-hidden card-wrapper">
      <template #header-extra>
        <TableHeaderOperation v-model:columns="columnChecks" :disabled-delete="checkedRowKeys.length === 0"
          :loading="loading" :is-add="false" :is-del="false" @refresh="getData">
          <template #prefix>
            <!-- <NButton size="small" @click="handleCreateTask()" type="primary" class="-mr-24px">
              创建任务
            </NButton> -->
          </template>
        </TableHeaderOperation>
      </template>
      <NDataTable v-model:checked-row-keys="checkedRowKeys" :columns="columns" :data="mockData"
        :flex-height="!appStore.isMobile" :scroll-x="962" :loading="loading" remote :row-key="(row) => row.taskId"
        :pagination="mobilePagination" class="sm:h-full" />
      <UserOperateDrawer v-model:visible="drawerVisible" :operate-type="operateType" :row-data="editingData"
        @submitted="getDataByPage" />
    </NCard>
  </div>
</template>

<style scoped></style>
