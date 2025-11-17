<script setup lang="tsx">
import { NButton, NPopconfirm, NProgress } from "naive-ui";
import { $t } from "@/locales";
import { useAppStore } from "@/store/modules/app";
import { useTable, useTableOperate } from "@/hooks/common/table";
import UserOperateDrawer from "./modules/user-operate-drawer.vue";
import UserSearch from "./modules/user-search.vue";
import { delTask, getTaskDetail, getTaskPage } from "@/service/api/expansion";
import { useBoolean } from "~/packages/hooks";
import BlackWhiteBoxOperateModal from "./modules/BlackWhiteBoxOperateModal.vue"
import { getBlackWhiteTestResult } from "@/service/api/third";
import MenuOperateModal from "./modules/menu-operate-modal.vue"

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
    taskName: null,
    taskStat: null,
    recordType: 4,
    taskTimeArr: null
  },
  columns: () => [
    {
      key: "taskId",
      title: "任务编号",
      align: "center",
      width: 120,
      fixed: "left"
    },
    {
      key: "taskInputName",
      title: "任务名称",
      align: "center",
    },
    {
      key: "taskName",
      title: "任务类型",
      align: "center",
      width: 180
    },
    {
      title: "任务状态",
      key: "taskStat",
      align: "center"
    },
    { title: "创建时间", key: "createTime", width: 180 },
    {
      fixed: 'right',
      key: "operate",
      title: $t("common.operate"),
      align: "center",
      width: 260,
      render: (row) => {
        return [
          h(
            "div",
            {
              class: "flex-row flex-center gap-8px",
            },
            [
              h(
                NButton,
                {
                  type: "primary",
                  ghost: true,
                  size: "small",
                  onClick: () => handleDetail(row.taskId),
                },
                "任务详情"
              ),
              h(
                NPopconfirm,
                {
                  onPositiveClick: () => handleDelete(row),
                },
                {
                  trigger: () =>
                    h(
                      NButton,
                      {
                        type: "primary",
                        ghost: true,
                        size: "small",
                      },
                      "删除"
                    ),
                  default: () => h("span", {}, "你确定要删除吗？"),
                }
              ),
            ]
          ),
        ];
      },
      fixed: "right",
    },
  ],
});

const {
  drawerVisible,
  operateType,
  editingData,
  checkedRowKeys,
  onBatchDeleted,
} = useTableOperate(data, getData);

const router = useRouter();
const { bool: visible, setTrue: openModal } = useBoolean();
const { bool: boxVisible, setTrue: openBoxModal } = useBoolean();
const detailData = ref<any>({});

const handleCreateTask = (): void => {
  openBoxModal()
};
async function handleDetail(id: number | string) {
  const res = await getBlackWhiteTestResult({
    taskId: id,
  });
  const { testResult } = res.data;
  const { acc, adv_acc } = testResult ? JSON.parse(testResult) : {};
  detailData.value = {
    ...res.data,
    acc,
    adv_acc
  };
  openModal();
}

const handleSubmit = () => {
  boxVisible.value = false;
  getDataByPage()
}
</script>

<template>
  <div class="min-h-500px flex-col-stretch gap-16px overflow-hidden lt-sm:overflow-auto">
    <UserSearch v-model:model="searchParams" @reset="resetSearchParams" @search="getDataByPage" />
    <NCard title="黑白盒" :bordered="false" size="small" class="sm:flex-1-hidden card-wrapper">
      <template #header-extra>
        <TableHeaderOperation v-model:columns="columnChecks" :disabled-delete="checkedRowKeys.length === 0"
          :loading="loading" :is-add="false" :is-del="false" @refresh="getData">
          <template #prefix>
            <NButton size="small" @click="handleCreateTask()" type="primary" class="-mr-24px">
              创建任务
            </NButton>
          </template>
        </TableHeaderOperation>
      </template>
      <NDataTable v-model:checked-row-keys="checkedRowKeys" :columns="columns" :data="data"
        :flex-height="!appStore.isMobile" :scroll-x="962" :loading="loading" remote :row-key="(row) => row.taskId"
        :pagination="mobilePagination" class="sm:h-full" />
      <UserOperateDrawer v-model:visible="drawerVisible" :operate-type="operateType" :row-data="editingData"
        @submitted="getDataByPage" />
      <MenuOperateModal v-model:visible="visible" :operate-type="operateType" :row-data="detailData" />
      <BlackWhiteBoxOperateModal v-model:visible="boxVisible" @submitted="handleSubmit"/>
    </NCard>
  </div>
</template>

<style scoped></style>
