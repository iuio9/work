<script setup lang="tsx">
import { NButton, NPopconfirm, NProgress, NTag } from "naive-ui";
import { $t } from "@/locales";
import { useAppStore } from "@/store/modules/app";
import { useTable, useTableOperate } from "@/hooks/common/table";
import UserOperateDrawer from "./modules/user-operate-drawer.vue";
import UserSearch from "./modules/user-search.vue";
import { getTaskDetail, getTaskPage, delTask } from "@/service/api/expansion";
import { useBoolean } from "~/packages/hooks";
import MenuOperateModal from "@/views/data-expansion/errArea/modules/menu-operate-modal.vue";

const appStore = useAppStore();

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
    // if you want to use the searchParams in Form, you need to define the following properties, and the value is null
    // the value can not be undefined, otherwise the property in Form will not be reactive
    taskName: null,
    taskStat: null,
    modelId: 3,
  },
  columns: () => [
    {
      key: "taskId",
      title: "任务编号",
      align: "center",
      width: 120,
    },
    { title: "数据集Id", key: "dataSetId" },
    {
      title: "版本",
      key: "version",
      width: 100,
      render: (row: any) => {
        return [h("span", {}, `v${row.version}`)];
      },
    },
    {
      title: "任务进度",
      key: "taskProgress",
      render: (row: any) => {
        const progress = row.taskProgress ? row.taskProgress.split("%")[0] : 0;
        return [
          h("div", {}, [
            h(NProgress, {
              type: "line",
              "indicator-placement": "inside",
              processing: true,
              percentage: progress,
            }),
          ]),
        ];
      },
    },
    { title: "任务状态", key: "taskStat" },
    { title: "错误原因", key: "taskException" },
    { title: "任务名称", key: "taskName" },
    {
      key: "operate",
      title: $t("common.operate"),
      align: "center",
      width: 260,
      // render: (row) => (
      //   <div class="flex-center gap-8px">
      //     <NButton
      //       type="primary"
      //       ghost
      //       size="small"
      //       onClick={() => navToExport(row)}
      //     >
      //       结果输出
      //     </NButton>
      //     <NButton
      //       type="primary"
      //       ghost
      //       size="small"
      //       onClick={() => handleDetail(row.taskId)}
      //     >
      //       任务详情
      //     </NButton>
      //   </div>
      // ),
      render: (row) => {
        const disabled =
          row.taskProgress && row.taskProgress !== "100%" ? true : false;
        return [
          h(
            "div",
            {
              class: "flex-center gap-8px",
            },
            [
              h(
                NButton,
                {
                  type: "primary",
                  ghost: true,
                  size: "small",
                  disabled,
                  onClick: () => navToExport(row),
                },
                "结果输出",
              ),
              h(
                NButton,
                {
                  type: "primary",
                  ghost: true,
                  size: "small",
                  onClick: () => handleDetail(row.taskId),
                },
                "任务详情",
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
                      "删除",
                    ),
                  default: () => h("span", {}, "你确定要删除吗？"),
                },
              ),
            ],
          ),
        ];
      },
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
const detailData = ref<any>({});
const handleCreateTask = (): void => {
  router.push({
    // name: "data-expansion_add",
    path: "/data-expansion/add",
    query: {
      name: "目标推理",
      modelId: "3",
    },
  });
  const query = {
    name: "目标推理",
    modelId: "3",
  };
  localStorage.setItem("row", JSON.stringify(query));
};
async function handleDetail(id: number | string) {
  const res = await getTaskDetail({
    taskId: id,
  });
  detailData.value = res.data;
  openModal();
}

const navToExport = (row: any) => {
  router.push({
    name: "data-expansion_exportres",
    query: {
      taskId: row.taskId,
    },
  });
};

const handleDelete = async (row) => {
  const res = await delTask({ taskId: row.taskId });
  if (res.data) {
    window.$message?.success?.("删除成功！");
    getDataByPage();
  }
};
</script>
<template>
  <div
    class="min-h-500px flex-col-stretch gap-16px overflow-hidden lt-sm:overflow-auto"
  >
    <UserSearch
      v-model:model="searchParams"
      @reset="resetSearchParams"
      @search="getDataByPage"
    />
    <NCard
      title="模型推理"
      :bordered="false"
      size="small"
      class="sm:flex-1-hidden card-wrapper"
    >
      <template #header-extra>
        <TableHeaderOperation
          v-model:columns="columnChecks"
          :disabled-delete="checkedRowKeys.length === 0"
          :loading="loading"
          :is-add="false"
          :is-del="false"
          @refresh="getData"
        >
          <template #prefix>
            <NButton
              size="small"
              @click="handleCreateTask()"
              type="primary"
              class="-mr-24px"
            >
              创建任务
            </NButton>
          </template>
        </TableHeaderOperation>
      </template>
      <NDataTable
        v-model:checked-row-keys="checkedRowKeys"
        :columns="columns"
        :data="data"
        :flex-height="!appStore.isMobile"
        :scroll-x="962"
        :loading="loading"
        remote
        :row-key="(row) => row.taskId"
        :pagination="mobilePagination"
        class="sm:h-full"
      />
      <UserOperateDrawer
        v-model:visible="drawerVisible"
        :operate-type="operateType"
        :row-data="editingData"
        @submitted="getDataByPage"
      />
      <MenuOperateModal
        v-model:visible="visible"
        :operate-type="operateType"
        :row-data="detailData"
      />
    </NCard>
  </div>
</template>

<style scoped></style>
