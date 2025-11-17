<script setup lang="tsx">
import { NButton, NPopconfirm, NProgress } from "naive-ui";
import { $t } from "@/locales";
import { useAppStore } from "@/store/modules/app";
import { useTable, useTableOperate } from "@/hooks/common/table";
import UserOperateDrawer from "./modules/user-operate-drawer.vue";
import UserSearch from "./modules/user-search.vue";
import { delTask, getTaskDetail, getTaskPage } from "@/service/api/expansion";
import { useBoolean } from "~/packages/hooks";
import MenuOperateModal from "@/views/data-expansion/sceneChange/modules/menu-operate-modal.vue";

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
    taskName: null,
    taskStat: null,
    modelId: 9,
    taskTimeArr: null,
    recordType: 3
  },
  columns: () => [
    {
      key: "taskId",
      title: "任务编号",
      align: "center",
      width: 120,
      fixed: "left",
    },
    {
      key: "taskInputName",
      title: "任务名称",
      align: "center",
      width: 180,
    },
    {
      key: "taskName",
      title: "任务类型",
      align: "center",
      width: 180
    },
    { title: "关联数据集", key: "groupVName", width: 180 },
    {
      width: 160,
      title: "任务进度",
      key: "taskProgress",
      render: (row: any) => {
        const progress = row.taskProgress ? row.taskProgress.split("%")[0] : 0;
        return [
          h("div", {}, [
            h(NProgress, {
              type: "line",
              "indicator-placement": "inside",
              processing: false,
              percentage: progress,
            }),
          ]),
        ];
      },
    },
    { title: "任务状态", key: "taskStat", align: "center", width: 120, },
    { title: "错误原因", key: "taskException", width: 220 },
    { title: "创建时间", key: "createTime", width: 180 },
    {
      key: "operate",
      title: $t("common.operate"),
      align: "center",
      width: 260,
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
const detailData = ref<any>({});
const handleCreateTask = (): void => {
  router.push({
    path: "/data-expansion/imgtask",
    query: {
      name: "图像任务",
      modelId: 9,
    },
  });
  const query = {
    name: "图像任务",
    modelId: "9",
  };
  localStorage.setItem("row", JSON.stringify(query));
};

const navToExport = (row: any) => {
  if (row.taskId) {
    router.push({
      name: "data-expansion_exportres",
      query: {
        taskId: row.taskId,
      },
    });
  }
};

const handleDelete = async (row) => {
  const res = await delTask({ taskId: row.taskId });
  if (res.data) {
    window.$message?.success?.("删除成功！");
    getDataByPage();
  }
};

async function handleDetail(id: number | string) {
  const res = await getTaskDetail({
    taskId: id,
  });
  detailData.value = res.data;
  openModal();
}

function initSocket(taskId: number | string) {
  const socket = {
    value: null,
  }
  const socketUrl = `${import.meta.env.VITE_WS_BASE_URL}/websocket/taskProgress/${taskId}`;
  socket.value = new WebSocket(socketUrl);
  socket.value.onopen = () => {
    console.log('WebSocket连接已打开');
  };
  socket.value.onmessage = (event) => {
    console.log('WebSocket')
    const obj = JSON.parse(event.data);
    // 根据obj.taskId查找data的索引
    const index = data.value.findIndex((item) => item.taskId === obj.taskId);
    if (index !== -1) {
      const progress = obj.taskProgress ? obj.taskProgress.split("%")[0] : 0;
      if (Number(progress) == 100) {
        socket.value?.close();
        getDataByPage();
      }
      data.value[index].taskProgress = progress;
    }
  };
  socket.value.onerror = (error) => {
    console.error('WebSocket错误:', error);
  };
  socket.value.onclose = () => {
    console.log('WebSocket连接已关闭');
  };
}

let timer = null;
onMounted(() => {

  getTaskPage({ page: 1, limit: 10, modelId: 9, recordType: 3 }).then((res) => {
    const list = res.data.records instanceof Array ? res.data.records : [];
    if (list.length > 0) {
      const dataList = list.filter((val) => {
        if (val?.taskProgress) {
          const progress = val.taskProgress.split("%")[0];
          return Number(progress) == 100;
        }
      });
      dataList.forEach((val: any) => {
        initSocket(val.taskId);
      })
    }
  })
});

onBeforeUnmount(() => {
  clearInterval(timer);
});
</script>

<template>
  <div class="min-h-500px flex-col-stretch gap-16px overflow-hidden lt-sm:overflow-auto">
    <UserSearch v-model:model="searchParams" @reset="resetSearchParams" @search="getDataByPage" />
    <NCard title="图像算子" :bordered="false" size="small" class="sm:flex-1-hidden card-wrapper">
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
    </NCard>
  </div>
</template>

<style scoped></style>
