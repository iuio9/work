<script setup lang="tsx">
import { NButton } from "naive-ui";
import { useAppStore } from "@/store/modules/app";
import { useTable, useTableOperate } from "@/hooks/common/table";
import {
  delExample,
  getExamplePage,
  getTaskPage,
  trainAssess,
  trainStop,
} from "@/service/api/dataManage";
import { useBoolean } from "~/packages/hooks";
import UserOperateDrawer from "./modules/user-operate-drawer.vue";
import UserSearch from "./modules/user-search.vue";
import MenuOperateModal from "./modules/menu-operate-modal.vue";
import { $t } from "@/locales";

const appStore = useAppStore();
const router = useRouter();
const route = useRoute();
const { bool: visible, setTrue: openModal } = useBoolean();
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
  sign: "id",
  apiFn: getTaskPage,
  showTotal: true,
  apiParams: {
    // isTrain: 1,
    current: 1,
    size: 10,
    taskInputName: null,
    recordType: 2,
    // modelId: route.query.modelId
  },
  columns: () => [
    {
      key: "taskId",
      title: "任务编号",
      align: "center",
      width: 80,
      fixed: "left",
      ellipsis: {
        tooltip: true
      }
    },
    {
      key: "taskInputName",
      title: "任务名称",
      align: "center",
      width: 150,
      ellipsis: {
        tooltip: true
      }
    },
    {
      key: "taskName",
      title: "任务类型",
      align: "center",
      width: 100,
      ellipsis: {
        tooltip: true
      }
    },
    {
      title: "关联数据集", key: "groupVName",
      align: "center",
      width: 150,
      ellipsis: {
        tooltip: true
      }
    },
    {
      title: "任务状态", key: "taskStat", align: "center",
      width: 120,
      ellipsis: {
        tooltip: true
      }
    },
    {
      title: "创建时间", key: "createTime",
      align: "center",
      width: 180,
      ellipsis: {
        tooltip: true
      }
    },
    {
      key: "operate",
      title: "操作",
      align: "center",
      width: 240,
      fixed: "right",
      render: (row: any) => {
        // const isOperate = Boolean(row.isDelete);
        const isTrainSuccess = Boolean(!!row.status ? row.status : false);
        return [
          h("div", { class: "flex-center gap-8px flex-wrap" }, [
            h(
              NButton,
              {
                disabled: isTrainSuccess,
                type: "primary",
                ghost: true,
                size: "small",
                // onClick: () => handleOperate("assess", row),
                onClick: () => handleOperate("trainResult", row),
              },
              "评估详情",
            ),
          ]),
        ];
      },
    },
  ],
});

const taskId = ref();
const trainType = ref(0);

const {
  drawerVisible,
  operateType,
  editingData,
  checkedRowKeys,
  onBatchDeleted,
  onDeleted,
  // closeDrawer
} = useTableOperate(data, getData);

async function handleBatchDelete() {
  // request
  const res = await delExample(checkedRowKeys.value);
  if (res.data >= 1 || !res.data) {
    onBatchDeleted();
  }
}

async function handleDelete(id: number) {
  // request
  const res = await delExample([id]);
  if (res.data >= 1 || !res.data) {
    onDeleted();
  }
}

const mapEditingData = (data: any) => {
  return { ...data, modelId: route.query.modelId };
};

function handleAdd() {
  operateType.value = "add";
  openModal();
}

function handleEdit(item: any) {
  item.query = item.params
    ? JSON.parse(item.params).map((val) => {
        return {
          // key: val.value,
          // value: val.label,
          ...val,
          key: val.serverKey, // val.value
          value: val.value ? val.value : null, // val.label
          type: val.type,
          valuePlaceholder: val.label,
        };
      })
    : [];
  item.responseQuery = item.responseParams
    ? JSON.parse(item.responseParams).map((val) => {
        return {
          // key: val.value,
          // value: val.label,
          ...val,
          key: val.serverKey, // val.value
          value: val.value ? val.value : null, // val.label
          type: val.type,
          valuePlaceholder: val.label,
        };
      })
    : [];
  operateType.value = "edit";
  editingData.value = { ...item };
  openModal();
}

const handleTrain = async (row: any) => {
  router.push({
    // name: "data-expansion_add",
    name: "model-manage_config",
  });
  const query = {
    id: row.id,
    modelId: row.modelId,
    params: row.params,
  };
  localStorage.setItem("row", JSON.stringify(query));
};

const navToExport = (row: any) => {
  router.push({
    name: "data-expansion_exportres",
    query: {
      taskId: row.taskId,
    },
  });
};

// 模型训练
const handleModelAssess = () => {
  // router.push({
  //   name: "model-manage_config",
  // });
  openModal();
};

// 任务进度
async function handleOperate(sign: string, row: any) {
  if (sign === "task") {
    taskId.value = row.taskId;
    trainType.value = row.trainType ?? 0;
    openModal();
  }
  if (sign === "detail") {
    if (row.trainUrl) {
      window.open(`${row.trainUrl}`, "_blank");
    }
  }
  // 评估
  if (sign === "assess") {
    const res = await trainAssess({ taskId: row.taskId });
    if (res.data) {
      const { status, result } = res.data;
      assessStatus.value = status;
      assessStr.value = result;
      assessShowModel.value = true;
    }
  }
  // 结束训练
  if (sign === "back") {
    const res = await trainStop({ taskId: row.taskId });
    console.log(res);
    if (res.data) {
      window.$message?.success?.("任务结束！");
    }
  }
  if(sign === "trainResult") {
    currentRow.value = row;
    trainResultShowModal.value = true;
  }
}

// socket 任务进度
const items = ref<any>([]);
const logSocket = ref<WebSocket>();

const sItems = ref<any>([]);
const sSocket = ref<WebSocket>();

const assessShowModel = ref<any>(false);
const assessStatus = ref<any>(0);
const assessStr = ref<any>("");

//  查看训练评估
const trainResultShowModal = ref<Boolean>();
const currentRow = ref<any>(null);
// 参数选择
const assParOptions = ref<any>([
  {
    value: "Rock'n'Roll Star",
    label: '训练参数1'
  },
  {
    value: 'Shakermaker',
    label: '训练参数2'
  },
  {
    value: 'Live Forever',
    label: '训练参数3'
  },
  {
    value: 'Up in the Sky',
    label: '训练参数4'
  },
]);
async function handleDefine() {
  await handleOperate("assess", currentRow.value);
}
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
      title="模型训练"
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
          @add="handleAdd"
          @delete="handleBatchDelete"
          @refresh="getData"
        >
          <template #prefix>
            <NButton
              size="small"
              @click="handleModelAssess()"
              type="primary"
              class="-mr-24px"
            >
              模型评估
            </NButton>
          </template>
        </TableHeaderOperation>
      </template>
      <NDataTable
        v-model:checked-row-keys="checkedRowKeys"
        :columns="columns"
        :data="data"
        size="small"
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
        :row-data="mapEditingData(editingData)"
        @submitted="getDataByPage"
      />
      <MenuOperateModal
        v-model:visible="visible"
        :taskId="taskId"
        :train-type="trainType"
        :items="items"
        :sItems="sItems"
        @submitted="getDataByPage"
      />
    </NCard>
    <!--评估-->
    <n-modal v-model:show="assessShowModel">
      <n-card
        style="width: 600px"
        title="模型评估"
        :bordered="false"
        size="huge"
        role="dialog"
        aria-modal="true"
      >
        <div class="wrap_content">
          <div class="w-full h-auto min-h-200px" v-if="assessStatus == 0">
            <span>{{ assessStr }}</span>
          </div>
          <div class="w-full h-auto min-h-200px" v-if="assessStatus == 1">
            <img :src="assessStr" alt="" class="w-full h-auto" />
          </div>
        </div>
        <template #footer>
          <NSpace justify="end" :size="16">
            <NButton @click="() => (assessShowModel = false)">{{
              $t("common.cancel")
            }}</NButton>
          </NSpace>
        </template>
      </n-card>
    </n-modal>
    <!-- 训练评估 -->
    <NModal v-model:show="trainResultShowModal">
      <NCard style="width: 600px" title="参数选择" :bordered="false" size="huge" role="dialog" aria-modal="true">
        <div class="wrap_content">
          <n-checkbox-group v-model:value="value">
            <n-space item-style="display: flex;">
              <n-checkbox  v-for="song in assParOptions" :key="song.value" :value="song.value" :label="song.label" />
            </n-space>
          </n-checkbox-group>
        </div>
        <template #footer>
          <NSpace justify="end" :size="16">
            <NButton type="primary" @click="handleDefine()">确认展示</NButton>
            <NButton @click="() => (trainResultShowModal = false)">取消返回</NButton>
          </NSpace>
        </template>
      </NCard>
    </NModal>
  </div>
</template>

<style scoped></style>
