<script setup lang="tsx">
import { NButton, NPopconfirm } from 'naive-ui';
import { useAppStore } from '@/store/modules/app';
import { useTable, useTableOperate } from '@/hooks/common/table';
import { delExample, getExamplePage, getTaskPage, trainAssess, trainStop } from '@/service/api/dataManage';
import { useBoolean } from '~/packages/hooks';
import { $t } from '@/locales';
import { useWebSocketStore } from '@/store/modules/websocket';
import { delTask, getDataSetListNoPage } from '@/service/api/expansion';
import UserOperateDrawer from './modules/user-operate-drawer.vue';
import UserSearch from './modules/user-search.vue';
import MenuOperateModal from './modules/menu-operate-modal.vue';

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
  resetSearchParams
} = useTable({
  sign: 'id',
  apiFn: getTaskPage,
  showTotal: true,
  apiParams: {
    isTrain: 1,
    current: 1,
    size: 10,
    // if you want to use the searchParams in Form, you need to define the following properties, and the value is null
    // the value can not be undefined, otherwise the property in Form will not be reactive
    algorithmName: null,
    taskInputName: null,
    modelName: null,
    recordType: 1
    // modelId: route.query.modelId
  },
  columns: () => [
    {
      key: 'taskId',
      title: '任务编号',
      align: 'center',
      width: 80,
      ellipsis: {
        tooltip: true
      },
      fixed: "left",
    },
    {
      key: 'taskInputName',
      title: '任务名称',
      align: 'center',
      width: 180,
      ellipsis: {
        tooltip: true
      }
    },
    {
      key: 'modelName',
      title: '模型名称',
      align: 'center',
      width: 140,
      ellipsis: {
        tooltip: true
      }
    },
    {
      key: 'modelVersion',
      title: '训练版本',
      align: 'center',
      width: 100,
      ellipsis: {
        tooltip: true
      }
    },
    {
      key: 'taskName',
      title: '任务类型',
      align: 'center',
      width: 100,
      ellipsis: {
        tooltip: true
      }
    },
    {
      title: '关联数据集', key: 'groupVName', width: 120,
      ellipsis: {
        tooltip: true
      }
    },
    {
      title: '任务状态', key: 'taskStat', align: 'center', width: 100,
      ellipsis: {
        tooltip: true
      }
    },
    // { title: "错误原因", key: "taskException", width: 220 },
    {
      title: '创建时间', key: 'createTime', width: 150,
      ellipsis: {
        tooltip: true
      }
    },
    {
      title: '结束时间', key: 'updateTime', width: 150,
      ellipsis: {
        tooltip: true
      }
    },
    {
      key: 'operate',
      title: '操作',
      align: 'center',
      width: 320,
      fixed: "right",
      render: (row: any) => {
        // const isOperate = Boolean(row.isDelete);
        const isTrainSuccess = Boolean(row.status ? row.status : false);
        const isStatus = row.taskStat === '进行中';
        // 结束
        const isFinish = row.taskStat.includes('结束');
        return [
          h('div', { class: 'flex-center gap-8px flex-wrap' }, [
            h(
              NButton,
              {
                disabled: !isStatus || isTrainSuccess,
                type: 'primary',
                ghost: true,
                size: 'small',
                style: {
                  display: isStatus ? 'block' : 'none'
                },
                onClick: () => handleOperate('task', row, 'success')
              },
              '正常控制台'
            ),
            h(
              NButton,
              {
                disabled: !isStatus || isTrainSuccess,
                type: 'error',
                ghost: true,
                size: 'small',
                style: {
                  display: row.trainType == 0 || !isStatus ? 'none' : 'block'
                },
                onClick: () => handleOperate('task', row, 'error')
              },
              '异常控制台'
            ),
            h(
              NButton,
              {
                disabled: !isFinish,
                type: 'primary',
                ghost: true,
                size: 'small',
                style: {
                  display: !isFinish ? 'none' : 'block'
                },
                onClick: () => handleOperate('detail', row)
              },
              '查看详情'
            ),
            // h(
            //   NButton,
            //   {
            //     disabled: isTrainSuccess,
            //     type: "primary",
            //     ghost: true,
            //     size: "small",
            //     onClick: () => handleOperate("assess", row),
            //   },
            //   "任务评估",
            // ),
            h(
              NButton,
              {
                disabled: isFinish,
                type: 'primary',
                ghost: true,
                size: 'small',
                style: {
                  display: isFinish ? 'none' : 'block'
                },
                onClick: () => handleOperate('back', row)
              },
              '中止训练'
            ),
            h(
              NPopconfirm,
              {
                onPositiveClick: () => handleDelete(row)
              },
              {
                trigger: () =>
                  h(
                    NButton,
                    {
                      type: 'primary',
                      ghost: true,
                      size: 'small'
                    },
                    '删除'
                  ),
                default: () => h('span', {}, '你确定要删除吗？')
              }
            )
          ])
        ];
      }
    }
  ]
});

const taskId = ref();
const trainType = ref(0);

const {
  drawerVisible,
  operateType,
  editingData,
  checkedRowKeys,
  onBatchDeleted,
  onDeleted
  // closeDrawer
} = useTableOperate(data, getData);

const assessShowModel = ref<any>(false);
const assessStatus = ref<any>(0);
const assessStr = ref<any>('');

async function handleBatchDelete() {
  // request
  const res = await delExample(checkedRowKeys.value);
  if (res.data >= 1 || !res.data) {
    onBatchDeleted();
  }
}

const handleDelete = async row => {
  const res = await delTask({ taskId: row.taskId });
  if (res.data) {
    window.$message?.success?.('删除成功！');
    getDataByPage();
  }
};

const mapEditingData = (data: any) => {
  return { ...data, modelId: route.query.modelId };
};

function handleAdd() {
  operateType.value = 'add';
  openModal();
}

function handleEdit(item: any) {
  item.query = item.params
    ? JSON.parse(item.params).map(val => {
        return {
          // key: val.value,
          // value: val.label,
          ...val,
          key: val.serverKey, // val.value
          value: val.value ? val.value : null, // val.label
          type: val.type,
          valuePlaceholder: val.label
        };
      })
    : [];
  item.responseQuery = item.responseParams
    ? JSON.parse(item.responseParams).map(val => {
        return {
          // key: val.value,
          // value: val.label,
          ...val,
          key: val.serverKey, // val.value
          value: val.value ? val.value : null, // val.label
          type: val.type,
          valuePlaceholder: val.label
        };
      })
    : [];
  operateType.value = 'edit';
  editingData.value = { ...item };
  openModal();
}

const handleTrain = async (row: any) => {
  router.push({
    // name: "data-expansion_add",
    name: 'model-manage_config'
  });
  const query = {
    id: row.id,
    modelId: row.modelId,
    params: row.params
  };
  localStorage.setItem('row', JSON.stringify(query));
};

const navToExport = (row: any) => {
  router.push({
    name: 'data-expansion_exportres',
    query: {
      taskId: row.taskId
    }
  });
};

// 模型训练
const handleModelTrain = () => {
  webSocketStore.reset();
  router.push({
    name: 'model-manage_config'
  });
};

const handleModelProgress = () => {
  openModal();
};

// 任务进度
async function handleOperate(sign: string, row: any, type: string) {
  if (sign === 'task') {
    taskId.value = row.taskId;
    // trainType.value = row.trainType ?? 0;
    trainType.value = type === 'success' ? 0 : 1;
    openModal();
  }
  if (sign === 'detail') {
    if (row.trainUrl) {
      window.open(`${row.trainUrl}`, '_blank');
    }
  }
  // 评估
  if (sign === 'assess') {
    const res = await trainAssess({ taskId: row.taskId });
    if (res.data) {
      const { status, result } = res.data;
      assessStatus.value = status;
      assessStr.value = result;
      assessShowModel.value = true;
    }
  }
  // 结束训练
  if (sign === 'back') {
    const res = await trainStop({ taskId: row.taskId });
    if (res.data) {
      window.$message?.success?.('任务结束！');
      webSocketStore.reset();
      getDataByPage();
    }
  }
}

// socket 任务进度
const items = ref<any>([]);
const logSocket = ref<WebSocket>();
const sItems = ref<any>([]);
const sSocket = ref<WebSocket>();
const webSocketStore = useWebSocketStore();

onMounted(() => {
  // ----------------------error 控制台--------------------------------
  // const socketUrl = `${import.meta.env.VITE_WS_BASE_URL}/websocket/exceptionTerminalProgress/1`;
  // logSocket.value = new WebSocket(socketUrl);
  // logSocket.value.onopen = () => {
  //   console.log("WebSocket连接已打开");
  // };
  // logSocket.value.onmessage = (event) => {
  //   // console.log("🚀 ~ onMounted ~ event error异常:", event);
  //   items.value = [...items.value, event.data.trim()];
  // };
  // logSocket.value.onerror = (error) => {
  //   console.error("WebSocket错误:", error);
  // };
  // logSocket.value.onclose = () => {
  //   console.log("WebSocket连接已关闭");
  // };
  // ----------------------success 控制台---------------------------------
  // const socketUrl0 = `${import.meta.env.VITE_WS_BASE_URL}/websocket/terminalProgress/1`;
  // sSocket.value = new WebSocket(socketUrl0);
  // sSocket.value.onopen = () => {
  //   console.log("WebSocket连接已打开");
  // };
  // sSocket.value.onmessage = (event) => {
  //   console.log("🚀 ~ onMounted ~ event success成功:", event);
  //   sItems.value = [...sItems.value, event.data.trim()];
  // };
  // sSocket.value.onerror = (error) => {
  //   console.error("WebSocket错误:", error);
  // };
  // sSocket.value.onclose = () => {
  //   console.log("WebSocket连接已关闭");
  // };

  // ----------pinia socket-----------
  webSocketStore.reset();
  webSocketStore.connect();
  // items.value = webSocketStore.items;
  // sItems.value = webSocketStore.sItems;
});

onUnmounted(() => {
  logSocket.value?.close();
  sSocket.value?.close();
});
</script>

<template>
  <div class="min-h-500px flex-col-stretch gap-16px overflow-hidden lt-sm:overflow-auto">
    <UserSearch v-model:model="searchParams" @reset="resetSearchParams" @search="getDataByPage" />
    <NCard title="模型训练" :bordered="false" size="small" class="sm:flex-1-hidden card-wrapper">
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
            <NButton size="small" type="primary" class="-mr-24px" @click="handleModelTrain()">创建任务</NButton>
            <!--
<NButton
              size="small"
              @click="handleModelProgress()"
              type="primary"
              class="-mr-24px ml-24px"
            >
              任务进度
            </NButton>
-->
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
        :row-key="row => row.modelId"
        :pagination="mobilePagination"
        class="sm:h-full"
      />
      <UserOperateDrawer
        v-model:visible="drawerVisible"
        :operate-type="operateType"
        :row-data="mapEditingData(editingData)"
        @submitted="getDataByPage"
      />
      <!--
 <MenuOperateModal v-model:visible="visible" :taskId="taskId" :train-type="trainType" :items="items"
        :sItems="sItems" @submitted="getDataByPage" />
-->
      <!-- pinia socket  -->
      <MenuOperateModal
        v-model:visible="visible"
        :task-id="taskId"
        :train-type="trainType"
        :items="webSocketStore.items"
        :s-items="webSocketStore.sItems"
        @submitted="getDataByPage"
      />
    </NCard>
    <!--评估-->
    <NModal v-model:show="assessShowModel">
      <NCard style="width: 600px" title="模型评估" :bordered="false" size="huge" role="dialog" aria-modal="true">
        <div class="wrap_content">
          <div v-if="assessStatus == 0" class="h-auto min-h-200px w-full">
            <span>{{ assessStr }}</span>
          </div>
          <div v-if="assessStatus == 1" class="h-auto min-h-200px w-full">
            <img :src="assessStr" alt="" class="h-auto w-full" />
          </div>
        </div>
        <template #footer>
          <NSpace justify="end" :size="16">
            <NButton @click="() => (assessShowModel = false)">{{ $t('common.cancel') }}</NButton>
          </NSpace>
        </template>
      </NCard>
    </NModal>
  </div>
</template>

<style scoped lang="scss">
::-webkit-scrollbar-button {
  background-color: #ccc;
}
</style>
