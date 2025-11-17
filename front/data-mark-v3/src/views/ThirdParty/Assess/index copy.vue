<script setup lang="tsx">
import {NButton, NPopconfirm, NProgress} from 'naive-ui';
import {useAppStore} from '@/store/modules/app';
import {useTable, useTableOperate} from '@/hooks/common/table';
import {delExample, getExamplePage, getTaskPage, trainAssess, trainStop} from '@/service/api/dataManage';
import {useBoolean} from '~/packages/hooks';
import {$t} from '@/locales';
import {useWebSocketStore} from '@/store/modules/websocket';
import {delTask, getDataSetListNoPage} from '@/service/api/expansion';
import UserSearch from './modules/user-search.vue';
import assReport from '@/assets/imgs/assReport.png';
import assessDetail from '@/assets/imgs/assessDetail.webp';
import assess0 from "@/assets/imgs/assess0.png";
import assess1 from "@/assets/imgs/assess1.png";
import assess2 from "@/assets/imgs/assess2.png";
import assess3 from "@/assets/imgs/assess3.png";
import ReportDetailModal from "./modules/ReportDetailModal.vue";

const appStore = useAppStore();
const router = useRouter();
const route = useRoute();
const {bool: visible, setTrue: openModal} = useBoolean();
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
    algorithmName: null,
    taskInputName: null,
    modelName: null,
    recordType: 1
    // modelId: route.query.modelId
  },
  columns: () => [
    {
      key: 'taskId',
      title: '编号',
      align: 'center',
      width: 50,
      fixed: 'left'
    },
    {
      key: 'taskInputName',
      title: '任务名称',
      align: 'center',
      width: 120,
      ellipsis: {
        tooltip: true
      },
      fixed: 'left'
    },
    {
      key: 'modelName',
      title: '模型名称',
      align: 'center',
      width: 120,
      ellipsis: {
        tooltip: true
      },
      fixed: 'left'
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
      title: "任务进度",
      key: "taskProgress",
      width: 160,
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
    {
      title: '任务状态',
      key: 'taskStat',
      align: 'center',
      width: 100,
      ellipsis: {
        tooltip: true
      }
    },
    {
      title: '关联数据集',
      key: 'groupVName',
      width: 120,
      ellipsis: {
        tooltip: true
      }
    },
    {title: '错误原因', key: 'taskException', width: 220},
    {
      title: '创建时间',
      key: 'createTime',
      width: 150,
      ellipsis: {
        tooltip: true
      }
    },
    // {
    //   title: '结束时间',
    //   key: 'updateTime',
    //   width: 150,
    //   ellipsis: {
    //     tooltip: true
    //   }
    // },
    {
      key: 'operate',
      title: '操作',
      align: 'center',
      width: 340,
      fixed: 'right',
      render: (row: any) => {
        const isTrainSuccess = Boolean(row.status ? row.status : false);
        // 结束
        const isFinish = row.taskStat.includes('结束');
        return [
          h('div', {class: 'flex-center gap-8px flex-wrap'}, [
            // h(
            //   NButton,
            //   {
            //     disabled: isTrainSuccess,
            //     type: 'primary',
            //     ghost: true,
            //     size: 'small',
            //     onClick: () => handleOperate('result', row)
            //   },
            //   '运行结果'
            // ),
            // h(
            //   NButton,
            //   {
            //     disabled: !isFinish,
            //     type: 'primary',
            //     ghost: true,
            //     size: 'small',
            //     style: {
            //       display: !isFinish ? 'none' : 'block'
            //     },
            //     onClick: () => handleOperate('detail', row)
            //   },
            //   '模型评估'
            // ),
            h(
              NButton,
              {
                disabled: isTrainSuccess,
                type: 'primary',
                ghost: true,
                size:'small',
                onClick: () => handleOperate('stop', row)
              },
              '暂停'
            ),
            h(
              NButton,
              {
                disabled: isTrainSuccess,
                type: 'primary',
                ghost: true,
                size:'small',
                onClick: () => handleOperate('stop', row)
              },
              '终止'
            ),
            h(
              NButton,
              {
                disabled: isTrainSuccess,
                type: 'primary',
                ghost: true,
                size:'small',
                onClick: () => handleOperate('restart', row)
              },
              '重新开始'
            ),
            h(
              NButton,
              {
                disabled: isTrainSuccess,
                type: 'primary',
                ghost: true,
                size: 'small',
                onClick: () => handleOperate('view', row)
              },
              '查看评估报告'
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
                onClick: () => handleOperate('report', row)
              },
              '生成评估报告'
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

const reportShowModal = ref<any>(false);
const detailShowModal = ref<any>(false);
const assessStatus = ref<any>(0);
const assessStr = ref<any>('');

// 使用流程
const assessTitle = ref<string>("使用流程");
const aInfoList = ref<any>([
  {
    name: '评估数据准备',
    info: '准备用于评估模型能力的数据集,并在 数据集管理 中导入和发布',
    btns: [],
    icon: 'data-collect',
    imgSrc: assess0
  },
  {
    name: '模型结果生成',
    info: '使用所选数据集，批量生成模型推理结果，以便进行下一步打分',
    btns: [],
    icon: 'data-qc',
    imgSrc: assess1
  },
  // {
  //   name: '人工在线评估',
  //   info: '点击评估任务操作栏「在线评估」按钮，对模型结果进行多维度人工评估',
  //   btns: [],
  //   icon: 'data-intellect',
  //   imgSrc: assess2
  // },
  {
    name: '评估指标计算',
    info: '根据所选自动评估方法，自动对推理结果进行评分，并汇总计算评估指标、产出评估报告',
    btns: [],
    icon: 'data-annotation',
    imgSrc: assess3
  }
]);
const aOpBtnList = ref<any>([
  {icon: 'material-symbols:add', name: '新增版本'},
  {icon: 'material-symbols-light:border-all', name: '所有版本'},
  {icon: 'icon-park-outline:merge', name: '合并版本'},
  {icon: 'material-symbols-light:delete', name: '删除'}
]);

const isModalVisible = ref(false);

const selectedTask = {
  name: '任务一',
  description: '这是一个示例任务的描述信息。',
  versionName: 'v1.0',
  versionDescription: '这是第一个版本的描述。',
  evaluationDataset: '数据集A',
  evaluationModel: '模型X',
  indicators: ["1", "2", "3", "4", "5", "6"]
};

function handleAdd() {
  operateType.value = 'add';
  openModal();
}

async function handleBatchDelete() {
  // request
  const res = await delExample(checkedRowKeys.value);
  if (res.data >= 1 || !res.data) {
    onBatchDeleted();
  }
}

const handleDelete = async row => {
  const res = await delTask({taskId: row.taskId});
  if (res.data) {
    window.$message?.success?.('删除成功！');
    getDataByPage();
  }
};

const mapEditingData = (data: any) => {
  return {...data, modelId: route.query.modelId};
};

// 模型训练
const handleCreateTask = () => {
  webSocketStore.reset();
  router.push({
    name: 'thirdparty_createtask'
  });
};

type OperateType = 'result' | 'detail' | 'report' | 'view' | 'stop' | 'start' | 'restart';

// switch
async function handleOperate(type: OperateType, row: any) {
  if (type === 'result') {
    router.push({
      name: 'thirdparty_info',
      query: {
        taskId: row.taskId,
        modelId: '3'
      }
    });
  }
  if (type === 'detail') {
    detailShowModal.value = true;
  }
  if (type === 'report') {
    isModalVisible.value = true;
  }
  if (type === 'view') {
    router.push({
      name: "thirdparty_info", query: { taskId: row.taskId }
    })
  }
}

// socket 任务进度
const logSocket = ref<WebSocket>();
const sSocket = ref<WebSocket>();
const webSocketStore = useWebSocketStore();

onMounted(() => {
  webSocketStore.reset();
  webSocketStore.connect();
});

onUnmounted(() => {
  logSocket.value?.close();
  sSocket.value?.close();
});
</script>

<template>
  <div class="min-h-500px flex-col-stretch gap-16px overflow-hidden lt-sm:overflow-auto">
    <NCard
      :bordered="false"
      size="small"
      class="card-wrapper"
    >
      <NCollapse>
        <NCollapseItem :title="assessTitle" name="user-search">
          <NFlex justify="space-between" class="wrap-container">
            <div
              v-for="(item, index) of aInfoList"
              :key="index"
              class="item-manage flex justify-center items-center"
            >
              <div class="item_main w-full">
                <div class="item-manage_icon">
                  <img :src="item.imgSrc" alt="" class="w-35%">
                  <div class="iconName">{{ item.name }}</div>
                </div>
                <div class="item-manage_info w-full flex justify-center items-center">{{ item.info }}</div>
                <div class="item-manage_btnC">
                  <NButton
                    v-for="(val, idx) of item.btns"
                    :key="idx"
                    quaternary
                    type="info"
                    @click="navTo(val)"
                  >
                    {{ val.name }}
                  </NButton>
                </div>
              </div>
              <div class="item_arrow" v-if="index !== aInfoList.length - 1">
                <div class="flow-arrow"><span class="aibp-custom-icon aibp-custom-icon-arrow">
                  <svg width="24" height="24"><path fill="#B8BABF" d="m8.053 3 9.192 9.192L8 21.437v-5.253l3.79-3.79L8 8.603V3.052L8.053 3Z"></path></svg></span>
                </div>
              </div>
            </div>
          </NFlex>
        </NCollapseItem>
      </NCollapse>
    </NCard>
    <UserSearch v-model:model="searchParams" @reset="resetSearchParams" @search="getDataByPage"/>
    <NCard title="评估列表" :bordered="false" size="small" class="sm:flex-1-hidden card-wrapper">
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
            <NButton size="small" type="primary" class="-mr-24px" @click="handleCreateTask()">创建评估任务</NButton>
          </template>
        </TableHeaderOperation>
      </template>
      <NDataTable
        v-model:checked-row-keys="checkedRowKeys"
        :columns="columns"
        :data="data"
        size="small"
        :flex-height="!appStore.isMobile"
        :scroll-x="1800"
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
    <!-- 评估报告 modal -->
    <NModal v-model:show="reportShowModal">
      <NCard style="width: 600px" title="评估报告" :bordered="false" size="huge" role="dialog" aria-modal="true">
        <div class="wrap_content">
          <img :src="assReport" alt="" class="w-full h-660px"/>
        </div>
        <template #footer>
          <NSpace justify="end" :size="16">
            <NButton type="primary">下载报告</NButton>
            <NButton @click="() => (reportShowModal = false)">关闭窗口</NButton>
          </NSpace>
        </template>
      </NCard>
    </NModal>
    <!-- 评估详情 modal -->
    <NModal v-model:show="detailShowModal">
      <NCard style="width: 600px" title="评估详情" :bordered="false" size="huge" role="dialog" aria-modal="true">
        <div class="wrap_content">
          <img :src="assessDetail" alt="" class="w-full h-660px"/>
        </div>
        <template #footer>
          <NSpace justify="end" :size="16">
            <NButton @click="() => (detailShowModal = false)">关闭窗口</NButton>
          </NSpace>
        </template>
      </NCard>
    </NModal>
    <!--生成评估报告-->
    <ReportDetailModal v-model:show="isModalVisible" :task="selectedTask" />
  </div>
</template>

<style scoped lang="scss">
::-webkit-scrollbar-button {
  background-color: #ccc;
}

.card-wrapper {
  border-radius: 8px;
}

.item-manage {
  flex: 1;

  .item-manage_icon {
    display: flex;
    flex-direction: column;
    justify-content: center;
    flex-wrap: wrap;
    align-items: center;

    .iconName {
      font-size: 14px;
      color: #151b26;
      line-height: 22px;
      margin: 10px 0 8px;
      text-align: center;
    }
  }

  .item-manage_info {
    font-size: 12px;
    color: #84868c;
    line-height: 20px;
    margin-bottom: 8px;
  }

  .item-manage_btnC {
    display: flex;
    justify-content: center;
    align-items: center;

    .btn {
      color: #2468f2;
      font-size: 12px;
    }
  }
}
</style>
