<script setup lang="tsx">
import { NButton, NPopconfirm, NPopover, NTag } from 'naive-ui';
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
import { resolveDirective, withDirectives } from 'vue';
import { getToken } from '@/store/modules/auth/shared';
import { downloadFile } from '@/utils/util';
import { getReportTaskList } from '@/service/api/third';

const appStore = useAppStore();
const router = useRouter();
const route = useRoute();
const { bool: visible, setTrue: openModal } = useBoolean();
enum TaskStatus {
  PENDING,     // 0
  RUNNING,     // 1
  COMPLETED,   // 2
  FAILED,      // 3
  TERMINATED,  // 4
  CONTINUED,    // 5
  PROCESSING
}


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
  apiFn: getReportTaskList,
  showTotal: true,
  apiParams: {
    isTrain: 1,
    current: 1,
    size: 10,
    algorithmName: null,
    taskName: null,
    modelName: null,
    recordType: 1
  },
  columns: () => [
    {
      key: 'applyForNum',
      title: '申请号',
      width: 240,
      fixed: 'left',
      rowSpan: (rowData, rowIndex) => getRowSpanForApplyForNum(rowData, rowIndex),
      render(row) {
        const taskTypeMap: any = {
          1: '测试',
          2: '评估',
        };
        const TaskStatusText = {
          [TaskStatus.PENDING]: '待评估',
          [TaskStatus.RUNNING]: '执行中',
          [TaskStatus.COMPLETED]: '已完成',
          [TaskStatus.FAILED]: '任务失败',
          [TaskStatus.TERMINATED]: '终止',
          [TaskStatus.CONTINUED]: '继续',
          // 待处理
          [TaskStatus.PROCESSING]: '待处理',
        };
        return [
          h(
            NPopover,
            { trigger: 'hover', placement: 'right' },
            {
              trigger: () => h(NTag, {
                class: 'cursor-pointer',
                type: 'success'
              }, row.applyForNum),
              default: () => h(
                'div',
                {
                  style: {
                    width: 'auto',
                    whiteSpace: 'normal',
                    wordBreak: 'break-all'
                  }
                },
                [
                  h('p', { style: { marginBottom: '8px' } }, `申请号: ${row.applyForNum}`),
                  h('p', { style: { marginBottom: '8px' } }, `申请单位: ${row.buildUnitName}`),
                  h('p', { style: { marginBottom: '8px' } }, `申请类型: ${taskTypeMap[row.applyForType]}`),
                  h('p', { style: { marginBottom: '8px' } }, `所属业务: ${row.btUnitName}`),
                  h('p', { style: { marginBottom: '8px' } }, `模型名称: ${row.modelName}`),
                  h('p', { style: { marginBottom: '8px' } }, `模型类型: ${row.modelType}`),
                  h('p', { style: { marginBottom: '8px' } }, `申请时间: ${row.applyForTime}`),
                  h('p', { style: { marginBottom: '8px' } }, `任务状态: ${TaskStatusText[row.taskStatus]}`)
                ]
              )
            }
          )
        ]
      }
    },
    {
      type: "selection"
    },
    {
      key: 'modelName',
      title: '模型名称',
    },
    {
      key: 'taskName',
      title: '任务名称',
    },
    {
      key: 'taskType',
      title: '任务类型',
      render: (row: any) => {
        const taskTypeMap: any = {
          1: '分类任务',
          2: '目标检测',
        };
        return taskTypeMap[row.taskType] || '未知';
      }
    },
    {
      key: 'taskProgress',
      title: '任务进度',
    },
    {
      key: 'taskStatus',
      title: '任务状态',
      render: (row: any) => {
        const TaskStatusText = {
          [TaskStatus.PENDING]: '待评估',
          [TaskStatus.RUNNING]: '执行中',
          [TaskStatus.COMPLETED]: '已完成',
          [TaskStatus.FAILED]: '任务失败',
          [TaskStatus.TERMINATED]: '终止',
          [TaskStatus.CONTINUED]: '继续',
          // 待处理
          [TaskStatus.PROCESSING]: '待处理',
        };
        return TaskStatusText[row.taskStatus] || '未知状态';
      }
    },
    {
      key: 'relatedDataset',
      title: '关联数据集',
    },
    // {
    //   title: '任务操作',
    //   key: 'taskOperation',
    //   width: 200,
    //   fixed: 'right',
    //   render(row) {
    //     const isDisabled = row?.taskStatus !== 2;
    //     const authDir = resolveDirective('hasPermi');
    //     const permission = `thirdparty:report:exportTask`;
    //     return withDirectives(h(NButton, {
    //       type: 'info', quaternary: false, loading: row?.loading ?? false,
    //       onClick: () => exportFile('task', row),
    //       disabled: isDisabled
    //     }, '导出任务评估报告'), [
    //       [
    //         authDir,
    //         permission
    //       ]
    //     ])
    //   }
    // },
    {
      title: '结果对比',
      key: 'taskResult',
      width: 200,
      fixed: 'right',
      render(row) {
        const authDir = resolveDirective('hasPermi');
        const permission = `thirdparty:report:result`;
        return withDirectives(h(NButton, {
          type: 'info', quaternary: false, loading: row?.loading ?? false,
          onClick: () => navToAno(row),
        }, '结果对比'), [
          [
            authDir,
            permission
          ]
        ])
      }
    },
    {
      title: '操作',
      key: 'operation',
      width: 200,
      fixed: 'right',
      rowSpan: (rowData, rowIndex) => getRowSpanForApplyForNum(rowData, rowIndex),
      render(row) {
        const authDir = resolveDirective('hasPermi');
        const permission = `thirdparty:report:exportApply`;
        return withDirectives(h(NButton, {
          type: 'info', quaternary: false, loading: row?.applyLoading ?? false,
          onClick: () => exportFile('apply', row)
        }, '导出申请号评估报告'), [
          [
            authDir,
            permission
          ]
        ])
      }
    }
  ]
});


// 模拟多条数据
const tableData = ref([
  {
    applyForNum: '20250522326',
    modelName: '输电模型',
    modelType: 'CNN(卷积神经网络模型)',
    buildUnitName: '天津国网天津市电力公司信息通信公司',
    btUnitName: '北京中电普华信息技术有限公司',
    applyForType: '模型新建',
    id: 1,
    taskName: '模型训练任务',
    taskType: '训练',
    taskProgress: '50%',
    taskStatus: '进行中',
    relatedDataset: '数据集A',
    taskId: 1
  },
  {
    applyForNum: '20250522326',
    modelName: '输电模型',
    modelType: 'CNN(卷积神经网络模型)',
    buildUnitName: '天津国网天津市电力公司信息通信公司',
    btUnitName: '北京中电普华信息技术有限公司',
    applyForType: '模型新建',
    id: 2,
    taskName: '模型评估任务',
    taskType: '评估',
    taskProgress: '30%',
    taskStatus: '未开始',
    relatedDataset: '数据集B'
  },
  {
    applyForNum: '20250521893',
    modelName: '基于大模型的现场作业智能管控场景模型',
    modelType: '图像识别',
    buildUnitName: '国网天津信通公司',
    btUnitName: '国电南瑞南京控制系统有限公司',
    applyForType: '模型优化',
    id: 3,
    taskName: '模型部署任务',
    taskType: '部署',
    taskProgress: '80%',
    taskStatus: '进行中',
    relatedDataset: '数据集C'
  }
]);

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

// 合并单元格方法
const getRowSpanForApplyForNum = (rowData, rowIndex) => {
  if (rowIndex === 0) {
    let rowspan = 1;
    for (let i = 1; i < data.value.length; i++) {
      if (data.value[i].applyForNum === data.value[0].applyForNum) {
        rowspan++;
      } else {
        break;
      }
    }
    return rowspan;
  }
  if (data.value[rowIndex].applyForNum === data.value[rowIndex - 1].applyForNum) {
    return 0;
  }
  let rowspan = 1;
  for (let i = rowIndex + 1; i < data.value.length; i++) {
    if (data.value[i].applyForNum === data.value[rowIndex].applyForNum) {
      rowspan++;
    } else {
      break;
    }
  }
  return rowspan;
};

const exportFile = async (sign: 'task' | 'apply', row: any) => {
  if (sign === 'task') {
    row.loading = true;
  }
  if (sign === 'apply') {
    row.applyLoading = true;
  }
  const baseUrl = import.meta.env.VITE_SERVICE_BASE_URL;
  const url = new URL('/api/download', baseUrl).toString();

  // 根据 sign 提取对应 ID
  const id = sign === 'task' ? row.taskId : row.applyForNum;

  if (id === undefined || id === null) {
    throw new Error('无效的导出 ID');
  }

  const headers = {
    Authorization: `Bearer ${getToken()}`
  };

  try {
    await downloadFile({
      url,
      params: { id, sign },
      headers
    });
    if (sign === 'task') {
      row.loading = false;
    }
    if (sign === 'apply') {
      row.applyLoading = false;
    }
  } catch (error) {
    console.error(`文件导出失败（${sign}）:`, error);
    alert('导出失败，请重试');
    throw error;
  }
}

const tableMemoryData = computed(() => {
  return data.value.map(val => {
    val.loading = false;
    val.applyLoading = false;
    return val;
  })
})


const navToAno = (row) => {
  router.push({
    // name: "data-ano_operation",
    // name: "data-ano_imgoperate",
    name: import.meta.env.VITE_TOGGLE_OPERATE === 'Y' ? 'data-ano_imgoperate' : 'data-ano_operation',
    params: {
      sign: "edit",
      row,
    },
    query: {
      taskId: row.taskId,
      id: row.sonId,
      markUserId: undefined,
      anoType: "result",
      sign: 'report'
    },
  });
}

const batchLoading = ref<Boolean>(false);

/**
 * 处理批量导出报告
 * @returns {Promise<void>}
 */
async function handleBatchExport() {
  // 检查是否选择了要导出的报告
  if (checkedRowKeys.value.length === 0) {
    window.$message?.warning?.('请选择要导出的报告');
    return;
  }

  // 定义API相关配置
  const API_PATH = '/api/batchDownload';
  const baseUrl = import.meta.env.VITE_SERVICE_BASE_URL;

  // 验证基础URL是否存在
  if (!baseUrl) {
    window.$message?.error?.('系统配置错误，未找到服务基础地址');
    return;
  }

  try {
    // 显示加载状态
    batchLoading.value = true;

    // 构建请求参数
    const taskIds = checkedRowKeys.value.join(',');
    const url = new URL(API_PATH, baseUrl).href;

    // 验证令牌是否存在
    const token = getToken();
    if (!token) {
      window.$message?.error?.('未获取到登录信息，请重新登录');
      return;
    }

    // 执行文件下载
    await downloadFile({
      url,
      params: { taskIds },
      headers: {
        Authorization: `Bearer ${token}`
      }
    });

    // 下载成功提示
    window.$message?.success?.('批量导出已开始');
  } catch (error) {
    // 错误处理
    console.error('批量导出失败:', error);
    window.$message?.error?.(
      error instanceof Error ? error.message : '批量导出过程中发生错误'
    );
  } finally {
    // 确保加载状态重置
    batchLoading.value = false;
  }
}
</script>

<template>
  <div class="flex-col-stretch gap-16px">
    <UserSearch v-model:model="searchParams" @reset="resetSearchParams" @search="getDataByPage" />
    <NCard title="测试评估报告" :bordered="false" size="small" class="sm:flex-1-hidden card-wrapper ">
      <template #header-extra>
        <TableHeaderOperation v-model:columns="columnChecks" :disabled-delete="checkedRowKeys.length === 0"
          :loading="loading" :is-add="false" :is-del="false" @add="handleAdd" @delete="handleBatchDelete"
          @refresh="getData">
          <template #prefix>
            <NButton size="small" @click="handleBatchExport()" type="primary" class="-mr-24px" :loading="batchLoading">
              批量导出报告
            </NButton>
          </template>
        </TableHeaderOperation>
      </template>
      <NDataTable v-model:checked-row-keys="checkedRowKeys" :columns="columns" :data="tableMemoryData" size="small"
        :flex-height="!appStore.isMobile" :scroll-x="1800" :loading="loading" remote :row-key="row => row.id"
        :pagination="mobilePagination" class="sm:h-full">
      </NDataTable>
      <UserOperateDrawer v-model:visible="drawerVisible" :operate-type="operateType"
        :row-data="mapEditingData(editingData)" @submitted="getDataByPage" />
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
    <!-- 导出评估报告 modal -->
    <NModal v-model:show="assessShowModel">
      <NCard style="width: 600px" title="导出评估报告" :bordered="false" size="huge" role="dialog" aria-modal="true">
        <NScrollbar class="h-280px pr-20px">
          <NForm ref="formRef" :model="exportModel" label-placement="left" :label-width="100">
            <NGrid responsive="screen" item-responsive class="ml-24px">
              <NFormItemGi span="24 m:24" label="导出类型:" path="createTime" class="h-42px">
              </NFormItemGi>
            </NGrid>
          </NForm>
        </NScrollbar>
      </NCard>
    </NModal>
  </div>
</template>

<style scoped lang="scss">
::-webkit-scrollbar-button {
  background-color: #ccc;
}
</style>
