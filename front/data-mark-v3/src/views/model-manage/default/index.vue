<script setup lang="tsx">
import { NButton } from 'naive-ui';
import { $t } from '@/locales';
import { useAppStore } from '@/store/modules/app';
import { useTable, useTableOperate } from '@/hooks/common/table';
import { delExample, delModel, getExamplePage, getModelPage } from '@/service/api/dataManage';
import { useBoolean } from '~/packages/hooks';
import { modelLastAssess } from '@/service/api/model-manage';
import UserOperateDrawer from './modules/user-operate-drawer.vue';
import UserSearch from './modules/user-search.vue';
import AssessModal from './modules/assess-modal.vue';

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
  sign: 'modelId',
  apiFn: getModelPage,
  showTotal: true,
  apiParams: {
    current: 1,
    size: 10,
    // if you want to use the searchParams in Form, you need to define the following properties, and the value is null
    // the value can not be undefined, otherwise the property in Form will not be reactive
    modelName: null
    // modelId: route.query.modelId
  },
  columns: () => [
    {
      type: 'selection',
      align: 'center',
      width: 48,
      fixed: 'left'
    },
    {
      key: 'index',
      title: $t('common.index'),
      align: 'center',
      width: 64
    },
    {
      title: '模型名称', key: 'modelName',
      width: 150,
      ellipsis: {
        tooltip: true
      }
    },
    // {
    //   key: 'modelVersion',
    //   title: '训练版本',
    //   align: 'center',
    //   width: 120
    // },
    {
      title: '训练状态', key: 'trainStat',
      width: 150,
      ellipsis: {
        tooltip: true
      }
    },
    {
      title: '保存路径',
      key: 'modelUrl',
      width: 180,
      ellipsis: {
        tooltip: true
      }},
    {
      title: '模型业务类型',
      key: 'modelBizType',
      width: 200,
      ellipsis: {
        tooltip: true
      }
    },
    {
      title: '模型描述信息', key: 'modelDesc',
      width: 220,
      ellipsis: {
        tooltip: true
      }
    },
    {
      title: '创建时间',
      key: 'createTime',
      width: 160,
      ellipsis: {
        tooltip: true
      }
    },
    {
      key: 'operate',
      title: $t('common.operate'),
      width: 320,
      align: 'center',
      fixed: "right",
      render: row => (
        <div class="flex flex-wrap gap-4px justify-center">
          <NButton type="primary" ghost size="small" onClick={() => handleOperate('trainResult', row)}>
            查看训练评估
          </NButton>
          <NButton type="primary" ghost size="small" onClick={() => handleOperate('list', row)}>
            评估列表
          </NButton>
          <NButton type="primary" ghost size="small" onClick={() => edit(row.modelId)}>
            {$t('common.edit')}
          </NButton>
          <NPopconfirm onPositiveClick={() => handleDelete(row.id)}>
            {{
              default: () => $t('common.confirmDelete'),
              trigger: () => (
                <NButton type="error" ghost size="small">
                  {$t('common.delete')}
                </NButton>
              )
            }}
          </NPopconfirm>
        </div>
      )
    }
  ]
});

const {
  drawerVisible,
  operateType,
  editingData,
  checkedRowKeys,
  onBatchDeleted,
  onDeleted,
  handleEdit
  // closeDrawer
} = useTableOperate(data, getData);

const assessShowModel = ref<any>(false);
const assessStatus = ref<any>(0);
const assessStr = ref<any>('');

function edit(id: number) {
  handleEdit(id);
}

async function handleBatchDelete() {
  // request
  const res = await delModel(checkedRowKeys.value);
  if (res.data >= 1 || !res.data) {
    onBatchDeleted();
  }
}

async function handleDelete(id: number) {
  // request
  const res = await delModel([id]);
  if (res.data >= 1 || !res.data) {
    onDeleted();
  }
}

function handleAdd() {
  operateType.value = 'add';
  openModal();
}

const handleTrain = async (row: any) => {
  router.push({
    name: 'data-expansion_add',
    query: {
      id: row.id,
      modelId: row.modelId,
      params: row.params
    }
  });
};

async function handleOperate(type: string, row: any) {
  switch (type) {
    case 'result':
      const res = await modelLastAssess({ modelId: row.modelId });
      if (res.data) {
        const { status, result } = res.data;
        assessStatus.value = status;
        assessStr.value = result;
        assessShowModel.value = true;
      }
      break;
    case 'list':
      editingData.value = { ...row };
      openModal();
      break;
    case 'trainResult':
      currentRow.value = row;
      trainResultShowModal.value = true;
      break;
    default:
      break;
  }
}

function handleAssess(data: any) {
  const { status, result } = data;
  assessStatus.value = status;
  assessStr.value = result;
  assessShowModel.value = true;
}

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
  await handleOperate("result", currentRow.value);
}
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
          @add="handleAdd"
          @delete="handleBatchDelete"
          @refresh="getData"
        />
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
        :row-data="editingData"
        @submitted="getDataByPage"
      />
      <AssessModal v-model:visible="visible" :row-data="editingData" @assess-modal="handleAssess" />
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
    </NCard>
  </div>
</template>

<style scoped></style>
