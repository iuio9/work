<script setup lang="tsx">
import { NButton } from 'naive-ui';
import { $t } from '@/locales';
import { useAppStore } from '@/store/modules/app';
import { useTable, useTableOperate } from '@/hooks/common/table';
import { delExample, getExamplePage } from '@/service/api/dataManage';
import { useBoolean } from '~/packages/hooks';
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
  apiFn: getExamplePage,
  showTotal: true,
  apiParams: {
    current: 1,
    size: 10,
    // if you want to use the searchParams in Form, you need to define the following properties, and the value is null
    // the value can not be undefined, otherwise the property in Form will not be reactive
    algorithmName: null
    // modelId: route.query.modelId
  },
  columns: () => [
    {
      type: 'selection',
      align: 'center',
      width: 48
    },
    { title: '模型名称', key: 'algorithmName' },
    // {title: "模型请求类型", key: "requestType"},
    // {title: "模型请求地址", key: "url"},
    { title: '模型描述信息', key: 'algorithmDesc' },
    {
      key: 'operate',
      title: $t('common.operate'),
      align: 'center',
      width: 260,
      fixed: "right",
      render: (row: any) => {
        const isOperate = Boolean(row.isDelete);
        const isTrainSuccess = Boolean(row.status ? row.status : false);
        return [
          h('div', { class: 'flex-center gap-8px' }, [
            // h(NButton, {
            //   disabled: isTrainSuccess,
            //   ghost: true,
            //   size: "small",
            //   onClick: () => handleTrain(row),
            // }, "模型训练"),
            h(
              NButton,
              {
                disabled: isOperate,
                type: 'primary',
                ghost: true,
                size: 'small',
                onClick: () => handleEdit(row)
              },
              $t('common.edit')
            )
            // h(NButton, {
            //   disabled: isTrainSuccess,
            //   type: "primary",
            //   ghost: true,
            //   size: "small",
            //   onClick: () => handleDetail(row),
            // }, "任务详情"),
            // h(NPopconfirm, {
            //   onPositiveClick: () => handleDelete(row.modelId),
            // }, {
            //   default: () => h("span", {}, $t('common.confirmDelete')),
            //   trigger: () => h(NButton, {
            //     disabled: isOperate,
            //     type: "error",
            //     ghost: true,
            //     size: "small",
            //   }, $t('common.delete'))
            // })
          ])
        ];
      }
    }
  ]
});

const {
  drawerVisible,
  operateType,
  editingData,
  checkedRowKeys,
  onBatchDeleted,
  onDeleted
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
  operateType.value = 'add';
  openModal();
}

function handleEdit(item: any) {
  item.query = item.params
    ? JSON.parse(item.params).map(val => {
        return {
          ...val,
          key: val.serverKey, // val.value
          value: val.value ? val.value : null, // val.label
          type: val.type,
          valuePlaceholder: val.label,
          dictId: undefined, // 字典ID
          isRelDict: false // 是否关联字典
        };
      })
    : [];
  item.responseQuery = item.responseParams
    ? JSON.parse(item.responseParams).map(val => {
        return {
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
    name: 'data-expansion_add',
    query: {
      id: row.id,
      modelId: row.modelId,
      params: row.params
    }
  });
};
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
        :row-key="row => row.id"
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
        :operate-type="operateType"
        :row-data="editingData"
        @submitted="getDataByPage"
      />
    </NCard>
  </div>
</template>

<style scoped></style>
