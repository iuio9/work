<script setup lang="tsx">
import { NButton, NPopconfirm } from 'naive-ui';
import { $t } from '@/locales';
import { useAppStore } from '@/store/modules/app';
import { useTable, useTableOperate } from '@/hooks/common/table';
import { getTeamList, teamAdd, teamEdit, teamRemove } from '@/service/api/ano';
import { useBoolean } from '~/packages/hooks';
import { modelLastAssess } from '@/service/api/model-manage';
import UserOperateDrawer from './modules/user-operate-drawer.vue';
import UserSearch from './modules/user-search.vue';
import AssessModal from './modules/assess-modal.vue';
import GroupOperateModal from "./modules/group-operate-modal.vue";

const appStore = useAppStore();
const router = useRouter();
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
  apiFn: getTeamList,
  showTotal: true,
  apiParams: {
    current: 1,
    size: 10,
    teamName: null
  },
  columns: () => [
    {
      type: 'selection',
      align: 'center',
      width: 48,
      fixed: 'left'
    },
    {
      title: '序号',
      key: 'index'
    },
    {
      title: '团队名称',
      key: 'teamName'
    },
    {
      title: '团队类型',
      key: 'teamType',
      render: (row: any) => {
        const teamTypes: any = {
          "1": "标注团队",
          "2": "审核团队",
        }
        const name = teamTypes[row.teamType];
        return h('span', { class: 'text-primary' }, name);
      }
    },
    {
      title: '团队成员数',
      key: 'teamCount'
    },
    {
      title: '团队描述',
      key: 'teamDec'
    },
    {
      title: '创建时间',
      key: 'createTime'
    },
    {
      key: 'operate',
      title: $t('common.operate'),
      width: 320,
      align: 'center',
      fixed: "right",
      render: row => {
        return [
          h(
            'div',
            { class: 'flex flex-wrap gap-4px justify-center' },
            [
              h(
                NButton,
                {
                  type: 'primary',
                  ghost: true,
                  size: 'small',
                  onClick: () => edit(row)
                },
                { default: () => $t('common.edit') }
              ),
              h(
                NPopconfirm,
                {
                  onPositiveClick: () => handleDelete(row.id)
                },
                {
                  default: () => $t('common.confirmDelete'),
                  trigger: () =>
                    h(
                      NButton,
                      {
                        type: 'error',
                        ghost: true,
                        size: 'small'
                      },
                      { default: () => $t('common.delete') }
                    )
                }
              )
            ]
          )
        ]
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
  onDeleted,
  handleEdit
  // closeDrawer
} = useTableOperate(data, getData);

const assessShowModel = ref<any>(false);
const assessStatus = ref<any>(0);
const assessStr = ref<any>('');

function edit(row: any) {
  operateType.value = 'edit';
  editingData.value = row;
  isGroupModel.value = true;
}

async function handleBatchDelete() {
  // request
  const res = await teamRemove({ ids: checkedRowKeys.value });
  if (res.data >= 1 || !res.data) {
    onBatchDeleted();
  }
}

async function handleDelete(id: number) {
  // request
  const params = {
    ids: [id]
  }
  const res = await teamRemove(params);
  if (res.data >= 1 || !res.data) {
    // onDeleted();
    getDataByPage();
  }
}

function handleAdd() {
  operateType.value = 'add';
  openModal();
}

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

//  查看训练评估
const trainResultShowModal = ref<Boolean>();
const currentRow = ref<any>(null);
const isGroupModel = ref<Boolean>(false);

async function handleCreateGroup() {
  operateType.value = 'add';
  editingData.value = {};
  isGroupModel.value = true;
}
</script>

<template>
  <div class="min-h-500px flex-col-stretch gap-16px overflow-hidden lt-sm:overflow-auto">
    <UserSearch v-model:model="searchParams" @reset="resetSearchParams" @search="getDataByPage" />
    <NCard title="" :bordered="false" size="small" class="sm:flex-1-hidden card-wrapper">
      <template #header>
        <div class="flex items-center">
          <span class="mr-16px">团队管理</span>
          <div class="flex items-center" @click="() => router.back()">
            <svg-icon local-icon="lets-icons--back" class="text-[16px]"></svg-icon>
            <span class="ml-8px cursor-pointer">返回</span>
          </div>
        </div>
      </template>
      <template #header-extra>
        <TableHeaderOperation v-model:columns="columnChecks" :disabled-delete="checkedRowKeys.length === 0"
          :loading="loading" :is-add="false" @delete="handleBatchDelete" @refresh="getData">
          <template #prefix>
            <NButton size="small" type="primary" class="-mr-12px" @click="handleCreateGroup()">创建团队</NButton>
          </template>
        </TableHeaderOperation>
      </template>
      <NDataTable v-model:checked-row-keys="checkedRowKeys" :columns="columns" :data="data" size="small"
        :flex-height="!appStore.isMobile" :scroll-x="962" :loading="loading" remote :row-key="row => row.id"
        :pagination="mobilePagination" class="sm:h-full" />
      <GroupOperateModal v-model:visible="isGroupModel" :operate-type="operateType" :row-data="editingData" @submitted="getDataByPage" />
    </NCard>
  </div>
</template>

<style scoped></style>
