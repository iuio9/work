<script setup lang="tsx">
import { NButton, NPopconfirm, NTag } from 'naive-ui';
import { $t } from '@/locales';
import { useAppStore } from '@/store/modules/app';
import { enableStatusRecord, userGenderRecord } from '@/constants/business';
import { useTable, useTableOperate } from '@/hooks/common/table';
import UserOperateDrawer from './modules/user-operate-drawer.vue';
import UserSearch from './modules/user-search.vue';
import {fetchDeptRemove, fetchGetDeptList} from "@/service/api/dept";

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
  resetSearchParams
} = useTable({
  apiFn: fetchGetDeptList,
  showTotal: true,
  apiParams: {
    current: 1,
    size: 10,
    status: null,
    deptName: null,
  },
  columns: () => [
    {
      key: 'deptName',
      title: "部门",
      align: 'center',
    },
    {
      key: 'sort',
      title: "排序",
      align: 'center',
    },
    {
      key: 'supt',
      title: "负责人",
      align: 'center',
    },
    {
      key: 'telePhone',
      title: "联系电话",
      align: 'center',
    },
    {
      key: 'email',
      title: "邮箱",
      align: 'center',
    },
    {
      key: 'status',
      title: "状态",
      align: 'center',
      render: row => {
        if (row.status === null) {
          return null;
        }

        const tagMap: Record<Api.Common.EnableStatus, NaiveUI.ThemeColor> = {
          1: 'success',
          2: 'warning'
        };

        const statusOptions: any = {
          0: "禁用",
          1: "启用"
        };

        const label = statusOptions[row.status];

        return <NTag type={tagMap[row.status]}>{label}</NTag>;
      }
    },
    {
      key: 'CreateTime',
      title: "创建时间",
      align: 'center',
    },
    {
      key: 'operate',
      title: $t('common.operate'),
      align: 'center',
      render: row => (
        <div class="flex-center gap-8px" style={{ 'display': row.isAllowDeletion == 2 ? 'block' : 'none' }}>
          <NButton type="primary" ghost size="small" onClick={() => edit(row.id)} class="mr-8px">
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
  handleAdd,
  handleEdit,
  checkedRowKeys,
  onBatchDeleted,
  onDeleted
  // closeDrawer
} = useTableOperate(data, getData);

async function handleBatchDelete() {
  // request
  console.log(checkedRowKeys.value);
  const res = await fetchDeptRemove(checkedRowKeys.value);
  if(res.data >=1) {
    onBatchDeleted();
  }
}

async function handleDelete(id: number) {
  // request
  const res = await fetchDeptRemove([id]);
  if(res.data >=1) {
    onDeleted();
  }
}

function edit(id: number) {
  handleEdit(id);
}
</script>

<template>
  <div class="min-h-500px flex-col-stretch gap-16px overflow-hidden lt-sm:overflow-auto">
    <UserSearch v-model:model="searchParams" @reset="resetSearchParams" @search="getDataByPage" />
    <NCard title="部门列表" :bordered="false" size="small" class="sm:flex-1-hidden card-wrapper">
      <template #header-extra>
        <TableHeaderOperation
          v-model:columns="columnChecks"
          :is-del="false"
          :disabled-delete="checkedRowKeys.length === 0"
          :loading="loading"
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
        default-expand-all
      />
      <UserOperateDrawer
        v-model:visible="drawerVisible"
        :operate-type="operateType"
        :row-data="editingData"
        @submitted="getDataByPage"
      />
    </NCard>
  </div>
</template>

<style scoped></style>
