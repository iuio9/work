<script setup lang="tsx">
import { NButton, NTag } from 'naive-ui';
import { $t } from '@/locales';
import { useAppStore } from '@/store/modules/app';
import {enableStatusRecord1} from '@/constants/business';
import { useTable, useTableOperate } from '@/hooks/common/table';
import UserOperateDrawer from './modules/user-operate-drawer.vue';
import MenuOperateModal from './modules/menu-operate-modal.vue';
import UserSearch from './modules/user-search.vue';
import {fetchGetLogDetail, fetchGetLogList} from "@/service/api/log";
import {useBoolean} from "~/packages/hooks";

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
  apiFn: fetchGetLogList,
  showTotal: true,
  apiParams: {
    current: 1,
    size: 10,
    title: "",
    businessType: null,
    operName: "",
    status: null,
    // timeArr: [new Date((new Date().getTime() - 24*60*60*1000)), new Date()],
    timeArr: [ 0, new Date()],
  },
  columns: () => [
    {
      key: 'index',
      title: "日志ID",
      align: 'center',
      width: 64
    },
    {
      key: 'title',
      title: "操作模块",
      align: 'center',
    },
    {
      key: 'businessType',
      title: "操作类型",
      align: 'center',
      minWidth: 100,
      render: (row: any) => {
        if (!row.businessType) {
          return null;
        }
        // 0其它 1后台用户 2手机端用户
        const types = {
          "0": "其它",
          "1": "新增",
          "2": "修改",
          "3": "删除",
        }
        const businessType = types[row.businessType];
        return (
          <div>
            <n-button type="info" quaternary>
              {businessType}
            </n-button>
          </div>
        )
      }
    },
    {
      key: 'operName',
      title: "操作人员",
      align: 'center',
    },
    {
      key: 'operIp',
      title: "操作地址",
      align: 'center',
    },
    {
      key: 'status',
      title: "操作状态",
      align: 'center',
      render: row => {
        if (row.status === null) {
          return null;
        }

        const tagMap: Record<Api.Common.EnableStatus, NaiveUI.ThemeColor> = {
          0: 'success',
          1: 'error'
        };

        const label = $t(enableStatusRecord1[row.status]);

        return <NTag type={tagMap[row.status]}>{label}</NTag>;
      }
    },
    {
      key: 'operTime',
      title: "操作日期",
      align: 'center',
    },
    {
      key: 'costTime',
      title: "消耗时间",
      align: 'center',
    },
    {
      key: 'operate',
      title: $t('common.operate'),
      align: 'center',
      width: 250,
      render: row => (
        <div class="flex-center gap-8px">
          <NButton type="primary" ghost size="small" onClick={() => handleDetail(row.operId)}>
            详情
          </NButton>
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
} = useTableOperate(data, getData);

const { bool: visible, setTrue: openModal } = useBoolean();
const detailData = ref<any>({});

async function handleDetail(operId: number | string) {
  const res = await fetchGetLogDetail({
    operId
  });
  detailData.value = res.data;
  openModal();
}

</script>

<template>
  <div class="min-h-500px flex-col-stretch gap-16px overflow-hidden lt-sm:overflow-auto">
    <UserSearch v-model:model="searchParams" @reset="resetSearchParams" @search="getDataByPage" />
    <NCard title="日志列表" :bordered="false" size="small" class="sm:flex-1-hidden card-wrapper">
      <template #header-extra>
        <TableHeaderOperation
          v-model:columns="columnChecks"
          :disabled-delete="checkedRowKeys.length === 0"
          :loading="loading"
          :is-add="false"
          :is-del="false"
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

