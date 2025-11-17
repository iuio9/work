<script setup lang="tsx">
import { useAppStore } from '@/store/modules/app';
import { useTable, useTableOperate } from '@/hooks/common/table';
import { fetchGetLoginList } from "@/service/api/log";
import UserSearch from './modules/user-search.vue';

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
  apiFn: fetchGetLoginList,
  showTotal: true,
  apiParams: {
    current: 1,
    size: 10,
    // if you want to use the searchParams in Form, you need to define the following properties, and the value is null
    // the value can not be undefined, otherwise the property in Form will not be reactive
    loginName: "",
    ipaddr: "",
  },
  columns: () => [
    {
      key: 'index',
      title: "访问编号",
      align: 'center',
    },
    {
      key: 'loginName',
      title: "账号",
      align: 'center',
    },
    {
      key: 'ipaddr',
      title: "登录IP地址",
      align: 'center',
    },
    {
      key: 'browser',
      title: "浏览器",
      align: 'center',
    },
    {
      key: 'os',
      title: "操作系统",
      align: 'center',
    },
    {
      key: 'loginTime',
      title: "访问时间",
      align: 'center',
    }
  ]
});

const {
  checkedRowKeys,
} = useTableOperate(data, getData);

onMounted(() => {
  console.log("mounted")
})


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
    </NCard>
  </div>
</template>

<style scoped></style>

