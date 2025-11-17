<script setup lang="tsx">
import {NButton} from "naive-ui";
import {$t} from "@/locales";
import {useAppStore} from "@/store/modules/app";
import {useTable, useTableOperate} from "@/hooks/common/table";
import {getSelectImportFileList} from "@/service/api/tag";

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
  resetSearchParams,
} = useTable({
  apiFn: getSelectImportFileList,
  showTotal: true,
  apiParams: {
    current: 1,
    size: 10,
    id: localStorage.getItem("fileId")
  },
  columns: () => [
    {
      key: "index",
      title: $t("common.index"),
      align: "center",
    },
    {
      title: "文件名称",
      key: "fdName",
      render: (row) => {
        return [
          h("span", {
            class: "truncate"
          }, `${row.fdName}`)
        ]
      },
    },
    {title: "文件大小", key: "fdSize"},
    {title: "状态", key: "status"},
    {title: "错误信息", key: "errorLog"},
  ],
});

const {
  drawerVisible,
  operateType,
  editingData,
  handleAdd,
  handleEdit,
  checkedRowKeys,
  onBatchDeleted,
  onDeleted,
  // closeDrawer
} = useTableOperate(data, getData);

onUnmounted(() => {
  localStorage.removeItem("fileId")
})

onMounted(() => {
  getDataByPage()
})
</script>

<template>
  <div
    class="min-h-500px flex-col-stretch gap-16px overflow-hidden lt-sm:overflow-auto"
  >
    <NCard
      title="图片上传信息"
      :bordered="false"
      size="small"
      style="width: 800px"
      class="sm:flex-1-hidden card-wrapper h-full"
    >
      <NDataTable
        :columns="columns"
        :data="data"
        :scroll-x="800"
        :loading="loading"
        remote
        :row-key="(row) => row.id"
        :pagination="mobilePagination"
        class="sm:h-full"
      />
    </NCard>
  </div>
</template>

<style scoped></style>
