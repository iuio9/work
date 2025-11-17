<script setup lang="tsx">
import { NModal, NUpload, NButton, NSpace, useMessage } from 'naive-ui';
import { $t } from "@/locales";
import { useAppStore } from "@/store/modules/app";
import { useTable, useTableOperate } from "@/hooks/common/table";
import UserOperateDrawer from "./modules/user-operate-drawer.vue";
import { fetchLabelList, fetchLabelRemove } from "@/service/api/tag";

const appStore = useAppStore();

const route = useRoute();
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
  apiFn: fetchLabelList,
  showTotal: true,
  apiParams: {
    current: 1,
    size: 10,
    // labelGroupId: route.query.id,
    labelGroupId: JSON.parse(localStorage.getItem("row")).id,
  },
  columns: () => [
    {
      type: "selection",
      align: "center",
      width: 48,
      fixed: "left",
    },
    {
      key: "onlyId",
      title: '唯一ID',
      align: "left",
      width: 240,
    },
    {
      title: "标签名称",
      key: "labelName",
      render: (row) => {
        return [
          h(
            "div",
            {
              class: "flex items-center",
            },
            [
              h(
                "span",
                {
                  class: "block w-8px h-20px mr-8px",
                  style: { backgroundColor: row.labelColor },
                },
                "",
              ),
              h("span", {}, row.labelName),
            ],
          ),
        ];
      },
    },
    {
      title: "标签名称(英文名)",
      key: "englishLabelName"
    },
    {
      title: "标签排序名",
      key: "labelSort"
    },
    { title: "创建时间", key: "createTime", width: 180 },
    { title: "更新时间", key: "updateTime", width: 180 },
    {
      key: "operate",
      title: $t("common.operate"),
      align: "center",
      render: (row) => {
        return [
          h(
            NButton,
            {
              text: true,
              type: "info",
              style: { marginRight: "10px", fontSize: "12px" },
              onClick: () => edit(row.id),
            },
            "编辑",
          ),
          h(
            NButton,
            {
              text: true,
              type: "info",
              style: { marginRight: "10px", fontSize: "12px" },
              onClick: () => handleDelete(row.id),
            },
            "删除",
          ),
        ];
      },
      fixed: "right",
    },
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

async function handleBatchDelete() {
  // request
  console.log(checkedRowKeys.value);

  const res = await fetchLabelRemove({
    ids: checkedRowKeys.value,
  });
  if (res.data >= 1) {
    onBatchDeleted();
  }
}

async function handleDelete(id: number) {
  // request
  const res = await fetchLabelRemove({ ids: [id] });
  if (res.data >= 1) {
    onDeleted();
  }
}

function edit(id: number) {
  handleEdit(id);
}

const router = useRouter();
const handleTagManage = (row: any) => {
  router.push({
    name: "data-ano_gtag",
    query: row,
  });
};

const rowQuery = ref<any>({});
onMounted(() => {
  const query = JSON.parse(localStorage.getItem("row"));

  rowQuery.value = query;
});

onBeforeUnmount(() => {
  localStorage.removeItem("row");
});

</script>

<template>
  <div class="min-h-500px flex-col-stretch gap-16px overflow-hidden lt-sm:overflow-auto">
    <NCard title="" :bordered="false" size="small" class="sm:flex-1-hidden card-wrapper">
      <template #header>
        <div class="flex items-center">
          <span class="mr-16px">标签列表</span>
          <div class="flex items-center" @click="() => router.back()">
            <svg-icon local-icon="lets-icons--back" class="text-[16px]"></svg-icon>
            <span class="ml-8px cursor-pointer">返回</span>
          </div>
        </div>
      </template>
      <template #header-extra>
        <TableHeaderOperation v-model:columns="columnChecks" :disabled-delete="checkedRowKeys.length === 0"
          :loading="loading" @add="handleAdd" @delete="handleBatchDelete" @refresh="getData">
        </TableHeaderOperation>
      </template>
      <NDataTable v-model:checked-row-keys="checkedRowKeys" :columns="columns" :data="data"
        :flex-height="!appStore.isMobile" :scroll-x="962" :loading="loading" remote :row-key="(row) => row.id"
        :pagination="mobilePagination" class="sm:h-full" />
      <UserOperateDrawer v-model:visible="drawerVisible" :operate-type="operateType" :row-data="editingData"
        @submitted="getDataByPage" />
    </NCard>

    <!-- 标签导入 -->
    <n-modal v-model:show="isModalVisible" preset="card" title="文件上传与模板下载" class="w-600px">
      <div class="">
        <n-space space="[16px]">
          <n-upload ref="uploadRef" :action="uploadAction" :before-upload="beforeUpload" accept=".xlsx, .xls"
            @finish="handleFinish">
            <n-button type="primary">上传文件</n-button>
          </n-upload>
        </n-space>
      </div>
      <template #footer>
        <n-space justify="end" space="[16px]">
          <n-button @click="closeModal" class="bg-gray-200 text-gray-800 hover:bg-gray-300">取消</n-button>
          <n-button type="primary" @click="handleSubmit">确认</n-button>
        </n-space>
      </template>
    </n-modal>
  </div>
</template>

<style scoped></style>
