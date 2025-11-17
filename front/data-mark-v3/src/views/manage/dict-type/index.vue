<script setup lang="tsx">
import { NButton, NPopconfirm, NTag } from "naive-ui";
import { fetchGetTreeData, fetchTwoDictRmove } from "@/service/api";
import { $t } from "@/locales";
import { useAppStore } from "@/store/modules/app";
import { enableStatusRecord, userGenderRecord } from "@/constants/business";
import { useTable, useTableOperate } from "@/hooks/common/table";
import UserOperateDrawer from "./modules/user-operate-drawer.vue";
import UserSearch from "./modules/user-search.vue";
import useDictStore from "@/store/modules/dict";

const appStore = useAppStore();
const route = useRoute();
const dictRow = ref<any>({});
const isChildAdd = ref(false);
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
  apiFn: fetchGetTreeData,
  showTotal: true,
  apiParams: {
    current: 1,
    size: 10,
    // if you want to use the searchParams in Form, you need to define the following properties, and the value is null
    // the value can not be undefined, otherwise the property in Form will not be reactive
    dictLabel: "",
    status: null,
    // typeId: JSON.parse(localStorage.getItem("dictRow")).id,
    typeId: route.query.id,
  },
  columns: () => [
    {
      type: "selection",
      align: "center",
      width: 48,
    },
    {
      key: "dictLabel",
      title: "字典标签",
      align: "left",
      width: 300
    },
    {
      key: "id",
      title: "字典ID",
      align: "left",
    },
    // {
    //   key: "dictValue",
    //   title: "字典键值",
    //   align: "center",
    //   minWidth: 100,
    //   render: (row: any) => {
    //     return (
    //       <div>
    //         <n-button quaternary type="info">
    //           {row.dictValue}
    //         </n-button>
    //       </div>
    //     );
    //   },
    // },
    {
      key: "dictSort",
      title: "排序",
      align: "center",
      minWidth: 100,
      render: (row: any) => {
        return (
          <div>
            <n-button quaternary type="info">
              {row.dictSort}
            </n-button>
          </div>
        );
      },
    },
    // {
    //   key: "status",
    //   title: $t("page.manage.user.userStatus"),
    //   align: "center",
    //   width: 100,
    //   render: (row) => {
    //     if (row.status === null) {
    //       return null;
    //     }
    //
    //     const tagMap: Record<Api.Common.EnableStatus, NaiveUI.ThemeColor> = {
    //       1: "success",
    //       2: "warning",
    //     };
    //
    //     const label = $t(enableStatusRecord[row.status]);
    //
    //     return <NTag type={tagMap[row.status]}>{label}</NTag>;
    //   },
    // },
    {
      key: "createTime",
      title: "创建时间",
      align: "center",
      width: 240,
    },
    {
      key: "remark",
      title: "备注信息",
      align: "center",
      minWidth: 160,
    },
    {
      key: "operate",
      title: $t("common.operate"),
      align: "center",
      width: 250,
      render: (row) => (
        <div class="flex-center gap-8px">
          <NButton
            type="primary"
            ghost
            size="small"
            onClick={() => add(row.id)}
          >
            {$t("common.add")}
          </NButton>
          <NButton
            type="primary"
            ghost
            size="small"
            onClick={() => edit(row.id, row.parentId)}
          >
            {$t("common.edit")}
          </NButton>
          <NPopconfirm onPositiveClick={() => handleDelete(row.id)}>
            {{
              default: () => $t("common.confirmDelete"),
              trigger: () => (
                <NButton type="error" ghost size="small">
                  {$t("common.delete")}
                </NButton>
              ),
            }}
          </NPopconfirm>
        </div>
      ),
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

// eslint-disable-next-line @typescript-eslint/no-shadow
const mapEditingData = (data: any) => {
  // return { ...data, typeId: dictRow.value.id };
  return { ...data, typeId: route.query.id };
};

async function handleBatchDelete() {
  // request
  const res = await fetchTwoDictRmove(checkedRowKeys.value);
  if (res.data >= 1) {
    onBatchDeleted();
  }
}

async function handleDelete(id: number) {
  // request
  const res = await fetchTwoDictRmove([id]);
  if (res.data >= 1) {
    onDeleted();
  }
}

function edit(id: number, parentId?: number) {
  handleEdit(id, parentId);
}

function add(id: number) {
  handleAdd(id);
}

function handleOperate(sign) {
  if(sign==="one") {
    handleAdd();
  }
  if(sign==="two") {}
}

const router = useRouter();

const handleBack = () => {
  router.back();
};

onMounted(() => {
  dictRow.value = JSON.parse(localStorage.getItem("dictRow") as string);
});

onBeforeUnmount(() => {
  localStorage.removeItem("dictRow");
});
</script>

<template>
  <div
    class="min-h-500px flex-col-stretch gap-16px overflow-hidden lt-sm:overflow-auto"
  >
    <UserSearch
      v-model:model="searchParams"
      @reset="resetSearchParams"
      @search="getDataByPage"
    />
    <NCard
      title="字典列表"
      :bordered="false"
      size="small"
      class="sm:flex-1-hidden card-wrapper"
    >
      <template #header-extra>
        <TableHeaderOperation
          v-model:columns="columnChecks"
          :disabled-delete="checkedRowKeys.length === 0"
          :loading="loading"
          @add="handleAdd"
          @delete="handleBatchDelete"
          @refresh="getData"
        >
          <template #prefix>
            <NButton size="small" ghost type="primary" @click="handleBack()">
              <template #icon>
                <!--<icon-ic-round-plus class="carbon:return" />-->
                <!--<SvgIcon icon="carbon:return" class="text-icon" />-->
                <SvgIcon local-icon="carbon--return" class="text-icon" />
              </template>
              <span>返回上一级</span>
            </NButton>
          </template>
        </TableHeaderOperation>
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
        :row-key="(row) => row.id"
        :pagination="mobilePagination"
        class="sm:h-full"
      />
      <!--:row-data="mapEditingData(editingData)"-->
      <UserOperateDrawer
        v-model:visible="drawerVisible"
        :operate-type="operateType"
        :row-data="mapEditingData(editingData)"
        @submitted="getDataByPage"
      />
    </NCard>
  </div>
</template>

<style scoped></style>
