<script setup lang="tsx">
import { NButton, NPopconfirm, NTag } from "naive-ui";
import {
  fetchDictRmove,
  fetchGetDictList,
  fetchGetUserList,
  fetchUserDel,
} from "@/service/api";
import { $t } from "@/locales";
import { useAppStore } from "@/store/modules/app";
import { enableStatusRecord, userGenderRecord } from "@/constants/business";
import { useTable, useTableOperate } from "@/hooks/common/table";
import UserOperateDrawer from "./modules/user-operate-drawer.vue";
import UserSearch from "./modules/user-search.vue";

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
  immediate: undefined,
  apiFn: fetchGetDictList,
  showTotal: true,
  apiParams: {
    current: 1,
    size: 10,
    // if you want to use the searchParams in Form, you need to define the following properties, and the value is null
    // the value can not be undefined, otherwise the property in Form will not be reactive
    dictName: "",
    status: null,
  },
  columns: () => [
    {
      type: "selection",
      align: "center",
      width: 48,
    },
    {
      key: "id",
      title: "字典ID",
      align: "center",
      width: 64,
    },
    {
      key: "dictName",
      title: "字典名称",
      align: "center",
      minWidth: 100,
      render: (row: any) => {
        return (
          <div>
            <n-button
              quaternary
              type="info"
              onClick={() => navigateToDict(row)}
            >
              {row.dictName}
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

    //     const tagMap: Record<Api.Common.EnableStatus, NaiveUI.ThemeColor> = {
    //       1: "success",
    //       2: "warning",
    //     };

    //     const label = $t(enableStatusRecord[row.status]);

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
      // render: (row) => {
      //   const disabled = `${row.id}` === "1" ? true: false;
      //   return (
      //     <div class="flex-center gap-8px">
      //       <NButton
      //         type="primary"
      //         ghost
      //         size="small"
      //         onClick={() => navigateToDict(row)}
      //       >
      //         列表
      //       </NButton>
      //       <NButton
      //         type="primary"
      //         ghost
      //         size="small"
      //         onClick={() => edit(row.id)}
      //       >
      //         {$t("common.edit")}
      //       </NButton>
      //       <NPopconfirm onPositiveClick={() => handleDelete(row.id)}>
      //         {{
      //           default: () => $t("common.confirmDelete"),
      //           trigger: () => (
      //             <NButton type="error" ghost size="small">
      //               {$t("common.delete")}
      //             </NButton>
      //           ),
      //         }}
      //       </NPopconfirm>
      //     </div>
      //   )
      // },
      render: (row) => {
        const disabled = `${row.id}` === "1" || Number(row.isAllowDeletion) == 1 ? true: false;
        return [
          h("div", {
            class: "flex-center gap-8px"
          }, [
            h(NButton, {
              type: "primary",
              ghost: true,
              size: "small",
              onClick: () => navigateToDict(row)
            }, '列表'),
            h(NButton, {
              type: "primary",
              ghost: true,
              size: "small",
              onClick: () => edit(row.id)
            }, $t("common.edit")),
            h(NPopconfirm, {
              onPositiveClick: () => handleDelete(row.id)
            }, {
              default: () => h("span", {}, $t("common.confirmDelete")),
              trigger: () => h(NButton, {
                disabled,
                type: "error",
                ghost: true,
                size: "small",
              }, $t("common.delete"))
            })
          ])
        ]
      }
    },
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
  onDeleted,
  // closeDrawer
} = useTableOperate(data, getData);

async function handleBatchDelete() {
  // request
  console.log(checkedRowKeys.value);
  const res = await fetchDictRmove(checkedRowKeys.value);
  if (res.data >= 1) {
    onBatchDeleted();
  }
}

async function handleDelete(id: number) {
  // request
  const res = await fetchDictRmove([id]);
  if (res.data >= 1) {
    onDeleted();
  }
}

function edit(id: number) {
  handleEdit(id);
}

const router = useRouter();
const navigateToDict = (row: any) => {
  router.push({
    name: "manage_dict-type",
    query: {
      id: row.id
    }
  });
  // localStorage.setItem("dictRow", JSON.stringify(row));
};
</script>

<template>
  <div class="min-h-500px flex-col-stretch gap-16px overflow-hidden lt-sm:overflow-auto">
    <UserSearch v-model:model="searchParams" @reset="resetSearchParams" @search="getDataByPage" />
    <NCard title="字典列表" :bordered="false" size="small" class="sm:flex-1-hidden card-wrapper">
      <template #header-extra>
        <TableHeaderOperation v-model:columns="columnChecks" :disabled-delete="checkedRowKeys.length === 0"
          :loading="loading" @add="handleAdd" @delete="handleBatchDelete" @refresh="getData" />
      </template>
      <NDataTable v-model:checked-row-keys="checkedRowKeys" :columns="columns" :data="data" size="small"
        :flex-height="!appStore.isMobile" :scroll-x="962" :loading="loading" remote :row-key="(row) => row.id"
        :pagination="mobilePagination" class="sm:h-full" />
      <UserOperateDrawer v-model:visible="drawerVisible" :operate-type="operateType" :row-data="editingData"
        @submitted="getDataByPage" />
    </NCard>
  </div>
</template>

<style scoped></style>
