<script setup lang="tsx">
import { NButton, NPopconfirm, NTag } from "naive-ui";
import { fetchGetUserList, fetchUserDel, resetDefaultPassword } from "@/service/api";
import { $t } from "@/locales";
import { useAppStore } from "@/store/modules/app";
import { enableStatusRecord, userGenderRecord } from "@/constants/business";
import { useTable, useTableOperate } from "@/hooks/common/table";
import UserOperateDrawer from "./modules/user-operate-drawer.vue";
import UserSearch from "./modules/user-search.vue";
import { useAuthStore } from "@/store/modules/auth";

const appStore = useAppStore();
const authStore = useAuthStore();


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
  apiFn: fetchGetUserList,
  showTotal: true,
  apiParams: {
    current: 1,
    size: 10,
    // if you want to use the searchParams in Form, you need to define the following properties, and the value is null
    // the value can not be undefined, otherwise the property in Form will not be reactive
    status: null,
    userName: null,
    userGender: null,
    nickName: null,
    userPhone: null,
    userEmail: null,
    deptId: null,
  },
  columns: () => [
    {
      type: "selection",
      align: "center",
      width: 48,
    },
    {
      key: "index",
      title: $t("common.index"),
      align: "center",
      width: 64,
    },
    {
      key: "userName",
      title: $t("page.manage.user.userName"),
      align: "center",
    },
    {
      key: "userGender",
      title: $t("page.manage.user.userGender"),
      align: "center",
      render: row => {
        if (row.userGender === null) {
          return null;
        }

        const tagMap: Record<Api.SystemManage.UserGender, NaiveUI.ThemeColor> = {
          1: 'primary',
          2: 'error'
        };

        const label = $t(userGenderRecord[row.userGender]);

        return <NTag type={tagMap[row.userGender]}>{label}</NTag>;
      }
    },
    {
      key: "nickName",
      title: $t("page.manage.user.nickName"),
      align: "center",
    },
    {
      key: "userPhone",
      title: $t("page.manage.user.userPhone"),
      align: "center",
    },
    {
      key: "userEmail",
      title: $t("page.manage.user.userEmail"),
      align: "center",
    },
    {
      key: "status",
      title: $t("page.manage.user.userStatus"),
      align: "center",
      render: (row) => {
        if (row.status === null) {
          return null;
        }

        const tagMap: Record<Api.Common.EnableStatus, NaiveUI.ThemeColor> = {
          1: "success",
          2: "warning",
        };

        const label = $t(enableStatusRecord[row.status]);

        return <NTag type={tagMap[row.status]}>{label}</NTag>;
      },
    },
    {
      key: "operate",
      title: $t("common.operate"),
      align: "center",
      width: 300,
      render: (row) => (
        <div
          style={{ display: row.isAllowDeletion == 2 ? "block" : "none" }}
          class="flex justify-center items-center gap-8px"
        >
          <NButton
            type="primary"
            ghost
            size="small"
            onClick={() => resetPasswd(row.id)}
            style={{ display: authStore.userInfo.isHide == 1 ? "" : "none" }}
            class="mr-8px"
          >
            重置密码
          </NButton>
          <NButton
            type="primary"
            ghost
            size="small"
            onClick={() => edit(row.id)}
            class="mr-8px"
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

async function handleBatchDelete() {
  // request
  console.log(checkedRowKeys.value);
  const res = await fetchUserDel(checkedRowKeys.value);
  if (res.data >= 1) {
    onBatchDeleted();
  }
}

async function handleDelete(id: number) {
  // request
  const res = await fetchUserDel([id]);
  if (res.data >= 1) {
    onDeleted();
  }
}

function edit(id: number) {
  handleEdit(id);
}

async function resetPasswd(id: number) {
  const res = await resetDefaultPassword({ id });
  if (res.data) {
    window.$message?.success("重置默认密码成功");
  }
}
</script>

<template>
  <div class="min-h-500px flex-col-stretch gap-16px overflow-hidden lt-sm:overflow-auto">
    <UserSearch v-model:model="searchParams" @reset="resetSearchParams" @search="getDataByPage" />
    <NCard :title="$t('page.manage.user.title')" :bordered="false" size="small" class="sm:flex-1-hidden card-wrapper">
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
