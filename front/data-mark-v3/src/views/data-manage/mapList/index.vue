<script setup lang="tsx">
import { NButton, NPopconfirm } from "naive-ui";
import { $t } from "@/locales";
import { useAppStore } from "@/store/modules/app";
import { useTable, useTableOperate } from "@/hooks/common/table";
import { getSelectDataSetDictList, getTreeLevelDictIds } from "@/service/api/dataManage";
import UserOperateDrawer from "./modules/user-operate-drawer.vue";
import { useRouteStore } from "@/store/modules/route";
const routeStore = useRouteStore();

const appStore = useAppStore();

const dataMapList = ref<any>([]);
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
  apiFn: getSelectDataSetDictList,
  showTotal: true,
  apiParams: {
    current: 1,
    size: 10,
    dictLabel: "",
  },
  columns: () => [
    {
      key: "dictLabel",
      title: "数据集类型",
      align: "left",
      minWidth: 260,
      render: (row: any) => {
        return (
          <n-button quaternary type="info" onClick={() => navigateToMap(row)}>
            {row.dictLabel}
          </n-button>
        );
      },
      fixed: "left"
    },
    {
      key: "number",
      title: "当前数据集数量",
      align: "center",
      minWidth: 160,
    },
    {
      key: "total",
      title: "数据集总数量",
      align: "center",
      minWidth: 160,
    },
    {
      key: "fileSumCount",
      title: "文件总数量",
      align: "center",
      minWidth: 160,
    },
    {
      key: "remark",
      title: "备注信息",
      align: "center",
      minWidth: 160,
    },
    {
      key: "createTime",
      title: "创建时间",
      align: "center",
      width: 240,
    },
    {
      key: "operate",
      title: $t("common.operate"),
      align: "center",
      width: 250,
      fixed: "right",
      render: (row) => (
        <div class="flex-center gap-8px">
          {/* v-hasPermi="system:maplist:goMap" */}
          <NButton
            type="primary"
            ghost
            size="small"
            onClick={() => navigateToMap(row)}
          >
            进入数据集
          </NButton>
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
const expandedRowKeys = ref<any>([]);

async function handleBatchDelete() {
  // request
  const res = await fetchDictRmove(checkedRowKeys.value);
  if (res.data >= 1) {
    onBatchDeleted();
  }
}

function sumValues(data) {
  let total = 0;
  for (const item of data) {
    total += item.number || 0;
    if (item.children) {
      total += sumValues(item.children);
    }
  }
  return total;
}

const recursionData = (data: any, label: any) => {
  // eslint-disable-next-line no-param-reassign
  data = data.map((item: any, index: string | number) => {
    if (item.children) {
      if (item.children.length > 0) {
        recursionData(item.children, item.dictLabel);
        // item.total = item.number + sumNestedValues(item.children)
        item.total = item.number + sumValues(item.children)
      };
      if (item.children.length === 0) {
        item.total = item.number;
        delete item.children;
      }
    } else {
      item.total = item.number;
    }
    item.label = label ? `${item.dictLabel}` : item.dictLabel;
    item.value = item.id;
    // expandedRowKeys.value.push(item.id);
    return item;
  });
  return data;
};

// function findObjectById(arr: any, id: any) {
//   for (let item of arr) {
//     if (item.id == id) {
//       return item;
//     } else if (Array.isArray(item.children) && item.children.length > 0) {
//       const found = findObjectById(item.children, id);
//       if (found) {
//         return found;
//       }
//     }
//   }
//   return null;
// }

async function getExpands() {
  const res = await getSelectDataSetDictList();
  // recursionData(res.data);
  dataMapList.value = recursionData(res.data);
  // if(route.query?.dataTypeId) {
  //   const row = findObjectById(dataMapList.value, route.query.dataTypeId);
  //   console.log(row);
  //   expandedRowKeys.value = [`${row.id}`];
  // }
}

// 获取expandedRowKeys
const route = useRoute();
async function getExpandedRowKeys() {
  if (route.query?.dataTypeId) {
    const res = await getTreeLevelDictIds({
      dataTypeId: route.query.dataTypeId
    });
    if (res.data) {
      expandedRowKeys.value = res.data.map(val => val);
    } else {
      expandedRowKeys.value = []
    }
  } else {
    expandedRowKeys.value = []
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
const navigateToMap = (row: any) => {
  router.push({
    name: "data-manage_map",
    query: {
      dataTypeId: row.id,
      dictLabel: row.dictLabel
    },
  });
  localStorage.setItem("mapRow", JSON.stringify(row));
};

onMounted(() => {
  getExpandedRowKeys();
  getExpands();
})
</script>

<template>
  <div class="min-h-500px flex-col-stretch gap-16px overflow-hidden lt-sm:overflow-auto">
    <!--<UserSearch
      v-model:model="searchParams"
      @reset="resetSearchParams"
      @search="getDataByPage"
    />-->
    <NCard title="数据集类型列表" :bordered="false" size="small" class="sm:flex-1-hidden card-wrapper">
      <template #header-extra>
        <TableHeaderOperation :is-add="false" :is-del="false" v-model:columns="columnChecks"
          :disabled-delete="checkedRowKeys.length === 0" :loading="loading" @add="handleAdd" @delete="handleBatchDelete"
          @refresh="getData" />
      </template>
      <!--:expanded-row-keys="expandedRowKeys"-->
      <NDataTable v-model:checked-row-keys="checkedRowKeys" v-model:expanded-row-keys="expandedRowKeys"
        :columns="columns" :data="dataMapList" size="small" :flex-height="!appStore.isMobile" :scroll-x="962"
        :loading="loading" remote :row-key="(row) => row.id" class="sm:h-full" default-expand-all="true" />
      <UserOperateDrawer v-model:visible="drawerVisible" :operate-type="operateType" :row-data="editingData"
        @submitted="getDataByPage" />
    </NCard>
  </div>
</template>

<style scoped></style>
