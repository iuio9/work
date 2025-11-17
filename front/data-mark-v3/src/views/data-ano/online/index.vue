<script setup lang="tsx">
import { NButton, NPopover } from "naive-ui";
import { useAppStore } from "@/store/modules/app";
import { useTable, useTableOperate } from "@/hooks/common/table";
import UserOperateDrawer from "./modules/user-operate-drawer.vue";
import UserSearch from "./modules/user-search.vue";
import { getDataSetMarkList } from "@/service/api/ano";
import { localStg } from "@/utils/storage";
import { getDataSetListNoPage } from "@/service/api/expansion";
import { useAnoStore } from "@/store/modules/ano";

const appStore = useAppStore();
const themeColor = localStg.get("themeColor") || "#646cff";
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
  apiFn: getDataSetMarkList,
  showTotal: true,
  apiParams: {
    current: 1,
    size: 10,
    status: null,
    deptName: null,
    dataTypeId: null,
  },
  columns: () => [
    {
      title: "数据集名称",
      key: "groupName",
      width: 260,
      render: (row) => {
        const color = themeColor;
        return row.isMany == 1 ? [
          h(
            "div",
            {
              class: "flex items-center",
            },
            [
              h(NPopover, { trigger: "hover", placement: "top" }, {
                trigger: () => [
                  h(
                    "span",
                    {
                      class: {
                        "text-[#e5e7eb]": row.count === 0,
                        'cursor-not-allowed': true,
                      },
                      style: {
                        color: '#b8babf',
                      },
                    },
                    `${row.groupName}`
                  ),
                ],
                default: () => [
                  h("span", { class: "truncate" }, '数据集多人标注中不支持任何编辑操作。')
                ]
              })
            ]
          ),
        ] : [
          h(
            "div",
            {
              class: "flex items-center",
            },
            [
              h(
                "span",
                {
                  class: {
                    "text-[#e5e7eb]": row.count === 0,
                  },
                  style: {
                    color,
                  },
                },
                `${row.groupName}`
              )
            ]
          ),
        ];
      },
    },
    {
      title: "数据集类型",
      key: "dataTypeName",
      width: 120,
    },
    {
      title: "版本",
      key: "version",
      width: 80,
      render: (row) => {
        return [h("span", {}, `${row.version}`)];
      },
    },
    {
      width: 120,
      title: "标注类型",
      key: "markType",
      render: (row) => {
        return (
          <div class="flex items-center">
            <span>{row.anoType == 0 ? "图像分割" : "物体检测"}</span>
          </div>
        );
      },
    },
    {
      title: "数据量",
      key: "count",
      align: "center",
      width: 110,
    },
    {
      title: "数据集ID",
      key: "sonId",
      width: 200,
    },
    {
      title: "标注进度",
      key: "status",
      width: 180
    },
    {
      title: "创建时间",
      key: "createTime",
      width: 180,
    },
    {
      width: 180,
      title: "操作",
      key: "operation",
      render(row) {
        const isMany = row.isMany == 1;
        return [
          h(
            NButton,
            {
              type: "primary",
              ghost: true,
              size: "small",
              style: { marginRight: "10px" },
              onClick: () => handleOnlineDetail(row),
              disabled: row.count === 0,
            },
            "详情"
          ),
          h(
            NButton,
            {
              type: "primary",
              ghost: true,
              size: "small",
              style: { marginRight: "10px" },
              onClick: () => handleOnlineAnnotation(row),
              disabled: row.count === 0 || isMany,
            },
            "标注"
          ),
          // h(
          //   NButton,
          //   {
          //     type: "primary",
          //     ghost: true,
          //     size: "small",
          //     style: { marginRight: "10px" },
          //     onClick: () => handleOnlineAnnotation1(row),
          //     disabled: row.count === 0 || isMany,
          //   },
          //   "标注测试"
          // ),
        ];
      },
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

// newCode
const router = useRouter();
const handleOnlineDetail = (row) => {
  router.push({
    name: "data-ano_detail",
    params: {
      sign: "detail",
      row,
    },
    query: {
      id: row.sonId,
      isMany: row.isMany,
      markType: row.anoType,  // 标注类型
    },
  });
};
const handleOnlineAnnotation = (row) => {
  const anoStore = useAnoStore();
  anoStore.updateAno(row.anoType ?? 1);
  router.push({
    // name: "data-ano_operation",
    // name: "data-ano_imgoperate",
    name: import.meta.env.VITE_TOGGLE_OPERATE === 'Y' ? 'data-ano_imgoperate' : 'data-ano_operation',
    params: {
      sign: "edit",
      row,
    },
    query: {
      id: row.sonId,
      anoType: "online",
      markType: row.anoType,  // 标注类型
    },
  });
};
const handleOnlineAnnotation1 = (row) => {
  const anoStore = useAnoStore();
  anoStore.updateAno(row.anoType ?? 1);
  router.push({
    name: "data-ano_imgoperate",
    params: {
      sign: "edit",
      row,
    },
    query: {
      id: row.sonId,
      anoType: "online",
    },
  });
};

const setOptions = ref<any[]>([]);

function renderLabel(option: { value?: string | number, label?: string }) {
  return [
    h("div", {
      class: "flex items-center"
    }, [
      h(NPopover, { trigger: "hover", placement: "top" }, {
        trigger: () => [
          h("span", { class: "truncate" }, `${option.label}`)
        ],
        default: () => [
          h("span", {}, `${option.label}`)
        ]
      })
    ])
  ]
}

// 数据集列表接口 noPage
async function getGroupList() {
  const recursionMapData = (data: any, label: any) => {
    const mapList = data.map((item: any, index: string | number) => {
      item.value = item.groupId || item.sonId;
      if (label) {
        // item.label = `${label} - ${item.groupName || `V${item.version}`}`;
        item.label = `${item.groupName || `V${item.version}`}`;
      } else {
        item.label = item.groupName || `V${item.version}`;
      }
      // item.label = item.groupName || `V${item.version}`;
      const children = item.dataSonResponseList || [];
      item.children = children.map((val: any) => {
        // 演示环境
        item.disabled = false;
        // val.disabled = val.count > 0 && val.progress == 100 ? false : true; // 正式环境
        val.disabled = false; // 演示环境
        return val;
      });
      if (item.children && item.children.length > 0) {
        recursionMapData(item.children, item.label);
      } else {
        delete item.children;
      }
      return item;
    });
    return mapList;
  };
  const res = await getDataSetListNoPage();
  const options = recursionMapData(res.data);
  setOptions.value = options;
}

onMounted(() => {
  getGroupList();
});
</script>

<template>
  <div class="min-h-500px flex-col-stretch gap-16px overflow-hidden lt-sm:overflow-auto">
    <UserSearch v-model:model="searchParams" @reset="resetSearchParams" @search="getDataByPage" />
    <NCard title="标注列表" :bordered="false" size="small" class="sm:flex-1-hidden card-wrapper">
      <template #header-extra>
        <TableHeaderOperation v-model:columns="columnChecks" :is-del="false" :is-add="false"
          :disabled-delete="checkedRowKeys.length === 0" :loading="loading" @add="handleAdd" @delete="handleBatchDelete"
          @refresh="getData" />
      </template>
      <NDataTable v-model:checked-row-keys="checkedRowKeys" :columns="columns" :data="data"
        :flex-height="!appStore.isMobile" :scroll-x="962" :loading="loading" remote :row-key="(row) => row.id"
        :pagination="mobilePagination" class="sm:h-full" default-expand-all />
    </NCard>
  </div>
</template>

<style scoped></style>
