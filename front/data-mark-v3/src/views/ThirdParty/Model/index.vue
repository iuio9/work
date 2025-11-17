<script setup lang="tsx">
import { NButton, NPopconfirm } from "naive-ui";
import { $t } from "@/locales";
import { useAppStore } from "@/store/modules/app";
import { useTable, useTableOperate } from "@/hooks/common/table";
import { delExample, getExamplePage } from "@/service/api/dataManage";
import { useBoolean } from "~/packages/hooks";
import UserOperateDrawer from "./modules/user-operate-drawer.vue";
import UserSearch from "./modules/user-search.vue";
import MenuOperateModal from "./modules/menu-operate-modal.vue";
import {enableStatusRecord} from "@/constants/business";

const appStore = useAppStore();
const router = useRouter();
const route = useRoute();
const { bool: visible, setTrue: openModal } = useBoolean();
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
  sign: "id",
  apiFn: getExamplePage,
  showTotal: true,
  apiParams: {
    current: 1,
    size: 10,
    // if you want to use the searchParams in Form, you need to define the following properties, and the value is null
    // the value can not be undefined, otherwise the property in Form will not be reactive
    algorithmName: null,
    // modelId: route.query.modelId
  },
  columns: () => [
    {
      type: "selection",
      align: "center",
      width: 48,
      fixed: 'left'
    },
    {
      key: "index",
      title: $t("common.index"),
      align: "center",
      width: 64,
      fixed: 'left'
    },
    {
      title: "模型名称",
      key: "algorithmName",
      width: 150,
      ellipsis: {
        tooltip: true,
      },
      fixed: 'left'
    },
    {
      title: "模型物理路径",
      key: "algorithmUrl",
      width: 220,
      ellipsis: {
        tooltip: true,
      },
    },
    // 模型输出类型
    {
      title: "模型输出类型",
      key: "type",
      width: 150,
      ellipsis: {
        tooltip: true,
      },
    },
    {
      key: 'status',
      title: "测试状态",
      align: 'center',
      width: 100,
    },
    {
      title: "输入地址",
      key: "importUrl",
      width: 220,
      ellipsis: {
        tooltip: true,
      },
    },
    {
      title: "输出地址",
      key: "exportUrl",
      width: 220,
      ellipsis: {
        tooltip: true,
      },
    },
    {
      title: "模型描述",
      key: "algorithmDesc",
      width: 240,
      ellipsis: {
        tooltip: true,
      },
    },
    {
      title: "创建时间",
      key: "createTime",
      width: 180,
      ellipsis: {
        tooltip: true,
      },
    },
    {
      key: "operate",
      title: $t("common.operate"),
      align: "center",
      width: 260,
      fixed: 'right',
      render: (row: any) => {
        const isOperate = Boolean(row.isDelete);
        const isTrainSuccess = Boolean(row.status ? row.status : false);
        return [
          h("div", { class: "flex-center gap-8px" }, [
            h(
              NButton,
              {
                disabled: isOperate,
                type: "primary",
                ghost: true,
                size: "small",
                onClick: () => createEditModel(row),
              },
              $t("common.edit")
            ),
            h(
              NPopconfirm,
              {
                onPositiveClick: () => handleDelete(row.modelId),
              },
              {
                default: () => h("span", {}, $t("common.confirmDelete")),
                trigger: () =>
                  h(
                    NButton,
                    {
                      disabled: isOperate,
                      type: "error",
                      ghost: true,
                      size: "small",
                    },
                    $t("common.delete")
                  ),
              }
            ),
          ]),
        ];
      },
    },
  ],
});

const {
  drawerVisible,
  operateType,
  editingData,
  checkedRowKeys,
  onBatchDeleted,
  onDeleted,
  handleAdd,
  handleEdit,
  // closeDrawer
} = useTableOperate(data, getData);

async function handleBatchDelete() {
  // request
  const res = await delExample(checkedRowKeys.value);
  if (res.data >= 1 || !res.data) {
    onBatchDeleted();
  }
}

async function handleDelete(id: number) {
  // request
  const res = await delExample([id]);
  if (res.data >= 1 || !res.data) {
    onDeleted();
  }
}

const mapEditingData = (data: any) => {
  return { ...data, modelId: route.query.modelId };
};

function handleOperate(type: string, row: any) {
  if (type === "edit") {
    handleEdit(row.id);
  } else if (type === "config") {
    row.query = [
      { key: "训练轮数(epochs)", value: "1", desc: "参数1描述" },
      { key: "元器件(异常)检测-epochs", value: "2", desc: "参数2描述" },
      { key: "元器件(异常)检测学习率", value: "3", desc: "参数3描述" },
    ];
    operateType.value = "edit";
    editingData.value = { ...row };
    openModal();
  } else if (type === "delete") {
    handleDelete(row.modelId);
  }
}

// 创建模型 编辑模型
function createEditModel(row: any) {
  router.push({
    name: "thirdparty_operate",
    query: {
      modelId: row?.modelId,
    },
  });
}
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
      title="模型列表"
      :bordered="false"
      size="small"
      class="sm:flex-1-hidden card-wrapper"
    >
      <template #header-extra>
        <TableHeaderOperation
          v-model:columns="columnChecks"
          :disabled-delete="checkedRowKeys.length === 0"
          :loading="loading"
          :is-add="true"
          addText="创建模型"
          :is-del="false"
          @add="createEditModel"
          @delete="handleBatchDelete"
          @refresh="getData"
        >
          <template #prefix>
            <NButton
              type="primary"
              ghost
              size="small"
            >
              <template #icon>
                <svg-icon
                  local-icon="ThirdParty_Test"
                  class="text-[24px]"
                ></svg-icon>
              </template>
              一键测试
            </NButton>
            <NButton
              type="primary"
              ghost
              size="small"
            >
              <template #icon>
                <svg-icon
                  local-icon="ix&#45;&#45;operate-plant"
                  class="text-[24px]"
                ></svg-icon>
              </template>
                接口文档说明
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
        :scroll-x="1800"
        :loading="loading"
        remote
        :row-key="(row) => row.modelId"
        :pagination="mobilePagination"
        class="sm:h-full"
      />
      <UserOperateDrawer
        v-model:visible="drawerVisible"
        :operate-type="operateType"
        :row-data="mapEditingData(editingData)"
        @submitted="getDataByPage"
      />
      <MenuOperateModal
        v-model:visible="visible"
        :operate-type="operateType"
        :row-data="editingData"
        @submitted="getDataByPage"
      />
    </NCard>
  </div>
</template>

<style scoped></style>
