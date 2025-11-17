<script setup lang="tsx">
import { NButton, NPopover, NTag, useDialog, useMessage } from "naive-ui";

import { useAppStore } from "@/store/modules/app";
import { useTable, useTableOperate } from "@/hooks/common/table";
import UserOperateDrawer from "./modules/user-operate-drawer.vue";
import UserSearch from "./modules/user-search.vue";
import { getDataSetMarkList } from "@/service/api/ano";
import { localStg } from "@/utils/storage";
import {deleteDataGroup, deleteDataSet, fetchGetDataSetList} from "@/service/api/dataManage";
import {useBoolean} from "~/packages/hooks";
import OperateModal from "@/views/data-manage/map/modules/operate-modal.vue";
import SvgIcon from "@/components/custom/svg-icon.vue";
import VersionInfo from "@/components/custom/version-info.vue";
import {SignType} from "@/views/data-manage/interface/map";
import {downloadByData} from "@/utils/common";
import axios from "axios";

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
  apiFn: fetchGetDataSetList,
  showTotal: true,
  apiParams: {
    current: 1,
    size: 10,
    groupName: null,
    dataTypeId: 1
  },
  columns: () => [
    {
      type: 'selection',
      width: 60
    },
    {
      title: "数据集组ID",
      key: "version",
      render: (row) => {
        return [h("span", {}, `${row.groupId}`)];
      },
    },
    {
      title: "数据集组名称",
      key: "groupName",
      width: "260",
      render: (row) => {
        const color = themeColor;
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
                  class: {
                    "text-[#e5e7eb]": row.count === 0,
                  },
                  style: {
                    color,
                  },
                },
                `${row.groupName}`,
              ),
            ],
          ),
        ];
      },
    },
    {
      title: "数据集类型",
      key: "dataTypeName",
    },
    {
      title: "数据量",
      key: "count",
      minWidth: 200,
      render: (row: any) => {
        // 根据row.dataSonResponseList里面的count统计
        let count = 0;
        if(row.dataSonResponseList instanceof Array && row.dataSonResponseList.length > 0) {
          count = row.dataSonResponseList.reduce((prev, item) => {
            return prev += item.count
          }, 0);
        } else {
          count = 0;
        }
        return (
          <span class="text-12px text-[#151b26]">
            {count}
          </span>
        );
      },
    },
    {
      title: "创建时间",
      key: "createTime",
    },
    {
      title: "操作",
      key: "operation",
      render(row) {
        return [
          h(
            NButton,
            {
              text: true,
              type: "info",
              style: { marginRight: "10px", fontSize: "12px" },
              onClick: () => handleHOperation({ name: "新增版本"}, row),
              disabled: row.count === 0,
            },
            "新增版本",
          ),
          h(
            NButton,
            {
              text: true,
              type: "info",
              style: { marginRight: "10px", fontSize: "12px" },
              onClick: () => handleHOperation({ name: "所有版本"}, row),
              disabled: row.count === 0,
            },
            "所有版本",
          ),
          h(
            NButton,
            {
              text: true,
              type: "info",
              style: { marginRight: "10px", fontSize: "12px" },
              onClick: () => handleHOperation({ name: "删除"}, row),
              disabled: row.count === 0,
            },
            "删除",
          ),
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
    },
  });
};
const handleOnlineAnnotation = (row) => {
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
    },
  });
};

// ----------0925 new------------
const dialog = useDialog();
const message = useMessage();
const rowData = ref<any>({});
const { bool: visible, setTrue: openModal } = useBoolean();
const isImportModal = ref<Boolean>(false);
let importList = reactive([]);
const importCols = reactive<any[]>([
  { key: "id", title: "ID" },
  { key: "fileSize", title: "文件大小" },
  { key: "count", title: "数据量" },
  { key: "nickName", title: "创建人" },
  { key: "importStartTime", title: "导入开始时间" },
  { key: "importEndTime", title: "导入完成时间" },
  {
    key: "status",
    title: "导入状态",
    render: (row: any) => {
      if (row.status === null) {
        return null;
      }

      const tagMap: Record<Api.Common.EnableStatus, NaiveUI.ThemeColor> = {
        1: "success",
        2: "warning",
      };
      const statusObj = {
        1: "导入完成",
        2: "导入失败",
      };

      const label = statusObj[row.status];

      return <NTag type={tagMap[row.status]}>{label}</NTag>;
    },
  },
]);
const isAllVerModal = ref<Boolean>(false);
const allVerCols = ref<any[]>([
  {
    title: "版本",
    key: "version",
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
                class: "text-[12px] text-[#151b26]",
              },
              `V${row.version}`,
            ),
            h(
              NPopover,
              { trigger: "hover", placement: "right" },
              {
                trigger: () =>
                  h("span", { class: "block ml-4px" }, [
                    h(SvgIcon, {
                      icon: "fluent:info-24-regular",
                      class: "text-[14px]",
                      localIcon: "fluent--info-24-regular",
                    }),
                  ]),
                default: () =>
                  h(VersionInfo, {
                    rowData: row,
                    onImport: (e) => handleImport(e),
                    onRemark: (e) => refresh(e),
                  }),
              },
            ),
          ],
        ),
      ];
    },
  },
  {
    title: "数据集ID",
    key: "sonId",
    render: (row) => {
      return <span class="text-12px text-[#151b26]">{row.sonId}</span>;
    },
  },
  {
    title: "数据集类型",
    key: "dataTypeName",
  },
  {
    title: "数据量",
    key: "count",
    minWidth: 200,
    render: (row) => {
      return (
        <span class="text-12px text-[#151b26]">
            {row.count ? row.count : 0}
          </span>
      );
    },
  },
  {
    title: "最近导入状态",
    key: "importStatus",
    minWidth: 200,
    render: (row) => {
      if (row.importStatus === null) {
        return null;
      }

      const tagMap: Record<Api.Common.EnableStatus, NaiveUI.ThemeColor> = {
        1: "success",
        2: "warning",
        0: "default",
      };
      const statusInfo = {
        0: "暂无导入记录",
        1: "已完成",
        2: "导入失败",
      };

      const label = statusInfo[row.importStatus];

      return <NTag type={tagMap[row.importStatus]}>{label}</NTag>;
    },
  },
  // {
  //   title: "标注类型 > 模版",
  //   key: "markType",
  //   minWidth: 200,
  //   render: (row) => {
  //     return (
  //       <div class="flex items-center">
  //         <span class="text-[12px] text-[#151b26]">{row.markType}</span>
  //         <span class="mx-2px text-[#151b26]"> {">"} </span>
  //         <span class="text-[12px] text-[#151b26]">{row.markTemp}</span>
  //       </div>
  //     );
  //   },
  // },
  {
    title: "标注状态",
    align: "left",
    key: "status",
    minWidth: 100,
  },
  {
    title: "操作",
    align: "left",
    width: 300,
    key: "操作",
    render(row: any) {
      const isHideBtn = +row.count == 0;
      const customClass = isHideBtn ? "hidden" : "inline-block";
      return [
        h(
          NButton,
          {
            type: "info",
            quaternary: true,
            onClick: () => handleOperation(SignType.detail, row),
            class: `h-16px ${customClass}`,
            size: "small",
          },
          "查看",
        ),
        h(
          NButton,
          {
            type: "info",
            quaternary: true,
            onClick: () => handleOperation(SignType.import, row),
            class: `h-16px`,
            size: "small",
          },
          "导入",
        ),
        h(
          NButton,
          {
            type: "info",
            quaternary: true,
            onClick: () => handleOperation(SignType.export, row),
            class: `h-16px ${customClass}`,
            size: "small",
          },
          "导出",
        ),
        h(
          NButton,
          {
            type: "info",
            quaternary: true,
            onClick: () => handleOperation(SignType.annotation, row),
            class: `h-16px ${customClass}`,
            size: "small",
          },
          "标注",
        ),
        h(
          NButton,
          {
            type: "info",
            quaternary: true,
            onClick: () => handleOperation(SignType.delete, row),
            class: "h-16px",
            size: "small",
          },
          "删除",
        ),
      ];
    },
  },
])
const handleHOperation = ({ name }: any, row) => {
  if (name === "新增版本") {
    rowData.value = row;
    openModal();
  } else if (name === "所有版本") {
    router.push({
      name: "data-manage_detail",
    });
    localStorage.setItem("rowData", JSON.stringify(row));
  } else {
    dialog.warning({
      title: "删除数据集",
      content:
        "操作删除后，数据集及全部版本数据都将会被删除且不可恢复，确认要删除吗？",
      positiveText: "确定",
      negativeText: "取消",
      onPositiveClick: () => {
        deleteDataGroup({
          groupId: row.groupId,
        }).then((res) => {
          console.log(res);
          if (res.data == 1) {
            window.$message?.success?.("删除成功！");
            getMapData();
          }
        });
      },
      onNegativeClick: () => {},
    });
  }
};
function downloadPost(config) {
  return new Promise((resolve, reject) => {
    axios({
      url: config.url, // 请求地址
      method: "post",
      data: config.data, // 参数
      responseType: "blob", // 表明返回服务器返回的数据类型
    })
      .then((res) => {
        resolve(res);
      })
      .catch((err) => {
        reject(err);
      });
  });
}
const handleOperation = async (sign: SignType, row) => {
  switch (sign) {
    case SignType.import:
      router.push({
        path: "/data-manage/import",
        query: {
          sonId: row.sonId,
          sign: "mapToImport",
        },
      });
      break;
    case SignType.export:
      const baseUrl = import.meta.env.VITE_SERVICE_BASE_URL;
      const config = {
        url: `${baseUrl}/file/download?sonId=${row.sonId}`,
        data: {
          sonId: row.sonId,
        },
      };
      const fileName = `数据集${row.sonId}.zip`;
      const res = await downloadPost(config);
      downloadByData(res.data, fileName);
      break;
    case SignType.annotation:
      router.push({
        // name: "data-ano_operation",
        // name: "data-ano_imgoperate",
        name: import.meta.env.VITE_TOGGLE_OPERATE === 'Y' ? 'data-ano_imgoperate' : 'data-ano_operation',
        params: {
          sign,
          id: row.sonId,
        },
        query: {
          id: row.sonId,
        },
      });
      break;
    case SignType.detail:
      router.push({
        name: "data-ano_detail",
        params: {
          sign,
        },
        query: {
          id: row.sonId,
        },
      });
      break;
    case SignType.delete:
      dialog.warning({
        title: "删除数据集",
        content:
          "操作删除后，数据集及全部版本数据都将会被删除且不可恢复，确认要删除吗？",
        positiveText: "确定",
        negativeText: "取消",
        onPositiveClick: () => {
          deleteDataSet({ sonId: row.sonId }).then((res) => {
            if (res.data == 1) {
              message.success("删除成功！");
              getMapData();
            }
          });
        },
        onNegativeClick: () => {
          message.error("取消");
        },
      });
      break;
    default:
      throw new Error("wrong operator");
  }
};

const refresh = () => {
  getDataByPage()
};
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
      title="数据集列表"
      :bordered="false"
      size="small"
      class="sm:flex-1-hidden card-wrapper"
    >
      <template #header-extra>
        <TableHeaderOperation
          v-model:columns="columnChecks"
          :is-del="false"
          :is-add="false"
          :disabled-delete="checkedRowKeys.length === 0"
          :loading="loading"
          @add="handleAdd"
          @delete="handleBatchDelete"
          @refresh="getData"
        />
      </template>
      <NDataTable
        v-model:checked-row-keys="checkedRowKeys"
        :columns="columns"
        :data="data"
        :flex-height="!appStore.isMobile"
        :scroll-x="962"
        :loading="loading"
        remote
        :row-key="(row) => row.id"
        :pagination="mobilePagination"
        class="sm:h-full"
        default-expand-all
      />
    </NCard>
    <OperateModal
      v-model:visible="visible"
      :row-data="rowData"
      @success="refresh"
    />
    <NModal v-model:show="isImportModal">
      <NCard
        style="width: 800px"
        title="导入记录"
        :bordered="false"
        size="huge"
        role="dialog"
        aria-modal="true"
      >
        <div class="content">
          <NDataTable
            :columns="importCols"
            :data="importList"
            :bordered="false"
          />
        </div>
        <template #footer>
          <NSpace justify="end" :size="16">
            <NButton type="primary" @click="() => (isImportModal = false)"
            >我知道了</NButton
            >
          </NSpace>
        </template>
      </NCard>
    </NModal>
  </div>
</template>

<style scoped></style>
