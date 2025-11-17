<script setup lang="ts">
import { NButton, NTag, useDialog, useMessage } from "naive-ui";
import { SignType } from "@/views/data-manage/interface/map";
import { useBoolean } from "~/packages/hooks";
import OperateModal from "../map/modules/operate-modal.vue";
import {
  deleteDataGroup,
  deleteDataSet,
  fetchImportList,
  updateDataSetName,
} from "@/service/api/dataManage";
import { NPopover } from "naive-ui";
import SvgIcon from "@/components/custom/svg-icon.vue";
import VersionInfo from "@/components/custom/version-info.vue";

interface DataManageObj {
  title: string;
  infoList: any[];
  operationBtns: any[];
}

const infoObj = ref({
  icon: "fluent:ios-arrow-24-filled",
  localIcon: "fluent--ios-arrow-24-filled",
  title: "",
});
const dataManageObj = ref<DataManageObj>({
  title: "介绍",
  infoList: [
    {
      name: "数据采集",
      info: "支持采集图片类数据，可以从本地接入视频抽帧图片或通过接入云服务调用数据接入图片",
      btns: [],
      icon: "data-collect",
    },
    {
      name: "数据质检",
      info: "支持对图像数据进行质检，质检报告中的指标可作为数据处理（标注、清洗）的重要参考",
      btns: [],
      icon: "data-qc",
    },
    {
      name: "数据智能处理",
      info: "⽀持对图⽚和⽂本数据进⾏清洗，以及对图⽚数据进⾏增强处理，您可按需选择数据智能处理功能",
      btns: [],
      icon: "data-intellect",
    },
    {
      name: "数据标注",
      info: "支持图片、文本、音频、视频数据标注，并支持丰富标注模板，个人在线标注及智能标注等多种方式",
      btns: [
        {
          name: "在线标注",
          routeName: "annotation",
        },
      ],
      icon: "data-annotation",
    },
  ],
  operationBtns: [
    // { icon: "material-symbols:add", name: "新增版本" },
    // { icon: "material-symbols-light:border-all", name: "所有版本" },
    // { icon: "material-symbols-light:delete", name: "删除" },
  ],
});
const tableObj = ref<any>({
  columns: [
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
                      onRemark: (e) => refresh(e, row),
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
        return [
          h(
            "span",
            {
              class: "text-[12px] text-[#151b26]",
            },
            `${row.sonId}`,
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
      render: (row) => {
        return [
          h(
            "span",
            {
              class: "text-12px text-[#151b26]",
            },
            `${!!row.count ? row.count : "0"}`,
          ),
        ];
      },
    },
    {
      title: "最近导入状态",
      key: "最近导入状态",
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
        return [
          h(
            NTag,
            {
              type: `${tagMap[row.importStatus]}`,
            },
            `${label}`,
          ),
        ];
      },
    },
    // {
    //   title: "标注类型 > 模版",
    //   key: "markType",
    //   minWidth: 200,
    //   render: (row) => {
    //     return [
    //       h(
    //         "div",
    //         {
    //           class: "flex items-center",
    //         },
    //         [
    //           h(
    //             "span",
    //             { class: "text-[12px] text-[#151b26]" },
    //             `${row.markType}`,
    //           ),
    //           h("span", { class: "mx-2px text-[#151b26]" }, " > "),
    //           h(
    //             "span",
    //             { class: "text-[12px] text-[#151b26]" },
    //             `${row.markTemp}`,
    //           ),
    //         ],
    //       ),
    //     ];
    //   },
    // },
    {
      title: "标注状态",
      align: "left",
      key: "status",
      minWidth: 100,
    },
    // {
    //   title: "操作",
    //   align: "center",
    //   key: "操作",
    //   render(row) {
    //     return [
    //       h(
    //         NButton,
    //         {
    //           type: "info",
    //           quaternary: true,
    //           onClick: () => handleOperation(SignType.detail),
    //           class: "h-16px",
    //           size: "small",
    //         },
    //         "查看",
    //       ),
    //       h(
    //         NButton,
    //         {
    //           type: "info",
    //           quaternary: true,
    //           onClick: () => handleOperation(SignType.import),
    //           class: "h-16px",
    //           size: "small",
    //         },
    //         "导入",
    //       ),
    //       h(
    //         NButton,
    //         {
    //           type: "info",
    //           quaternary: true,
    //           onClick: () => handleOperation(SignType.export),
    //           class: "h-16px",
    //           size: "small",
    //         },
    //         "导出",
    //       ),
    //       h(
    //         NButton,
    //         {
    //           type: "info",
    //           quaternary: true,
    //           onClick: () => handleOperation(SignType.annotation),
    //           class: "h-16px",
    //           size: "small",
    //         },
    //         "标注",
    //       ),
    //       h(
    //         NButton,
    //         {
    //           type: "info",
    //           quaternary: true,
    //           onClick: () => handleOperation(SignType.delete, row),
    //           class: "h-16px",
    //           size: "small",
    //         },
    //         "删除",
    //       ),
    //     ];
    //   },
    // },
  ],
  pagination: false,
  data: [{}],
});
const router = useRouter();
const dialog = useDialog();
const message = useMessage();
const { bool: visible, setTrue: openModal } = useBoolean();

const dataList = ref<any>([]);
const rowData = ref<any>({});

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
      // <NTag type = {tagMap[row.status]} > {label} < /NTag>
      return [
        h(
          NTag,
          {
            type: `${tagMap[row.status]}`,
          },
          `${label}`,
        ),
      ];
    },
  },
]);

// methods
const handleBack = () => {
  router.back();
};
const handleOperation = (sign: SignType, row) => {
  switch (sign) {
    case SignType.import:
      router.push({
        name: "data-manage_import",
        params: {
          sign,
        },
      });
      break;
    case SignType.export:
      router.push({
        name: "data-manage_export",
        params: {
          sign,
        },
      });
      break;
    case SignType.annotation:
      router.push({
        // name: "data-ano_operation",
        // name: "data-ano_imgoperate",
        name: import.meta.env.VITE_TOGGLE_OPERATE === 'Y' ? 'data-ano_imgoperate' : 'data-ano_operation',
        params: {
          sign,
        },
      });
      break;
    case SignType.detail:
      router.push({
        name: "data-manage_detail",
        params: {
          sign,
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
              message.success("删除成功");
              dataList.value[0].dataSonResponseList =
                dataList.value[0].dataSonResponseList.filter((val) => {
                  return val.sonId !== row.sonId;
                });
              const len = dataList.value[0].dataSonResponseList.length;
              if (len == 0) {
                router.back();
                localStorage.remove("rowData");
              }
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
          if (res.data == 1) {
            window.$message?.success?.("删除成功！");
            dataList.value = [];
            localStorage.removeItem("rowData");
            router.back();
          }
        });
      },
      onNegativeClick: () => {},
    });
  }
};

const refresh = (e, row) => {
  // getMapData({page: 1, limit: 10});
  row.remark = e.remark;
};

const handleImport = async (e) => {
  const res = await fetchImportList({
    sonId: e.sonId,
  });
  importList = [...res.data];
  isImportModal.value = true;
};

// update groupName
const handleDefine = async (item: any) => {
  // request
  const res = await updateDataSetName({
    groupId: item.groupId,
    groupName: item.groupName,
  });
  if (res.data == 1) {
    window.$message?.success("数据集名称修改成功！");
    item.isGroup = false;
  }
};
const handleCancel = (item: any) => {
  item.isGroup = false;
};

const handleUpdateGroup = (item: any) => {
  item.isGroup = true;
};

onMounted(() => {
  dataList.value = [JSON.parse(localStorage.getItem("rowData"))].map((item) => {
    item.isGroup = false;
    return item;
  });
});

onBeforeRouteLeave((to) => {
  if (to.path === "/data-manage/map") {
    localStorage.removeItem("rowData");
  }
});
</script>

<template>
  <div class="wrap_container flex_col_start">
    <div class="header">
      <div class="h_back flex_start" @click="handleBack()">
        <!--<SvgIcon
          :icon="infoObj.icon"
          :local-icon="infoObj.localIcon"
          class="inline-block align-text-bottom text-16px"
        />-->
        <SvgIcon
          local-icon="carbon--return"
          class="inline-block align-text-bottom text-16px text-[#000000]"
        />
        <span>返回</span>
      </div>
      <div class="h_title">{{ infoObj.title }}</div>
    </div>
    <div class="content flex_col_start">
      <n-card size="small" class="card-wrapper">
        <div
          class="flex-vertical-stretch gap-16px overflow-hidden <sm:overflow-auto"
        >
          <NCard
            :bordered="false"
            size="small"
            class="sm:flex-1-hidden card-wrapper"
          >
            <div class="table_content w-full flex-1">
              <div
                v-for="(item, index) of dataList"
                class="item-table mb-16px"
                :key="item.id"
              >
                <div class="item-table_header">
                  <div class="left">
                    <span
                      class="block text-[#252933] cursor-pointer"
                      v-if="!item.isGroup"
                      @click="handleUpdateGroup(item)"
                      >{{ item.groupName }}</span
                    >
                    <div class="flex items-center" v-else="item.isGroup">
                      <n-input
                        v-model:value="item.groupName"
                        placeholder="请输入数据集名称"
                      />
                      <div class="ml-14px flex items-center gap-4px">
                        <n-button
                          quaternary
                          type="info"
                          @click="handleDefine(item)"
                          size="small"
                        >
                          确定
                        </n-button>
                        <n-button
                          quaternary
                          @click="handleCancel(item)"
                          size="small"
                        >
                          取消
                        </n-button>
                      </div>
                    </div>
                    <span>数据集组ID: {{ item.groupId }}</span>
                  </div>
                  <div class="right">
                    <div
                      v-for="(val, index) of dataManageObj.operationBtns"
                      :key="index"
                      class="item_btn cursor-pointer"
                      @click="handleHOperation(val, item)"
                    >
                      <SvgIcon
                        :icon="val.icon"
                        class="inline-block align-text-bottom text-18px"
                      />
                      <span>{{ val.name }}</span>
                    </div>
                  </div>
                </div>
                <n-data-table
                  size="small"
                  :columns="tableObj.columns"
                  :data="item.dataSonResponseList"
                  :pagination="tableObj.pagination"
                  :bordered="false"
                />
              </div>
            </div>
          </NCard>
        </div>
      </n-card>
    </div>
    <OperateModal
      v-model:visible="visible"
      :rowData="rowData"
      @success="refresh"
    />
    <n-modal v-model:show="isImportModal">
      <n-card
        style="width: 800px"
        title="导入记录"
        :bordered="false"
        size="huge"
        role="dialog"
        aria-modal="true"
      >
        <div class="content">
          <n-data-table
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
      </n-card>
    </n-modal>
  </div>
</template>

<style scoped lang="scss">
.flex_start {
  display: flex;
  justify-content: flex-start;
  align-items: center;
}

.flex_center {
  display: flex;
  justify-content: center;
  align-items: center;
}

.flex_between {
  display: flex;
  justify-content: space-around;
  align-items: center;
}

.flex_around {
  display: flex;
  justify-content: space-around;
  align-items: center;
}

.flex_col_start {
  display: flex;
  flex-direction: column;
  justify-content: flex-start;
  align-items: center;
}

.flex_col_center {
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
}

.wrap_container {
  padding: 0;
  width: 100%;
  height: 100%;
  background-color: #f7f7f9;

  .header {
    width: 100%;
    padding: 0 16px;
    box-sizing: border-box;
    height: 48px;
    background-color: #fff;
    display: flex;
    justify-content: flex-start;
    align-items: center;

    .h_back {
      span {
        color: #303540;
        font-size: 12px;
        cursor: pointer;
      }
    }

    .h_title {
      margin-left: 16px;
      font-size: 16px;
      color: #151b26;
      font-weight: 500;
    }
  }

  .content {
    padding: 16px 24px;
    box-sizing: border-box;
    width: 100%;
    flex: 1;

    .item_card {
      width: 100%;
      min-height: 240px;

      .form_annotationType {
        padding: 8px;
        box-sizing: border-box;
        width: 128px;
        height: 128px;
        border: 1px solid #1a73e8;
        border-radius: 4px;
      }
    }
  }

  .footer {
    width: 100%;
    height: 60px;
    background-color: #fff;
    padding: 0 24px;
    box-sizing: border-box;

    .n-button {
      margin-right: 24px;
    }
  }
}

.card-wrapper {
  border-radius: 8px;
}

.item-manage {
  flex: 1;

  .item-manage_icon {
    display: flex;
    flex-direction: column;
    justify-content: center;
    flex-wrap: wrap;
    align-items: center;

    .iconName {
      font-size: 14px;
      color: #151b26;
      line-height: 22px;
      margin: 10px 0 8px;
      text-align: center;
    }
  }

  .item-manage_info {
    font-size: 12px;
    color: #84868c;
    line-height: 20px;
    margin-bottom: 8px;
    overflow: hidden;
    text-overflow: ellipsis;
    display: -webkit-box;
    -webkit-line-clamp: 3;
    -webkit-box-orient: vertical;
  }

  .item-manage_btnC {
    display: flex;
    justify-content: center;
    align-items: center;

    .btn {
      color: #2468f2;
      font-size: 12px;
    }
  }
}

.table_header {
  display: flex;
  justify-content: space-between;
  align-items: center;

  .header_l {
  }

  .header_r {
    display: flex;
    justify-content: flex-start;
    align-items: center;

    .header_r_ipt {
      width: 200px;
    }

    .header_r_btn {
      margin-left: 4px;
      height: 34px;
    }
  }
}

.table_content {
  margin-top: 16px;

  .item-table_header {
    height: 52px;
    padding: 15px;
    background-color: #f7f7f7;
    border-bottom: 1px solid #eee;
    display: flex;
    justify-content: space-between;
    align-items: center;

    .left {
      display: flex;
      justify-content: space-between;
      align-items: center;

      span {
        font-size: 12px;
      }

      span:nth-of-type(2) {
        margin-left: 8px;
      }
    }

    .right {
      display: flex;
      justify-content: flex-start;
      align-items: center;

      .item_btn {
        margin-right: 24px;
        display: flex;
        justify-content: center;
        align-items: center;

        span {
          font-size: 12px;
          margin-left: 3px;
        }
      }
    }
  }

  .item-operation {
    .n-button {
      margin-right: 14px;
    }
  }
}
</style>
