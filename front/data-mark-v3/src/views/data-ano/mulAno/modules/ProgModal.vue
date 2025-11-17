<script setup lang="ts">
import { watch } from "vue";
import { useTable, useTableOperate } from "@/hooks/common/table";
import { delAssessTask } from "@/service/api/model-manage";
import { NButton, NCascader, NPopconfirm, NPopover } from "naive-ui";
import {
  getByTaskIdTeamList,
  getDeptByUserList,
  taskShift,
  viewProgress,
  endUserTask,
} from "@/service/api/ano";

defineOptions({
  name: "ProgModal",
});

export type OperateType = NaiveUI.TableOperateType | "addChild";

interface Props {
  rowData?: any;
}

// Vue3 props withDefaults
const props = withDefaults(defineProps<Props>(), {
  rowData: {},
});

interface Emits {
  (e: "submitted"): void;
  (e: "close"): void;
}

const emit = defineEmits<Emits>();

const visible = defineModel<boolean>("visible", {
  default: false,
});

const id = defineModel<boolean>("id", {
  default: null,
});

const row = defineModel<any>("row", {
  default: null,
});

function closeDrawer() {
  visible.value = false;
}

const {
  columns,
  columnChecks,
  data,
  getData,
  getDataByPage,
  loading,
  mobilePagination,
  searchParams,
  updateSearchParams,
  resetSearchParams,
} = useTable({
  apiFn: viewProgress,
  showTotal: true,
  apiParams: {
    current: 1,
    size: 10,
    id: id.value,
  },
  columns: () => [
    {
      title: "标注员",
      key: "nickName",
      align: "center",
    },
    {
      title: "已标注/待标注",
      key: "markNum",
      align: "center",
    },
    {
      title: "标注进度",
      key: "progress",
      align: "center",
    },
    {
      title: "标注状态",
      key: "markState",
      align: "center",
    },
    { title: "创建时间", key: "createTime" },
    {
      key: "operate",
      title: "操作",
      align: "center",
      render: (row: any, index: string | number) => {
        const isHide = row.isHide == 1;
        return [
          h("div", { class: "flex-center gap-8px flex-wrap" }, [
            h(
              NButton,
              {
                type: "primary",
                ghost: true,
                size: "small",
                onClick: () => handleOperate("detail", row),
              },
              "详情",
            ),
            h(
              NPopconfirm,
              {
                onPositiveClick: () => handleOperate("assess", row),
              },
              {
                trigger: () =>
                  h(
                    NButton,
                    {
                      type: "primary",
                      ghost: true,
                      size: "small",
                      style: {
                        display: isHide ? "none" : "inline",
                      },
                    },
                    "结束标注",
                  ),
                default: () => h("span", {}, "是否结束标注？"),
              },
            ),
            h(
              NPopover,
              {
                trigger: "manual",
                placement: "right",
                show: row.showPopover,
              },
              {
                trigger: () => [
                  h(
                    NButton,
                    {
                      type: "primary",
                      ghost: true,
                      size: "small",
                      style: {
                        display: isHide ? "none" : "inline",
                      },
                      onClick: () => handleOperate("passTo", row),
                    },
                    "任务转交",
                  ),
                ],
                default: () => [
                  h(
                    "div",
                    {
                      class: "w-200px h-auto",
                    },
                    [
                      h(
                        "span",
                        {
                          class: "flex-center gap-8px",
                          style: "height: 32px;",
                        },
                        "请先选择转交人",
                      ),
                      h(NCascader, {
                        value: row.shiftId,
                        multiple: false,
                        cascade: true,
                        clearable: true,
                        checkStrategy: "child",
                        options: options.value,
                        onChange: (value: any) => {
                          row.shiftId = value;
                        },
                      }),
                      h(
                        "div",
                        {
                          class: "w-full flex-center gap-8px mt-8px",
                          style: "height: 32px;",
                        },
                        [
                          h(
                            NButton,
                            {
                              type: "primary",
                              ghost: true,
                              size: "small",
                              onClick: () => {
                                handleOperate("passDefine", row);
                              },
                            },
                            "确定",
                          ),
                          h(
                            NButton,
                            {
                              type: "error",
                              ghost: true,
                              size: "small",
                              onClick: () => {
                                handleOperate("passCancel", row);
                              },
                            },
                            "取消",
                          ),
                        ],
                      ),
                    ],
                  ),
                ],
              },
            ),
          ]),
        ];
      },
    },
  ],
});

const mapData = computed(() => {
  return data.value.map((item: any) => {
    item.showPopover = false;
    item.shiftId = undefined;
    return item;
  });
});

const {
  drawerVisible,
  operateType,
  editingData,
  checkedRowKeys,
  onBatchDeleted,
  onDeleted,
  // closeDrawer
} = useTableOperate(data, getData);

const options = ref<any>([]);
const shiftId = ref<string | null>(null);

const router = useRouter();

watch(visible, () => {
  if (visible.value) {
    const params = {
      current: 1,
      size: 10,
      id: id.value,
    };
    updateSearchParams(params);
    getData();
  }
});

async function handleOperate(sign: string, rowData: any) {
  try {
    switch (sign) {
      case "assess":
        const assessRes = await endUserTask({ id: rowData.id });
        if (assessRes.data) {
          window.$message?.success("标注已结束");
          getData();
        }
        break;
      case "delete":
        await delAssessTask(rowData);
        getData();
        break;
      case "detail":
        router.push({
          name: "data-ano_detail",
          params: {
            sign: "detail",
            row: rowData,
          },
          query: {
            id: row.value.sonId,
            isMany: rowData.isMany,
            markUserId: rowData.id,
          },
        });
        break;
      case "passTo":
        removeDisabledFromChildren(options.value);
        const userId = rowData.userId;
        findAndDisableById(options.value, userId);
        data.value = data.value.map((item: any) => {
          item.showPopover = false;
          return item;
        });
        await nextTick();
        rowData.showPopover = !rowData.showPopover;
        break;
      case "passDefine":
        const res = await taskShift({
          currentUserId: rowData.id,
          shiftId: rowData.shiftId,
        });
        if (res.data) {
          window.$message?.success("任务已转交");
          getData();
        }
        break;
      case "passCancel":
        removeDisabledFromChildren(options.value);
        rowData.showPopover = false;
        emit("close");
        break;
      default:
        console.warn(`未知的操作标识: ${sign}`);
    }
  } catch (error) {
    console.error(`执行操作 "${sign}" 时出错:`, error);
    window.$message?.error(`执行操作 "${sign}" 时出错，请稍后重试`);
  }
}

function convertToCascaderOptions(data) {
  return data.map((item) => {
    const option = {
      label: item.nickName,
      value: item.userId,
      // children: item.userList.map(user => ({
      //   label: user.userName,
      //   value: user.userId
      // }))
    };
    return option;
  });
}

function findAndDisableById(data, id) {
  for (let i = 0; i < data.length; i++) {
    const item = data[i];
    if (item.value === id) {
      item.disabled = true;
      return;
    }
    if (item.children && item.children.length > 0) {
      findAndDisableById(item.children, id);
    }
  }
}

function removeDisabledFromChildren(data) {
  for (let i = 0; i < data.length; i++) {
    const item = data[i];
    item.disabled = false;
    if (item.children && item.children.length > 0) {
      removeDisabledFromChildren(item.children);
    }
  }
  return data;
}

const getUserData = async () => {
  // const res = await getDeptByUserList();
  // options.value = convertToCascaderOptions(res.data);
  const res = await getByTaskIdTeamList({ taskId: row.value.id, teamType: "1" });
  options.value = convertToCascaderOptions(res.data);
};

onMounted(() => {
  getUserData();
});
</script>

<template>
  <NModal
    v-model:show="visible"
    title="任务进度"
    preset="card"
    class="w-1000px"
  >
    <div class="w-full h-auto">
      <NForm
        ref="formRef"
        :model="model"
        label-placement="left"
        :label-width="80"
      >
        <NGrid responsive="screen" item-responsive>
          <NFormItemGi span="24 m:8" label="数据集ID:" path="datasetId">
            {{ row.sonId }}
          </NFormItemGi>
          <NFormItemGi span="24 m:6" label="任务名称:" path="datasetId">
            {{ row.taskName }}
          </NFormItemGi>
          <!-- 任务总量 -->
          <NFormItemGi span="24 m:6" label="任务总量:" path="datasetId">
            {{ row.fileNum }}
          </NFormItemGi>
        </NGrid>
        <NGrid responsive="screen" item-responsive class="-mt-12px">
          <!-- 团队名称 -->
           <NFormItemGi span="24 m:8" label="团队名称:" path="datasetId">
            {{ row.teamName }}
          </NFormItemGi>
          <!-- 团队成员 -->
          <NFormItemGi span="24 m:6" label="团队成员:" path="datasetId">
            {{ row.userNum }}
          </NFormItemGi>
        </NGrid>
      </NForm>
      <NDataTable
        v-model:checked-row-keys="checkedRowKeys"
        :columns="columns"
        :data="mapData"
        size="small"
        :loading="loading"
        remote
        :row-key="(row) => row.id"
        :pagination="mobilePagination"
        class="sm:h-full"
      />
    </div>
    <template #footer>
      <NSpace justify="end" :size="16">
        <NButton @click="closeDrawer">关闭</NButton>
        <!-- <NButton type="primary" @click="handleSubmit">{{ $t('common.confirm') }}</NButton> -->
      </NSpace>
    </template>
  </NModal>
</template>

<style scoped></style>
