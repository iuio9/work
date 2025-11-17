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
  distributionExamine,
  examineTeamInfo,
  confirmAudit,
  examineDetails,
  examineTaskShift
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

const isAuditLoading = defineModel<boolean>("isAuditLoading", {
  default: false,
});

const id = defineModel<boolean>("id", {
  default: null,
});

const row = defineModel<any>("row", {
  default: null,
});


// watch 监听visible变化，调用examineTeamInfo
watch(
  visible,
  async (newValue, oldValue) => {
    const res = await distributionExamine({ taskId: id.value });
    if (res.data) {
      data.value = [];
      isAuditLoading.value = false;
      if (newValue) {
        examineTeamInfo({
          id: id.value,
        }).then((res) => {
          row.value = res.data;
        });
      };
      await getDataByPage();
    }
    // if (newValue) {
    //   examineTeamInfo({
    //     id: id.value,
    //   }).then((res) => {
    //     console.log('res: ', res);
    //     row.value = res.data;
    //   });
    // }
  },
  {
    immediate: true,
  },
);


function closeDrawer() {
  visible.value = false;
  emit("close");
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
  apiFn: examineDetails,
  showTotal: true,
  apiParams: {
    current: 1,
    size: 10,
    taskId: id.value,
  },
  columns: () => [
    {
      title: "标注员",
      key: "markName",
      align: "center",
    },
    {
      title: "审核员",
      key: "auditName",
      align: "center",
    },
    {
      title: "审核数量",
      key: "auditNum",
      align: "center",
    },
    {
      title: "进度",
      key: "progress",
      align: "center",
    },
    {
      title: "审核状态",
      key: "auditState",
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
  immediate: false
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
        console.log("assessRes: ", assessRes);
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
        await getUserData()
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
        const res = await examineTaskShift({
          currentUserId: rowData.id,
          shiftId: rowData.shiftId,
          id: id.value
        });
        if (res.data) {
          window.$message?.success("任务已转交");
          getData();
        }
        break;
      case "passCancel":
        removeDisabledFromChildren(options.value);
        rowData.showPopover = false;
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
  const res = await getByTaskIdTeamList({ taskId: id.value, teamType: "2" });
  options.value = convertToCascaderOptions(res.data);
};

const handleAuditDefine = async () => {
  try {
    const res = await confirmAudit({ id: id.value });
    if (res.data) {
      window.$message?.success?.("审核分配成功！");
      emit("submitted");
    }
  } catch (error) {
    console.error("审核分配出错:", error);
    window.$message?.error?.("审核分配失败，请稍后重试！");
  }
};

onMounted(() => {
  getUserData();
});
</script>

<template>
  <NModal v-model:show="visible" title="分配审核任务" preset="card" class="w-1000px">
    <div class="w-full h-auto relative">
      <div v-show="isAuditLoading" class="mask-layer">
        <div class="loading-spinner">
          <NSpin size="large" description="" />
        </div>
      </div>
      <NForm ref="formRef" :model="model" label-placement="left" :label-width="120">
        <NGrid responsive="screen" item-responsive>
          <NFormItemGi span="24 m:8" label="任务名称:" path="datasetId">
            {{ row.taskName }}
          </NFormItemGi>
          <NFormItemGi span="24 m:8" label="数据集ID:" path="datasetId">
            {{ row.sonId }}
          </NFormItemGi>
          <!-- 标注团队名称 -->
          <NFormItemGi span="24 m:8" label="标注团队名称:" path="markTeamName">
            {{ row.markTeamName }}
          </NFormItemGi>
        </NGrid>
        <NGrid responsive="screen" item-responsive class="-mt-12px">
          <!-- 标注团队人数 -->
          <NFormItemGi span="24 m:8" label="标注团队人数:" path="markTeamNumber">
            {{ row.markTeamNumber }}
          </NFormItemGi>
          <!-- 审核团队名称-->
          <NFormItemGi span="24 m:8" label="审核团队名称:" path="auditTeamName">
            {{ row.auditTeamName }}
          </NFormItemGi>
          <!-- 审核团队人数 -->
          <NFormItemGi span="24 m:8" label="审核团队人数:" path="auditTeamNumber">
            {{ row.auditTeamNumber }}
          </NFormItemGi>
        </NGrid>

      </NForm>
      <NDataTable v-model:checked-row-keys="checkedRowKeys" :columns="columns" :data="mapData" size="small"
        :loading="loading" remote :row-key="(row) => row.id" :pagination="mobilePagination" class="sm:h-full" />
    </div>
    <template #footer>
      <NSpace justify="end" :size="16">
        <NButton @click="closeDrawer">关闭</NButton>
        <NButton type="primary" @click="handleAuditDefine" :disabled="isAuditLoading">确认分配</NButton>
      </NSpace>
    </template>
  </NModal>
</template>

<style scoped>
.mask-layer {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  background-color: rgba(255, 255, 255, 0.7);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 9999;
}
</style>
