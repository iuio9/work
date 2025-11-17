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
  examineDetails,
  examineReturn,
  approved
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
  (e: "audit"): void;
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
      type: 'selection'
    },
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
            // h(
            //   NPopconfirm,
            //   {
            //     onPositiveClick: () => handleOperate("assess", row),
            //   },
            //   {
            //     trigger: () =>
            //       h(
            //         NButton,
            //         {
            //           type: "primary",
            //           ghost: true,
            //           size: "small",
            //           style: {
            //             display: isHide ? "none" : "inline",
            //           },
            //         },
            //         "结束标注",
            //       ),
            //     default: () => h("span", {}, "是否结束标注？"),
            //   },
            // ),
            // h(
            //   NPopover,
            //   {
            //     trigger: "manual",
            //     placement: "right",
            //     show: row.showPopover,
            //   },
            //   {
            //     trigger: () => [
            //       h(
            //         NButton,
            //         {
            //           type: "primary",
            //           ghost: true,
            //           size: "small",
            //           style: {
            //             display: isHide ? "none" : "inline",
            //           },
            //           onClick: () => handleOperate("passTo", row),
            //         },
            //         "任务转交",
            //       ),
            //     ],
            //     default: () => [
            //       h(
            //         "div",
            //         {
            //           class: "w-200px h-auto",
            //         },
            //         [
            //           h(
            //             "span",
            //             {
            //               class: "flex-center gap-8px",
            //               style: "height: 32px;",
            //             },
            //             "请先选择转交人",
            //           ),
            //           h(NCascader, {
            //             value: row.shiftId,
            //             multiple: false,
            //             cascade: true,
            //             clearable: true,
            //             checkStrategy: "child",
            //             options: options.value,
            //             onChange: (value: any) => {
            //               row.shiftId = value;
            //             },
            //           }),
            //           h(
            //             "div",
            //             {
            //               class: "w-full flex-center gap-8px mt-8px",
            //               style: "height: 32px;",
            //             },
            //             [
            //               h(
            //                 NButton,
            //                 {
            //                   type: "primary",
            //                   ghost: true,
            //                   size: "small",
            //                   onClick: () => {
            //                     handleOperate("passDefine", row);
            //                   },
            //                 },
            //                 "确定",
            //               ),
            //               h(
            //                 NButton,
            //                 {
            //                   type: "error",
            //                   ghost: true,
            //                   size: "small",
            //                   onClick: () => {
            //                     handleOperate("passCancel", row);
            //                   },
            //                 },
            //                 "取消",
            //               ),
            //             ],
            //           ),
            //         ],
            //       ),
            //     ],
            //   },
            // ),
            h(
              NButton,
              {
                type: "primary",
                ghost: true,
                size: "small",
                onClick: () => handleDetail(row),
              },
              "详情",
            ),
            h(NPopconfirm, {
              style: {
                display: isHide ? "none" : "",
              },
              onPositiveClick: (event) => handleOperate("taskReturn", row)
            }, {
              default: () => h("span", {}, "是否打回任务？"),
              trigger: () => h(
                NButton,
                {
                  type: "primary",
                  ghost: true,
                  size: "small",
                  style: {
                    display: isHide ? "none" : "inline",
                  },
                },
                "任务打回",
              ),
            })
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
        break;
      case "taskReturn":
        const returnRes = await examineReturn({
          taskId: id.value,
          ids: [rowData.id],
        })
        if (returnRes.data) {
          window.$message?.success("任务已打回");
          getData();
        }
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
  const res = await getByTaskIdTeamList({
    taskId: row.value.id,
    teamType: "1",
  });
  options.value = convertToCascaderOptions(res.data);
};

const handleBatchBack = async () => {

  const res = await examineReturn({
    taskId: id.value,
    ids: checkedRowKeys.value
  });
  if (res.data >= 1) {
    window.$message?.success("批量打回成功");
    getData();
  }
};

const handleAllBack = async () => {
  const res = await examineReturn({
    taskId: id.value,
  });
  if (res.data >= 1) {
    window.$message?.success("全部打回成功");
    getData();
  }
};

const handleAuditDefine = async () => {
  const res = await approved({
    taskId: id.value,
  });
  if (res.data) {
    window.$message?.success("审核通过");
    getData();
    emit('audit');
    visible.value = false;
  }
}

const handleDetail = (rowData: any) => {
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
      sign: 'audit'
    },
  });
}

onMounted(() => {
  getUserData();
});
</script>

<template>
  <NModal v-model:show="visible" title="审核详情" preset="card" class="w-1000px">
    <div class="w-full h-auto">
      <NForm ref="formRef" :model="model" label-placement="left" :label-width="120">
        <NGrid responsive="screen" item-responsive>
          <NFormItemGi span="24 m:6" label="任务名称:" path="datasetId">
            {{ row.taskName }}
          </NFormItemGi>
          <!-- 标注团队名称 -->
          <NFormItemGi span="24 m:6" label="标注团队名称:" path="markTeamName">
            {{ row.markTeamName }}
          </NFormItemGi>
          <!-- 标注团队人数 -->
          <NFormItemGi span="24 m:6" label="标注团队人数:" path="markTeamNumber">
            {{ row.markTeamNumber }}
          </NFormItemGi>
        </NGrid>
        <NGrid responsive="screen" item-responsive class="-mt-12px">
          <!-- 审核团队名称-->
          <NFormItemGi span="24 m:6" label="审核团队名称:" path="auditTeamName">
            {{ row.auditTeamName }}
          </NFormItemGi>
          <!-- 审核团队人数 -->
          <NFormItemGi span="24 m:6" label="审核团队人数:" path="auditTeamNumber">
            {{ row.auditTeamNumber }}
          </NFormItemGi>
          <NFormItemGi span="24 m:8" label="数据集ID:" path="datasetId">
            {{ row.sonId }}
          </NFormItemGi>
        </NGrid>
      </NForm>
      <TableHeaderOperation v-model:columns="columnChecks" :disabled-delete="checkedRowKeys.length === 0" :isAdd="false"
        :isDel="false" :loading="loading" @refresh="getData" class="mb-8px">
        <template #prefix>
          <NButton size="small" ghost type="primary" @click="handleBatchBack()" class="" :disabled="checkedRowKeys.length == 0">
            <span>批量打回</span>
          </NButton>
          <NButton size="small" ghost type="primary" @click="handleAllBack()" class="-mr-24px">
            <span>全部打回</span>
          </NButton>
        </template>
      </TableHeaderOperation>
      <NDataTable v-model:checked-row-keys="checkedRowKeys" :columns="columns" :data="mapData" size="small"
        :loading="loading" remote :row-key="(row) => row.id" :pagination="mobilePagination" class="sm:h-full" />
    </div>
    <template #footer>
      <NSpace justify="end" :size="16">
        <n-popconfirm @positive-click="handleAuditDefine">
          <template #trigger>
            <NButton type="primary">审核通过</NButton>
          </template>
          <span>是否确定审核通过？</span>
        </n-popconfirm>
        <NButton @click="closeDrawer">关闭</NButton>
      </NSpace>
    </template>
  </NModal>
</template>

<style scoped></style>
