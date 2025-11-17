<script setup lang="tsx">
// 第三方库
import { NButton, NPopconfirm, NProgress } from "naive-ui";
import { useBoolean } from "~/packages/hooks";

// 自定义库和工具
import { useAppStore } from "@/store/modules/app";
import { useTable, useTableOperate } from "@/hooks/common/table";

// API 服务
import {
  delExample,
  getExamplePage,
  getTaskPage,
  trainAssess,
  trainStop,
} from "@/service/api/dataManage";
import {
  endTask,
  deleteTask,
  getMyReceiveList,
  getMyCreateTaskList,
  isManyTask,
  submitTask,
  submitTaskPrompt,
  distributionExamine,
  myExamineTaskList,
  submitTaskPrompt2,
  submitExamineTask2,
  returnTask,
  verifyComplete,
  examineTeamInfo,
  examineReturn
} from "@/service/api/ano";

// 组件
import ProgModal from "./modules/ProgModal.vue";
import ReviewDetailModal from "./modules/ReviewDetailModal.vue";
import AuditModal from "./modules/AuditModal.vue";
import SvgIcon from "@/components/custom/svg-icon.vue";
import TagOperateModal from "./modules/TagOperateModal.vue";

const appStore = useAppStore();
const { bool: visible, setTrue: openModal } = useBoolean();
const { bool: visible1, setTrue: openModal1 } = useBoolean();
const { bool: tagVisible, setTrue: openTagModal } = useBoolean();
const {
  columns,
  columnChecks,
  data,
  getData,
  getDataByPage,
  loading,
  mobilePagination,
  resetSearchParams,
} = useTable({
  sign: "id",
  apiFn: getMyCreateTaskList,
  showTotal: true,
  apiParams: {
    current: 1,
    size: 10,
  },
  columns: () => [
    {
      title: "任务ID",
      key: "id",
      width: 100,
      align: "center",
      fixed: "left",
    },
    {
      title: "任务名称",
      key: "taskName",
      width: 200,
      align: "center",
      fixed: "left",
    },
    {
      title: "数据集ID",
      key: "sonId",
      width: 150,
      align: "center",
    },
    {
      title: "来源数据集",
      key: "sonName",
      width: 200,
      align: "center",
    },
    {
      title: "标注类型",
      key: "anoType",
      width: 150,
      align: "center",
    },
    {
      title: "任务状态",
      key: "taskState",
      width: 150,
      align: "center",
      render: (row: any) => {
        const taskStatusMap: Record<string, string> = {
          "1": "任务分配中",
          "2": "未开始",
          "3": "标注中",
          "4": "标注已完成",
          "5": "待审核",
          "6": "审核中",
          "7": "审核完成",
          "8": "任务结束",
        };
        return taskStatusMap[row.taskState] || "";
      },
    },
    {
      title: "创建时间",
      key: "createTime",
      width: 200,
      align: "center",
    },
    {
      key: "operate",
      title: "操作",
      align: "center",
      width: 300,
      fixed: "right",
      render: (row: any) => {
        const taskState = +row.taskState;
        return getButtonsByStatus(taskState, row);
      },
    },
  ],
});

const {
  columns: columns1,
  columnChecks: columnChecks1,
  data: data1,
  getData: getData1,
  getDataByPage: getDataByPage1,
  loading: loading1,
  mobilePagination: mobilePagination1,
  searchParams: searchParams1,
  resetSearchParams: resetSearchParams1,
} = useTable({
  apiFn: getMyReceiveList,
  showTotal: true,
  apiParams: {
    current: 1,
    size: 10,
  },
  columns: () => [
    { title: "子任务ID", key: "id" },
    { title: "任务名称", key: "taskName" },
    { title: "标注进度", key: "progress" },
    { title: "标注状态", key: "markState" },
    { title: "任务创建者", key: "creator" },
    { title: "创建时间", key: "createTime" },
    {
      key: "operate",
      title: "操作",
      align: "center",
      width: 240,
      fixed: "right",
      render: (row: any) => {
        return [
          h("div", { class: "flex-center gap-8px flex-wrap" }, [
            h(
              NButton,
              {
                type: "primary",
                ghost: true,
                size: "small",
                style: {
                  display: row.isHide == 1 ? "none" : "block",
                },
                onClick: () => handleOperate(NewOperateType.SUBMIT, row),
              },
              "提交任务",
            ),
            h(
              NButton,
              {
                type: "primary",
                ghost: true,
                size: "small",
                style: {
                  display: row.isHide == 1 ? "none" : "block",
                },
                onClick: () => handleOperate(NewOperateType.VIEW, row),
              },
              "前往标注",
            ),
          ]),
        ];
      },
    },
  ],
});

const {
  columns: columns2,
  columnChecks: columnChecks2,
  data: data2,
  getData: getData2,
  getDataByPage: getDataByPage2,
  loading: loading2,
  mobilePagination: mobilePagination2,
  searchParams: searchParams2,
  resetSearchParams: resetSearchParams2,
} = useTable({
  apiFn: myExamineTaskList,
  showTotal: true,
  apiParams: {
    current: 1,
    size: 10,
  },
  columns: () => [
    { title: "子任务ID", key: "id" },
    { title: "任务名称", key: "taskName" },
    { title: "标注员", key: "markName" },
    { title: "已审核/未审核", key: "auditNum" },
    { title: "审核进度", key: "progress" },
    { title: "审核状态", key: "markState" },
    { title: "任务创建者", key: "creator" },
    { title: "创建时间", key: "createTime" },
    {
      key: "operate",
      title: "操作",
      align: "center",
      width: 240,
      fixed: "right",
      render: (row: any) => {
        return [
          h("div", { class: "flex-center gap-8px flex-wrap" }, [
            h(
              NButton,
              {
                type: "primary",
                ghost: true,
                size: "small",
                style: {
                  display: row.isHide == 1 ? "none" : "block",
                },
                onClick: () => handleOperate(NewOperateType.AUDITSUBMIT, row),
              },
              "提交任务",
            ),
            h(
              NButton,
              {
                type: "primary",
                ghost: true,
                size: "small",
                style: {
                  display: row.isHide == 1 ? "none" : "block",
                },
                onClick: () => handleOperate(NewOperateType.AUDIT, row),
              },
              "前往审核",
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
  // closeDrawer
} = useTableOperate(data, getData);

const assessTitle = ref<string>("使用流程");
const aInfoList = ref<any>([
  {
    name: "选择数据集版本",
    info: "您只可选择由您创建的、此时未被共享、清洗、标注的数据集版本，并提前设定数据集版本的标注标签，任务过程中标注标签不支持增删改查",
    btns: [],
    icon: "data-collect",
    imgSrc: "mul_step01",
  },
  {
    name: "选择标注团队",
    info: "您可以通过添加团队成员的方式自定义标注团队成员，一个标注团队成员上限20人",
    btns: [],
    icon: "data-qc",
    imgSrc: "mul_step02",
  },
  {
    name: "标注任务分发",
    info: "选定标注团队后系统将根据任务总数随机分配个人任务",
    btns: [],
    icon: "data-annotation",
    imgSrc: "mul_step03",
  },
]);

function handleAdd() {
  operateType.value = "add";
  openModal();
}

function mapData(data: any) {
  return data.map(item => {
    return {
      ...item,
    }
  })
}

async function handleBatchDelete() {
  // request
  const res = await delExample(checkedRowKeys.value);
  if (res.data >= 1 || !res.data) {
    onBatchDeleted();
  }
}
const handleTabUpdate = (value: string | number) => {
  console.log("value: ", value);
  tabName.value = value;
  if (value === "我发起的任务") {
    getDataByPage();
  }
  if (value === "我接收的任务") {
    getDataByPage1();
  }
  if (value === "我审核的任务") {
    getDataByPage2();
  }
};

const handleGroupMange = () => {
  router.push({
    name: "data-ano_groupmanage",
  });
};

// newCode
// 定义操作类型的枚举
enum NewOperateType {
  VIEW = "view",
  PROGRESS = "progress",
  DELETE = "delete",
  END = "end",
  VALIDATE = "validate",
  DETAIL = "detail",
  SUBMIT = "submit",
  AUDIT = "audit",
  REVIEW = "review",
  REVIEWDETAIL = "reviewDetail",
  COMPLETE = "complete",
  ISIMMEDIATE = "isimmediate",
  AUDITSUBMIT = "auditSubmit",
  ADDTAG = "addTag",
}

const router = useRouter();
const route = useRoute();
const id = ref<number | string | null>(null);
const rowData = ref<any>(null);
const tabName = ref<string>("我发起的任务");
const tabArr = ref<any>(["我发起的任务", "我接收的任务", "我审核的任务"]);
const submitShowModal = ref<Boolean>(false);
const submitTooltipText = ref<string>("");
const submitRow = ref<any>(null);
const submitSign = ref<any>("submit");

// 验收完成
const isShowValidate = ref<Boolean>(false);
const validateStatus = ref<String>("1");
const validateModel = ref<any>({
  verifyState: null,
  returnState: "",
  message: null,
});
const statusOptions = ref([
  // { value: "0", label: "打回全部数据 " },
  { value: "1", label: "保存全部数据 " },
  { value: "2", label: "仅保存验收通过的数据 " },
]);
const repulseOptions = ref([
  { value: "1", label: "未验收的数据 " },
  { value: "2", label: "验收不通过的数据 " },
  { value: "3", label: "未验收+验收不通过的数据" },
]);

const isShowAudit = ref<Boolean>(false);
const isAuditLoading = ref<Boolean>(false);

const validateTitle = computed(() => {
  if (validateStatus.value === "0") {
    return "剩余验收通过";
  } else if (validateStatus.value === "1") {
    return "验收完成";
  } else if (validateStatus.value === "2") {
    return "打回任务";
  }
});

const handleCreateTask = () => {
  router.push({
    name: "data-ano_mulanotask",
  });
};

// 按钮工厂函数
const createButton = (
  type: keyof typeof NewOperateType,
  text: string,
  row: any,
  props?: any,
) => {
  return h(
    NButton,
    {
      type: "primary",
      ghost: true,
      size: "small",
      onClick: (event) => {
        event.stopPropagation();
        handleOperate(NewOperateType[type], row);
      },
      ...props,
    },
    text,
  );
};

// 确认弹窗工厂函数
const createConfirmButton = (
  type: keyof typeof NewOperateType,
  text: string,
  confirmText: string,
  row: any,
) => {
  return h(
    NPopconfirm,
    {
      onPositiveClick: (event: Event) => {
        event.stopPropagation();
        handleOperate(NewOperateType[type], row);
      },
    },
    {
      trigger: () => createButton(NewOperateType.ISIMMEDIATE, text, row),
      default: () => h("span", {}, confirmText),
    },
  );
};

// 状态按钮配置
const buttonConfigs: Record<number, (row: any) => any[]> = {
  1: () => [
    createButton("ADDTAG", "新增标签", row),
    createConfirmButton("DELETE", "删除", "你确定要删除吗？", row),
  ], // 任务分配中
  2: (row) => [
    createButton("DETAIL", "查看数据集", row),
    createButton("ADDTAG", "新增标签", row),
    // 未开始
    createButton("PROGRESS", "标注详情", row),
    createConfirmButton("END", "结束标注", "你确定要结束当前标注吗？", row),
    createConfirmButton("DELETE", "删除", "你确定要删除吗？", row),
  ],
  3: (row) => [
    createButton("DETAIL", "查看数据集", row),
    createButton("ADDTAG", "新增标签", row),
    // 标注中
    createButton("PROGRESS", "标注详情", row),
    createConfirmButton("END", "结束标注", "你确定要结束当前标注吗？", row),
    createConfirmButton("DELETE", "删除", "你确定要删除吗？", row),
  ],
  4: (row) => [
    createButton("DETAIL", "查看数据集", row),
    createButton("ADDTAG", "新增标签", row),
    // 标注已完成
    createButton("PROGRESS", "标注详情", row),
    createButton("REVIEW", "分配审核任务", row),
    // createConfirmButton('REVIEW', '分配审核任务', '确定要分配嘛？', row)
    createConfirmButton("DELETE", "删除", "你确定要删除吗？", row),
  ],
  5: (row) => [
    // 待审核
    createButton("DETAIL", "查看数据集", row),
    createButton("ADDTAG", "新增标签", row),
    createButton("REVIEWDETAIL", "审核详情", row),
    createConfirmButton("DELETE", "删除", "你确定要删除吗？", row),
  ],
  6: (row) => [
    // 审核中
    createButton("DETAIL", "查看数据集", row),
    createButton("ADDTAG", "新增标签", row),
    createButton("PROGRESS", "标注详情", row),
    createButton("REVIEWDETAIL", "审核详情", row),
    createConfirmButton("DELETE", "删除", "你确定要删除吗？", row),
  ],
  7: (row) => [
    // 审核完成
    createButton("DETAIL", "查看数据集", row),
    createButton("COMPLETE", "验收保存", row),
    createConfirmButton("DELETE", "删除", "你确定要删除吗？", row),
  ],
  8: (row) => [
    // 任务结束
    createButton("DETAIL", "查看数据集", row),
    createConfirmButton("DELETE", "删除", "你确定要删除吗？", row),
  ],
};

// render主函数
const getButtonsByStatus = (taskState: number, row: any) => {
  const buttons = buttonConfigs[taskState]?.(row) || [createConfirmButton("DELETE", "删除", "你确定要删除吗？", row)];
  return h("div", { class: "flex-center gap-8px flex-wrap" }, buttons);
};

// 操作处理函数
async function handleOperate(type: NewOperateType, row: any) {
  switch (type) {
    case NewOperateType.VIEW:
      router.push({
        // name: "data-ano_operation",
        // name: "data-ano_imgoperate",
        name: import.meta.env.VITE_TOGGLE_OPERATE === 'Y' ? 'data-ano_imgoperate' : 'data-ano_operation',
        query: {
          id: row.sonId,
          anoType: "validateUser",
          markUserId: row.id,
          // taskId: row.taskId,
        },
      });
      break;
    case NewOperateType.PROGRESS:
      id.value = row.id;
      rowData.value = row;
      openModal();
      break;
    case NewOperateType.DELETE:
      try {
        const res = await deleteTask({ id: row.id });
        if (res.data) {
          window.$message?.success?.("删除成功！");
          getDataByPage();
        }
      } catch (error) {
        console.error("删除任务时出错:", error);
        window.$message?.error?.("删除失败，请稍后重试！");
      }
      break;
    case NewOperateType.VALIDATE:
      router.push({
        // name: "data-ano_operation",
        // name: "data-ano_imgoperate",
        name: import.meta.env.VITE_TOGGLE_OPERATE === 'Y' ? 'data-ano_imgoperate' : 'data-ano_operation',
        query: {
          id: row.sonId,
          anoType: "validate",
          taskId: row.id,
        },
      });
      break;
    case NewOperateType.DETAIL:
      router.push({
        name: "data-ano_detail",
        params: {
          sign: "detail",
          row,
        },
        query: {
          id: row.sonId,
          isMany: row.isMany
        },
      });
      break;
    case NewOperateType.END:
      try {
        const res = await endTask({ id: row.id });
        if (res.data) {
          window.$message?.success?.("结束成功！");
          getDataByPage();
        }
      } catch (error) {
        console.error("结束任务时出错:", error);
        window.$message?.error?.("结束失败，请稍后重试！");
      }
      break;
    case NewOperateType.REVIEW:
      try {
        isShowAudit.value = true;
        isAuditLoading.value = true;
        id.value = row.id;
        rowData.value = row;
      } catch (error) {
        console.error("审核分配出错:", error);
        window.$message?.error?.("审核分配失败，请稍后重试！");
      }
      break;
    case NewOperateType.ISIMMEDIATE:
      console.log("没有逻辑");
      break;
    case NewOperateType.REVIEWDETAIL:
      const res2 = await examineTeamInfo({ id: row.id });
      id.value = row.id;
      rowData.value = res2.data;
      openModal1();
      break;
    case NewOperateType.SUBMIT:
      const res1 = await submitTaskPrompt({ id: row.id });
      if (res1.data) {
        submitSign.value = "submit";
        submitRow.value = row;
        submitShowModal.value = true;
        submitTooltipText.value = res1.data;
      }
      break;
    case NewOperateType.AUDIT:
      router.push({
        // name: "data-ano_operation",
        // name: "data-ano_imgoperate",
        name: import.meta.env.VITE_TOGGLE_OPERATE === 'Y' ? 'data-ano_imgoperate' : 'data-ano_operation',
        query: {
          id: row.sonId,
          anoType: "audit",
          markUserId: row.id,
          taskId: row.taskId,
        },
      });
      break;
    case NewOperateType.AUDITSUBMIT:
      const res = await submitTaskPrompt2({ id: row.id });
      if (res.data) {
        submitSign.value = "auditSubmit";
        submitRow.value = row;
        submitShowModal.value = true;
        submitTooltipText.value = res.data;
      }
      break;
    case NewOperateType.COMPLETE:
      submitRow.value = row;
      isShowValidate.value = true;
      break;
    case NewOperateType.ADDTAG:
      rowData.value = row;
      openTagModal();
      break;
    default:
      console.warn(`未知的操作类型: ${type}`);
  }
}

const handleValidateSuccess = async () => {
  if (validateStatus.value === "0") {
    const res = await examineReturn({
      taskId: submitRow.value.id,
    });
    if (res.data) {
      window.$message?.success?.(`操作成功`);
      isShowValidate.value = false;
      getDataByPage();
    }
  }
  if (validateStatus.value === "1") {
    // if (validateModel.value.verifyState === '0') {
    //   const res = await examineReturn({
    //     taskId: submitRow.value.id,
    //   });
    //   if (res.data) {
    //     window.$message?.success?.(`操作成功`);
    //     isShowValidate.value = false;
    //     getDataByPage();
    //   }
    // } else {
    //   const res = await verifyComplete({
    //     taskId: submitRow.value.id,
    //     verifyState: validateModel.value.verifyState,
    //   });
    //   if (res.data) {
    //     window.$message?.success?.(`操作成功`);
    //     isShowValidate.value = false;
    //     getDataByPage();
    //   }
    // }

    const taskId = submitRow.value.id;
    const verifyState = validateModel.value.verifyState;

    let res;
    if (verifyState === '0') {
      res = await examineReturn({ taskId });
    } else {
      res = await verifyComplete({ taskId, verifyState });
    }

    if (res.data) {
      window.$message?.success?.('操作成功');
      isShowValidate.value = false;
      getDataByPage();
    }
  }
  if (validateStatus.value === "2") {
    const res = await returnTask({
      taskId: submitRow.value.id,
      returnState: validateModel.value.returnState,
    });
    if (res.data) {
      window.$message?.success?.(`操作成功`);
      router.back();
    }
  }
};

const handleModalCancel = (sign: string) => {
  if (sign === "validate") {
    isShowValidate.value = false;
  }
};

const handleModalDefine = (sign: string) => {
  if (sign === "validate") {
    handleValidateSuccess();
  }
};

// 权限控制tab
const setTabArr = async () => {
  const res = await isManyTask();
  const tabMap: any = {
    "0": "我发起的任务",
    "1": "我接收的任务",
    "2": "我审核的任务",
  };
  if (res.data) {
    tabArr.value = res.data.map((item: any) => tabMap[item]);
    tabName.value = tabArr.value[0];
  } else {
    tabArr.value = ["我接收的任务"];
    tabName.value = "我接收的任务";
  }
};

const handleSubmitDefine = async () => {
  if (submitSign.value === "submit") {
    const res = await submitTask({ id: submitRow.value.id });
    if (res.data) {
      window.$message?.success?.("任务提交成功！");
      submitShowModal.value = false;
      getDataByPage1();
    }
  }
  if (submitSign.value === "auditSubmit") {
    const res = await submitExamineTask2({ id: submitRow.value.id });
    if (res.data) {
      window.$message?.success?.("任务提交成功！");
      submitShowModal.value = false;
      getDataByPage2();
    }
  }
};

const handleSubmitClose = () => {
  submitShowModal.value = false;
};

const handleAuditSubmitted = () => {
  isShowAudit.value = false;
  getDataByPage();
};

const handleAuditDefine = () => {
  getDataByPage();
};

const handleClose = () => {
  getDataByPage();
}

onMounted(() => {
  setTabArr();
});
</script>

<template>
  <div class="min-h-500px flex-col-stretch gap-16px overflow-hidden lt-sm:overflow-auto">
    <!-- <UserSearch v-model:model="searchParams" @reset="resetSearchParams" @search="getDataByPage" /> -->
    <NCard :bordered="false" size="small" class="card-wrapper">
      <NCollapse>
        <NCollapseItem :title="assessTitle" name="user-search">
          <NFlex justify="space-between" class="wrap-container">
            <div v-for="(item, index) of aInfoList" :key="index" class="item-manage flex justify-center items-center">
              <div class="item_main w-full flex-col justify-around items-center">
                <div class="item-manage_icon">
                  <SvgIcon :local-icon="item.imgSrc" class="text-#000 text-42px" />
                  <div class="iconName">{{ item.name }}</div>
                </div>
                <div class="item-manage_info w-66% flex justify-center items-center">
                  {{ item.info }}
                </div>
              </div>
              <div class="item_arrow" v-if="index !== aInfoList.length - 1">
                <div class="flow-arrow">
                  <span class="aibp-custom-icon aibp-custom-icon-arrow">
                    <svg width="24" height="24">
                      <path fill="#B8BABF" d="m8.053 3 9.192 9.192L8 21.437v-5.253l3.79-3.79L8 8.603V3.052L8.053 3Z">
                      </path>
                    </svg></span>
                </div>
              </div>
            </div>
          </NFlex>
        </NCollapseItem>
      </NCollapse>
    </NCard>
    <n-tabs type="line" animated @update:value="handleTabUpdate">
      <n-tab-pane v-for="(item, index) of tabArr" :key="index" :name="item" :tab="item"></n-tab-pane>
      <template #suffix>
        <n-button type="primary" @click="handleGroupMange" v-hasPermi="'system:mul:manage'">
          管理多人标注团队
        </n-button>
      </template>
    </n-tabs>
    <NCard title="任务列表" :bordered="false" size="small" class="sm:flex-1-hidden card-wrapper"
      v-if="tabName === '我发起的任务'">
      <template #header-extra>
        <TableHeaderOperation v-model:columns="columnChecks" :disabled-delete="checkedRowKeys.length === 0"
          :loading="loading" :is-add="false" :is-del="false" @add="handleAdd" @delete="handleBatchDelete"
          @refresh="getData">
          <template #prefix>
            <NButton size="small" type="primary" class="-mr-24px" @click="handleCreateTask()"
              v-hasPermi="'system:mul:createTask'">创建任务</NButton>
          </template>
        </TableHeaderOperation>
      </template>
      <NDataTable v-model:checked-row-keys="checkedRowKeys" :columns="columns" :data="data" size="small"
        :flex-height="!appStore.isMobile" :scroll-x="1800" :loading="loading" remote :row-key="(row) => row.id"
        :pagination="mobilePagination" class="sm:h-full" />
    </NCard>
    <NCard title="任务列表" :bordered="false" size="small" class="sm:flex-1-hidden card-wrapper"
      v-if="tabName === '我接收的任务'">
      <NDataTable v-model:checked-row-keys="checkedRowKeys1" :columns="columns1" :data="data1" size="small"
        :flex-height="!appStore.isMobile" :scroll-x="1800" :loading="loading1" remote :row-key="(row) => row.id"
        :pagination="mobilePagination1" class="sm:h-full" />
    </NCard>
    <NCard title="任务列表" :bordered="false" size="small" class="sm:flex-1-hidden card-wrapper"
      v-if="tabName === '我审核的任务'">
      <NDataTable v-model:checked-row-keys="checkedRowKeys2" :columns="columns2" :data="data2" size="small"
        :flex-height="!appStore.isMobile" :scroll-x="1800" :loading="loading2" remote :row-key="(row) => row.id"
        :pagination="mobilePagination2" class="sm:h-full" />
    </NCard>
    <ProgModal v-model:visible="visible" v-model:id="id" v-model:row="rowData" :row-data="editingData" @close="handleClose" v-if="!!id" />
    <ReviewDetailModal v-model:visible="visible1" v-model:id="id" v-model:row="rowData" :row-data="editingData"
      @audit="handleAuditDefine"  @close="handleClose" v-if="!!id" />
    <AuditModal v-model:visible="isShowAudit" v-model:isAuditLoading="isAuditLoading" v-model:id="id"
      v-model:row="rowData" :row-data="editingData" @submitted="handleAuditSubmitted" @close="handleClose"
      v-if="!!id" />
    <!-- 提交任务 -->
    <n-modal :show="submitShowModal">
      <n-card style="width: 600px" title="提交任务" size="huge" :bordered="false" role="dialog" aria-modal="true">
        <div>{{ submitTooltipText }}</div>
        <template #footer>
          <div class="h-auto w-full flex items-center justify-end gap-24px">
            <NButton type="primary" @click="handleSubmitDefine()">确定提交</NButton>
            <NButton @click="handleSubmitClose()">关闭窗口</NButton>
          </div>
        </template>
      </n-card>
    </n-modal>
    <!--验收完成-->
    <NModal v-model:show="isShowValidate" :close-on-esc="false" class="wrap-tag-modal">
      <NCard :title="validateTitle" class="w-640px" :bordered="false" size="huge" role="dialog" aria-modal="true">
        <NForm ref="formRef" :model="validateModel">
          <div class="h-full w-full">
            <div v-if="validateStatus === '1'">
              <NFormItem label="保存类型" path="type">
                <NRadioGroup v-model:value="validateModel.verifyState">
                  <NRadio v-for="item in statusOptions" :key="item.value" :value="item.value" :label="item.label" />
                </NRadioGroup>
              </NFormItem>
              <div class="bg-[#fff4e6] p-8px py-8px">
                注:验收完成后任务就会结束 不能再进行操作
              </div>
            </div>
            <div v-if="validateStatus === '2'">
              <NFormItem label="打回类型" path="repulseType">
                <NRadioGroup v-model:value="validateModel.returnState">
                  <NRadio v-for="item in repulseOptions" :key="item.value" :value="item.value" :label="item.label" />
                </NRadioGroup>
              </NFormItem>
            </div>
          </div>
        </NForm>
        <template #footer>
          <div class="h-auto w-full flex items-center justify-end gap-24px">
            <NButton type="primary" @click="handleModalDefine('validate')">确定</NButton>
            <NButton @click="handleModalCancel('validate')">取消</NButton>
          </div>
        </template>
      </NCard>
    </NModal>

    <TagOperateModal v-model:visible="tagVisible" v-model:rowData="rowData" />
  </div>
</template>

<style scoped lang="scss">
::-webkit-scrollbar-button {
  background-color: #ccc;
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
</style>
