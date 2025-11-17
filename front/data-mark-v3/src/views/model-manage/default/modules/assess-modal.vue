<script setup lang="ts">
import { watch } from 'vue';
import { $t } from '@/locales';
import { useTable, useTableOperate } from '@/hooks/common/table';
import { getAssessLst, delAssessTask } from '@/service/api/model-manage';
import { NButton } from 'naive-ui';
import { trainAssess } from '@/service/api/dataManage';

defineOptions({
  name: 'MenuOperateModal'
});

export type OperateType = NaiveUI.TableOperateType | 'addChild';

interface Props {
  rowData?: any;
}

// Vue3 props withDefaults
const props = withDefaults(defineProps<Props>(), {
  rowData: {}
});


interface Emits {
  (e: 'submitted'): void;
}

const emit = defineEmits<Emits>();

const visible = defineModel<boolean>('visible', {
  default: false
});

const row = defineModel<any>('row', {
  default: null
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
  sign: "taskId",
  apiFn: getAssessLst,
  showTotal: true,
  apiParams: {
    current: 1,
    size: 10,
  },
  columns: () => [
    {
      key: "taskId",
      title: "任务编号",
      align: "center",
      width: 100,
    },
    {
      key: "taskInputName",
      title: "任务名称",
      align: "center",
    },
    {
      key: "taskName",
      title: "任务类型",
      align: "center",
    },
    { title: "关联数据集", key: "groupVName" },
    { title: "任务状态", key: "taskStat", align: "center" },
    { title: "创建时间", key: "createTime" },
    {
      key: "operate",
      title: "操作",
      align: "center",
      fixed: 'right',
      width: 200,
      render: (row: any) => {
        // const isOperate = Boolean(row.isDelete);
        const isTrainSuccess = Boolean(!!row.status ? row.status : false);
        return [
          h("div", { class: "flex-center gap-8px flex-wrap" }, [
            h(
              NButton,
              {
                disabled: isTrainSuccess,
                type: "primary",
                ghost: true,
                size: "small",
                onClick: () => handleOperate("assess", row),
              },
              "评估详情",
            ),
            h(
              NButton,
              {
                disabled: isTrainSuccess,
                type: "primary",
                ghost: true,
                size: "small",
                onClick: () => handleOperate("delete", row),
              },
              "删除",
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

watch(visible, () => {
  if (visible.value) {
    const params = {
      current: 1,
      size: 10,
      ...props.rowData
    }
    updateSearchParams(params);
    getData();
  }
});

async function handleOperate(sign: string, row: any) {
  if (sign === "assess") {
    const res = await trainAssess({ taskId: row.taskId });
    if (res.data) {
      const { status, result } = res.data;
      emit('assessModal', {
        status, result
      })
    }
  }
  if (sign === "delete") {
    await delAssessTask(row);
    getData();
  }
}


</script>

<template>
  <NModal v-model:show="visible" :title="title" preset="card" class="w-1000px">
    <div class="w-full h-auto">
      <NDataTable v-model:checked-row-keys="checkedRowKeys" :columns="columns" :data="data" size="small" :scroll-x="962"
        :loading="loading" remote :row-key="(row) => row.taskId" :pagination="mobilePagination" class="sm:h-full" />
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
