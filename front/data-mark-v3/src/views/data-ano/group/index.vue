<script setup lang="tsx">
import { NButton, NPopover, NTag, UploadFileInfo } from 'naive-ui';
import { $t } from '@/locales';
import { useAppStore } from '@/store/modules/app';
import { useTable, useTableOperate } from '@/hooks/common/table';
import { enableStatusOptions } from '@/constants/business';
import { fetchLabelGroupList, fetchLabelGroupRemove } from '@/service/api/tag';
import { useBoolean } from '~/packages/hooks/src';
import UserOperateDrawer from './modules/user-operate-drawer.vue';
import UserSearch from './modules/user-search.vue';
import MenuOperateModal from './modules/menu-operate-modal.vue';
import { getDataSetListNoPage, copyLabelGroup, assocDataSet, getDataSonLabelStatus, importLabel, TemDownload } from "@/service/api/expansion";
import { getToken } from '@/store/modules/auth/shared';
import axios from "axios";

const appStore = useAppStore();

const yesOrNoRecord: Record<string, string> = {
  1: '是',
  2: '否'
};

const {
  columns,
  columnChecks,
  data,
  getData,
  getDataByPage,
  loading,
  mobilePagination,
  searchParams,
  resetSearchParams
} = useTable({
  apiFn: fetchLabelGroupList,
  showTotal: true,
  apiParams: {
    current: 1,
    size: 10,
    // if you want to use the searchParams in Form, you need to define the following properties, and the value is null
    // the value can not be undefined, otherwise the property in Form will not be reactive
    labelGroupName: null
  },
  columns: () => [
    {
      type: 'selection',
      align: 'center',
      width: 48,
      fixed: 'left'
    },
    {
      key: 'index',
      title: $t('common.index'),
      align: 'center',
      width: 64
    },
    {
      title: '标签组名称', key: 'labelGroupName', width: 120, ellipsis: {
        tooltip: true
      }
    },
    {
      title: '标签组名称（英文）', key: 'englishLabelGroupName', width: 120, ellipsis: {
        tooltip: true
      }
    },
    {
      title: '标签组描述', key: 'labelGroupDesc', width: 140, ellipsis: {
        tooltip: true
      }
    },
    { title: '创建时间', key: 'createTime', width: 120 },
    { title: '更新时间', key: 'updateTime', width: 120 },
    {
      key: 'operate',
      width: 300,
      title: $t('common.operate'),
      align: 'center',
      render: row => {
        return [
          h(
            NButton,
            {
              type: 'primary',
              ghost: true,
              size: 'small',
              style: { marginRight: '10px' },
              onClick: () => handleOperate('copy')
            },
            '复制当前标签组'
          ),
          h(
            NButton,
            {
              type: 'primary',
              ghost: true,
              size: 'small',
              style: { marginRight: '10px' },
              onClick: () => handleOperate('relevance', row)
            },
            '关联数据集'
          ),
          h(
            NButton,
            {
              type: 'primary',
              ghost: true,
              size: 'small',
              style: { marginRight: '10px' },
              onClick: () => handleTagManage(row)
            },
            '标签列表'
          ),
          h(
            NButton,
            {
              type: 'primary',
              ghost: true,
              size: 'small',
              style: { marginRight: '10px' },
              onClick: () => edit(row.id)
            },
            '编辑'
          ),
          h(
            NButton,
            {
              type: 'primary',
              ghost: true,
              size: 'small',
              style: { marginRight: '10px' },
              onClick: () => handleDelete(row.id)
            },
            '删除'
          )
        ];
      },
      fixed: "right"
    }
  ]
});

const {
  drawerVisible,
  operateType,
  editingData,
  handleAdd,
  handleEdit,
  checkedRowKeys,
  onBatchDeleted,
  onDeleted
  // closeDrawer
} = useTableOperate(data, getData);


async function handleBatchDelete() {
  // request
  console.log(checkedRowKeys.value);

  const res = await fetchLabelGroupRemove({
    ids: checkedRowKeys.value
  });
  if (res.data >= 1) {
    onBatchDeleted();
  }
}

async function handleDelete(id: number) {
  // request
  const res = await fetchLabelGroupRemove({ ids: [id] });
  if (res.data >= 1) {
    onDeleted();
  }
}

function edit(id: number) {
  handleEdit(id);
}

const router = useRouter();
const handleTagManage = (row: any) => {
  router.push({
    name: 'data-ano_gtag'
    // query: row,
  });
  localStorage.setItem('row', JSON.stringify(row));
};

// eslint-disable-next-line @typescript-eslint/no-redeclare
type operateType = 'groupOperate' | 'createTag' | 'copy' | 'relevance';
const { bool: visible, setTrue: openModal } = useBoolean();
// 创建标签Modal
const isCreateModal = ref<Boolean>(false);
const createModel = ref<any>({});
const tagOptions = ref<any>([]);

// 复制标签组操作
const isCopyModal = ref<Boolean>(false);
const copyModel = ref<any>({});
const copyOptions = ref<any>([
  { value: '1', label: '全部覆盖' },
  { value: '2', label: '数据叠加' }
]);

// 关联数据集
const isDataSetModal = ref<Boolean>(false);
const dataSetModel = ref<any>({});
const setOptions = ref<any>([]);
const checkStrategyIsChild = ref(true);
const cascade = ref(false);
const showPath = ref(true);
const hoverTrigger = ref(true);
const filterable = ref(false);
const responsiveMaxTagCount = ref(true);
const clearFilterAfterSelect = ref(true);
const labelGroupId = ref<string | null>('');

function handleOperate(sign: operateType, row: any) {
  switch (sign) {
    case 'groupOperate':
      // handle group operate
      openModal();
      break;
    case 'createTag':
      // handle create tag
      isCreateModal.value = true;
      createModel.value = {};
      break;
    case 'copy':
      // handle copy
      isCopyModal.value = true;
      copyModel.value = {};
      break;
    case 'relevance':
      // handle relevance
      console.log(editingData);
      labelGroupId.value = row.id;
      isDataSetModal.value = true;
      dataSetModel.value = {};
      break;
    default:
      break;
  }
}

// 数据集列表接口 noPage
async function getGroupList() {
  const recursionMapData = (data: any, label: any) => {
    const mapList = data.map((item: any, index: string | number) => {
      item.value = item.groupId || item.sonId;
      if (label) {
        item.label = `${item.groupName || `V${item.version}`}`;
      } else {
        item.label = item.groupName || `V${item.version}`;
      }
      const children = item.dataSonResponseList || [];
      item.children = children.map((val: any) => {
        // 演示环境
        item.disabled = false;
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

// 确认关联
async function handleTagOperate(type: 'cascader' | 'update' | 'copy') {
  if (type === 'cascader') {
    const params = {
      labelGroupId: labelGroupId.value,
      dataSetIdList: dataSetModel.value.dataSetIdList || []
    };
    const res = await assocDataSet(params);
    if (res.data) {
      window.$message?.success?.("关联成功！");
      isDataSetModal.value = false;
      await getData();
    }
  }
  if (type === 'update') {
    // update
  }
  if (type === 'copy') {
    // copy
    const { groupInfo, ...rest } = copyModel.value;
    const res = await copyLabelGroup(rest);
    if (res.data) {
      window.$message?.success?.("复制标签组成功！");
      isCopyModal.value = false;
      await getData();
    }
  }
}

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

interface CascaderEvent {
  value: string;
}

async function handleUpdateValue(event: CascaderEvent, type: string) {
  if (type === 'copy') {
    const res = await getDataSonLabelStatus({ sonId: event });
    if (res.data) {
      copyModel.value.groupInfo = res.data;
    }
  }
}

onMounted(() => {
  getGroupList();
})

// 标签导入
const isModalVisible = ref(false);
const uploadRef = ref<any>(null);
const uploadAction = `${import.meta.env.VITE_SERVICE_BASE_URL}/temp/anyUpload`;
const headers = reactive<any>({
  Authorization: `Bearer ${getToken()}`,
});
const fileList = ref<any>([]);
const fileIds = ref<any>([]);

const beforeUpload = (file: File) => {
  if (!['application/vnd.ms-excel', 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'].includes(file.type)) {
    window.$message?.error('只能上传 .xlsx 或 .xls 文件');
    return false;
  }
  return true;
};

const handleFinish = (options: { file: UploadFileInfo, event?: ProgressEvent }) => {
  const res = JSON.parse(options.event.currentTarget.response);
  if (res.code === 200) {
    window.$message?.success('文件上传成功');
    fileIds.value.push(res.data[0]);
  }
};

const handleRemove = (options: { file: UploadFileInfo, fileList: Array<UploadFileInfo>, index: number }) => {
  fileIds.value.splice(options.index, 1);
};

const downloadTemplate = async () => {
  // try {
  //   const response = await TemDownload({ type: 1 });

  //   const blob = response.data;
  //   const url = window.URL.createObjectURL(blob);
  //   const link = document.createElement('a');
  //   link.href = url;
  //   link.download = 'template.xlsx';
  //   document.body.appendChild(link);
  //   link.click();
  //   document.body.removeChild(link);
  //   window.URL.revokeObjectURL(url);

  // } catch (error) {
  //   console.error('下载失败:', error);
  //   window.$message?.error(error.message || '下载模板文件失败');
  // }

  const fileUrl = '/static/template.xlsx'; // 注意这里的路径是相对于public文件夹

  axios({
    url: fileUrl,
    method: 'get',
    responseType: 'blob', // 设置响应类型为blob
  }).then((response) => {
    const fileBlob = new Blob([response.data], { type: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet' });
    const link = document.createElement('a');
    link.href = URL.createObjectURL(fileBlob);
    link.download = '标签导入模板.xlsx';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  }).catch(error => {
    console.error("文件加载失败", error);
  });
};

const openTagModal = () => {
  isModalVisible.value = true;
};

const closeTagModal = () => {
  isModalVisible.value = false;
};

const handleSubmit = async () => {
  try {
    const fileList = uploadRef.value.fileList;
    if (fileList.length === 0) {
      window.$message?.warning('请选择要上传的文件');
      return;
    }

    // const formData = new FormData();
    // fileList.forEach((fileItem: any) => {
    //   formData.append('files', fileItem.file);
    // });

    const res = await importLabel({ fileIds: fileIds.value.map(val => val.id) });
    if (res.data) {
      window.$message?.success('文件提交成功');
      closeTagModal();
      getDataByPage();
    }
  } catch (error) {
    window.$message?.error((error as Error).message);
  }
};
</script>

<template>
  <div class="min-h-500px flex-col-stretch gap-16px overflow-hidden lt-sm:overflow-auto">
    <UserSearch v-model:model="searchParams" @reset="resetSearchParams" @search="getDataByPage" />
    <NCard title="标签组" :bordered="false" size="small" class="sm:flex-1-hidden card-wrapper">
      <template #header-extra>
        <TableHeaderOperation v-model:columns="columnChecks" :disabled-delete="checkedRowKeys.length === 0"
          :loading="loading" add-text="创建标签组" @add="handleAdd" @delete="handleBatchDelete" @refresh="getData">
          <!-- <template #prefix>
            <NButton type="primary" ghost size="small" @click="handleOperate('groupOperate')">
              <template #icon>
                <svg-icon local-icon="ix&#45;&#45;operate-plant" class="text-[24px]"></svg-icon>
              </template>
标签组操作
</NButton>
<NButton type="primary" ghost size="small" @click="handleOperate('copy')">
  <template #icon>
                <svg-icon local-icon="solar&#45;&#45;copy-linear" class="text-[24px]"></svg-icon>
              </template>
  复制当前标签组
</NButton>
<NButton size="small" type="primary" class="" @click="handleOperate('createTag')">创建标签</NButton>
</template> -->
          <template #prefix>
            <!-- mdi--home-import-outline -->
            <NButton type="primary" @click="openTagModal" size="small">
              <template #icon>
                <SvgIcon local-icon="mdi--home-import-outline" class="text-24px text-[#fff]" />
              </template>
              导入标签
            </NButton>
            <!-- ic--baseline-download -->
            <NButton type="primary" @click="downloadTemplate" size="small">
              <template #icon>
                <SvgIcon local-icon="ic--baseline-download" class="text-24px text-[#fff]" />
              </template>
              模板下载
            </NButton>
          </template>
        </TableHeaderOperation>
      </template>
      <NDataTable v-model:checked-row-keys="checkedRowKeys" :columns="columns" :data="data"
        :flex-height="!appStore.isMobile" :scroll-x="962" :loading="loading" remote :row-key="row => row.id"
        :pagination="mobilePagination" class="sm:h-full" />
      <UserOperateDrawer v-model:visible="drawerVisible" :operate-type="operateType" :row-data="editingData"
        @submitted="getDataByPage" />
      <MenuOperateModal v-model:visible="visible" @create="handleGroupCreate" />
      <!--标签操作管理-->
      <NModal v-model:show="isCreateModal" title="标签操作管理" preset="card" class="w-600px">
        <NScrollbar class="h-auto pr-20px">
          <NForm ref="formRef" :model="createModel" label-placement="left" :label-width="100">
            <NGrid responsive="screen" item-responsive>
              <NFormItemGi span="24 m:24" label="标签组名称" path="groupName">
                <NSelect v-model:value="createModel.groupName" multiple :options="tagOptions" placeholder="请选择标签组名称" />
              </NFormItemGi>
              <NFormItemGi span="24 m:24" label="标签名称" path="tagName">
                <NInput v-model:value="createModel.tagName" placeholder="请输入标签名称" />
              </NFormItemGi>
              <NFormItemGi span="24 m:24" label="排序号" path="sort">
                <NInputNumber v-model:value="createModel.sort" clearable />
              </NFormItemGi>
              <NFormItemGi span="24 m:24" label="是否使用" path="isUse">
                <NSelect v-model:value="createModel.isUse" multiple :options="tagOptions" placeholder="是否使用" />
              </NFormItemGi>
            </NGrid>
          </NForm>
        </NScrollbar>
        <template #footer>
          <NSpace justify="end" :size="16">
            <NButton @click="() => (isCreateModal = false)">{{ $t('common.cancel') }}</NButton>
            <NButton type="primary" @click="handleTagOperate('update')">修改</NButton>
          </NSpace>
        </template>
      </NModal>
      <!--复制标签组管理-->
      <NModal v-model:show="isCopyModal" title="复制标签组管理" preset="card" class="wrap_modal w-600px">
        <NScrollbar class="h-auto pr-20px">
          <NForm ref="formRef" :model="copyModel" label-placement="left" :label-width="140">
            <NGrid responsive="screen" item-responsive>
              <NFormItemGi span="24 m:24" label="当前选择数据集名称" path="sonId">
                <NCascader v-model:value="copyModel.sonId" clearable expand-trigger="hover" check-strategy="child"
                  placeholder="请选择数据集" :options="setOptions" :render-label="renderLabel"
                  @update:value="handleUpdateValue($event, 'copy')"></NCascader>
              </NFormItemGi>
              <NFormItemGi span="24 m:24" label="当前选择标签组信息" path="groupInfo">
                <span v-if="copyModel.groupInfo">{{ copyModel.groupInfo }}</span>
                <span class="text-red" v-else>请先选择上方数据集名称</span>
              </NFormItemGi>
              <NFormItemGi span="24 m:24" label="选择需要复制标签组" path="copyGroup">
                <NCascader v-model:value="copyModel.copySonId" clearable expand-trigger="hover" check-strategy="child"
                  placeholder="选择被复制标签组数据集名称" :options="setOptions" :render-label="renderLabel"></NCascader>
              </NFormItemGi>
              <NFormItemGi span="24 m:24" label="复制标签组业务类型" path="businessType">
                <NRadioGroup v-model:value="copyModel.businessType">
                  <NRadio v-for="item in copyOptions" :key="item.value" :value="item.value" :label="item.label" />
                </NRadioGroup>
              </NFormItemGi>
            </NGrid>
          </NForm>
        </NScrollbar>
        <template #footer>
          <NSpace justify="end" :size="16">
            <NButton @click="() => (isCopyModal = false)">取消关闭</NButton>
            <NButton type="primary" @click="handleTagOperate('copy')">开始复制</NButton>
          </NSpace>
        </template>
      </NModal>
      <!--关联数据集-->
      <NModal v-model:show="isDataSetModal" title="关联数据集" preset="card" class="wrap_modal w-800px">
        <NScrollbar class="h-auto pr-20px">
          <NForm ref="formRef" :model="dataSetModel" label-placement="left" :label-width="100">
            <NGrid responsive="screen" item-responsive>
              <NFormItemGi span="24 m:24" label="所选数据集" path="dataSet">
                <n-cascader v-model:value="dataSetModel.dataSetIdList" multiple clearable placeholder="选择数据集"
                  :max-tag-count="responsiveMaxTagCount ? 'responsive' : undefined"
                  :expand-trigger="hoverTrigger ? 'hover' : 'click'" :options="setOptions" :cascade="cascade"
                  :check-strategy="checkStrategyIsChild ? 'child' : 'all'" :show-path="showPath"
                  :filterable="filterable" :clear-filter-after-select="clearFilterAfterSelect" />
              </NFormItemGi>
            </NGrid>
          </NForm>
        </NScrollbar>
        <template #footer>
          <NSpace justify="end" :size="16">
            <NButton @click="() => (isDataSetModal = false)">关闭窗口</NButton>
            <NButton type="primary" @click="handleTagOperate('cascader')">确认关联</NButton>
          </NSpace>
        </template>
      </NModal>
    </NCard>
    <!-- 标签导入 -->
    <n-modal v-model:show="isModalVisible" preset="card" title="标签导入" class="w-600px">
      <div class="w-full">
        <n-space space="[16px] w-full">
          <n-upload class="!w-400px" ref="uploadRef" :action="uploadAction" :headers="headers"
            :before-upload="beforeUpload" accept=".xlsx, .xls" @finish="handleFinish" @remove="handleRemove"
            v-model:file-list="fileList">
            <n-button type="primary">上传文件</n-button>
          </n-upload>
        </n-space>
      </div>
      <template #footer>
        <n-space justify="end" space="[16px]">
          <n-button @click="closeTagModal" class="bg-gray-200 text-gray-800 hover:bg-gray-300">取消</n-button>
          <n-button type="primary" @click="handleSubmit">确认</n-button>
        </n-space>
      </template>
    </n-modal>
  </div>
</template>

<style scoped></style>
