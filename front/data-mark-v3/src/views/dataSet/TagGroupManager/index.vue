<script setup lang="tsx">
import { NButton, NForm, NInput, NModal, NPopconfirm, NSelect, NTag } from 'naive-ui';
import { fetchGetUserList } from '@/service/api';
import { $t } from '@/locales';
import { useAppStore } from '@/store/modules/app';
import { enableStatusOptions, enableStatusRecord, userGenderRecord } from '@/constants/business';
import { useTable, useTableOperate } from '@/hooks/common/table';
import { useBoolean } from '~/packages/hooks';
import UserOperateDrawer from './modules/user-operate-drawer.vue';
import MenuOperateModal from './modules/menu-operate-modal.vue';
import UserSearch from './modules/user-search.vue';

type operateType = 'groupOperate' | 'createTag' | 'copy';

const yesOrNoRecord: Record<string, string> = {
  '1': '是',
  '2': '否'
};

const appStore = useAppStore();

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
  resetSearchParams
} = useTable({
  apiFn: fetchGetUserList,
  showTotal: true,
  apiParams: {
    current: 1,
    size: 10,
    // if you want to use the searchParams in Form, you need to define the following properties, and the value is null
    // the value can not be undefined, otherwise the property in Form will not be reactive
    status: null,
    userName: null,
    userGender: null,
    nickName: null,
    userPhone: null,
    userEmail: null
  },
  columns: () => [
    {
      type: 'selection',
      align: 'center',
      width: 48
    },
    {
      key: 'index',
      title: $t('common.index'),
      align: 'center',
      width: 64
    },
    {
      key: 'userName',
      title: '场景名称',
      align: 'center',
      minWidth: 100
    },
    {
      key: 'nickName',
      title: '标签名称',
      align: 'center',
      minWidth: 100
    },
    {
      key: 'userGender',
      title: '是否使用',
      align: 'center',
      width: 140,
      render: row => {
        if (row.userGender === null) {
          return null;
        }

        const tagMap: Record<Api.SystemManage.UserGender, NaiveUI.ThemeColor> = {
          1: 'primary',
          2: 'error'
        };

        const label = $t(yesOrNoRecord[row.userGender]);

        return <NTag type={tagMap[row.userGender]}>{label}</NTag>;
      }
    },
    {
      key: 'userPhone',
      title: '创建时间',
      align: 'center',
      width: 200
    },
    {
      key: 'userPhone',
      title: '更新时间',
      align: 'center',
      width: 200
    },
    {
      key: 'operate',
      title: $t('common.operate'),
      align: 'center',
      width: 180,
      render: row => (
        <div class="flex-center gap-8px">
          <NButton type="primary" ghost size="small" onClick={() => edit(row.id)}>
            {$t('common.edit')}
          </NButton>
          <NPopconfirm onPositiveClick={() => handleDelete(row.id)}>
            {{
              default: () => $t('common.confirmDelete'),
              trigger: () => (
                <NButton type="error" ghost size="small">
                  {$t('common.delete')}
                </NButton>
              )
            }}
          </NPopconfirm>
        </div>
      )
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

// 创建标签Modal
const isCreateModal = ref<Boolean>(false);
const createModel = ref<any>({});
const tagOptions = ref<any>([]);

// 复制场景操作
const isCopyModal = ref<Boolean>(false);
const copyModel = ref<any>({});
const copyOptions = ref<any>([
  { value: '1', label: '全部覆盖' },
  { value: '0', label: '数据叠加' }
]);

async function handleBatchDelete() {
  // request
  console.log(checkedRowKeys.value);

  onBatchDeleted();
}

function handleDelete(id: number) {
  // request
  console.log(id);

  onDeleted();
}

function edit(id: number) {
  handleEdit(id);
}

function handleOperate(sign: string) {
  switch (sign) {
    case 'groupOperate':
      // handle group operate
      // drawerVisible.value = true;
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
    default:
      break;
  }
}

function handleGroupCreate() {}

function handleTagOperate() {}

const router = useRouter();
function handleBack() {
  router.back();
}
</script>

<template>
  <div class="min-h-500px flex-col-stretch gap-16px overflow-hidden lt-sm:overflow-auto">
    <UserSearch v-model:model="searchParams" @reset="resetSearchParams" @search="getDataByPage" />
    <NCard title="" :bordered="false" size="small" class="sm:flex-1-hidden card-wrapper">
      <template #header>
        <div class="flex justify-start items-center gap-16px">
          <span>场景管理</span>
          <NButton type="primary" ghost size="small" @click="handleBack()">
            <template #icon>
              <SvgIcon local-icon="oui--return-key" class="text-[24px]"></SvgIcon>
            </template>
            返回数据集列表
          </NButton>
        </div>
      </template>
      <template #header-extra>
        <TableHeaderOperation
          v-model:columns="columnChecks"
          :disabled-delete="checkedRowKeys.length === 0"
          :loading="loading"
          add-text="创建标签"
          @add="handleAdd"
          @delete="handleBatchDelete"
          @refresh="getData"
        >
          <template #prefix>
            <NButton type="primary" ghost size="small" @click="handleOperate('groupOperate')">
              <template #icon>
                <svg-icon local-icon="ix--operate-plant" class="text-[24px]"></svg-icon>
              </template>
              场景操作
            </NButton>
            <NButton type="primary" ghost size="small" @click="handleOperate('copy')">
              <template #icon>
                <svg-icon local-icon="solar--copy-linear" class="text-[24px]"></svg-icon>
              </template>
              复制当前场景
            </NButton>
            <!-- <NButton size="small" type="primary" class="" @click="handleOperate('createTag')">创建标签</NButton> -->
          </template>
        </TableHeaderOperation>
      </template>
      <NDataTable
        v-model:checked-row-keys="checkedRowKeys"
        :columns="columns"
        :data="data"
        size="small"
        :flex-height="!appStore.isMobile"
        :scroll-x="962"
        :loading="loading"
        remote
        :row-key="row => row.id"
        :pagination="mobilePagination"
        class="sm:h-full"
      />
      <UserOperateDrawer
        v-model:visible="drawerVisible"
        :operate-type="operateType"
        :row-data="editingData"
        @submitted="getDataByPage"
      />
      <MenuOperateModal v-model:visible="visible" @create="handleGroupCreate" />
      <NModal v-model:show="isCreateModal" title="标签操作管理" preset="card" class="w-600px">
        <NScrollbar class="h-auto pr-20px">
          <NForm ref="formRef" :model="createModel" label-placement="left" :label-width="100">
            <NGrid responsive="screen" item-responsive>
              <NFormItemGi span="24 m:24" label="场景名称" path="groupName">
                <NSelect
                  v-model:value="createModel.groupName"
                  multiple
                  :options="tagOptions"
                  placeholder="请选择场景名称"
                />
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
            <NButton type="primary" @click="handleTagOperate()">修改</NButton>
          </NSpace>
        </template>
      </NModal>
      <NModal v-model:show="isCopyModal" title="复制场景管理" preset="card" class="wrap_modal w-600px">
        <NScrollbar class="h-auto pr-20px">
          <NForm ref="formRef" :model="copyModel" label-placement="left" :label-width="140">
            <NGrid responsive="screen" item-responsive>
              <NFormItemGi span="24 m:24" label="当前选择数据集名称" path="groupName">数据集001</NFormItemGi>
              <NFormItemGi span="24 m:24" label="当前选择场景信息" path="groupName">***</NFormItemGi>
              <NFormItemGi span="24 m:24" label="选择需要复制数据集" path="groupName">
                <NSelect
                  v-model:value="copyModel.groupName"
                  multiple
                  :options="tagOptions"
                  placeholder="请选择场景名称"
                />
              </NFormItemGi>
              <NFormItemGi span="24 m:24" label="复制场景业务类型" path="tagName">
                <NRadioGroup v-model:value="copyModel.status">
                  <NRadio v-for="item in copyOptions" :key="item.value" :value="item.value" :label="item.label" />
                </NRadioGroup>
              </NFormItemGi>
            </NGrid>
          </NForm>
        </NScrollbar>
        <template #footer>
          <NSpace justify="end" :size="16">
            <NButton @click="() => (isCopyModal = false)">取消关闭</NButton>
            <NButton type="primary" @click="handleTagOperate()">开始复制</NButton>
          </NSpace>
        </template>
      </NModal>
    </NCard>
  </div>
</template>

<style scoped lang="scss">
.wrap_modal :deep(.n-card-header__main) {
  .n-card-header__main {
    font-weight: 650 !important;
  }
}
</style>
