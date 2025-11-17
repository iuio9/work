<script setup lang="ts">
import { NButton } from 'naive-ui';
import type { DataTableColumns, FormInst } from 'naive-ui';
import callTagGroup from '@/assets/imgs/callTagGroup.png';
import createTagGroup from '@/assets/imgs/createTagGroup.png';
import manageTagGroup from '@/assets/imgs/manageTagGroup.png';
type RowData = {
  key: number;
  name: string;
  age: string;
  address: string;
};
interface SearchPar {
  name: string;
}
interface DataManageObj {
  title: string;
  infoList: any[];
  dataList: any[];
  columns: DataTableColumns<RowData>;
  searchPar: Partial<SearchPar>;
  isShowModal: boolean;
  modalTitle: string;
  sign: 'add' | 'edit';
}
const dataManageObj = ref<DataManageObj>({
  title: '标签组管理',
  infoList: [
    {
      name: '新建标签组',
      info: '点击“创建标签组”按钮，根据需要输入标签组名称和描述。',
      btns: [],
      icon: createTagGroup
    },
    {
      name: '管理标签组',
      info: '支持手动或批量“添加/删除/修改”标签，您可上传csv、xls、txt格式文件批量添加标签。',
      btns: [],
      icon: manageTagGroup
    },
    {
      name: '调用标签组',
      info: '在线标注数据时，您可一键导入“标签组”，使用组内标签进行标注。',
      btns: [],
      icon: callTagGroup
    }
  ],
  dataList: [{}],
  columns: [
    {
      title: '标签组名称',
      key: 'name'
    },
    {
      title: '标签组描述',
      key: 'description'
    },
    {
      title: '创建时间',
      key: 'createTime'
    },
    {
      title: '更新时间',
      key: 'updateTime'
    },
    {
      title: '操作',
      key: 'operation'
    },
    {
      title: '标签组名称',
      key: 'name',
      render() {
        return [
          h(
            NButton,
            {
              text: true,
              type: 'info',
              style: { marginRight: '10px', fontSize: '12px' },
              onClick: () => handleTagManage()
            },
            '标签组管理'
          ),
          h(
            NButton,
            {
              text: true,
              type: 'info',
              style: { marginRight: '10px', fontSize: '12px' },
              onClick: () => handleTagEdit()
            },
            '编辑'
          ),
          h(
            NButton,
            {
              text: true,
              type: 'info',
              style: { marginRight: '10px', fontSize: '12px' },
              onClick: () => handleTagDelete()
            },
            '删除'
          )
        ];
      }
    }
  ],
  searchPar: {
    name: ''
  },
  isShowModal: false,
  modalTitle: '创建标签组',
  sign: 'add'
});
const formRef = ref<FormInst | null>(null);

// computed watch
watch(
  () => dataManageObj.value.sign,
  newVal => {
    if (newVal === 'add') {
      dataManageObj.value.modalTitle = '创建标签组';
    } else if (newVal === 'edit') {
      dataManageObj.value.modalTitle = '编辑标签组';
    } else {
      // return false;
    }
  }
);
const router = useRouter();

// methods
const handleTagEdit = () => {
  dataManageObj.value.isShowModal = true;
  dataManageObj.value.sign = 'edit';
};
const handleTagManage = () => {
  router.push({
    name: 'data-ano_gtag'
  });
};
const handleTagDelete = () => {};
const handleClickModal = signVal => {
  dataManageObj.value.sign = signVal;
  dataManageObj.value.isShowModal = true;
};
const handleCloseModal = () => {
  dataManageObj.value.isShowModal = false;
};
const model = ref<any>({
  name: null,
  desc: null
});
const rules = computed(() => {
  return {
    name: {
      required: true,
      trigger: ['blur', 'input'],
      message: '请输入标签组名称'
    }
  };
});

// newCode
// cancel 取消
const handleCancel = () => {
  dataManageObj.value.isShowModal = false;
};
</script>

<template>
  <div class="flex-vertical-stretch gap-16px overflow-hidden <sm:overflow-auto">
    <n-card :title="dataManageObj.title" :bordered="false" size="small" class="card-wrapper">
      <n-flex justify="space-around" class="wrap-container">
        <div v-for="(item, index) of dataManageObj.infoList" :key="index" class="item-manage">
          <div class="item-manage_icon" flex-center>
            <img :src="item.icon" alt="" />
            <div class="iconName">{{ item.name }}</div>
          </div>
          <div class="item-manage_info">{{ item.info }}</div>
          <div class="item-manage_btnC">
            <NButton v-for="(val, idx) of item.btns" :key="idx" quaternary type="info">{{ val }}</NButton>
          </div>
        </div>
      </n-flex>
    </n-card>
    <n-card class="wrap_content card-wrapper">
      <n-flex justify="space-between" class="header_flex">
        <NButton type="info" @click="handleClickModal('add')">创建标签组</NButton>
        <n-flex justify="end" :wrap="false">
          <n-input v-model:value="dataManageObj.searchPar.name" type="text" placeholder="输入标签组名称">
            <template #suffix>
              <SvgIcon icon="ion:search-outline" class="inline-block align-text-bottom text-16px" />
            </template>
          </n-input>
          <NButton>
            <SvgIcon icon="material-symbols:refresh" class="inline-block align-text-bottom text-18px" />
          </NButton>
        </n-flex>
      </n-flex>
      <n-data-table
        :columns="dataManageObj.columns"
        :data="dataManageObj.dataList"
        :pagination="false"
        :bordered="false"
      />
    </n-card>
    <n-modal v-model:show="dataManageObj.isShowModal" :mask-closable="false">
      <n-card
        style="width: 600px"
        :title="dataManageObj.modalTitle"
        :bordered="false"
        size="huge"
        role="dialog"
        aria-modal="true"
      >
        <template #header-extra>
          <div @click="handleCloseModal()">
            <SvgIcon icon="material-symbols:close" class="inline-block align-text-bottom text-16px" />
          </div>
        </template>
        <n-form
          ref="formRef"
          :model="model"
          :rules="rules"
          label-placement="left"
          label-width="auto"
          require-mark-placement="right-hanging"
          size="small"
          :style="{
            maxWidth: '640px'
          }"
        >
          <n-form-item label="标签组名称" path="name">
            <n-input v-model:value="model.name" placeholder="请输入标签组名称" />
          </n-form-item>
          <n-form-item label="标签组描述" path="desc">
            <n-input
              v-model:value="model.desc"
              placeholder="请输入标签组描述"
              type="textarea"
              :autosize="{
                minRows: 3,
                maxRows: 5
              }"
            />
          </n-form-item>
        </n-form>
        <template #footer>
          <n-flex justify="flex-end">
            <NButton @click="handleCancel()">取 消</NButton>
            <NButton type="info">确 认</NButton>
          </n-flex>
        </template>
      </n-card>
    </n-modal>
  </div>
</template>

<style scoped lang="scss">
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
    .img {
      width: 140px;
      height: 72px;
      margin-bottom: 20px;
      align-self: center;
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

.wrap-container {
  .item-manage {
    display: flex;
    flex-direction: column;
    align-items: center;
    .item-manage_icon {
      img {
        width: 140px;
        height: 72px;
        margin-top: 20px;
      }
      .iconName {
        font-size: 14px;
        color: #151b26;
        line-height: 20px;
        margin-bottom: 4px;
        align-self: center;
        display: flex;
        align-items: center;
      }
      .item-manage_info {
        font-size: 12px;
        color: #84868c;
        line-height: 20px;
        text-align: center;
      }
    }
  }
}
.wrap_content {
  .header_flex {
    margin-bottom: 16px;
  }
}
</style>
