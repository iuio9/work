<script setup lang="ts">
import { NButton } from 'naive-ui';
import { usePagination } from 'vue-request';
import noTag from '@/assets/imgs/noTag.png';
import SvgIcon from '@/components/custom/svg-icon.vue';
import { deleteResultFile, getResultDataSetSave, getTaskResult, saveResult } from '@/service/api/ano';
import { delTask, getTaskDetail, getTaskPage } from "@/service/api/expansion";
import { fetchGetTreeData } from '@/service/api';
import { getSelectDataSetDictList } from '@/service/api/dataManage';

import aType from "@/assets/imgs/aType.png";
import { useFormRules, useNaiveForm } from '@/hooks/common/form';

const route = useRoute();
const router = useRouter();
const imgList = ref<any[]>([]);
const total = ref<number>(0);
const tabConfig = ref<any>({
  state: 0,
  tabNum: {
    all: undefined,
    haveAno: undefined,
    noAno: undefined
  }
});

const dictMapList = ref<any>([]);
const getData = async (
  params: any = {
    page: 1,
    limit: 24,
    taskId: route.query.taskId
    // state: tabConfig.value.state,
  }
) => {
  // getTaskResult
  const res = await getTaskResult(params);
  if (res.data) {
    const records = res.data.records.map((item: any) => {
      return {
        labels: item.labels,
        imgSrc: item.imgPath,
        isMark: item.isMark,
        isCheck: false,
        ...item
      };
    });
    imgList.value = [...records];
    // 演示环境
    // imgList.value = [
    //   { httpFilePath: aType, fdName: "演示数据001"},
    //   { httpFilePath: aType, fdName: "演示数据002"},
    // ];
    total.value = res.data.total;
  } else {
    imgList.value = [];
    // 演示环境
    // imgList.value = [
    //   { httpFilePath: aType, fdName: "演示数据001"},
    //   { httpFilePath: aType, fdName: "演示数据002"},
    // ];
    total.value = 0;
  }
};

const { current, pageSize, run } = usePagination(getData, {
  defaultParams: [
    {
      limit: 24,
      taskId: route.query.taskId
    }
  ],
  pagination: {
    currentKey: 'page',
    pageSizeKey: 'limit',
    totalKey: 'total'
  }
});

const isAllCheck = computed(() => {
  return imgList.value.length > 0 && imgList.value.every(val => val.isCheck);
});

const visible = ref(false); // 遮罩mask

const isShowSave = ref<Boolean>(true);

const detailData = ref<any>({});

// methods
// const handleOperate = async (sign: 'save' | 'saveNew' | 'cancel' | 'toList') => {
//   if (sign === 'save') {
//     router.push({
//       path: '/data-expansion/addmap',
//       query: {
//         taskId: route.query.taskId
//       }
//     });
//     // request
//     // const fileIds = imgList.value.map((val) => val.id).join(",");
//     // const res = await getResultDataSetSave({ fileIds });
//     // if (res.data) {
//     //   window.$message?.success?.("保存成功！");
//     //   router.back();
//     // }
//   }

//   if(sign === "toList") {
//     router.back()
//   }
//   if(sign === "cancel") {

//   }
// };

const handleSelCurPage = () => {
  // eslint-disable-next-line @typescript-eslint/no-shadow
  const isAllCheck = imgList.value.every(val => val.isCheck);
  if (isAllCheck) {
    imgList.value = imgList.value.map(item => {
      item.isCheck = false;
      return item;
    });
  } else {
    imgList.value = imgList.value.map(item => {
      item.isCheck = true;
      return item;
    });
  }
};

const handleBatchDel = async () => {
  const fileIds = imgList.value
    .filter(val => val.isCheck)
    .map(item => {
      return item.id;
    });
  const res = await deleteResultFile({
    fileIds
  });
  if (res.data >= 1) {
    await getData();
    window.$message?.success?.('删除成功！');
  }
};

const labeLToArr = (str: any) => {
  return str.split(',');
};

const handleOperateMouse = (sign: string, row: any) => {
  if (sign === 'enter') {
    row.isHover = true;
  }
  if (sign === 'leave') {
    row.isHover = false;
  }
};

const handleImgOperate = (sign: string, row: any, index: any) => {
  if (sign === 'edit') {
    router.push({
      // name: 'data-ano_operation',
      // name: 'data-ano_imgoperate',
      name: import.meta.env.VITE_TOGGLE_OPERATE === 'Y' ? 'data-ano_imgoperate' : 'data-ano_operation',
      query: {
        id: route.query.id,
        imgSrc: row.imgSrc,
        isMark: row.isMark,
        fileId: row.fileId,
        imgIdx: index,
        markType: row.anoType ?? 1,  // 标注类型
      }
    });
  }
  if (sign === 'preview') {
  }
};

const handleBack = () => {
  router.back();
};

const handleSave = async () => {
  // router.push({
  //   path: '/data-expansion/addmap',
  //   query: {
  //     taskId: route.query.taskId
  //   }
  // });
  const fileIds = imgList.value.filter(val => val.isCheck).map((val) => val.id).join(",");
  const params = Object.assign({}, versionModel.value, { fileIds, type: 1, taskId: route.query.taskId, });
  const res = await saveResult(params);
  if (res.data) {
    window.$message?.success?.("保存版本成功！");
    const { dataTypeId, dictLabel } = res.data;
    router.push({
      name: "data-manage_map",
      query: {
        dataTypeId,
        dictLabel
      },
    });
  }
};

const handleSaveNew = async () => {
  // 实现保存为新记录的逻辑
  console.log('执行保存为新记录操作');
  router.push({
    path: '/data-expansion/addmap',
    query: {
      taskId: route.query.taskId
    }
  });
};

const handleToList = () => {
  router.back();
};

const handleCancel = () => {
  // 可以添加取消操作的具体逻辑，例如重置表单等
  console.log('执行取消操作');
  const params = {
    page: 1,
    limit: 24,
    taskId: route.query.taskId
  };

  run(params);
};

const handleOperate = async (sign: 'save' | 'saveNew' | 'cancel' | 'toList') => {
  switch (sign) {
    case 'save':
      await handleSave();
      break;
    case 'saveNew':
      // await handleSaveNew();
      versionShowModal.value = true;
      break;
    case 'toList':
      handleToList();
      break;
    case 'cancel':
      handleCancel();
      break;
    default:
      break;
  }
};

const getDetailInfo = async () => {
  const res = await getTaskDetail({
    taskId: route.query.taskId
  });
  if (res.data) {
    detailData.value = res.data;
  }
}

watch(() => route.query.taskId, () => {
  getDetailInfo();
}, {
  immediate: true
})

onMounted(() => {
  isShowSave.value = !route.query?.modelId;

  getMapOptions();
});

/* ---------------------newCode----------------------- */
interface VersionModel {
  dataTypeId: string | undefined;
  groupName: string;
  anoType: string;
}

const versionShowModal = ref<Boolean>(false);
const versionModel = ref<VersionModel>({
  dataTypeId: undefined,
  groupName: '',
  anoType: ''
});
const anoOptions = [
  { value: "0", label: "图像分割" },
  { value: "1", label: "物体检测" },
];
const mapOptions = ref<any>([]);

const { formRef, validate, restoreValidation } = useNaiveForm();
const { createRequiredRule } = useFormRules();

const rules: Record<string, App.Global.FormRule> = {
  dataTypeId: createRequiredRule('请选择数据集类型'),
  groupName: createRequiredRule('请输入数据集名称'),
  anoType: createRequiredRule('请选择标注类型')
};

// 定义数据项的类型
type DataItem = {
  id: string | number;
  dictLabel: string;
  children?: DataItem[];
};

// 定义递归处理函数
const processDataItem = (item: DataItem): DataItem => {
  const { children, ...rest } = item;
  const newItem = {
    ...rest,
    label: item.dictLabel,
    value: item.id
  };

  if (children && children.length > 0) {
    newItem.children = children.map(processDataItem);
  } else if (children) {
    // 若 children 为空数组，不添加到新对象中
  }

  return newItem;
};

// 定义获取地图选项的函数
const getMapOptions = async () => {
  try {
    const res = await getSelectDataSetDictList();
    if (Array.isArray(res.data)) {
      mapOptions.value = res.data.map(processDataItem);
    } else {
      console.error('返回的数据不是数组类型:', res.data);
    }
  } catch (error) {
    console.error('获取数据集字典列表时出错:', error);
  }
};

const saveVersion = async () => {
  await validate();
  const fileIds = imgList.value.filter(val => val.isCheck).map((val) => val.id).join(",");
  const params = Object.assign({}, versionModel.value, { fileIds: fileIds, type: 2, taskId: route.query.taskId, });
  const res = await saveResult(params);
  if (res.data) {
    window.$message?.success?.("保存新版本成功！");
    const { dataTypeId, dictLabel } = res.data;
    router.push({
      name: "data-manage_map",
      query: {
        dataTypeId,
        dictLabel
      },
    });
  }
};
</script>

<template>
  <div class="wrap-container relative box-border h-full w-full flex flex-col items-start justify-start p-24px">
    <div class="box-border w-full flex-1 p-24px flex-col justify-start items-center">
      <NCard class="h-auto w-full !mb-8px">
        <div class="box-border h-auto w-full p-16px">
          <NForm :model="model" label-placement="left" :label-width="120">
            <NGrid responsive="screen" item-responsive>
              <NFormItemGi span="24 s:12 m:6" label="任务名称:" path="tagGroupName" class="pr-24px">
                {{ detailData.taskName }}
              </NFormItemGi>
              <NFormItemGi span="24 s:12 m:6" label="开始时间:" path="tagName" class="pr-24px">{{ detailData.createTime }}
              </NFormItemGi>
              <NFormItemGi span="24 s:12 m:6" label="结束时间:" path="userStatus" class="pr-24px">{{ detailData.updateTime
              }}
              </NFormItemGi>
              <NFormItemGi span="24 s:12 m:6" label="任务状态:" path="userStatus" class="pr-24px">{{ detailData.taskStat }}
              </NFormItemGi>
              <NFormItemGi span="24 s:12 m:12" label="关联数据集名称:" path="userStatus" class="pr-24px">
                {{ detailData.dataSetName }}
              </NFormItemGi>
              <NFormItemGi span="24 s:12 m:6" label="数据总量" path="userStatus" class="pr-24px">
                {{ detailData.dataSetTotal }}
              </NFormItemGi>
            </NGrid>
          </NForm>
        </div>
      </NCard>
      <NCard class="flex-1 w-full overflow-y-auto">
        <div class="box-border h-full w-full flex flex-col items-start justify-start p-24px">
          <div class="header mb-16px h-36px w-full flex flex items-center">
            <div class="item_return mr-16px h-full w-auto flex cursor-pointer items-center" @click="handleBack()">
              <SvgIcon local-icon="oui--return-key" class="inline-block align-text-bottom text-18px color-[#000]">
              </SvgIcon>
              <span class="ml-[4px] block h-full w-auto flex items-center text-[12px] text-[#84868c]">返回</span>
            </div>
            <NButton class="w-88px" @click="handleBatchDel">删除已选</NButton>
            <NCheckbox v-model:checked="isAllCheck" class="ml-24px" @click="handleSelCurPage">选择本页</NCheckbox>
            <div class="h-full flex-1">
              <div class="h-full w-full flex items-center justify-end gap-24px">
                <NButton v-if="isShowSave" type="info" class="w-88px" @click="handleOperate('save')">保存</NButton>
                <NButton type="info" class="min-w-88px" @click="handleOperate('saveNew')">保存新版本</NButton>
                <NButton type="info" class="min-w-88px" @click="handleOperate('cancel')">取消本次结果</NButton>
                <NButton type="info" class="min-w-88px" @click="handleOperate('toList')">返回列表</NButton>
              </div>
            </div>
          </div>
          <div class="mb-16px w-full flex-1 min-h-0 overflow-y-auto">
            <div v-if="imgList.length !== 0" class="imgList flex flex-wrap items-start justify-start gap-32px">
              <NImageGroup>
                <div v-for="(item, index) of imgList" :key="index"
                  class="relative box-border h-auto w-144px bg-[#eee] py-8px"
                  @mouseenter="handleOperateMouse('enter', item)" @mouseleave="handleOperateMouse('leave', item)">
                  <!--无效数据-->
                  <div v-show="item.isInvalid == 0" class="invalid-tip">无效数据</div>
                  <div class="h-24px w-full flex items-start">
                    <NCheckbox v-model:checked="item.isCheck" class="ml-8px"></NCheckbox>
                  </div>
                  <div class="img h-88px w-full flex items-center justify-center">
                    <!--<img :src="item.imgSrc" alt="" class="h-100% w-100%" />-->
                    <NImage width="100%" height="100%" :src="item.httpFilePath" />
                  </div>
                  <div
                    class="item_tag_con box-border h-auto min-h-44px w-full flex items-center justify-between px-8px pt-8px">
                    <div class="w-fulltext-[#151b26]">
                      {{ item.fdName }}
                    </div>
                    <div v-show="false" class="w-[35%] flex items-center justify-end gap-[8px]">
                      <div @click="handleImgOperate('edit', item, index)">
                        <SvgIcon local-icon="lucide--edit" class="text-[16px]"></SvgIcon>
                      </div>
                    </div>
                  </div>
                </div>
              </NImageGroup>
            </div>
            <div v-else class="imgList h-full w-full flex flex-col items-center justify-center">
              <img :src="noTag" alt="" class="block" />
              <div class="mt-24px text-[14px] text-[#666]">暂无可用数据 没有找到数据！</div>
            </div>
          </div>
          <div class="main-pagination h-auto w-full flex items-center justify-end">
            <NPagination v-model:page="current" v-model:page-size="pageSize" v-model:item-count="total"
              :page-count="total" :page-slot="5" :page-sizes="[1, 24, 48, 72]" show-size-picker />
          </div>
        </div>
      </NCard>
    </div>

    <!-- <div class="footer box-border w-full flex items-center justify-start gap-24px bg-[#fff] px-24px py-12px">
      <NButton type="info" class="w-88px" @click="handleOperate('save')">保存</NButton>
    </div> -->

    <!-- save newVersion modal-->
    <NModal v-model:show="versionShowModal" title="保存新版本" preset="card" class="w-600px">
      <NScrollbar class="h-auto pr-20px">
        <NForm ref="formRef" :model="versionModel" :rules="rules" label-placement="left" :label-width="100">
          <NGrid responsive="screen" item-responsive>
            <!-- 数据集名称 -->
            <NFormItemGi span="24 m:24" label="数据集名称:" path="groupName">
              <NInput v-model:value="versionModel.groupName" placeholder="请输入数据集名称" />
            </NFormItemGi>
            <!-- 数据集类型 -->
            <NFormItemGi span="24 m:24" label="数据集类型:" path="dataTypeId">
              <NCascader v-model:value="versionModel.dataTypeId" clearable placeholder="请选择数据集类型" :options="mapOptions"
                check-strategy="all" />
            </NFormItemGi>
            <!-- 标注类型 -->
            <NFormItemGi span="24 m:24" label="标注类型:" path="anoType">
              <NRadioGroup v-model:value="versionModel.anoType">
                <NRadioButton v-for="item in anoOptions" :key="item.value" :value="item.value" :label="item.label" />
              </NRadioGroup>
            </NFormItemGi>
          </NGrid>
        </NForm>
      </NScrollbar>
      <template #footer>
        <NSpace justify="end" :size="16">
          <NButton @click="() => (versionShowModal = false)">{{ $t('common.cancel') }}</NButton>
          <NButton type="primary" @click="saveVersion">保存</NButton>
        </NSpace>
      </template>
    </NModal>
  </div>
</template>

<style scoped lang="scss">
:deep(.n-transfer-list--target) {
  display: none !important;
}

.wrap-container {
  padding: 0 !important;
}

:deep(.n-card__content) {
  padding: 0 !important;
}

// :deep(.n-form-item) {
//   height: 32px !important;
// }
</style>
