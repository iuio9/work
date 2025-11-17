<script setup lang="ts">
import { usePagination } from "vue-request";
import { FlashOutline } from "@vicons/ionicons5";
import { NButton } from "naive-ui";
import SvgIcon from "@/components/custom/svg-icon.vue";
import {
  DataDetailsCount,
  addDataSetAndLabel,
  addSaveLabel,
  deleteDataSetLabel,
  deleteFile,
  revokeFile,
  getDataDetails,
  getSelectGroupLabel,
  selectDataSetLabel,
  withdraw
} from "@/service/api/ano";
import noTag from "@/assets/imgs/noTag.png";
import { fetchLabelEdit } from "@/service/api/tag";
import _ from "lodash";

enum OperateType {
  "import",
  "annotation",
}

interface SearchObj {
  sourceList: string[];
  importTimes: string[];
  anoTimes: string[];
  tags: any[];
  params: {
    source?: string;
    importTime?: string[];
    anoTime?: string[];
    tag?: string;
  };
}

// data
const route = useRoute();
const router = useRouter();
const searchObj = ref<Partial<SearchObj>>({
  sourceList: ["全部", "本地上传", "数据增强"],
  importTimes: [],
  anoTimes: [],
  tags: [],
  params: {
    source: "",
  },
});

const query = ref<any>({});

// methods
const handleOperation = (sign: OperateType) => {
  switch (sign) {
    case OperateType.annotation:
      router.push({
        // name: "data-ano_operation",
        // name: "data-ano_imgoperate",
        name: import.meta.env.VITE_TOGGLE_OPERATE === 'Y' ? 'data-ano_imgoperate' : 'data-ano_operation',
      });
      break;
    case OperateType.import:
      router.push({
        name: "data-manage_import",
      });
      break;
    default:
      throw new Error("Invalid operate type");
  }
};

const handleTagMoute = (sign: string, row: any) => {
  if (sign === "enter") {
    row.isHover = true;
  }
  if (sign === "leave") {
    row.isHover = false;
  }
};

const handleOperateMouse = (sign: string, row: any) => {
  if (sign === "enter") {
    row.isHover = true;
  }
  if (sign === "leave") {
    row.isHover = false;
  }
};

const handleImgOperate = async (sign: string, row: any, index: any) => {
  if (sign === "edit") {
    router.push({
      // name: "data-ano_operation",
      // name: "data-ano_imgoperate",
      name: import.meta.env.VITE_TOGGLE_OPERATE === 'Y' ? 'data-ano_imgoperate' : 'data-ano_operation',
      query: {
        id: route.query.id,
        imgSrc: row.imgSrc,
        isMark: row.isMark,
        fileId: row.fileId,
        imgIdx: index,
        markType: route.query.markType,  // 标注类型
      },
    });
  }
  if (sign === "preview") {
  }
  if (sign === "revoke") {
    const res = await withdraw({
      taskId: route.query.markUserId,
      ids: [row.id],
      sign: route.query?.sign
    });
    if (res.data >= 1) {
      await getDataDetailsCount();
      await getData();
      window.$message?.success?.("撤销成功！");
    }
  }
};

const handlePositiveClick = async (row, index) => {
  const res = await withdraw({
    taskId: route.query.markUserId,
    ids: [row.id],
    sign: route.query?.sign
  });
  if (res.data >= 1) {
    await getDataDetailsCount();
    await getData();
    window.$message?.success?.("撤销成功！");
  }
}

// newCode
const imgList = ref<any[]>([]);
const total = ref<number>(0);
const tagConfig = ref<any>({
  isEditTag: true,
  sign: "group",
  params: {
    val: undefined,
    color: "#000000",
  },
  options: [],
  isShow: false,
  tagList: [],
  deepTagList: [],
});
const tabConfig = ref<any>({
  state: 0,
  tabNum: {
    all: undefined,
    haveAno: undefined,
    noAno: undefined,
    invalid: undefined,
  },
});
const selectRef = ref(null);
const getData = async (
  params: any = {
    page: 1,
    limit: 24,
    sonId: route.query.id,
    state: tabConfig.value.state,
    markUserId: route.query?.markUserId,
    sign: route.query?.sign
  },
) => {
  params.state = tabConfig.value.state;

  const res = await getDataDetails(params);
  if (res.data) {
    const records = res.data.records.map((item: any) => {
      return {
        labels: item.labels,
        imgSrc: item.imgPath,
        isMark: item.isMark,
        isCheck: false,
        ...item,
      };
    });
    imgList.value = [...records];
    total.value = res.data.total;
  } else {
    imgList.value = [];
    total.value = 0;
  }
};

const { current, pageSize, run, changeCurrent, changePageSize } = usePagination(getData, {
  defaultParams: [
    {
      limit: 24,
      sonId: route.query.id,
      state: tabConfig.value.state,
      markUserId: route.query?.markUserId,
      sign: route.query?.sign
    },
  ],
  pagination: {
    currentKey: "page",
    pageSizeKey: "limit",
    totalKey: "total",
  },
  manual: false
});

// watch
watch(
  () => tabConfig.value.state,
  (newVal) => {
    const params = {
      page: 1,
      limit: 24,
      sonId: route.query.id,
      state: newVal,
      markUserId: route.query?.markUserId,
      sign: route.query?.sign
    };
    changeCurrent(1);
    // getData(params);
  }
);

// add tag
const handleAddTag = (sign = "group") => {
  tagConfig.value.isEditTag = true;
  tagConfig.value.sign = sign;
};

const handleCancel = () => {
  tagConfig.value.isEditTag = false;
};

const getSelectData = async () => {
  const params = {
    sonId: route.query.id,
  };
  const res = await getSelectGroupLabel(params);
  tagConfig.value.options = res.data.map((item: any) => {
    return {
      label: item.label,
      value: item.id,
      count: item.count,
    };
  });
};
const navToTagGroup = () => {
  router.push({
    name: "data-ano_group",
  });
};

const handleSelectChange = (val) => {
  // tagConfig.value.isShow = true;
  console.log(val);
};
const handleSelectFocus = () => {
  tagConfig.value.params.val = undefined;
  tagConfig.value.isShow = true;
};

const getTagList = async () => {
  const res = await selectDataSetLabel({ sonId: route.query.id });
  // const tagObj = {name: '测试001', isOperate: true, tagIdx: `${tagIdx}`, isHover: false};
  const dataList = res.data.map((item, index) => {
    return {
      name: item.labelName,
      // color: convertToHexValue(extractRGBValues(item.labelColor)),
      color: item.labelColor,
      isOperate: true,
      tagIdx: index,
      isHover: false,
      count: item.labelCount,
      labelId: item.labelId,
      ...item,
    };
  });
  tagConfig.value.tagList = _.uniqBy(dataList, "name");
  // deepTagList
  tagConfig.value.deepTagList = dataList;
};

const handleDefine = async () => {
  if (tagConfig.value.sign === "group") {
    const params = {
      sonId: route.query.id,
      labelGroupId: tagConfig.value.params.val,
    };
    const res = await addDataSetAndLabel(params);
    if (res.data >= 1) {
      window.$message?.success?.("添加标签组成功！");
      tagConfig.value.isShow = false;
      // eslint-disable-next-line @typescript-eslint/no-use-before-define
      await getTagList();
    }
  }
  if (tagConfig.value.sign === "tag") {
    const params = {
      sonId: route.query.id,
      labelColor: tagConfig.value.params.color,
      labelName: tagConfig.value.params.val,
    };
    const res = await addSaveLabel(params);
    if (res.data >= 1) {
      window.$message?.success?.("添加标签成功！");
      tagConfig.value.isEditTag = false;
      tagConfig.value.params.val = "";
      await getTagList();
    }
  }
};

const handleInput = (val: any) => {
  const dataList = tagConfig.value.deepTagList.filter((item: any) => {
    return item.name.includes(val);
  });
  tagConfig.value.tagList = dataList;
};

const handleTagOperate = async (sign: string, row: any) => {
  if (sign === "delete") {
    // const idx = tagList.findIndex((item: any) => item.tagIdx === row.tagIdx);
    // if (idx !== -1) {
    //   tagList.splice(idx, 1);
    // }
    deleteDataSetLabel({
      sonId: route.query.id,
      labelId: row.labelId,
    }).then(async (res) => {
      if (res.data >= 1) {
        window.$message?.success?.("删除成功！");
        await getTagList();
      }
    });
  }
  if (sign === "edit") {
    row.isOperate = !row.isOperate;
  }
  if (sign === "confirm") {
    const { labelId, color, labelGroupId, name } = row;
    const res = await fetchLabelEdit({
      id: labelId,
      labelColor: color,
      labelGroupId,
      labelName: name,
    });
    if (res.data >= 1) {
      window.$message?.success("修改标签成功！");
      await getTagList();
      row.isOperate = !row.isOperate;
    }
  }
  if (sign === "cancel") {
    row.isOperate = !row.isOperate;
  }
};

const handleBack = () => {
  router.back();
};

// tab change
const handleTabChange = (val: number | any) => {
  tabConfig.value.state = val;
};

const getDataDetailsCount = async () => {
  const res = await DataDetailsCount({ sonId: route.query.id, markUserId: route.query?.markUserId, sign: route.query?.sign });
  tabConfig.value.tabNum.all = res.data.all ?? 0;
  tabConfig.value.tabNum.haveAno = res.data.haveAno ?? 0;
  tabConfig.value.tabNum.noAno = res.data.noAno ?? 0;
  tabConfig.value.tabNum.invalid = res.data?.invalid ?? 0;
};

const handleSelCurPage = () => {
  const isAllCheck = imgList.value.every((val) => val.isCheck);
  if (isAllCheck) {
    imgList.value = imgList.value.map((item) => {
      item.isCheck = false;
      return item;
    });
  } else {
    imgList.value = imgList.value.map((item) => {
      item.isCheck = true;
      return item;
    });
  }
};
const handleBatchDel = async () => {
  const fileIds = imgList.value
    .filter((val) => val.isCheck)
    .map((item) => {
      return item.fileId;
    });
  const res = await deleteFile({
    sonId: route.query.id,
    fileIds,
  });
  if (res.data >= 1) {
    await getDataDetailsCount();
    await getData();
    window.$message?.success?.("撤回成功！");
  }
};
// handleBatchRevoke
const handleBatchRevoke = async () => {
  const fileIds = imgList.value
    .filter((val) => val.isCheck)
    .map((item) => {
      return item.id;
    });
  if (fileIds.length > 0) {
    const res = await withdraw({
      taskId: route.query.markUserId,
      ids: fileIds,
      sign: route.query?.sign
    });
    if (res.data >= 1) {
      await getDataDetailsCount();
      await getData();
      window.$message?.success?.("撤销成功！");
    }
  } else {
    window.$message?.error?.("请选择要撤回的图片！");
  }
}

const labeLToArr = (str: any) => {
  return str.split(",");
};

// computed
const isAllCheck = computed(() => {
  return imgList.value.length > 0 && imgList.value.every((val) => val.isCheck);
});

const nImg = ref(null);
const handleImgChange = (item) => {
  // item.imgUrl = item.previewImgPath;
  const nImg = document.querySelector(".wrap-img");
  // 触发该元素的点击事件
  if (nImg) {
    nImg.click();
  }
};

// 用于存储元素的ref
const itemRefs = ref({});
// 设置ref的函数
const setItemRef = (el, id) => {
  if (el) {
    itemRefs.value[`tag_${id}`] = el;
  }
};
const handleTagClick = (row: any) => {
  tagConfig.value.tagList.forEach((row) => {
    itemRefs.value[`tag_${row.labelId}`].style.border = "1px solid #eee";
  });
  // 根据row.labelId找到ref 设置边框样式
  const refName = `tag_${row.labelId}`;
  itemRefs.value[refName].style.border = "1px solid blue";
  const params = {
    page: 1,
    limit: 24,
    sonId: route.query.id,
    state: tabConfig.value.state,
    labelId: row.labelId,
    sign: route.query?.sign
  };
  run(params);
};

const handleTagdbClick = (row: any) => {
  // 根据row.labelId找到ref 设置边框样式
  const refName = `tag_${row.labelId}`;
  itemRefs.value[refName].style.border = "1px solid #eee";
  const params = {
    page: 1,
    limit: 24,
    sonId: route.query.id,
    state: tabConfig.value.state,
  };
  run(params);
};
onMounted(() => {
  query.value = route.query;
  getDataDetailsCount();
  getSelectData();
  getTagList();
});

</script>

<template>
  <div style="padding: 0 !important"
    class="wrap_main h-full w-full flex flex-col items-center justify-start bg-[#f7f7f9] p-0">
    <div class="header box-border box-border h-[48px] w-full flex items-center bg-[#fff] px-16px py-0">
      <div class="item_return h-full w-auto flex cursor-pointer items-center" @click="handleBack()">
        <SvgIcon local-icon="oui--return-key" class="inline-block align-text-bottom text-18px color-[#000]"></SvgIcon>
        <span class="ml-[4px] block h-full w-auto flex items-center text-[12px] text-[#84868c]">
          返回
        </span>
      </div>
      <div class="item_name ml-[12px] h-full w-auto flex items-center text-16px text-[#151b26] font-[500]">
        详情
      </div>
      <div class="item_rBtn_con h-full flex-1">
        <div class="h-full w-full flex items-center justify-end gap-[24px]">
        </div>
      </div>
    </div>
    <div class="header1 box-border box-border h-[48px] w-full flex items-center bg-[#fff] px-16px py-0">
      <NTabs type="line" animated class="wrap_tabs ml-14px w-auto" @update:value="handleTabChange"
        v-if="!route.query.sign">
        <NTabPane name="0" tab="全部">
          <template #tab>全部({{ tabConfig.tabNum.all }})</template>
        </NTabPane>
        <NTabPane name="1" tab="有标注信息">
          <template #tab>有标注信息({{ tabConfig.tabNum.haveAno }})</template>
        </NTabPane>
        <NTabPane name="2" tab="无标注信息">
          <template #tab>无标注信息({{ tabConfig.tabNum.noAno }})</template>
        </NTabPane>
        <NTabPane name="3" tab="无效数据信息">
          <template #tab>无效数据信息({{ tabConfig.tabNum.invalid }})</template>
        </NTabPane>
      </NTabs>
      <NTabs type="line" animated class="wrap_tabs ml-14px w-auto" @update:value="handleTabChange" v-else>
        <NTabPane name="0" tab="全部">
          <template #tab>未审核信息({{ tabConfig.tabNum.all }})</template>
        </NTabPane>
        <NTabPane name="1" tab="有标注信息">
          <template #tab>审核通过({{ tabConfig.tabNum.haveAno }})</template>
        </NTabPane>
        <NTabPane name="2" tab="无标注信息">
          <template #tab>审核未通过({{ tabConfig.tabNum.noAno }})</template>
        </NTabPane>
      </NTabs>
    </div>
    <div class="content w-full flex-1 p-[16px] overflow-auto">
      <NCard title="" class="h-full w-full">
        <div class="h-full w-full flex items-center justify-start border-1 border-[#eee]"
          v-if="route.query.isMany == '0'">
          <!--left tag || input content-->
          <div class="h-full w-20% border-r-1 border-r-[#eee] flex-col justify-start">
            <!--top select -->
            <div class="h-[10%] w-full border-b-1 border-b-[#eeee]">
              <div v-if="!tagConfig.isEditTag" class="h-full w-full flex items-center justify-around">
                <div class="flex items-center text-[16px] text-[#000]">
                  标签栏
                </div>
                <div class="flex items-center gap-4px">
                  <!--<NButton
                    type="primary"
                    class="add"
                    size="small"
                    @click="handleAddTag('tag')"
                    >添加标签</NButton
                  >-->
                  <NButton type="primary" class="add" size="small" @click="handleAddTag()">选择标签组</NButton>
                </div>
              </div>
              <div v-else class="box-border h-full w-full flex items-center justify-center">
                <div v-if="tagConfig.sign === 'group'" class="select w-[94%]">
                  <!--<NSelect
                    ref="selectRef"
                    v-model:value="tagConfig.params.val"
                    :show="tagConfig.isShow"
                    placeholder="请选择标签组"
                    :options="tagConfig.options"
                    @update:value="handleSelectChange"
                    @focus="handleSelectFocus"
                  >
                    <template #action>
                      <div class="w-full flex items-center justify-between">
                        <div class="l w-60% flex items-center">
                          <NButton
                            quaternary
                            type="info"
                            size="tiny"
                            @click="navToTagGroup()"
                            >创建标签组</NButton
                          >
                        </div>
                        <div class="r w-35% flex items-center justify-end">
                          <NButton
                            quaternary
                            type="info"
                            size="tiny"
                            @click="handleDefine()"
                            >确定</NButton
                          >
                          <NButton
                            quaternary
                            size="tiny"
                            @click="handleCancel()"
                            >取消</NButton
                          >
                        </div>
                      </div>
                    </template>
                  </NSelect>-->
                  <NSelect ref="selectRef" v-model:value="tagConfig.params.val" placeholder="请选择标签组"
                    :options="tagConfig.options" @update:value="handleSelectChange" @focus="handleSelectFocus">
                  </NSelect>
                </div>
                <div v-else class="select w-[94%] flex items-center">
                  <div class="flex items-center">
                    <NColorPicker v-model:value="tagConfig.params.color" :show-alpha="false" :actions="['confirm']"
                      class="custom-color-picker" />
                  </div>
                  <div class="w-[70%]">
                    <NInput v-model:value="tagConfig.params.val" class="w-full" placeholder="请输入标签名" />
                  </div>
                  <div class="ml-4px w-[25%] flex items-center">
                    <NButton quaternary type="info" size="tiny" @click="handleDefine()">确定</NButton>
                    <NButton quaternary size="tiny" @click="handleCancel()">取消</NButton>
                  </div>
                </div>
              </div>
            </div>
            <!--center search -->
            <!--<div
              class="box-border h-[12%] w-full flex items-center border-b-1 border-b-[#eee] px-[24px]"
            >
              <NInput
                placeholder="搜索"
                class="h-50% w-full flex items-center"
                @input="handleInput"
              >
                <template #suffix>
                  <NIcon :component="FlashOutline" />
                </template>
              </NInput>
            </div>-->
            <!--bottom tag list-->
            <!--class="box-border h-[90%] w-full overflow-y-auto px-24px py-24px"-->
            <div v-if="tagConfig.tagList.length !== 0"
              class="box-border flex-1 flex flex-col w-full overflow-auto px-24px py-24px">
              <!--<div class="w-full h-auto overflow-y-auto">
                <div
                  v-for="(item, index) of tagConfig.tagList"
                  :key="index"
                  class="mb-8px box-border h-40px w-full flex items-center overflow-hidden border-1 border-[#eee] rounded-[4px]"
                  :ref="(el) => setItemRef(el, item.labelId)"
                >
                  <div
                    class="mr-12px h-full w-10px"
                    :style="{ background: item.color }"
                  ></div>
                  &lt;!&ndash;default&ndash;&gt;
                  <div
                    v-show="item.isOperate"
                    class="default h-full w-full flex items-center"
                    @mouseenter="handleTagMoute('enter', item)"
                    @mouseleave="handleTagMoute('leave', item)"
                  >
                    <div
                      class="w-[80%]"
                      @click="handleTagClick(item)"
                      @dblclick="handleTagdbClick(item)"
                    >
                      {{ item.name }}
                    </div>
                    <div
                      v-show="item.isHover"
                      class="h-full w-[20%] flex items-center gap-[8px]"
                    >
                      <div @click="handleTagOperate('edit', item)">
                        &lt;!&ndash;<SvgIcon icon="lucide:edit" class="text-[16px]"></SvgIcon>&ndash;&gt;
                        <SvgIcon
                          local-icon="lucide&#45;&#45;edit"
                          class="text-[16px]"
                        ></SvgIcon>
                      </div>
                      <div @click="handleTagOperate('delete', item)">
                        &lt;!&ndash;<SvgIcon
                          icon="material-symbols-light:delete-outline"
                          class="text-[20px]"
                        ></SvgIcon>&ndash;&gt;
                        <SvgIcon
                          local-icon="material-symbols-light&#45;&#45;delete-outline"
                          class="text-[20px]"
                        ></SvgIcon>
                      </div>
                    </div>
                  </div>
                  <div
                    v-show="!item.isOperate"
                    class="h-full w-full flex items-center"
                  >
                    <div class="item_ipt_con h-full w-[80%] flex items-center">
                      <NInput
                        v-model:value="item.name"
                        type="text"
                        placeholder="tag"
                        class="border-none outline-none"
                      />
                    </div>
                    <div class="item_ipt_con h-full w-[40%] flex items-center">
                      <NButton
                        quaternary
                        type="info"
                        size="tiny"
                        @click="handleTagOperate('confirm', item)"
                      >
                        确定
                      </NButton>
                      <NButton
                        quaternary
                        type="default"
                        size="tiny"
                        @click="handleTagOperate('cancel', item)"
                      >
                        取消
                      </NButton>
                    </div>
                  </div>
                </div>
              </div>-->
              <div class="w-full h-full overflow-hidden">
                <n-virtual-list class="!h-full" :item-size="42" :items="tagConfig.tagList">
                  <template #default="{ item }">
                    <div
                      class="mb-8px box-border h-40px w-full flex items-center overflow-hidden border-1 border-[#eee] rounded-[4px]"
                      :ref="(el) => setItemRef(el, item.labelId)">
                      <div class="mr-12px h-full w-10px" :style="{ background: item.color }"></div>
                      <!--default-->
                      <div v-show="item.isOperate" class="default h-full w-full flex items-center cursor-pointer"
                        @mouseenter="handleTagMoute('enter', item)" @mouseleave="handleTagMoute('leave', item)">
                        <div class="w-[80%]" @click="handleTagClick(item)" @dblclick="handleTagdbClick(item)">
                          <span>{{ item.name }}</span>
                          <span class="ml-4px">(ID: {{ item.labelId }})</span>
                        </div>
                        <div v-show="item.isHover" class="h-full w-[20%] flex items-center gap-[8px]">
                          <div @click="handleTagOperate('edit', item)">
                            <SvgIcon local-icon="lucide--edit" class="text-[16px]"></SvgIcon>
                          </div>
                          <div @click="handleTagOperate('delete', item)">
                            <SvgIcon local-icon="material-symbols-light--delete-outline" class="text-[20px]"></SvgIcon>
                          </div>
                        </div>
                      </div>
                      <div v-show="!item.isOperate" class="h-full w-full flex items-center">
                        <div class="item_ipt_con h-full w-[80%] flex items-center">
                          <NInput v-model:value="item.name" type="text" placeholder="tag"
                            class="border-none outline-none" />
                        </div>
                        <div class="item_ipt_con h-full w-[40%] flex items-center">
                          <NButton quaternary type="info" size="tiny" @click="handleTagOperate('confirm', item)">
                            确定
                          </NButton>
                          <NButton quaternary type="default" size="tiny" @click="handleTagOperate('cancel', item)">
                            取消
                          </NButton>
                        </div>
                      </div>
                    </div>
                  </template>
                </n-virtual-list>
              </div>
            </div>
            <div v-else class="box-border h-[75%] w-full flex flex-col items-center justify-center px-24px py-24px">
              <img :src="noTag" alt="" />
              <div class="mt-24px text-[14px] text-[#666]">
                暂无可用标签 ，请点击上方按钮添加！
              </div>
              <!--<div class="cursor-pointer" @click="navToTagGroup()">跳转</div>-->
            </div>
          </div>
          <!--right content-->
          <div class="box-border h-full w-80% flex flex-col items-start justify-start p-24px">
            <div class="header mb-16px h-36px w-full flex flex items-center">
              <NButton class="w-88px" @click="handleBatchDel">删除</NButton>
              <NCheckbox v-model:checked="isAllCheck" class="ml-24px" @click="handleSelCurPage">选择本页</NCheckbox>
            </div>
            <div class="mb-16px w-full flex-1 shrink-1 overflow-y-auto">
              <n-image-group>
                <div v-if="imgList.length !== 0" class="imgList grid grid-cols-8 grid-rows-3 gap-16px content-start">
                  <div v-for="(item, index) of imgList" :key="index"
                    class="row-span-1 relative box-border bg-[#eee] py-8px"
                    @mouseenter="handleOperateMouse('enter', item)" @mouseleave="handleOperateMouse('leave', item)">
                    <!--无效数据-->
                    <div v-show="item.isInvalid == 0" class="invalid-tip">
                      无效数据
                    </div>
                    <div class="h-24px w-full flex items-start">
                      <NCheckbox v-model:checked="item.isCheck" class="ml-8px"></NCheckbox>
                    </div>
                    <div class="img h-68px w-full flex items-center justify-center" @click="handleImgChange(item)">
                      <!--<img :src="item.imgSrc" alt="" class="h-100% w-100%" />-->
                      <NImage :ref="nImg" lazy class="wrap-img" width="100%" height="100%" :src="item.imgSrc"
                        :preview-src="item.previewImgPath" />
                    </div>
                    <NPopover v-if="!!item.fileName" trigger="hover">
                      <template #trigger>
                        <div class="w-full h-auto px-8px pt-8px box-border line-clamp-2 text-12px font-bold">
                          {{ item.fileName }}
                        </div>
                      </template>
                      <div class="w-full h-auto px-8px pt-8px box-border line-clamp-2 text-12px font-bold">
                        {{ item.fileName }}
                      </div>
                    </NPopover>
                    <!-- <div class="w-full h-auto px-8px pt-8px box-border line-clamp-2 text-12px font-bold">
                      {{ item.fileName }}
                    </div> -->
                    <div class="item_tag_con box-border h-34px w-full flex items-center justify-between px-8px">
                      <NPopover v-if="!!item.labels" trigger="hover">
                        <template #trigger>
                          <div class="w-70% truncate cursor-pointer text-[#151b26]">
                            {{ labeLToArr(item.labels)[0] }}
                          </div>
                        </template>
                        <div class="flex items-center gap-14px">
                          <span v-for="(item, index) of labeLToArr(item.labels)" :index="index">{{ item }}</span>
                        </div>
                      </NPopover>
                      <div v-else class="w-50% text-[#151b26]">无标签</div>
                      <div v-show="item.isHover" class="w-[35%] flex items-center justify-end gap-[8px]">
                        <div @click="handleImgOperate('edit', item, index)">
                          <SvgIcon local-icon="lucide--edit" class="text-[16px]"></SvgIcon>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
                <div v-else class="imgList h-full w-full flex flex-col items-center justify-center">
                  <img :src="noTag" alt="" class="block" />
                  <div class="mt-24px text-[14px] text-[#666]">
                    暂无可用数据 没有找到数据！
                  </div>
                </div>
              </n-image-group>
            </div>
            <div class="main-pagination h-auto w-full flex items-center justify-end">
              <NPagination v-model:page="current" v-model:page-size="pageSize" v-model:item-count="total"
                :page-count="total" :page-slot="5" :page-sizes="[1, 24, 48, 72]" show-size-picker />
            </div>
          </div>
        </div>
        <div class="h-full w-full flex items-center justify-start border-1 border-[#eee]" v-else>
          <div class="box-border h-full w-full flex flex-col items-start justify-start p-24px">
            <!-- <div class="header mb-16px h-36px w-full flex flex items-center">
              <NButton class="w-88px" @click="handleBatchRevoke">批量撤销</NButton>
              <NCheckbox v-model:checked="isAllCheck" class="ml-24px" @click="handleSelCurPage">选择本页</NCheckbox>
            </div> -->
            <div class="mb-16px w-full flex-1 shrink-1 overflow-y-auto">
              <n-image-group>
                <div v-if="imgList.length !== 0" class="imgList grid grid-cols-8 grid-rows-3 gap-16px content-start">
                  <div v-for="(item, index) of imgList" :key="index"
                    class="row-span-1 relative box-border bg-[#eee] py-8px"
                    @mouseenter="handleOperateMouse('enter', item)" @mouseleave="handleOperateMouse('leave', item)">
                    <!--无效数据-->
                    <div v-show="item.isInvalid == 0" class="invalid-tip">
                      无效数据
                    </div>
                    <div class="h-24px w-full flex items-start">
                      <NCheckbox v-model:checked="item.isCheck" class="ml-8px"></NCheckbox>
                    </div>
                    <div class="img h-68px w-full flex items-center justify-center" @click="handleImgChange(item)">
                      <NImage :ref="nImg" lazy class="wrap-img" width="100%" height="100%" :src="item.imgSrc"
                        :preview-src="item.previewImgPath" />
                    </div>
                    <!-- <div class="w-full h-auto px-8px pt-8px box-border line-clamp-2 text-12px font-bold">
                      {{ item.fileName }}
                    </div> -->
                    <NPopover v-if="!!item.fileName" trigger="hover">
                      <template #trigger>
                        <div class="w-full h-auto px-8px pt-8px box-border line-clamp-2 text-12px font-bold">
                          {{ item.fileName }}
                        </div>
                      </template>
                      <div class="w-full h-auto px-8px pt-8px box-border line-clamp-2 text-12px font-bold">
                        {{ item.fileName }}
                      </div>
                    </NPopover>
                    <div class="item_tag_con box-border h-34px w-full flex items-center justify-between px-8px">
                      <NPopover v-if="!!item.labels" trigger="hover">
                        <template #trigger>
                          <div class="w-50% cursor-pointer text-[#151b26]">
                            {{ labeLToArr(item.labels)[0] }}
                          </div>
                        </template>
                        <div class="flex items-center gap-14px">
                          <span v-for="(item, index) of labeLToArr(item.labels)" :index="index">{{ item }}</span>
                        </div>
                      </NPopover>
                      <div v-else class="w-50% text-[#151b26]">无标签</div>
                      <div class="w-[35%] flex items-center justify-end gap-[8px]">
                        <!-- <n-popover trigger="hover">
                          <template #trigger>
                            <div @click="handleImgOperate('revoke', item, index)" v-show="item.id">
                              <SvgIcon local-icon="icon-park--back-one" class="text-[16px] text-[#000]"></SvgIcon>
                            </div>
                          </template>
                          <span>撤回当前标注</span>
                        </n-popover> -->
                        <!-- <n-popconfirm @positive-click="handlePositiveClick(item, index)"
                          @negative-click="handleNegativeClick">
                          <template #trigger>
                            <div v-show="item.id">
                              <SvgIcon local-icon="icon-park--back-one" class="text-[16px] text-[#000]"></SvgIcon>
                            </div>
                          </template>
                          <span>是否确认撤销当前标注?</span>
                        </n-popconfirm> -->
                      </div>
                    </div>
                  </div>
                </div>
                <div v-else class="imgList h-full w-full flex flex-col items-center justify-center">
                  <img :src="noTag" alt="" class="block" />
                  <div class="mt-24px text-[14px] text-[#666]">
                    暂无可用数据 没有找到数据！
                  </div>
                </div>
              </n-image-group>
            </div>
            <div class="main-pagination h-auto w-full flex items-center justify-end">
              <NPagination v-model:page="current" v-model:page-size="pageSize" v-model:item-count="total"
                :page-count="total" :page-slot="5" :page-sizes="[1, 24, 48, 72]" show-size-picker />
            </div>
          </div>
        </div>
      </NCard>
    </div>
  </div>
</template>

<style scoped lang="scss">
:deep(.wrap_tabs) {
  .n-tabs-nav-scroll-content {
    border: none !important;
  }

  .n-tabs-pane-wrapper {
    display: none !important;
  }
}

.invalid-tip {
  z-index: 666;
  position: absolute;
  right: 0;
  top: 0;
  width: 40px;
  height: 40px;
  background-color: rgba(0, 0, 0, 0.4);
  color: #fff;
  font-size: 12px;
  padding: 4px;
  text-align: center;
  user-select: none;

  &:before {
    position: absolute;
    content: " ";
    width: 10px;
    height: 5px;
    bottom: -20px;
    overflow: hidden;
    left: 0;
    border-left: 0;
    border-top: 10px solid rgba(0, 0, 0, 0.4);
    border-right: 20px solid transparent;
    border-bottom: 10px solid transparent;
    box-sizing: border-box;
  }

  &:after {
    position: absolute;
    content: " ";
    width: 10px;
    height: 5px;
    bottom: -20px;
    overflow: hidden;
    right: 0;
    border-right: 0;
    border-left: 20px solid transparent;
    border-top: 10px solid rgba(0, 0, 0, 0.4);
    border-bottom: 10px solid transparent;
    box-sizing: border-box;
  }
}

:deep(.custom-color-picker) {
  width: 24px !important;
  height: 24px !important;

  .n-color-picker-trigger {
    border: none !important;
  }

  .n-color-picker-trigger__value {
    display: none;
  }
}

:deep(.wrap-img) {
  //width: 100%;
  height: 100%;

  .n-image {
    //width: 100%;
    height: 100%;
  }
}

:deep(.n-card__content) {
  height: 100% !important;
  // padding: 0 !important;
}
</style>
