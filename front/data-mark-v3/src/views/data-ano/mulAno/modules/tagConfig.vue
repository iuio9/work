<template>
  <div class="w-full h-full border-1 border-[#eee]">
    <div class="h-full w-full">
      <!-- 标签组选择或自定义标签部分 -->
      <div class="min-h-60px w-full flex items-center">
        <div v-if="!isEditTag" class="h-full w-full flex items-center justify-around">
          <div class="flex items-center text-[16px] text-[#000]">标签组选择</div>
          <div class="flex items-center gap-4px">
            <NButton type="primary" class="add" size="small" @click="handleAddTag('tag')">自定义标签</NButton>
            <NButton type="primary" class="add" size="small" @click="handleAddTag()">选择标签组</NButton>
            <NButton type="primary" class="add" size="small" @click="handleAddTag('single')">选择单标签</NButton>
          </div>
        </div>
        <!-- 编辑模式下的内容 -->
        <div v-else class="w-full flex items-center px-16px box-border">
          <div v-if="tagConfig.sign === 'group'" class="select w-full">
            <NSelect ref="selectRef" v-model:value="tagConfig.params.val" placeholder="请选择标签组" :show="tagConfig.isShow"
              :options="tagConfig.options" @update:value="handleSelectChange" @focus="handleSelectFocus">
              <template #action>
                <div class="w-full flex items-center justify-between">
                  <div class="l w-60% flex items-center">
                    <NButton quaternary type="info" size="tiny" @click="navToTagGroup()">标签组管理</NButton>
                  </div>
                  <div class="r w-35% flex items-center justify-end">
                    <NButton quaternary type="info" size="tiny" @click="handleDefine()">确定</NButton>
                    <NButton quaternary size="tiny" @click="handleCancel()">取消</NButton>
                  </div>
                </div>
              </template>
            </NSelect>
          </div>
          <div v-else-if="tagConfig.sign === 'tag'" class="w-full flex items-center px-16px box-border">
            <n-input-group class="item_iptGroup w-full flex justify-start items-center gap-8px">
              <div class="flex items-center">
                <NColorPicker v-model:value="tagConfig.params.color" :show-alpha="false" class="custom-color-picker"
                  :actions="['confirm']" />
              </div>
              <!-- <NInput v-model:value="tagConfig.params.val" class="w-full" placeholder="请输入标签名" /> -->
              <div class="w-60% flex items-center gap-8px">
                <NInput v-model:value="tagConfig.params.val" class="w-full" placeholder="请输入标签名" />
                <NInput v-model:value="tagConfig.params.englishVal" class="w-full" placeholder="请输入标签名(英文名)" />
              </div>
              <div class="ml-4px w-[16%] flex items-center">
                <NButton quaternary type="info" size="tiny" @click="handleDefine()">确定</NButton>
                <NButton quaternary size="tiny" @click="handleCancel()">取消</NButton>
              </div>
            </n-input-group>
          </div>
          <div v-else-if="tagConfig.sign === 'single'" class="w-full flex items-center px-16px box-border">
            <NCascader v-model:value="singleIds" multiple clearable placeholder="选择指定标签组下的标签（*可多选）" :max-tag-count="5"
              :expand-trigger="hoverTrigger ? 'hover' : 'click'" :options="singleTagOptions" cascade="true"
              check-strategy="child" show-path="true" filterable="false"
              :clear-filter-after-select="clearFilterAfterSelect">
              <template #action>
                <div class="ml-4px w-[16%] flex items-center">
                  <NButton quaternary type="primary" size="small" @click="handleDefine()">确定</NButton>
                  <NButton quaternary size="small" @click="handleCancel()">取消</NButton>
                </div>
              </template>
            </NCascader>
          </div>
        </div>
      </div>

      <!-- 标签列表 -->
      <div v-if="tagConfig.tagList.length !== 0" class="box-border w-full h-[360px] overflow-y-auto px-14px">
        <div v-for="(item, index) of tagConfig.tagList" :key="index"
          class="mb-8px box-border h-40px w-full flex justify-around items-center overflow-hidden border-1 border-[#eee] rounded-[4px]"
          @click="handleTagActive(index)">
          <div v-show="item.isOperate" class="default w-full h-full flex items-center box-border px-6px"
            @mouseenter="handleTagMoute('enter', item)" @mouseleave="handleTagMoute('leave', item)">
            <n-input-group class="flex justify-start items-center gap-8px">
              <NColorPicker v-model:value="item.color" :show-alpha="false" class="custom-color-picker" disabled
                :actions="['confirm']" />
              <div class="w-[80%]">
                <span>{{ item.name }}</span>
                <span class="ml-4px">(ID: {{ item.labelId }})</span>
              </div>
              <div class="flex justify-center items-center gap-8px" v-show="item.isHover">
                <div @click="handleTagOperate('edit', item)">
                  <SvgIcon local-icon="lucide--edit" class="text-[16px]"></SvgIcon>
                </div>
                <div @click="handleTagOperate('delete', item)">
                  <SvgIcon icon="material-symbols-light:delete-outline" class="text-[20px]"></SvgIcon>
                </div>
              </div>
            </n-input-group>
          </div>
          <div v-show="!item.isOperate" class="w-full h-full flex items-center box-border px-6px">
            <n-input-group class="flex justify-start items-center gap-8px">
              <NColorPicker v-model:value="item.color" :show-alpha="false" class="custom-color-picker"
                :actions="['confirm']" />
              <!-- <NInput :style="{ width: '80%' }" v-model:value="item.name" type="text" placeholder=""
                class="border-none outline-none" /> -->
              <div class="flex-1 flex items-center gap-8px">
                <NInput v-model:value="item.twoLabelName" type="text" placeholder="请输入标签名"
                  class="border-none outline-none" />
                <NInput v-model:value="item.englishLabelName" type="text" placeholder="请输入标签英文名"
                  class="border-none outline-none" />
              </div>
              <div class="w-[16%] flex justify-center items-center">
                <NButton quaternary type="info" size="tiny" @click="handleTagOperate('confirm', item)">确定</NButton>
                <NButton quaternary type="default" size="tiny" @click="handleTagOperate('cancel', item)">取消</NButton>
              </div>
            </n-input-group>
          </div>
        </div>
      </div>
      <div v-else class="box-border h-[75%] w-full flex flex-col items-center justify-center px-24px py-24px">
        <img :src="noTag" alt="">
        <div class="mt-24px text-[14px] text-[#666]">暂无可用标签 ，请点击上方按钮添加！</div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed } from 'vue';
import {
  NButton,
  NSelect,
  NColorPicker,
  NInput,
  NCascader
} from 'naive-ui';
import noTag from "@/assets/imgs/noTag.png";
import {
  addDataSetAndLabel,
  addSaveLabel,
  deleteDataSetLabel,
  getSelectGroupLabel,
  selectDataSetLabel,
  selectLabelList,
  bindLabel
} from "@/service/api/ano";
import { fetchLabelEdit } from '@/service/api/tag';

interface TagItem {
  name: string;
  labelId: number;
  color: string;
  isCheck: boolean;
  isHover: boolean;
  isOperate: boolean;
  [key: string]: any
}

const emit = defineEmits(['update:visible', 'define', 'cancel']);

const router = useRouter();
const route = useRoute();

// newCode
const sonId = defineModel<boolean>('sonId', {
  default: null
});
const isEditTag = ref(false);
const tagConfig = ref<any>({
  isEditTag: false,
  sign: "group",
  params: {
    val: undefined,
    englishVal: undefined,
    color: "#000000",
  },
  options: [],
  isShow: false,
  tagList: [],
  deepTagList: [],
});

// single
const singleIds = ref<any>([]);
const showPath = ref<Boolean>(true);
const cascade = ref<Boolean>(true);
const responsiveMaxTagCount = ref<Boolean>(false);
const filterable = ref<Boolean>(false);
const hoverTrigger = ref<Boolean>(false);
const clearFilterAfterSelect = ref<Boolean>(true);
const singleTagOptions = ref<any>([]);

watch(() => sonId.value, async () => {
  if (sonId.value) {
    // 清空
    tagConfig.value.tagList = [];
    tagConfig.value.deepTagList = [];
    await nextTick();
    await getTagList();
    await getGroupList();
  }
}, {
  immediate: true
});

const getTagList = async () => {
  const res = await selectDataSetLabel({ sonId: sonId.value });
  const dataList = res.data.map((item, index) => {
    return {
      name: item.labelName,
      color: item.labelColor,
      isOperate: true,
      isCheck: false,
      tagIdx: index,
      isHover: false,
      count: item.labelCount,
      labelId: item.labelId,
      ...item,
    };
  });
  tagConfig.value.tagList = dataList;
  tagConfig.value.deepTagList = dataList;
};

const getGroupList = async () => {
  const params = {
    sonId: sonId.value,
  };
  const res = await getSelectGroupLabel(params);
  if (res.data) {
    tagConfig.value.options = res.data.map((item) => {
      return {
        label: item.label,
        value: item.id,
        count: item.count,
      };
    });
  }
};

async function handleDefine() {
  if (tagConfig.value.sign === "group") {
    const params = {
      sonId: sonId.value,
      labelGroupId: tagConfig.value.params.val,
    };
    const res = await addDataSetAndLabel(params);
    if (res.data >= 1) {
      window.$message?.success?.("标签组切换成功！");
      tagConfig.value.isShow = false;
      tagConfig.value.params.val = null;
      await getTagList();
    }
  }
  if (tagConfig.value.sign === "tag") {
    if (!tagConfig.value.params.englishVal) {
      window.$message?.error?.('标签组英文名必填');
      return;
    }
    const regex = /^[a-zA-Z0-9_\-\/]+$/;
    if (tagConfig.value.params.englishVal && !regex.test(tagConfig.value.params.englishVal)) {
      window.$message?.error?.('请输入符合格式要求的英文名，仅允许包含字母、数字、下划线、连字符和斜杠。');
      return;
    }
    const params = {
      sonId: sonId.value,
      labelColor: tagConfig.value.params.color,
      labelName: tagConfig.value.params.val,
      englishLabelName: tagConfig.value.params.englishVal,
    };
    const res = await addSaveLabel(params);
    if (res.data >= 1) {
      window.$message?.success?.("添加标签成功！");
      isEditTag.value = false;
      await getTagList();
    }
  }

  if (tagConfig.value.sign === "single") {
    if (singleIds.value.length === 0) {
      window.$message?.warning?.('请选择标签');
      return;
    }

    const params = {
      sonId: sonId.value,
      labelIds: singleIds.value.join(','),
    };
    const res = await bindLabel(params);
    if (res.data >= 1) {
      window.$message?.success?.("绑定标签成功！");
      isEditTag.value = false;
      await getTagList();
    }
  }
}

function handleCancel() {
  isEditTag.value = false;
}

function handleSelectChange(value: string) {
  tagConfig.value.isShow = true;
}

function handleSelectFocus() {
  tagConfig.value.params.val = null;
  tagConfig.value.isShow = true;
}

function handleAddTag(type: string = 'group') {
  if (type === 'tag') {
    tagConfig.value.params.val = '';
    tagConfig.value.params.englishVal = '';
  }
  tagConfig.value.sign = type;
  isEditTag.value = true;
}

async function handleTagOperate(sign: string, row: TagItem) {
  if (sign === "delete") {
    deleteDataSetLabel({
      sonId: sonId.value,
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
    if (!row.englishLabelName) {
      window.$message?.error?.('标签组英文名必填');
      return;
    }
    const regex = /^[a-zA-Z0-9_\-\/]+$/;
    if (row.englishLabelName && !regex.test(row.englishLabelName)) {
      window.$message?.error?.('请输入符合格式要求的英文名，仅允许包含字母、数字、下划线、连字符和斜杠。');
      return;
    }
    const res = await fetchLabelEdit({
      id: labelId,
      labelColor: color,
      labelGroupId,
      labelName: row.twoLabelName,
      englishLabelName: row.englishLabelName,
      sonId: sonId.value,
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
}

function handleTagActive(index: number) {
  tagConfig.value.activeIdx = index;
  tagConfig.value.tagList = tagConfig.value.tagList.map((item) => {
    item.isCheck = false;
    return item;
  });
  tagConfig.value.tagList[index].isCheck = true;
}

function handleTagMoute(sign: string, row: TagItem) {
  if (sign === "enter") {
    row.isHover = true;
  }
  if (sign === "leave") {
    row.isHover = false;
  }
}

function navToTagGroup() {
  router.push({
    name: "dataset_taggroupmanager",
  });
}

const validateEnglishVal = () => {
  const regex = /^[a-zA-Z0-9_\-\/]+$/;
  if (!regex.test(tagConfig.value.params.englishVal)) {
    window.$message?.error?.('请输入符合格式要求的英文名，仅允许包含字母、数字、下划线、连字符和斜杠。');
  }
};

function transformData(data: any[], type: string = 'group'): LabelValueChildrenItem[] {
  return data.map((group) => ({
    label: group.labelGroupName,
    value: group.id.toString(), // 确保 value 是字符串类型
    disabled: false, // 禁用状态
    children:
      group.list.length > 0
        ? group.list.map((item) => ({
          label: item.labelName,
          value: type === 'group' ? `${group.id}-${item.id}` : item.id.toString(), // 组合父级和子级的 id
          color: item.labelColor, // 添加颜色属性
          disabled: type === 'group' ? true : false,
        }))
        : [],
  }));
}

async function getTagGroupList() {
  const res = await selectLabelList();
  let dataList: any;
  let singleTagList: any;
  if (res.data) {
    singleTagList = transformData(res.data, 'single');
  } else {
    dataList = [];
    singleTagList = [];
  }
  singleTagOptions.value = [...singleTagList];
}

onMounted(() => {
  getTagGroupList()
})

</script>

<style scoped lang="scss">
/* 添加样式 */
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
</style>
