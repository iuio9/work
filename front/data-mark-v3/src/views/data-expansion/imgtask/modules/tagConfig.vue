<template>
  <div class="w-full h-full border-1 border-[#eee]">
    <!-- Header -->
    <div class="h-full w-full">
      <!-- 标签组选择或自定义标签部分 -->
      <div class="my-16px h-[15%] w-full">
        <div v-if="!isEditTag" class="h-full w-full flex items-center justify-around">
          <div class="flex items-center text-[16px] text-[#000]">标签组选择</div>
          <div class="flex items-center gap-4px">
            <NButton type="primary" class="add" size="small" @click="handleAddTag('tag')">自定义标签</NButton>
            <NButton type="primary" class="add" size="small" @click="handleAddTag()">选择标签组</NButton>
          </div>
        </div>
        <!-- 编辑模式下的内容 -->
        <div v-else class="box-border h-full w-full flex items-center justify-center">
          <!-- 根据sign值显示不同的编辑内容 -->
          <div v-if="tagConfig.sign === 'group'" class="select w-full">
            <NSelect ref="selectRef" v-model:value="tagConfig.params.val" placeholder="请选择标签组"
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
          <div v-else-if="tagConfig.sign === 'tag'" class="select w-[94%] flex items-center">
            <div class="flex items-center">
              <NColorPicker v-model:value="tagConfig.params.color" :show-alpha="false" class="custom-color-picker"
                :actions="['confirm']" />
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

      <!-- 标签列表 -->
      <div v-if="tagConfig.tagList.length !== 0" class="mt-24px box-border h-[360px] w-full overflow-y-auto">
        <div v-for="(item, index) of tagConfig.tagList" :key="index"
          class="mb-8px box-border h-40px w-full flex justify-around items-center overflow-hidden border-1 border-[#eee] rounded-[4px]"
          @click="handleTagActive(index)">
          <div class="ml-24px mr-8px h-full w-10px" :style="{ background: item.color }"></div>
          <div v-show="item.isOperate" class="default h-full w-full flex items-center"
            @mouseenter="handleTagMoute('enter', item)" @mouseleave="handleTagMoute('leave', item)">
            <!-- <div class="mr-8px flex items-center">
              <NCheckbox v-model:checked="item.isCheck" />
            </div> -->
            <div class="w-[80%]">
              <span>{{ item.name }}</span>
              <span class="ml-4px">(ID: {{ item.labelId }})</span>
            </div>
            <div v-show="item.isHover" class="h-full w-[20%] flex items-center gap-[8px]">
              <div @click="handleTagOperate('edit', item)">
                <SvgIcon local-icon="lucide--edit" class="text-[16px]"></SvgIcon>
              </div>
              <div @click="handleTagOperate('delete', item)">
                <SvgIcon icon="material-symbols-light:delete-outline" class="text-[20px]"></SvgIcon>
              </div>
            </div>
          </div>
          <div v-show="!item.isOperate" class="h-full w-full flex items-center">
            <div class="item_ipt_con h-full w-[80%] flex items-center">
              <NInput v-model:value="item.name" type="text" placeholder="" class="border-none outline-none" />
            </div>
            <div class="ml-24px item_ipt_con h-full w-[40%] flex items-center">
              <NButton quaternary type="info" size="tiny" @click="handleTagOperate('confirm', item)">确定</NButton>
              <NButton quaternary type="default" size="tiny" @click="handleTagOperate('cancel', item)">取消</NButton>
            </div>
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
  NModal,
  NCard,
  NButton,
  NSelect,
  NColorPicker,
  NInput,
  NCheckbox,
  SelectOption,
} from 'naive-ui';
import noTag from "@/assets/imgs/noTag.png";
import {
  DataDetailsCount,
  MarkFileUpload,
  addDataMarkInfo,
  addDataSetAndLabel,
  addSaveLabel,
  deleteDataSetLabel,
  getDataDetailsNoMarkFilePath,
  getSelectGroupLabel,
  segmentStart,
  selectDataSetLabel,
} from "@/service/api/ano";

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
const isEditTag = ref(false);
const tagConfig = ref<any>({
  isEditTag: false,
  sign: "group",
  params: {
    val: undefined,
    color: "#000000",
  },
  options: [],
  isShow: false,
  tagList: [
    { name: "标签1", labelId: 1, color: "#FF0000", isCheck: false, isHover: false, isOperate: true },
    { name: "标签2", labelId: 2, color: "#00FF00", isCheck: false, isHover: false, isOperate: false },
    { name: "标签3", labelId: 3, color: "#0000FF", isCheck: false, isHover: false, isOperate: false },
  ],
  deepTagList: [],
});

function handleAddTag(type?: string = 'group') {
  tagConfig.value.sign = type;
  isEditTag.value = true;
}

function handleSelectChange(value: string) {
  tagConfig.value.isShow = true;
}

function handleSelectFocus() {
  // Handle focus event if needed
  tagConfig.value.params.val = null;
  tagConfig.value.isShow = true;
}

async function handleDefine() {
  if (tagConfig.value.sign === "group") {
    const params = {
      sonId: route.query.id,
      labelGroupId: tagConfig.value.params.val,
    };
    const res = await addDataSetAndLabel(params);
    if (res.data >= 1) {
      window.$message?.success?.("标签组切换成功！");
      tagConfig.value.isShow = false;
      tagConfig.value.params.val = null;
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
      isEditTag.value = false;
      await getTagList();
    }
  }
}

function handleCancel() {
  isEditTag.value = false;
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

async function handleTagOperate(sign: string, row: TagItem) {
  if (sign === "delete") {
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
    // fetchLabelEdit
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
      changeLayerText(row);
      await getTagList();
      row.isOperate = !row.isOperate;
    }
  }
  if (sign === "cancel") {
    row.isOperate = !row.isOperate;
  }
}

function navToTagGroup() {
  // Implement navigation to tag group management page
  router.push({
    name: "dataset_taggroupmanager",
  });
}
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
