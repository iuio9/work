<template>
  <div class="w-full h-full relative flex items-center gap-24px px-18px box-border">
    <button @click="prevPage" :disabled="currentPage === 1" class="p-2 hover:bg-gray-200 rounded-l font-bold">
      ←
    </button>
    <div class="grid grid-cols-8 gap-4 h-92% flex-1 min-w-0">
      <div v-for="(img, index) in displayedImages" :key="img.fileId"
        :style="{ 'border': isImageSelected(img) ? '2px solid #2468f2' : '' }"
        class="w-full cursor-pointer py-4px box-border border-2 border-gray-3000 flex justify-center items-center box-border"
        @click="handleImageClick(img)">
        <img :src="img.previewImgPath" alt="" class="h-92% object-contain border-2 border-gray-300 border-transparent">
      </div>
    </div>
    <button @click="nextPage" :disabled="currentPage * 8 >= total" class="p-2 hover:bg-gray-200 rounded-r">
      →
    </button>
  </div>
</template>

<script setup lang="ts">
defineOptions({
  name: "ImageSelector"
})

import { ref, computed, watchEffect, defineProps, defineEmits } from 'vue';

interface ImageItem {
  id: null | number;
  fileId: string;
  version: number;
  markFileId: null | number;
  imgPath: string;
  previewImgPath: string;
  isMark: string;
  labels: string;
  markInfo: null;
  labelMarkInfo: null;
  width: number;
  height: number;
  operateWidth: null | number;
  operateHeight: null | number;
  fileName: null | string;
  notPassMessage: null | string;
}

const props: any = defineProps({
  selectedIds: {
    type: Array as () => string[] | number[],
    default: () => []
  },
  total: {
    type: Number,
    required: true
  }
});

const emit = defineEmits(['update:selectedIds', "update:page"]);

const currentPage = defineModel<number>('currentPage', { required: true, default: 1 });
const imageList = defineModel<any>('imageList', { required: true, default: [] });

const selectedImages = ref(new Set<string>(props.selectedIds));

// 监听props变化
watchEffect(() => {
  selectedImages.value = new Set(props.selectedIds);
});

// 计算当前页显示的图片
const displayedImages = computed(() => {
  return imageList.value;
});

// 上一页
const prevPage = () => {
  if (currentPage.value > 1) {
    currentPage.value--;
    emit('update:page', currentPage.value);
  }
};

// 下一页
const nextPage = () => {
  if (currentPage.value * 8 < props.total) {
    currentPage.value++;
    emit('update:page', currentPage.value);
  }
};

// 处理图片点击
const handleImageClick = (img: ImageItem) => {
  const isSelected = selectedImages.value.has(img.fileId);
  selectedImages.value.clear();
  if (!isSelected) {
    selectedImages.value.add(img.fileId);
  }
  emit('update:selectedIds', Array.from(selectedImages.value));
};

// 判断是否选中
const isImageSelected = (img: ImageItem) => selectedImages.value.has(img.fileId);
</script>

<style scoped>
/* 可添加自定义样式 */
</style>
