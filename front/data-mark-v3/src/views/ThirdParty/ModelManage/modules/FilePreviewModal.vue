<script setup lang="ts">
import { ref, shallowRef } from 'vue';
import VuePdfEmbed from 'vue-pdf-embed';
import { useLoading } from '@sa/hooks';

//引入VueOfficeDocx组件
import VueOfficeExcel from'@vue-office/excel/lib/v3/vue-office-excel.mjs'
import '@vue-office/excel/lib/v3/index.css'

defineOptions({
  name: 'FilePreviewModal'
});

const visible = defineModel<boolean>('visible', {
  default: false
});

const previewPath = defineModel<string>('previewPath', {
  default: ''
});

const previewFileSuffix = defineModel<string>('previewFileSuffix', {
  default: ''
});

const { loading, endLoading } = useLoading(true);

const pdfRef = shallowRef<InstanceType<typeof VuePdfEmbed> | null>(null);
const source = `https://xiaoxian521.github.io/hyperlink/pdf/Cookie%E5%92%8CSession%E5%8C%BA%E5%88%AB%E7%94%A8%E6%B3%95.pdf`;

const showAllPages = ref(false);
const currentPage = ref<undefined | number>(1);
const pageCount = ref(1);

function onPdfRendered() {
  endLoading();

  if (pdfRef.value?.doc) {
    pageCount.value = pdfRef.value.doc.numPages;
  }
}

function showAllPagesChange() {
  currentPage.value = showAllPages.value ? undefined : 1;
}

const rotations = [0, 90, 180, 270];
const currentRotation = ref(0);

function handleRotate() {
  currentRotation.value = (currentRotation.value + 1) % 4;
}

async function handlePrint() {
  await pdfRef.value?.print(undefined, 'test.pdf', true);
}

async function handleDownload() {
  await pdfRef.value?.download('test.pdf');
}

function closeModal() {
  visible.value = false;
}

const getFileExtension = (url: string): string => {
  try {
    const pathname = new URL(url).pathname;
    const basename = pathname.split('/').pop() || '';
    const lastDotIndex = basename.lastIndexOf('.');

    return lastDotIndex > 0
      ? basename.substring(lastDotIndex + 1)
      : '';
  } catch (error) {
    console.error('Invalid URL:', error);
    return '';
  }
};

// function determineFileType(filePath: string, fileSuffix: string): string {
//   const extension = getFileExtension(filePath).toLowerCase();
//   const imageExtensions = ['jpg', 'jpeg', 'png', 'gif', 'bmp'];
//   const documentExtensions = ['pdf', 'docx', 'doc'];

//   if (imageExtensions.includes(extension)) {
//     return 'image';
//   } else if (documentExtensions.includes(extension)) {
//     return 'pdf';
//   }
//   return '其他';
// }

function determineFileType(filePath: string, fileSuffix: string = previewFileSuffix.value): string {
  // 获取文件扩展名（优先使用fileSuffix，如果为空则从filePath提取）
  const extension = (fileSuffix || getFileExtension(filePath)).toLowerCase();

  // 定义支持的文件类型及其扩展名
  const SUPPORTED_TYPES = {
    image: ['jpg', 'jpeg', 'png', 'gif', 'bmp', 'webp', 'svg'],
    pdf: ['pdf'],
    excel: ['xlsx', 'xls', 'csv'],
    word: ['docx'],
    audio: ['mp3', 'wav', 'ogg', 'flac'],
    video: ['mp4', 'webm', 'ogg', 'mov'],
    code: [
      'js', 'javascript', 'ts', 'typescript', 'vue', 'html', 'css',
      'scss', 'sass', 'less', 'json', 'xml', 'java', 'python',
      'php', 'c', 'cpp', 'cs', 'go', 'ruby', 'sql', 'bash', 'sh'
    ]
  };

  // 遍历SUPPORTED_TYPES查找匹配的类型
  for (const [type, extensions] of Object.entries(SUPPORTED_TYPES)) {
    if (extensions.includes(extension)) {
      return type;
    }
  }

  // 如果没有匹配的类型，返回'其他'
  return '其他';
}
</script>

<template>
  <n-modal v-model:show="visible" preset="card" title="文件查看" class="w-800px">
    <!-- pdf预览 -->
    <div class="h-full flex-col-stretch" v-if="determineFileType(previewPath) === 'pdf'">
      <div class="flex-y-center justify-end gap-12px">
        <NCheckbox v-model:checked="showAllPages" @update:checked="showAllPagesChange">显示所有页面</NCheckbox>
        <ButtonIcon tooltip-content="旋转90度" @click="handleRotate" class="text-24px">
          <icon-material-symbols-light:rotate-90-degrees-ccw-outline-rounded />
        </ButtonIcon>
        <ButtonIcon tooltip-content="打印" @click="handlePrint" class="text-24px">
          <icon-mdi:printer />
        </ButtonIcon>
        <ButtonIcon tooltip-content="下载" @click="handleDownload" class="text-24px">
          <icon-charm:download />
        </ButtonIcon>
      </div>
      <NScrollbar class="flex-1-hidden">
        <NSkeleton v-if="loading" size="small" class="mt-12px" text :repeat="12" />
        <VuePdfEmbed ref="pdfRef" class="overflow-auto container" :class="{ 'h-0': loading }"
          :rotation="rotations[currentRotation]" :page="currentPage" :source="previewPath" @rendered="onPdfRendered" />
      </NScrollbar>
      <div class="flex-y-center justify-between">
        <div v-if="showAllPages" class="text-18px font-medium">共{{ pageCount }}页</div>
        <NPagination v-else v-model:page="currentPage" :page-count="pageCount" :page-size="1" />
      </div>
    </div>
    <!-- 图片预览 -->
    <img :src="previewPath" alt="" class="w-full h-full object-cover"
      v-else-if="determineFileType(previewPath) === 'image'" />
    <!-- excel预览 -->
    <div class="h-full flex-col-stretch" v-else-if="determineFileType(previewPath) === 'excel'">
      <vue-office-excel :src="previewPath" style="height: 600px; width: 100%;" />
    </div>
    <template #footer>
      <n-space justify="end" space="[16px]">
        <n-button @click="closeModal" class="bg-gray-200 text-gray-800 hover:bg-gray-300">取消</n-button>
      </n-space>
    </template>
  </n-modal>
</template>

<style scoped></style>
