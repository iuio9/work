<script setup lang="ts">
import type { UploadFile, UploadFiles, UploadInstance, UploadProps } from 'element-plus';
import { UploadRawFile } from 'element-plus';
import { ArchiveOutline as ArchiveIcon } from '@vicons/ionicons5';
import axios from 'axios';
import { NSpin } from 'naive-ui';
import _ from 'lodash';
import { $t } from '@/locales';
import { temDownload } from '@/service/api/dataManage';
// import path from 'path';

defineOptions({
  name: 'UploadOperateDrawer'
});

interface Emits {
  (e: 'submitted'): void;
}

const emit = defineEmits<Emits>();

// data
const isImgImport = ref<Boolean>(false);
const uploaderFiles = ref<any>([]); // 图片列表
const fileGroup = ref<any>({
  // imgList: [],
  // infoList: [],
}); // json || xml列表
const uploadRef = ref<UploadInstance>();
const visible = defineModel<boolean>('visible', {
  default: false
});

// 标注状态
const markStatus = defineModel<string>('markStatus', {
  default: '0'
});
const uploadedCount = ref<Number>(0);
const totalFilesCount = ref<Number>(0);
const loading = ref(false);
const temList = ref<any>([
  { name: 'json模板', key: 'json' },
  { name: 'xml模板', key: 'xml' }
]); // json || xml

let worker = null;

// methods
function closeDrawer() {
  isImgImport.value = false;
  visible.value = false;
}

const startProcessingFile = (uploadFiles: UploadFiles) => {
  // lodash深拷贝
  const files = _.cloneDeep(uploadFiles);
  worker = new Worker(new URL('./fileHandler.worker.ts', import.meta.url));
  worker.onmessage = ({ data }) => {
    uploaderFiles.value = data.uploaderFiles;
  };
  worker.postMessage({
    files,
    markStatus: markStatus.value
  }); // 每片1MB
};

const fileOnChange = (_uploadFile: UploadFile, uploadFiles: UploadFiles) => {
  // test
  // startProcessingFile(uploadFiles);

  uploaderFiles.value = _.uniqBy(uploadFiles, 'name').map((item: any, index) => {
    const postfix = item.name.split('.')[1].toLowerCase();
    item.sortIdx = index + 1;
    item.isHover = false;
    item.errMessage = '暂未上传，请点击下方确认按钮开始上传！';
    if (markStatus.value === '0') {
      item.type = `image/${postfix}`;
      fileValidation(item);
    }
    if (markStatus.value === '1') {
      item.lastModified = item.raw.lastModified;
      item.type = item.name.split('.')[1].toLowerCase();
      item.prevfix = item.name.split('.')[0];
      fileValidation1(item);
    }
    return item;
  });
  totalFilesCount.value = uploaderFiles.value.length;
};

const upload = (command: string) => {
  if (command === 'image') {
    uploadRef.value.$el.querySelector('input').value = '';
    uploadRef.value.$el.querySelector('input').click();
  }
};

const handleMouseEnter = (options: any) => {
  for (let i = 0; i < uploaderFiles.value.length; i++) {
    const item = uploaderFiles.value[i];
    if (item.uid === options.uid) {
      uploaderFiles.value[i].isHover = true;
      return false;
    }
  }
};
const handleMouseLeave = (options: any) => {
  for (let i = 0; i < uploaderFiles.value.length; i++) {
    const item = uploaderFiles.value[i];
    if (item.uid === options.uid) {
      uploaderFiles.value[i].isHover = false;
      return false;
    }
  }
};

const clearFiles = () => {
  isImgImport.value = false;
  uploaderFiles.value = [];
  // uploadRef.value!.clearFiles();
  emit('submitted', {
    fileList: []
  });
};

// 文件超出限制触发
const handleExceed: UploadProps['onExceed'] = () => {};

const handleSuccess: UploadProps['onSuccess'] = (res, file) => {
  file.status = 'success';
};

const removeFile = (row: any) => {
  uploaderFiles.value.splice(
    uploaderFiles.value.findIndex(item => item.uid === row.uid),
    1
  );
  const fileList =
    markStatus.value === '0'
      ? uploaderFiles.value.filter((val: { status: string }) => val.status === 'finished')
      : uploaderFiles.value.filter((val: { status: string }) => val.status === 'finished' && isImg(val));
  emit('submitted', {
    fileList
  });
};

const getFileGroupKeys = (uploaderFiles: any) => {
  const fileGroup = _.groupBy(_.uniqBy(uploaderFiles, 'name'), 'prevfix');
  const mapKeys = Object.entries(fileGroup).map(item => {
    return {
      key: item[0],
      // 一对一
      list:
        item[1].length == 2
          ? item[1].sort((a, b) => a.lastModified - b.lastModified)
          : [
              isImg(item[1][0])
                ? item[1][0]
                : {
                    status: 'error',
                    name: '请上传与标注信息相关联的图片'
                  },
              isInfo(item[1][0])
                ? item[1][0]
                : {
                    status: 'error',
                    name: '请上传与图片相关联的标注文件'
                  }
            ],
      // 一对多
      mapping: {
        imgList: item[1].filter(val1 => isImg(val1)),
        infoList: item[1].filter(val1 => isInfo(val1))
      }
    };
  });
  console.log('mapKeys');
  console.log(mapKeys);

  return mapKeys;
};

const isImg = (row: any) => {
  const postfix = row.name.split('.')[1].toLowerCase();
  return ['jpg', 'jpeg', 'png', 'webp', 'bmp'].includes(postfix);
};

const isInfo = (row: any) => {
  const postfix = row.name.split('.')[1].toLowerCase();
  return ['json', 'xml'].includes(postfix);
};

function asyncApiCall(val: any) {
  const uploadUrl = `${import.meta.env.VITE_SERVICE_BASE_URL}/temp/anyUpload`;
  const formData = new FormData();
  formData.append('file', val.raw);
  uploadedCount.value++;
  return new Promise((resolve, reject) => {
    axios
      .post(uploadUrl, formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      })
      .then(res => {
        if (res.data.code == 500) {
          const index = uploaderFiles.value.findIndex((item: any) => item.uid === val.uid);
          if (index !== -1) {
            uploaderFiles.value[index].status = 'error';
            uploaderFiles.value[index].errMessage = res.data.data;
          }
          resolve(res.data.data);
        }
        if (res.data.code == 200) {
          // 根据val.id查找model.value.fileList的索引
          const index = uploaderFiles.value.findIndex((item: any) => item.uid === val.uid);
          if (index !== -1) {
            uploaderFiles.value[index].status = 'finished';
            uploaderFiles.value[index].id = res.data.data[0].id;
            uploaderFiles.value[index].path = res.data.data[0].path;
            uploaderFiles.value[index].errMessage = '图片上传成功！';
          }
          resolve(res.data.data[0]);
        }
      })
      .catch(error => {
        console.log('文件上传失败', error);
        const index = uploaderFiles.value.findIndex((item: any) => item.uid === val.uid);
        if (index !== -1) {
          uploaderFiles.value[index].status = 'error';
        }
        resolve(error);
      });
  });
}

function* concurrentApiCallGenerator(urls: any, maxConcurrency = 5) {
  const chunks = [];
  for (let i = 0; i < urls.length; i += maxConcurrency) {
    chunks.push(urls.slice(i, i + maxConcurrency));
  }

  for (const chunk of chunks) {
    const promises = chunk.map((val: any) => asyncApiCall(val));
    yield Promise.allSettled(promises);
  }
}

async function run(generator: any) {
  let result: any[] = [];
  for await (const chunk of generator) {
    result = result.concat(chunk);
  }
}

const beforeUpload = (file: any) => {
  const allowedExtensions = ['image/jpg', 'image/jpeg', 'image/png', 'image/webp', 'image/bmp'];
  const isImage = file.type.startsWith('image/');
  if (!isImage) {
    console.error('只能上传图片');
    return false;
  }
  if (!allowedExtensions.includes(file.type)) {
    console.error('只能上传jpg、jpeg、png、webp或bmp格式的图片');
    return false;
  }
  const MAX_FILE_SIZE = 20 * 1024 * 1024; // 20MB
  if (file.size > MAX_FILE_SIZE) {
    console.error('文件大小不能超过 20MB');
    return false;
  }
  return false;
};

// file validation
const fileValidation = (file: any) => {
  const allowedExtensions = ['image/jpg', 'image/jpeg', 'image/png', 'image/webp', 'image/bmp'];
  const isImage = file.type.startsWith('image/');
  if (!isImage) {
    file.status = 'error';
    file.errMessage = '只能上传图片';
    console.error('只能上传图片');
    return false;
  }
  if (!allowedExtensions.includes(file.type)) {
    file.status = 'error';
    file.errMessage = '只能上传jpg、jpeg、png、webp或bmp格式的图片';
    console.error('只能上传jpg、jpeg、png、webp或bmp格式的图片');
    return false;
  }
  const MAX_FILE_SIZE = 20 * 1024 * 1024; // 20MB
  if (file.size > MAX_FILE_SIZE) {
    file.status = 'error';
    file.errMessage = '文件大小不能超过 20MB';
    console.error('文件大小不能超过 20MB');
    return false;
  }
};

// 图片 + 标注信息校验
const fileValidation1 = (file: any) => {
  const allowedExtensions = ['jpg', 'jpeg', 'png', 'webp', 'bmp', 'json', 'xml'];
  const isBool = allowedExtensions.includes(file.type);
  if (!isBool) {
    file.status = 'error';
    file.errMessage = '只能上传json、xml、jpg、jpeg、png、webp或bmp格式的文件';
    return false;
  }
};

async function handleSubmit() {
  loading.value = true;
  // 并发控制
  const uploaderFileList = uploaderFiles.value.filter(val => {
    return val.status !== 'finished' && val.status !== 'error';
  });
  const generator = concurrentApiCallGenerator(uploaderFileList, 50);
  await run(generator);
  loading.value = false;
  isImgImport.value = false;
  // if (uploadedCount.value === totalFilesCount.value) {
  //   console.log('所有文件都上传完成了');
  // }
  // window.$message?.success(props.operateType === 'add' ? "新增成功" : $t('common.updateSuccess'));
  // closeDrawer();
  const fileList =
    markStatus.value === '0'
      ? uploaderFiles.value.filter((val: { status: string }) => val.status === 'finished')
      : uploaderFiles.value.filter((val: { status: string }) => val.status === 'finished' && isImg(val));
  emit('submitted', {
    fileList
  });
}

// resetUploadFiles
function resetFiles() {
  uploaderFiles.value = [];
  uploadedCount.value = 0;
}

// 模板下载
async function handleTemDownload(row: any) {
  const { key } = row;
  const res = await temDownload({ sign: key });
  if (res.data) {
    const url = window.URL.createObjectURL(new Blob([res.data]));
    const link = document.createElement('a');
    link.href = url;
    link.setAttribute('download', '模板文件.zip');
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    window.URL.revokeObjectURL(url);
  }
}

/*
  分片上传 start
*/
const CHUNK_SIZE = 10 * 1024 * 1024; // 1MB

const uploadChunk = async (file, chunk, chunkIndex, totalChunks) => {
  const formData = new FormData();
  formData.append('file', chunk);
  formData.append('fileName', file.name);
  formData.append('chunkIndex', chunkIndex);
  formData.append('totalChunks', totalChunks);

  try {
    const response = await fetch('/upload-chunk', {
      method: 'POST',
      body: formData
    });
    return response.ok;
  } catch (error) {
    console.error('Chunk upload failed:', error);
    return false;
  }
};

const uploadFileInChunks = async file => {
  const totalChunks = Math.ceil(file.size / CHUNK_SIZE);
  for (let chunkIndex = 0; chunkIndex < totalChunks; chunkIndex++) {
    const start = chunkIndex * CHUNK_SIZE;
    const end = Math.min(start + CHUNK_SIZE, file.size);
    const chunk = file.slice(start, end);

    const success = await uploadChunk(file, chunk, chunkIndex, totalChunks);
    if (!success) {
      console.error('Failed to upload chunk', chunkIndex);
      return;
    }
  }
  console.log('File upload complete');
};

const handleFileUpload = async files => {
  for (const file of files) {
    await uploadFileInChunks(file);
  }
};

const startUpload = () => {
  const files = uploaderFiles.value.map(item => item.raw);
  handleFileUpload(files);
};
// ---------------------------------------- end --------------------------------------------

defineExpose({
  resetFiles
});
</script>

<template>
  <NDrawer v-model:show="visible" display-directive="show" :width="markStatus === '0' ? '540' : '740'">
    <NDrawerContent title="上传" :native-scrollbar="false" closable>
      <div class="h-full w-full flex flex-col items-start justify-start">
        <div class="h-auto w-full flex items-center justify-between">
          <div class="flex items-center justify-start">
            <NButton type="primary" @click="upload('image')">
              <span v-if="markStatus === '0'">上传图片</span>
              <span v-else>上传图片 + 标注信息</span>
            </NButton>
          </div>
          <div class="flex items-center justify-start">
            <NButton @click="clearFiles">清空列表</NButton>
          </div>
        </div>
        <div class="wrap-upload mt-24px w-full">
          <ElUpload
            ref="uploadRef"
            v-model:file-list="uploaderFiles"
            drag
            action="#"
            :auto-upload="false"
            :on-change="fileOnChange"
            :on-exceed="handleExceed"
            :on-success="handleSuccess"
            :before-upload="beforeUpload"
            :show-file-list="false"
            multiple
          >
            <div class="">
              <div style="margin-bottom: 12px">
                <NIcon size="48" :depth="3">
                  <ArchiveIcon />
                </NIcon>
              </div>
              <NText style="font-size: 16px">点击或者拖动文件到该区域来上传</NText>
              <NP depth="3" style="margin: 8px 0 0 0">
                1. 图片类型为jpg/png/bmp/jpeg/webp
                <br />
              </NP>
              <NP depth="3" style="margin: 8px 0 0 0">2. 图片大小限制在20M内</NP>
              <NP depth="3" style="margin: 8px 0 0 0">3. 标注信息文件后缀支持格式 json/xml</NP>
              <!--
 <n-p depth="3" style="margin: 8px 0 0 0" class="flex justify-center" v-if="markStatus === '1'">
                <div class="flex items-center">
                  <span>示例模板下载：</span>
                  <div class="flex items-center">
                    <n-button quaternary type="info" v-for="(item, index) of temList" :key="item.key"
                      @click="handleTemDownload(item)">
                      {{ item.name }}
                    </n-button>
                  </div>
                </div>
              </n-p>
-->
            </div>
          </ElUpload>
        </div>
        <div class="w-full overflow-y-auto !flex-1">
          <div v-show="isImgImport" class="mask-layer">
            <div class="loading-spinner">
              <!-- 这里可以放置任何你想要的加载动画 -->
              <NSpin size="large" description="图片导入中... 请稍等" />
            </div>
          </div>
          <NVirtualList v-if="markStatus === '0' || markStatus === '1'" :item-size="42" :items="uploaderFiles" class="">
            <template #default="{ item }">
              <div class="h-auto w-full flex flex-col items-start justify-start">
                <div
                  :key="item.key"
                  class="item h-44px w-full flex items-center justify-between py-8px hover:bg-[#ebf7ed]"
                  @mouseenter="handleMouseEnter(item)"
                  @mouseleave="handleMouseLeave(item)"
                >
                  <NPopover trigger="hover" placement="left">
                    <template #trigger>
                      <!-- 图片列表 markStatus === 1 -->
                      <div class="w-90% flex items-center justify-start">
                        <span
                          class="ml-4px w-90% truncate"
                          :style="{
                            color: item.status && item.status === 'error' ? 'red' : ''
                          }"
                        >
                          {{ item.name }}
                        </span>
                        <NButton v-show="item.status && item.status === 'finished'" quaternary type="primary">
                          已完成
                        </NButton>
                        <NButton v-show="item.status && item.status === 'error'" quaternary type="error">失败</NButton>
                      </div>
                      <!-- 图片 + info标注信息 markStatus === 2 -->
                    </template>
                    <div
                      v-show="item.errMessage"
                      class=""
                      :style="{
                        color: item.status && item.status === 'error' ? 'red' : ''
                      }"
                    >
                      {{ item.errMessage }}
                    </div>
                  </NPopover>
                  <div v-show="item.isHover" class="mr-14px" @click.stop="removeFile(item)">
                    <SvgIcon local-icon="imgDel" class="text-[24px]"></SvgIcon>
                  </div>
                </div>
              </div>
            </template>
          </NVirtualList>
          <NVirtualList v-else :item-size="42" :items="getFileGroupKeys(uploaderFiles)" class="">
            <template #default="{ item }">
              <div class="h-auto w-full flex flex-col items-start justify-start">
                <div :key="item.key" class="h-auto w-full flex items-center justify-around">
                  <div v-for="(val, idx) of item.list" :key="idx" class="w-50% border-b-2 border-[#eeeeee]">
                    <div
                      :key="val.key"
                      class="item h-44px w-full flex items-center justify-between py-8px hover:bg-[#ebf7ed]"
                      :style="{
                        'border-left': idx == 1 ? '2px solid #eeeeee' : ''
                      }"
                      @mouseenter="handleMouseEnter(val)"
                      @mouseleave="handleMouseLeave(val)"
                    >
                      <NPopover trigger="hover" placement="left">
                        <template #trigger>
                          <!-- 图片列表 markStatus === 0 -->
                          <div class="w-90% flex items-center justify-start">
                            <span
                              class="ml-4px w-90% truncate"
                              :style="{
                                color: val.status && val.status === 'error' ? 'red' : ''
                              }"
                            >
                              {{ val.name }}
                            </span>
                            <NButton v-show="val.status && val.status === 'finished'" quaternary type="primary">
                              已完成
                            </NButton>
                            <NButton v-show="val.status && val.status === 'error'" quaternary type="error">
                              失败
                            </NButton>
                          </div>
                          <!-- 图片 + info标注信息 markStatus === 1 -->
                        </template>
                        <div
                          v-show="val.errMessage"
                          class=""
                          :style="{
                            color: val.status && val.status === 'error' ? 'red' : ''
                          }"
                        >
                          {{ val.errMessage }}
                        </div>
                      </NPopover>
                      <div v-show="val.isHover" class="mr-14px" @click.stop="removeFile(val)">
                        <SvgIcon local-icon="imgDel" class="text-[24px]"></SvgIcon>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </template>
          </NVirtualList>
        </div>
      </div>
      <template #footer>
        <NSpace :size="16">
          <NButton @click="closeDrawer">关闭</NButton>
          <NButton type="primary" :loading="loading" @click="handleSubmit">确认上传</NButton>
        </NSpace>
      </template>
    </NDrawerContent>
  </NDrawer>
</template>

<style scoped lang="scss">
:deep(.n-scrollbar-content) {
  height: 100% !important;
}
</style>
