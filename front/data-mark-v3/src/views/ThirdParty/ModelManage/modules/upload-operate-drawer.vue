<script setup lang="ts">
import type { UploadFile, UploadFiles, UploadInstance, UploadProps } from 'element-plus';
import { UploadRawFile } from 'element-plus';
import { ArchiveOutline as ArchiveIcon } from '@vicons/ionicons5';
import axios from 'axios';
import { NSpin } from 'naive-ui';
import _ from 'lodash';
import SparkMD5 from 'spark-md5';
import { $t } from '@/locales';
import { temDownload, uploadZipChunk } from '@/service/api/dataManage';
import { nanoid } from '~/packages/utils/src';

defineOptions({
  name: 'UploadOperateDrawer'
});

interface Emits {
  (e: 'submitted', data: any): void;
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

const isUpSuccess = defineModel<boolean>('isUpSuccess', {
  default: false
});

// 标注状态
const markStatus = defineModel<string>('markStatus', {
  default: '0'
});

const importMode = defineModel<string>('importMode', {
  default: '0-0'
});
const uploadedCount = ref<Number>(0);
const totalFilesCount = ref<Number>(0);
const loading = ref(false);
const temList = ref<any>([
  { name: 'json模板', key: 'json' },
  { name: 'xml模板', key: 'xml' }
]); // json || xml


const imgToolText = computed(() => {
  // uploaderFiles.value  判断是否为空
  const uploaderFilesLength = uploaderFiles.value.length;
  // const uploadedCountValue = uploadedCount.value;
  const uploadedCountValue = uploaderFiles.value.filter((val: any) => val.status === 'finished').length;

  const totalFilesCountValue = totalFilesCount.value;
  return uploaderFilesLength === 0 ? '本次上次文件数: 0, 已完成传输数: 0' : `本次上次文件数: ${uploaderFilesLength}, 已完成传输数: ${uploadedCountValue}/${totalFilesCountValue}`;
});

const tooltipText = ref<string>(null);

// methods
function closeDrawer() {
  isImgImport.value = false;
  visible.value = false;
}

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

/*
  压缩包上传
  1. 压缩包支持zip/tar.gz/rar格式，压缩前源文件大小限制5G以内
  2. 压缩包内图片格式要求为：图片类型为jpg/png/bmp/jpeg
*/
const fileValidation2 = (file: any) => {
  const allowedExtensions = ['zip'];
  const isBool = allowedExtensions.includes(file.name.split('.')[1].toLowerCase());
  if (!isBool) {
    file.status = 'error';
    file.errMessage = '只能上传zip、tar.gz、rar格式的压缩包';
    return false;
  }
  const MAX_FILE_SIZE = 10 * 1024 * 1024 * 1024; // 4G
  if (file.size > MAX_FILE_SIZE) {
    file.status = 'error';
    file.errMessage = '文件大小不能超过 10G';
    return false;
  }
};

const fileOnChange = (_uploadFile: UploadFile, uploadFiles: UploadFiles) => {
  // 上传图片
  if (importMode.value === '0-0') {
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
  }
  // 上传压缩包
  if (importMode.value === '0-1') {
    // 判断是否上传过该文件 根据_uploadFile.name
    const existFile = uploaderFiles.value.find((item: any) => item.name === _uploadFile.name);
    if (existFile) {
      window.$message?.error?.('该文件已上传，请勿重复上传！');
      return;
    }
    uploaderFiles.value = _.uniqBy(uploadFiles, 'name').map((item: any, index) => {
      item.sortIdx = index + 1;
      item.isHover = false;
      item.percentage = 0;
      item.chunks = Array.from({ length: Math.ceil(item.size / CHUNK_SIZE) }, (_, i) => ({
        index: i,
        uploaded: false
      }));
      item.errMessage = '暂未上传，请点击下方确认按钮开始上传！';
      fileValidation2(item);
      return item;
    });
    totalFilesCount.value = uploaderFiles.value.length;
    const uploaderFilesLength = uploaderFiles.value.length;
    const uploadedCountValue = uploaderFiles.value.filter((val: any) => val.status === 'finished').length;
    const totalFilesCountValue = totalFilesCount.value;
    tooltipText.value = uploaderFilesLength === 0 ? '本次上次文件数: 0, 已完成传输数: 0' : `本次上次文件数: ${uploaderFilesLength}, 已完成传输数: ${uploadedCountValue}/${totalFilesCountValue}`;
  }
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
  emit('submitted', {
    fileList: []
  });
};

// 文件超出限制触发
const handleExceed: UploadProps['onExceed'] = () => {
  window.$message?.error?.('每次最多只能上传2000个文件');
};

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

async function handleSubmit() {
  loading.value = true;
  isUpSuccess.value = true;
  // 并发控制
  const uploaderFileList = uploaderFiles.value.filter(val => {
    return val.status !== 'finished' && val.status !== 'error';
  });
  const generator = concurrentApiCallGenerator(uploaderFileList, 50);
  await run(generator);
  loading.value = false;
  isUpSuccess.value = false;
  isImgImport.value = false;
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
const CHUNK_SIZE = 5 * 1024 * 1024; // 5MB
const fileProgressMap = ref(new Map<File, { loaded: number; total: number }>());

// 初始化文件进度
const initFileProgress = (file: File) => {
  fileProgressMap.value.set(file, { loaded: 0, total: file.size });
};

const calculateFileHash = async (file: File): Promise<string> => {
  return new Promise(resolve => {
    const spark = new SparkMD5.ArrayBuffer();
    const reader = new FileReader();
    reader.readAsArrayBuffer(file);
    reader.onload = () => {
      spark.append(reader.result as ArrayBuffer);
      resolve(spark.end());
    };
  });
};

// 获取文件上传进度百分比
const getPercentage = (file: File) => {
  const progress = fileProgressMap.value.get(file);
  if (progress) {
    return Math.round((progress.loaded / progress.total) * 100).toFixed(0);
  }
  return 0;
};

const uploadChunk = async (
  file: File,
  chunk: Blob,
  chunkIndex: number,
  totalChunks: number,
  fileIdx: number | string
) => {
  const formData = new FormData();
  const fileHash = await calculateFileHash(file);
  const chunkHash = await calculateFileHash(chunk);
  formData.append('file', chunk);
  formData.append('fileName', file.name);
  formData.append('index', chunkIndex);
  formData.append('chunkCount', totalChunks);
  formData.append('fileHash', fileHash);
  formData.append('chunkHash', chunkHash);
  let response;
  const res = await uploadZipChunk(formData);
  if (res.data && !`${res.data}`.includes('successfully')) {
    uploaderFiles.value[fileIdx].id = res.data;
  }
  response = res.data !== 'Failed to upload chunk';
  if (!response) {
    console.error('Failed to upload chunk', chunkIndex);
  } else {
    uploaderFiles.value[fileIdx].chunks[chunkIndex].uploaded = true;
    const file = uploaderFiles.value[fileIdx].raw;
    if (file) {
      const progress = fileProgressMap.value.get(file);
      if (progress) {
        progress.loaded += chunk.size;
      }
    }
  }
};

const uploadFileInChunks = async (file: File, fileIdx: any) => {
  const totalChunks = Math.ceil(file.size / CHUNK_SIZE);
  for (let chunkIndex = 0; chunkIndex < totalChunks; chunkIndex++) {
    const start = chunkIndex * CHUNK_SIZE;
    const end = Math.min(start + CHUNK_SIZE, file.size);
    const chunk = file.slice(start, end);
    const fileIndex = uploaderFiles.value.findIndex((val: any) => val.name === file.name);

    try {
      await uploadChunk(file, chunk, chunkIndex, totalChunks, fileIndex);
    } catch (error) {
      console.error('上传分片失败', error);
    }
  }
  console.log('File upload complete');
  const fileItem = uploaderFiles.value.find((val: any) => val.name === file.name);
  fileItem.status = 'finished';
  fileItem.errMessage = `文件上传完成！`;

  const isAllUploaded = uploaderFiles.value
    .filter(val => val.status !== 'finished')
    .every((val: any) => {
      return getPercentage(val.raw) == 100;
    });
  if (isAllUploaded) {
    loading.value = false;
    isUpSuccess.value = false;
    const fileList = uploaderFiles.value.filter((val: { status: string }) => val.status === 'finished');
    emit('submitted', {
      fileList
    });
  }
};

// 检查已上传的分片
const checkUploadedChunks = async (fileHash: string, fileName: string) => {
  const formData = new FormData();
  formData.append('hash', fileHash);
  formData.append('fileName', fileName);
  try {
    const response = await fetch('/upload/checkChunks', {
      method: 'POST',
      body: formData
    });
    const data = await response.json();
    return data;
  } catch (error) {
    console.error('检查已上传分片出错:', error);
    return [];
  }
};

// 检查文件是否已存在（用于秒传判断）
const checkFileExist = async (fileHash: string) => {
  const formData = new FormData();
  formData.append('hash', fileHash);
  try {
    const response = await fetch('/upload/checkFileExist', {
      method: 'POST',
      body: formData
    });
    const data = await response.json();
    return data.exists;
  } catch (error) {
    console.error('检查文件是否已存在出错:', error);
    return false;
  }
};

// 对文件进行分片
const sliceFile = (file: File, params: any): Blob[] => {
  const chunks: Blob[] = [];
  let start = 0;
  while (start < file.size) {
    const end = Math.min(start + CHUNK_SIZE, file.size);
    const chunk = file.slice(start, end);
    const data = { ...params, chunk, chunkIndex: start / CHUNK_SIZE, file };
    chunks.push(data);
    start = end;
  }
  return chunks;
};

const startUpload = async () => {
  loading.value = true;
  isUpSuccess.value = true;
  const files = uploaderFiles.value.filter(val => val.status !== 'finished').map(item => item.raw);
  files.forEach(async (file: File) => {
    initFileProgress(file);
  });
  await nextTick();
  const uploaderFilesLength = uploaderFiles.value.length;
  const uploadedCountValue = uploaderFiles.value.filter((val: any) => val.status === 'finished').length;
  const totalFilesCountValue = totalFilesCount.value;
  tooltipText.value = uploaderFilesLength === 0 ? '本次上次文件数: 0, 已完成传输数: 0' : `本次上次文件数: ${uploaderFilesLength}, 已完成传输数: ${uploadedCountValue}/${totalFilesCountValue}`;
  const generator = asyncChunkGenerator(files, 3);
  await runChunk(generator);
};

const asyncUploadChunk = async (file: File) => {
  return new Promise(async (resolve, reject) => {
    // initFileProgress(file);
    const fileIdx = uploaderFiles.value.findIndex((val: any) => val.name === file.name);
    const fileHash = nanoid();
    const totalChunks = Math.ceil(file.size / CHUNK_SIZE);
    const chunkList = sliceFile(file, { totalChunks, fileIdx, fileHash });
    const generator = concurrentApiChunkCallGenerator(chunkList, 20);
    await runChunk(generator);
    const fileItem = uploaderFiles.value.find((val: any) => val.name === file.name);
    fileItem.status = 'finished';
    fileItem.errMessage = `文件上传完成！`;
    tooltipText.value = getUploadText();

    const isAllUploaded = uploaderFiles.value
      .filter(val => val.status !== 'finished')
      .every((val: any) => {
        return getPercentage(val.raw) == 100;
      });
    if (isAllUploaded) {
      loading.value = false;
      isUpSuccess.value = false;
      const fileList = uploaderFiles.value.filter((val: { status: string }) => val.status === 'finished');
      emit('submitted', {
        fileList
      });
    }
    resolve(true);
  });
};

function* asyncChunkGenerator(urls: any, maxConcurrency = 5) {
  const chunks = [];
  for (let i = 0; i < urls.length; i += maxConcurrency) {
    chunks.push(urls.slice(i, i + maxConcurrency));
  }

  for (const chunk of chunks) {
    const promises = chunk.map((val: any) => asyncUploadChunk(val));
    yield Promise.allSettled(promises);
  }
}

const asyncApiChunkCall = async (row: any, sign: string) => {
  const uploadUrl = `${import.meta.env.VITE_SERVICE_BASE_URL}/bigFileUpload/uploadChunk`;
  const formData = new FormData();
  formData.append('file', row.chunk);
  formData.append('fileName', row.file.name);
  formData.append('index', row.chunkIndex);
  formData.append('chunkCount', row.totalChunks);
  formData.append('fileHash', row.fileHash);
  formData.append('chunkHash', '00');
  return new Promise((resolve, reject) => {
    if (sign === 'success') {
      resolve({ success: true });
    } else {
      axios
        .post(uploadUrl, formData, {
          headers: {
            'Content-Type': 'multipart/form-data'
          }
        })
        .then(res => {
          if (res.data.data) {
            if (res?.data?.data?.id) {
              uploaderFiles.value[row.fileIdx].id = res.data.data.id;
            }
            uploaderFiles.value[row.fileIdx].chunks[row.chunkIndex].uploaded = true;
            const file = uploaderFiles.value[row.fileIdx].raw;
            if (file) {
              const progress = fileProgressMap.value.get(file);
              if (progress) {
                progress.loaded += row.chunk.size;
              }
            }
            resolve(res.data.data);
          }
        })
        .catch(err => {
          console.error('上传分片出错:', err);
          reject(err);
        });
    }
  });
};

function* concurrentApiChunkCallGenerator(urls: any, maxConcurrency = 5, sign: string) {
  const chunks = [];
  for (let i = 0; i < urls.length; i += maxConcurrency) {
    chunks.push(urls.slice(i, i + maxConcurrency));
  }

  for (const chunk of chunks) {
    const promises = chunk.map((val: any) => asyncApiChunkCall(val, sign));
    yield Promise.allSettled(promises);
  }
}

async function runChunk(generator: any) {
  let result: any[] = [];
  for await (const chunk of generator) {
    result = result.concat(chunk);
  }
}

function getUploadText() {
  const uploaderFilesLength = uploaderFiles.value.length;
  const uploadedCountValue = uploaderFiles.value.filter((val: any) => val.status === 'finished').length;
  const totalFilesCountValue = totalFilesCount.value;
  return uploaderFilesLength === 0 ? '本次上次文件数: 0, 已完成传输数: 0' : `本次上次文件数: ${uploaderFilesLength}, 已完成传输数: ${uploadedCountValue}/${totalFilesCountValue}`;
}
// ---------------------------------------- end --------------------------------------------

defineExpose({
  resetFiles
});
</script>

<template>
  <NDrawer v-model:show="visible" display-directive="show" width="600">
    <NDrawerContent title="上传" :native-scrollbar="false" closable>
      <div class="h-full w-full flex flex-col items-start justify-start">
        <div class="h-auto w-full flex items-center justify-between">
          <div class="flex items-center justify-start">
            <NButton type="primary" ghost @click="upload('image')">
              <template #icon>
                <SvgIcon local-icon="game-icons--click" class="text-[24px]"></SvgIcon>
              </template>
              点击上传
            </NButton>
          </div>
          <div class="flex items-center justify-start">
            <NButton @click="clearFiles">清空列表</NButton>
          </div>
        </div>
        <div class="wrap-upload mt-24px w-full">
          <ElUpload ref="uploadRef" v-model:file-list="uploaderFiles" drag action="#"
            :limit="importMode === '0-0' ? 2000 : 100" :auto-upload="false" :on-change="fileOnChange"
            :on-exceed="handleExceed" :on-success="handleSuccess" :before-upload="beforeUpload" :show-file-list="false"
            multiple>
            <div v-show="importMode === '0-0'" class="">
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
              <NP depth="3" style="margin: 8px 0 0 0">3. 为了保证最佳性能，每次最多可上传2000个文件</NP>
            </div>
            <div v-show="importMode === '0-1'" class="">
              <div style="margin-bottom: 12px">
                <NIcon size="48" :depth="3">
                  <ArchiveIcon />
                </NIcon>
              </div>
              <NText style="font-size: 16px">点击或者拖动文件到该区域来上传</NText>
              <NP depth="3" style="margin: 8px 0 0 0">
                1. 文件类型为zip
                <br />
              </NP>
              <NP depth="3" style="margin: 8px 0 0 0">2. 为了保证最佳性能，每次最多可上传2000个文件</NP>
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
          <NVirtualList v-if="importMode === '0-0'" :item-size="42" :items="uploaderFiles" class="">
            <template #default="{ item }">
              <div class="h-auto w-full flex flex-col items-start justify-start">
                <div :key="item.key"
                  class="item cursor:pointer h-44px w-full flex items-center justify-between py-8px hover:bg-[#ebf7ed]"
                  @mouseenter="handleMouseEnter(item)" @mouseleave="handleMouseLeave(item)">
                  <NPopover trigger="hover" placement="left">
                    <template #trigger>
                      <!-- 图片列表 markStatus === 1 -->
                      <div class="w-90% flex items-center justify-start">
                        <span class="ml-4px w-90% truncate" :style="{
                          color: item.status && item.status === 'error' ? 'red' : ''
                        }">
                          {{ item.name }}
                        </span>
                        <NButton v-show="item.status && item.status === 'finished'" quaternary type="primary">
                          已完成
                        </NButton>
                        <NButton v-show="item.status && item.status === 'error'" quaternary type="error">失败</NButton>
                      </div>
                      <!-- 图片 + info标注信息 markStatus === 2 -->
                    </template>
                    <div v-show="item.errMessage" class="" :style="{
                      color: item.status && item.status === 'error' ? 'red' : ''
                    }">
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
          <NVirtualList v-if="importMode === '0-1'" :item-size="42" :items="uploaderFiles" class="">
            <template #default="{ item }">
              <div class="h-auto w-full flex flex-col items-start justify-start">
                <div :key="item.key"
                  class="item h-44px w-full w-full flex cursor-pointer items-center justify-between py-8px hover:bg-[#ebf7ed]"
                  @mouseenter="handleMouseEnter(item)" @mouseleave="handleMouseLeave(item)">
                  <NPopover trigger="hover" placement="left">
                    <template #trigger>
                      <div class="w-90% flex-col items-center justify-center">
                        <div class="w-90% flex items-center justify-start">
                          <span class="ml-4px w-90% truncate" :style="{
                            color: item.status && item.status === 'error' ? 'red' : ''
                          }">
                            {{ item.name }}
                          </span>
                          <NButton v-show="item.status && item.status === 'finished'" quaternary type="primary">
                            已完成
                          </NButton>
                          <NButton v-show="item.status && item.status === 'error'" quaternary type="error">
                            失败
                          </NButton>
                        </div>
                        <div v-show="getPercentage(item.raw) != 100 && getPercentage(item.raw) != 0" class="w-90%">
                          <NProgress type="line" :percentage="getPercentage(item.raw)" indicator-placement="inside"
                            processing />
                        </div>
                      </div>
                    </template>
                    <div v-show="item.errMessage" class="" :style="{
                      color: item.status && item.status === 'error' ? 'red' : ''
                    }">
                      {{ item.errMessage }}
                    </div>
                  </NPopover>
                  <div v-show="item.isHover" :style="{
                    cursor: getPercentage(item.raw) != 100 ? 'not-allowed' : 'pointer'
                  }" class="mr-14px" @click.stop="removeFile(item)">
                    <SvgIcon local-icon="imgDel" class="text-[24px]"></SvgIcon>
                  </div>
                </div>
              </div>
            </template>
          </NVirtualList>
        </div>
      </div>
      <template #footer>
        <div class="w-full flex items-center justify-between">
          <NSpace :size="16" v-if="importMode === '0-0'">{{ imgToolText }}</NSpace>
          <NSpace :size="16" v-if="importMode === '0-1'">{{ tooltipText }}</NSpace>
          <NSpace :size="16">
            <NButton @click="closeDrawer">关闭当前窗口</NButton>
            <NButton v-show="importMode === '0-0'" type="primary" :loading="isUpSuccess" @click="handleSubmit">
              开始上传
            </NButton>
            <NButton v-show="importMode === '0-1'" type="primary" :loading="isUpSuccess" @click="startUpload">
              开始上传
            </NButton>
          </NSpace>
        </div>
      </template>
    </NDrawerContent>
  </NDrawer>
</template>

<style scoped lang="scss">
:deep(.n-scrollbar-content) {
  height: 100% !important;
}
</style>
