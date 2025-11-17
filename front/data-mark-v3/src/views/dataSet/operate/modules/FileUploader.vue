<template>
  <div class="h-full w-full flex flex-col rounded">
    <div ref="dashboardContainer" class="uppy-dashboard flex-grow overflow-hidden"></div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, watch, onMounted, onBeforeUnmount } from 'vue';
import Uppy from '@uppy/core';
import Dashboard from '@uppy/dashboard';
import XHRUpload from '@uppy/xhr-upload';
import ImageEditor from '@uppy/image-editor';
import Chinese from '@uppy/locales/lib/zh_CN.js';
import '@uppy/core/dist/style.css';
import '@uppy/dashboard/dist/style.css';
import '@uppy/image-editor/dist/style.min.css';
import SparkMD5 from 'spark-md5';
import { getToken } from '@/store/modules/auth/shared';

defineOptions({
  name: 'FileUploader',
});

interface UploadFile {
  id: string;
  name: string;
  size: number;
  type: string;
  progress: number;
  status: 'pending' | 'uploading' | 'success' | 'error';
  preview?: string;
  md5?: string;
  description?: string;
}

const props = defineProps<{
  uploadUrl: string;
  maxConcurrentUploads?: number;
  maxFileSize?: number;
  maxTotalFiles?: number;
  allowedTypes?: string[];
  enableMd5Check?: boolean;
  autoUpload?: boolean;
  initialFiles?: UploadFile[];
  fileType?: 'image' | 'archive';
  dashboardWidth?: number | string;
  dashboardHeight?: number | string;
}>();

const emit = defineEmits<{
  (e: 'success', files: UploadFile[]): void;
  (e: 'error', error: Error): void;
  (e: 'progress', percent: number): void;
  (e: 'file-added', file: UploadFile): void;
  (e: 'file-removed', file: UploadFile): void;
  (e: 'file-updated', file: UploadFile): void;
}>();

const dashboardContainer = ref<HTMLDivElement | null>(null);
const uppy = ref<Uppy | null>(null);
const uploadedFiles = ref<UploadFile[]>([]);
const totalProgress = ref<number>(0);

// 生成友好的文件大小显示
const getFileSizeText = (bytes: number): string => {
  if (bytes < 1024) return `${bytes}B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)}KB`;
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)}MB`;
  return `${(bytes / (1024 * 1024 * 1024)).toFixed(1)}GB`;
};

// 生成 Uppy 的 note 文本
const generateUppyNote = () => {
  const fileTypeText = props.fileType === 'image' ? '图片' : '压缩包';
  const sizeText = getFileSizeText(props.maxFileSize || 1024 * 1024 * 50);
  return `最多 ${props.maxTotalFiles || 10} 个${fileTypeText}，单个不超过 ${sizeText}`;
};

// 销毁现有的 Uppy 实例
const destroyUppy = () => {
  if (uppy.value) {
    uppy.value.cancelAll();
    uppy.value.clear();
    uppy.value.destroy();
    uppy.value = null;
  }
};

// 手动分片上传
const uploadChunks = async (file: File, fileId: string) => {
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
  const chunkSize = 5 * 1024 * 1024;
  const chunks = Math.ceil(file.size / chunkSize);
  let uploadedChunks = 0;

  for (let i = 0; i < chunks; i++) {
    const start = i * chunkSize;
    const end = Math.min(start + chunkSize, file.size);
    const chunk = file.slice(start, end);
    const fileHash = await calculateFileHash(file);

    const formData = new FormData();
    formData.append('file', chunk);
    formData.append('fileName', file.name);
    formData.append('index', i.toString());
    formData.append('chunkCount', chunks.toString());
    formData.append('fileHash', fileHash);

    try {
      const uploadUrl = `${import.meta.env.VITE_SERVICE_BASE_URL}/bigFileUpload/uploadChunk`
      const response = await fetch(`${uploadUrl}`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${getToken()}`,
        },
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`上传分片 ${i} 失败: ${response.statusText}`);
      }

      uploadedChunks++;
      const progress = (uploadedChunks / chunks) * 100;
      totalProgress.value = progress;
      emit('progress', progress);
    } catch (error) {
      console.error('上传分片出错:', error);
      emit('error', error as Error);
      return;
    }
  }

  // 所有分片上传完成
  const index = uploadedFiles.value.findIndex(f => f.id === fileId);
  if (index !== -1) {
    uploadedFiles.value[index].status = 'success';
    uploadedFiles.value[index].progress = 100;
    emit('success', [...uploadedFiles.value]);
  }
};

// 初始化 Uppy 配置
const configureUppy = () => {
  if (!dashboardContainer.value) return;

  uppy.value = new Uppy({
    debug: true,
    autoProceed: props.autoUpload !== false,
    locale: Chinese,
    restrictions: {
      maxFileSize: props.maxFileSize || 1024 * 1024 * 50,
      maxNumberOfFiles: props.maxTotalFiles || 1500,
      allowedFileTypes: props.allowedTypes || ['image/*', 'application/pdf'],
    },
  });

  if (props.fileType === 'archive') {
    // 手动处理上传
    uppy.value.on('upload', (fileIds, files) => {
      console.log('files: ', files);
      files.forEach(async (file) => {
        if (file) {
          await uploadChunks(file.data as File, file.id);
        }
      });
    });
  } else {
    uppy.value.use(XHRUpload, {
      endpoint: props.uploadUrl,
      formData: true,
      fieldName: 'file',
      headers: {
        'Authorization': `Bearer ${getToken()}`,
      },
      limit: props.maxConcurrentUploads || 20,
    });
  }

  // 配置Dashboard插件
  uppy.value.use(Dashboard, {
    width: props.dashboardWidth || 700,
    height: props.dashboardHeight || 500,
    target: dashboardContainer.value,
    inline: true,
    showProgressDetails: true,
    note: generateUppyNote(),
    proudlyDisplayPoweredByUppy: false,
    metaFields: [
      // { id: 'description', name: '描述', placeholder: '添加描述信息' }
    ],
  });

  // 仅为图片类型启用ImageEditor
  if (props.fileType === 'image') {
    uppy.value.use(ImageEditor, {
      id: 'ImageEditor',
      title: '图片编辑器',
      actions: {
        revert: true,
        rotate: true,
        flip: true,
        zoomIn: true,
        zoomOut: true,
        crop: true,
        resize: true,
      },
    });
  }
};

// 添加初始文件
const addInitialFiles = () => {
  if (props.initialFiles) {
    props.initialFiles.forEach(file => {
      if (uppy.value) {
        uppy.value.addFile({
          id: file.id,
          name: file.name,
          type: file.type,
          data: new Blob(),
          size: file.size,
          meta: {
            description: file.description || ''
          },
          progress: {
            percentage: 100
          },
          uploadComplete: true,
        });
      }
      uploadedFiles.value.push(file);
    });
  }
};

// 初始化 Uppy
const initUppy = () => {
  destroyUppy();
  configureUppy();
  addInitialFiles();
};

// 监听Uppy事件
const listenUppyEvents = () => {
  if (!uppy.value) return;

  uppy.value.on('file-added', async file => {
    const newFile: UploadFile = {
      id: file.id,
      name: file.name,
      size: file.size,
      type: file.type || '',
      progress: 0,
      status: 'pending'
    };

    if (props.enableMd5Check) {
      try {
        newFile.md5 = await calculateMD5(file.data as File);
      } catch (error) {
        console.error('计算MD5失败:', error);
      }
    }

    uploadedFiles.value.push(newFile);
    emit('file-added', newFile);
  });

  uppy.value.on('progress', progress => {
    totalProgress.value = progress.overall;
    emit('progress', progress.overall);
  });

  uppy.value.on('upload-success', (file, response) => {
    const index = uploadedFiles.value.findIndex(f => f.id === file.id);
    if (index !== -1) {
      uploadedFiles.value[index].status = 'success';
      uploadedFiles.value[index].progress = 100;

      if (file.meta?.description) {
        uploadedFiles.value[index].description = file.meta.description;
      }

      emit('success', [...uploadedFiles.value]);
    }
  });

  uppy.value.on('error', (error: Error) => {
    console.error('上传错误:', error);
    emit('error', error);
  });

  uppy.value.on('file-removed', file => {
    const removedFile = uploadedFiles.value.find(f => f.id === file.id);
    if (removedFile) {
      uploadedFiles.value = uploadedFiles.value.filter(f => f.id !== file.id);
      emit('file-removed', removedFile);
    }
  });

  uppy.value.on('file-updated', (file) => {
    const index = uploadedFiles.value.findIndex(f => f.id === file.id);
    if (index !== -1) {
      uploadedFiles.value[index].name = file.name;
      uploadedFiles.value[index].description = file.meta?.description || '';

      emit('file-updated', uploadedFiles.value[index]);
    }
  });
};

// 计算文件MD5
const calculateMD5 = (file: File): Promise<string> => {
  return new Promise((resolve, reject) => {
    const chunkSize = 2 * 1024 * 1024;
    const chunks = Math.ceil(file.size / chunkSize);
    let currentChunk = 0;
    const spark = new SparkMD5.ArrayBuffer();
    const fileReader = new FileReader();

    fileReader.onload = (e) => {
      if (e.target?.result instanceof ArrayBuffer) {
        spark.append(e.target.result);
        currentChunk++;

        if (currentChunk < chunks) {
          loadNext();
        } else {
          resolve(spark.end());
        }
      }
    };

    fileReader.onerror = () => {
      reject(new Error('无法读取文件'));
    };

    function loadNext() {
      const start = currentChunk * chunkSize;
      const end = start + chunkSize >= file.size ? file.size : start + chunkSize;
      fileReader.readAsArrayBuffer(file.slice(start, end));
    }

    loadNext();
  });
};

// 监听关键props变化，重新初始化Uppy
watch([
  () => props.fileType,
  () => props.maxFileSize,
  () => props.maxTotalFiles,
  () => props.allowedTypes,
  () => props.dashboardWidth,
  () => props.dashboardHeight
], () => {
  initUppy();
  listenUppyEvents();
});

// 生命周期钩子
onMounted(() => {
  initUppy();
  listenUppyEvents();
});

onBeforeUnmount(() => {
  destroyUppy();
});
</script>

<style scoped></style>
