<script setup lang="ts">
import { $t } from "@/locales";
import { modelBackFillModelEvaluation, upload } from "@/service/api/third";
import { LogInst, VirtualListInst } from "naive-ui";

defineOptions({
  name: "ModelApplyModal",
});

export type OperateType = NaiveUI.TableOperateType | "addChild";

interface Props {
  taskId: string;
  trainType: string;
  items: any;
  sItems: any;
}

const props = defineProps<Props>();

interface Emits {
  (e: "submitted"): void;
}

const emit = defineEmits<Emits>();

const visible = defineModel<boolean>("visible", {
  default: false,
});

const id = defineModel<string>("id", {
  default: "",
});

// fileList
const fileList = ref([]);


function closeDrawer() {
  visible.value = false;
};

const handleOperate = async (sign: 'define') => {
  if (sign === 'define') {
    const formData = new FormData();
    formData.append('file', fileList.value[0].file);
    formData.append('type', "3");
    const res = await upload(formData);
    if (res.data) {
      const params = {
        id: id.value,
        filePath: res.data
      };
      const res1 = await modelBackFillModelEvaluation(params);
      if (res1.data) {
        window.$message?.success("上传报告成功");
        emit('submitted');
        visible.value = false;
      }
    }
  }
}

// 允许的 MIME 类型
const allowedMimeTypes = [
  // Word
  'application/msword', // .doc
  'application/vnd.openxmlformats-officedocument.wordprocessingml.document', // .docx

  // PDF
  'application/pdf', // .pdf

  // 图片
  'image/jpeg',
  'image/png',
  'image/gif',
  'image/bmp',
  'image/webp',
  'image/svg+xml'
];

// 允许的扩展名（正则表达式兜底）
const fileExtensionRegex = /\.(doc|docx|pdf|jpe?g|png|gif|bmp|webp|svg)$/i;

// beforeUpload 钩子函数
const beforeUpload = ({ file }) => {
  console.log('file: ', file);
  const isValidType = allowedMimeTypes.includes(file.type);
  const isValidExtension = fileExtensionRegex.test(file.name);

  if (!isValidType && !isValidExtension) {
    window.$message?.error(`文件 ${file.name} 类型不支持，请上传 Word、PDF 或图片文件`);
    return false; // 阻止上传
  }

  // 可选：限制文件大小（例如最大 10MB）
  // const isValidSize = file.size <= 1024 * 1024 * 10;
  // if (!isValidSize) {
  //   window.$message?.error('文件太大，请上传小于 10MB 的文件');
  //   return false;
  // }

  return true; // 允许上传
};
</script>

<template>
  <NModal v-model:show="visible" title="上传报告" preset="card" class="w-600px">
    <div>
      <n-upload v-model:file-list="fileList" @before-upload="beforeUpload">
        <n-button>上传文件</n-button>
      </n-upload>
    </div>
    <template #footer>
      <NSpace justify="end" :size="16">
        <NButton type="primary" @click="handleOperate('define')">确定</NButton>
        <NButton @click="closeDrawer">关闭窗口</NButton>
      </NSpace>
    </template>
  </NModal>
</template>

<style scoped lang="scss"></style>
