<script setup lang="ts">
import { NSpin, useMessage } from "naive-ui";
import type { UploadCustomRequestOptions, UploadFileInfo } from "naive-ui";
import aType from "@/assets/imgs/aType.png";
import axios from "axios";
import {
  dataSetImport,
  fetchDataSetAdd,
  fileUpload,
  fileUploadDel,
} from "@/service/api/dataManage";
import { getToken } from "@/store/modules/auth/shared";
import { ArchiveOutline as ArchiveIcon } from "@vicons/ionicons5";
import { useBoolean } from "~/packages/hooks";
import UploadOperateDrawer from './modules/upload-operate-drawer.vue';

const infoObj = ref({
  icon: "fluent:ios-arrow-24-filled",
  title: "导入数据",
});
const formInfo = ref({
  model: {
    inputValue: null,
    dataType: "",
    annotationType: "",
    anoTemVal: "",
  },
  formRef: null,
  rules: {
    inputValue: {
      required: true,
      trigger: ["blur", "input"],
      message: "请输入 inputValue",
    },
  },
  imgPaths: {
    aType,
  },
});
const router = useRouter();
const route = useRoute();
const message = useMessage();
const { bool: drawerVisible, setTrue: openDrawer } = useBoolean();

const sign = ref<string>("default");
const isCreSuccess = ref<Boolean>(false);
const isImport = ref<Boolean>(false);
const isUpAllSuccess = ref<Boolean>(false); // 是否全部上传完成

// methods
const handleBack = () => {
  router.back();
};
// data
const model = ref<any>({
  markStatus: "0",
  fileList: [],
  imgList: [],
  uploadList: [],
});
const isStartImport = ref<Boolean>(false);
const uploadedFilesCount = ref(0);
// eslint-disable-next-line @typescript-eslint/no-shadow
const handleFileUpload = async (options: UploadCustomRequestOptions) => {
  const { file } = options;
  const formData = new FormData();
  formData.append("files", file.file as File);
  // request
  const res = await fileUpload(formData);
  if (res.data) {
    model.value.imgList = [...model.value.imgList, ...res.data];
    model.value.fileList = model.value.fileList.map((item, index) => {
      const obj = {
        fileId: model.value.imgList[index].id,
      };
      return { ...item, ...obj };
    });
  }
};
const handleUploadRemove = async (options: {
  file: UploadFileInfo;
  fileList: Array<UploadFileInfo>;
  index: number;
}) => {
  const delList = model.value.imgList.slice(options.index, options.index + 1);
  const res = await fileUploadDel({
    filIds: delList.map((item) => item.id),
  });
  model.value.imgList.splice(options.index, 1);
  model.value.fileList.splice(options.index, 1);
};

// file-upload
const maxUploadCount = ref(1);
const fileAction = `${import.meta.env.VITE_SERVICE_BASE_URL}/temp/anyUpload`;
const headers = reactive<any>({
  Authorization: `Bearer ${getToken()}`,
  // 'Content-Type': 'multipart/form-data'
});
const handleChange = (fileList) => {
  if (fileList.length > maxUploadCount) {
    message.error(`最多只能上传 ${maxUploadCount} 个文件`);
    // model.value.fileList = fileList.slice(0, maxUploadCount); // 限制文件列表数量
  }
};
const handleImgChange = (options) => {
  isStartImport.value = true;
}
// naive ui 设置最多一次上传100张图片
const handleRemove = async (options: any) => {
  const { file } = options;
  // 根据file.id过滤model.value.uploadList
  for (let i = 0; i < model.value.uploadList.length; i++) {
    const item = model.value.uploadList[i];
    if (item.id === file.id) {
      uploadedFilesCount.value++
      model.value.uploadList.splice(i, 1);
      return false;
    }
  };
  for (let i = 0; i < model.value.fileList.length; i++) {
    const item = model.value.fileList[i];
    if (item.id === file.id) {
      model.value.fileList.splice(i, 1);
      const delList = model.value.imgList.splice(i, 1);
      // eslint-disable-next-line no-await-in-loop
      await fileUploadDel({
        ids: delList.map((item) => item.id),
      });
      return false;
    }
  }
};
// 移入移出
const handleMouseEnter = (options: any) => {
  for (let i = 0; i < model.value.uploadList.length; i++) {
    const item = model.value.uploadList[i];
    if (item.id === options.id) {
      model.value.uploadList[i].isHover = true;
      return false;
    }
  }
};
const handleMouseLeave = (options: any) => {
  for (let i = 0; i < model.value.uploadList.length; i++) {
    const item = model.value.uploadList[i];
    if (item.id === options.id) {
      model.value.uploadList[i].isHover = false;
      return false;
    }
  }
};
const handleImgDel = (row: any) => {
  // 根据row.id 删除model.value.uploadList
  model.value.uploadList.splice(
    model.value.uploadList.findIndex((item) => item.id === row.id),
    1
  );
};
const beforeUpload = (options: any) => {
  isStartImport.value = true;
  // -----------------2024.10.12 new----------------------
  // model.value.uploadList = [...model.value.uploadList, options.file].map((item, index) => {
  //   item.sortIdx = index + 1;
  //   item.isHover = false;
  //   return item;
  // });
  // ---------------------------------------
  const { file } = options;
  const isLtSize = file.file.size / 1024 / 1024 < 20;
  if (!isLtSize) {
    window.$message?.error("上传图片大小不能超过 20 MB!");
    return false;
  }
  const types = [
    "image/jpeg",
    "image/png",
    "image/jpg",
    "image/webp",
    "image/bmp",
  ];
  const isAllowedType = types.includes(file.type);
  if (!isAllowedType) {
    message.error("只能上传 JPG/PNG/JPEG/PNG/WEBP/BMP 格式的图片！");
    return false; // 停止上传
  }
  return true;
};
const handleUploadSuccess = (options: any) => {
  const res = JSON.parse(options.event.currentTarget.response);
  if (res.code == 200) {
    model.value.imgList = [...model.value.imgList, ...res.data];
  } else {
    console.log(res);
  }
};
const handleUploadError = (error) => {
  console.log(error);
};

const handleUploadFileList = (fileList: any) => { };
const handleDefineCancel = async () => {
  isImport.value = true;
  // 10.12 new 并发控制
  // const generator = concurrentApiCallGenerator(model.value.uploadList, 20);
  // await run(generator);
  // ---------------------
  if (sign.value === "default") {
    const params = {
      ...route.query,
      isMarkInfo: model.value.markStatus,
      fileIds: model.value.imgList.filter(val => val.id).map((item) => item.id).join(","),
    };
    const res = await fetchDataSetAdd(params);
    if (res.data?.status == 1) {
      isImport.value = false;
      window.$message?.success("创建数据集成功！");
      router.replace({
        name: "data-manage_map",
        query: {
          dataTypeId: route.query.dataTypeId,
          sonId: res.data.sonId,
          groupId: res.data.groupId,
        },
      });
    }
  }

  if (sign.value === "mapToImport") {
    const params = {
      sonId: route.query.sonId,
      fileIds: model.value.imgList.filter(val => val.id).map((item) => item.id).join(","),
    };
    const res = await dataSetImport(params);
    if (res.data?.status == 1) {
      window.$message?.success("创建数据集成功！");
      localStorage.setItem("isImport", "true");
      // ----------------------------------------------------------------
      router.replace({
        name: "data-manage_map",
        query: {
          dataTypeId: route.query.dataTypeId,
          sonId: res.data.sonId,
          groupId: res.data.groupId,
        },
      });
    }
  }
};

// ----------------------newCode---------------------------
const handleImgFinish = () => {
  if (model.value.fileList.length == uploadedFilesCount.value) {
    model.value.uploadList = model.value.fileList.map((item, index) => {
      item.sortIdx = index + 1;
      item.isHover = false;
      return item;
    });
    setTimeout(() => {
      isStartImport.value = false;
    }, 300)
  }
}
// 自定义请求
const customRequest = ({
  file,
  data,
  headers,
  withCredentials,
  action,
  onFinish,
  onError,
  onProgress,
}: UploadCustomRequestOptions) => {
  uploadedFilesCount.value++;
  onFinish();
};

function asyncApiCall(val: any) {
  const uploadUrl = `${import.meta.env.VITE_SERVICE_BASE_URL}/temp/anyUpload`;
  const formData = new FormData();
  formData.append("files", val.file);
  return new Promise((resolve, reject) => {
    axios
      .post(uploadUrl, formData, {
        headers: {
          "Content-Type": "multipart/form-data",
        },
      })
      .then((res) => {
        // 根据val.id查找model.value.fileList的索引
        const index = model.value.fileList.findIndex((item: any) => item.id === val.id);
        if (index !== -1) {
          model.value.fileList[index].status = "finished";
        }
        resolve(res.data.data[0]);
      })
      .catch((error) => {
        console.log("文件上传失败", error);
        const index = model.value.fileList.findIndex((item: any) => item.id === val.id);
        if (index !== -1) {
          model.value.fileList[index].status = "error";
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
  };
  model.value.imgList = result.map(val => {
    val.id = val.value.id;
    return val;
  });
  // model.value.fileList = result.map(val => {
  //   return {
  //     id: val.value.id,
  //     name: val.value?.name ?? "default",
  //     url: val.value.path,
  //     status: val?.status === "fulfilled" ? "finished" : "error",
  //   }
  // });
}

function uploadSubmit(data: any) {
  model.value.imgList = data.fileList;
}

const childRef = ref(null);

function handleRadioChange() {
  childRef.value.resetFiles();
  model.value.imgList = [];
}

onMounted(() => {
  if (route.query.sign === "mapToImport") {
    sign.value = "mapToImport";
  } else {
    sign.value = "default";
  }
});
</script>

<template>
  <div class="wrap_container flex_col_start relative">
    <div v-show="isImport" class="mask-layer">
      <div class="loading-spinner">
        <!-- 这里可以放置任何你想要的加载动画 -->
        <NSpin size="large" description="创建数据集中... 请稍等" />
      </div>
    </div>
    <div class="header">
      <div class="h_back flex_start" @click="handleBack()">
        <!--<SvgIcon :icon="infoObj.icon" class="inline-block align-text-bottom text-16px"/>-->
        <SvgIcon local-icon="oui--return-key" class="inline-block align-text-bottom text-16px text-[#000000]" />
        <span>返回</span>
      </div>
      <div class="h_title">{{ infoObj.title }}</div>
    </div>
    <UploadOperateDrawer ref="childRef" v-model:visible="drawerVisible" v-model:markStatus="model.markStatus"
      @submitted="uploadSubmit" />
    <div class="content f-c-c min-h-500px flex-col-stretch gap-16px overflow-hidden lt-sm:overflow-auto">
      <div class="item_card h-full">
        <n-card title="导入配置" class="h-full overflow-y-auto">
          <n-form class="w-80% h-auto" ref="formRef" :model="formInfo.model" :rules="formInfo.rules" label-width="100"
            label-align="left" label-placement="left">
            <n-form-item label="数据标注状态" path="version">
              <n-radio-group v-model:value="model.markStatus" name="radiogroup" @update:value="handleRadioChange">
                <n-space>
                  <n-radio value="0" class="flex items-center">
                    <span>无标注信息</span>
                  </n-radio>
                  <n-radio value="1" class="flex items-center">
                    <span>有标注信息</span>
                  </n-radio>
                </n-space>
              </n-radio-group>
            </n-form-item>
            <n-form-item label="导入方式" path="version" class="w-[70%]">
              本地上传
            </n-form-item>
            <n-form-item label="上传文件" path="version" class="w-[100%] wrap-upload">
              <n-button type="primary" @click="() => openDrawer()">
                <span v-if="model.markStatus == 0">图片列表</span>
                <span v-else>图片列表 + 标注信息</span>
              </n-button>
              <div class="ml-16px flex justify-start items-center" v-show="model.imgList.length !== 0">
                <span>已上传</span>
                <span>{{ model.imgList.length }}</span>
                <span>张图片</span>
              </div>
              <!--<template #label>
                <div class="w-auto flex justify-start items-center gap-4px">
                  <span>上传图片</span>
                  <n-popover trigger="hover" placement="right">
                    <template #trigger>
                      <div>
                        <svg-icon local-icon="markTol" class="text-22px"></svg-icon>
                      </div>
                    </template>
<div class="w-auto p-14px box-border flex flex-col justify-start items-start">
  <n-text style="font-size: 14px">
    1. 图片类型为jpg/png/bmp/jpeg/webp，单次上传限制100个文件<span class="text-red">(只能上传图片类型)</span>
  </n-text>
  <n-text style="font-size: 14px">
    2. 图片大小限制在20M内
  </n-text>
</div>
</n-popover>
</div>
</template>-->
              <!--<n-image-group>
                <n-upload
                  multiple
                  directory-dnd
                  name="files"
                  :action="fileAction"
                  accept=".jpg,.png,.jpeg,.bmp,.webp"
                  :headers="headers"
                  list-type="image-card"
                  @before-upload="beforeUpload"
                  @finish="handleUploadSuccess"
                  @remove="handleRemove"
                  @update:file-list="handleUploadFileList"
                  v-model:file-list="model.fileList"
                >
                  <n-upload-dragger>
                    <div style="margin-bottom: 12px">
                      <n-icon size="48" :depth="3">
                        <ArchiveIcon />
                      </n-icon>
                    </div>
                    <n-text style="font-size: 16px">
                      点击或者拖动文件到该区域来上传
                    </n-text>
                    <n-p depth="3" style="margin: 8px 0 0 0">
                      1. 图片类型为jpg/png/bmp/jpeg/webp，单次上传限制100个文件
                      <br />
                      <span class="text-red">(只能上传图片类型)</span>
                    </n-p>
                    <n-p depth="3" style="margin: 8px 0 0 0">
                      2. 图片大小限制在20M内
                    </n-p>
                  </n-upload-dragger>
                </n-upload>
              </n-image-group>-->
              <!-- <n-upload
                multiple
                directory-dnd
                name="files"
                :action="fileAction"
                accept=".jpg,.png,.jpeg,.bmp,.webp"
                :headers="headers"
                @before-upload="beforeUpload"
                @finish="handleUploadSuccess"
                @remove="handleRemove"
                @update:file-list="handleUploadFileList"
                v-model:file-list="model.fileList"
              > -->
              <!--<div class="w-full h-auto flex flex-col justify-start items-start">
                <n-upload
                  multiple
                  directory-dnd
                  name="files"
                  :action="fileAction"
                  accept=".jpg,.png,.jpeg,.bmp,.webp"
                  :headers="headers"
                  :show-file-list="false"
                  @before-upload="beforeUpload"
                  @remove="handleRemove"
                  @change="handleImgChange"
                  @finish="handleImgFinish"
                  @update:file-list="handleUploadFileList"
                  :custom-request="customRequest"
                  v-model:file-list="model.fileList"
                >
                  <n-upload-dragger>
                    <div style="margin-bottom: 12px">
                      <n-icon size="48" :depth="3">
                        <ArchiveIcon />
                      </n-icon>
                    </div>
                    <n-text style="font-size: 16px">
                      点击或者拖动文件到该区域来上传
                    </n-text>
                    <n-p depth="3" style="margin: 8px 0 0 0">
                      1. 图片类型为jpg/png/bmp/jpeg/webp
                      <br />
                      <span class="text-red">(只能上传图片类型)</span>
                    </n-p>
                    <n-p depth="3" style="margin: 8px 0 0 0">
                      2. 图片大小限制在20M内
                    </n-p>
                  </n-upload-dragger>
                </n-upload>
                <div class="mt-8px"></div>
                <div v-show="isStartImport" class="mask-layer">
                  <div class="loading-spinner">
                    &lt;!&ndash; 这里可以放置任何你想要的加载动画 &ndash;&gt;
                    <NSpin size="large" description="图片导入中... 请稍等" />
                  </div>
                </div>
                <n-virtual-list style="max-height: 240px" :item-size="42" :items="model.uploadList">
                  <template #default="{ item }">
                    <div
                      :key="item.key"
                      class="item w-full h-44px flex justify-between items-center hover:bg-[#ebf7ed] py-8px"
                      @mouseenter="handleMouseEnter(item)"
                      @mouseleave="handleMouseLeave(item)"
                    >
                      <div class="flex justify-start items-center">
                        &lt;!&ndash;<span>{{ item.sortIdx }}.</span>&ndash;&gt;
                        <span class="ml-4px"> {{ item.name }}</span>
                      </div>
                      <div class="mr-14px" @click.stop="handleImgDel(item)" v-show="item.isHover">
                        <svg-icon local-icon="imgDel" class="text-[24px]"></svg-icon>
                      </div>
                    </div>
                  </template>
                </n-virtual-list>
              </div>-->
            </n-form-item>
          </n-form>
        </n-card>
      </div>
    </div>
    <div class="footer flex_start">
      <n-button @click="handleDefineCancel()" type="primary">确认并返回</n-button>
    </div>
  </div>
</template>

<style scoped lang="scss">
.mask-layer {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  background-color: rgba(255, 255, 255, 0.7);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 9999;
}

.flex_start {
  display: flex;
  justify-content: flex-start;
  align-items: center;
}

.flex_center {
  display: flex;
  justify-content: center;
  align-items: center;
}

.flex_between {
  display: flex;
  justify-content: space-around;
  align-items: center;
}

.flex_around {
  display: flex;
  justify-content: space-around;
  align-items: center;
}

.flex_col_start {
  display: flex;
  flex-direction: column;
  justify-content: flex-start;
  align-items: center;
}

.flex_col_center {
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
}

.wrap_container {
  padding: 0;
  width: 100%;
  height: 100%;
  background-color: #f7f7f9;

  .header {
    width: 100%;
    padding: 0 16px;
    box-sizing: border-box;
    height: 48px;
    background-color: #fff;
    display: flex;
    justify-content: flex-start;
    align-items: center;

    .h_back {
      span {
        color: #303540;
        font-size: 12px;
        cursor: pointer;
      }
    }

    .h_title {
      margin-left: 16px;
      font-size: 16px;
      color: #151b26;
      font-weight: 500;
    }
  }

  .content {
    padding: 16px 24px;
    box-sizing: border-box;
    width: 100%;
    flex: 1;

    .item_card {
      width: 100%;
      min-height: 240px;

      .form_annotationType {
        padding: 8px;
        box-sizing: border-box;
        width: 128px;
        height: 128px;
        border: 1px solid #ddd;
        border-radius: 4px;
      }
    }
  }

  .footer {
    width: 100%;
    height: 60px;
    background-color: #fff;
    padding: 0 24px;
    box-sizing: border-box;

    .n-button {
      margin-right: 24px;
    }
  }
}

:deep(.wrap-upload) {
  .n-upload-file--success-status {

    //border: 1px solid #2d7a67;
    .n-upload-file-info__name {
      color: #2d7a67;
    }
  }

  .n-upload-file-list {
    // display: none !important;
  }
}
</style>
