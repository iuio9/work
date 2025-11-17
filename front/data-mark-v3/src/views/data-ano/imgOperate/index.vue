<template>
  <div class="image-annotation-container font-inter bg-gray-50 w-full h-full flex flex-col">
    <div class="w-full h-auto flex justify-between items-center relative">
      <div class="w-full">
        <AnoTypeTabs v-model:activeTab="activeTab" v-model:curTabNum="curTabNum" v-model:tabConfig="tabConfig"
          @tabChange="handleTabChange" ref="anoTabRef" />
      </div>
      <div class="w-auto absolute right-0 bottom-24px flex items-center gap-24px">
        <ValidationButtons v-model:validateStatus="validateStatus" @show-validate-modal="openValidateModal"
          @refresh-data="refreshAllData" />
        <!-- 返回 -->
        <NButton type="primary" @click="handleBack">返回</NButton>
      </div>
    </div>
    <!-- 主内容区 -->
    <main class="flex-1 min-h-0 w-full px-4 py-6 flex flex-col gap-6">
      <!-- 工具栏和状态信息 -->
      <div class="w-full flex justify-start items-center">
        <div class="flex items-center gap-4">
          <h2 class="text-lg font-semibold text-gray-800">当前图片: {{ currentImageName || '暂无图片' }}</h2>
          <h2
            :class="{ 'text-red-500': currentImageResult?.status === 'error', 'text-green-500': currentImageResult?.status === 'success' }"
            v-if="shouldShowResult && currentImageResult?.message">（{{ route.query.anoType === 'audit' ? '审核结果' :
              '验收结果' }}: {{ currentImageResult?.message }}）</h2>
        </div>
      </div>

      <!-- 标注区域 -->
      <!-- route.query.anoType === 'audit' || route.query.anoType === 'result' ? '!grid-cols-[25%_75%]' : 'grid-cols-[25%_50%_25%]' -->
      <div :class="[
        'annotation-area flex-1 min-h-0 w-full bg-gray-100 rounded-lg border border-gray-200 overflow-hidden grid grid-cols-[25%_50%_25%]'
      ]">
        <!-- 图片列表 -->
        <div class="image-list flex flex-col panel-border left-panel h-full overflow-hidden">
          <div class="panel-title">
            <h3 class="text-sm">图片列表</h3>
          </div>

          <!-- 图片内容区域 - 占据剩余空间 -->
          <div class="flex-1 panel-content p-4 overflow-y-auto">
            <div class="image-grid">
              <template v-for="(image, imageIndex) in pagImgList" :key="image.id">
                <div class="image-item flex justify-center items-center"
                  :class="{ 'ring-2 ring-primary ring-offset-2': image.fileId === activeImageId }"
                  @click="selectImage(image, imageIndex)">
                  <img :src="image.previewImgPath" :alt="image.fileName || '未命名图片'"
                    class="w-full h-full object-cover transition-transform duration-500 group-hover:scale-110"
                    loading="lazy" :crossOrigin="'anonymous'">
                </div>
              </template>
            </div>
          </div>

          <!-- 分页控件 - 固定在底部 -->
          <div class="p-4 border-t border-gray-200 flex justify-center items-center">
            <NPagination v-model:page="imgListCurrent" v-model:page-size="imgListPageSize"
              :page-count="imgListTotalPage" v-if="pagImgList.length > 0" :page-slot="3" show-quick-jumper
              @update:page="handlePageChange" />
          </div>
        </div>

        <!-- 标注容器 -->
        <div class="relative col-span-1 h-full overflow-hidden flex-col justify-start items-center">
          <!-- 状态指示器 -->
          <div
            class="absolute top-3 left-3 bg-white/80 backdrop-blur-sm rounded-lg shadow-sm px-3 py-1.5 text-sm flex items-center gap-3 z-20">
            <span class="font-medium">当前工具: {{ getModeName(state.currentMode) }}</span>
          </div>

          <!-- 无效数据 -->
          <div v-show="imgInvalidValue" class="invalid-tip">无效数据</div>
          <!-- 后台保存中 -->
          <div v-show="isBackendSave" class="mask-layer">
            <div class="loading-spinner">
              <NSpin size="large" description="后台保存中.. 请稍等" />
            </div>
          </div>
          <!-- 图片加载中 -->
          <div v-show="state.isImgRender" class="mask-layer">
            <div class="loading-spinner">
              <NSpin size="large" description="图片加载中.. 请稍等" />
            </div>
          </div>
          <!-- 自动标注中 -->
          <div v-show="state.isAutoLabel" class="mask-layer">
            <div class="loading-spinner">
              <NSpin size="large" description="自动标注中... 请稍等" />
            </div>
          </div>

          <!-- 标注容器 - 设置宽高为100% -->
          <div ref="annotationContainer" class="ailabel-container w-full h-full" id="annotationContainer"></div>

          <!-- 中间工具栏 -->
          <div v-if="route.query?.anoType !== 'audit' && route.query?.anoType !== 'result'"
            class="annotation-toolbar absolute bottom-4 left-1/2 transform -translate-x-1/2 bg-white/90 backdrop-blur-sm rounded-full shadow-md border border-gray-100 px-3 py-1.5 flex items-center transition-all duration-300 hover:shadow-lg z-20">
            <n-popover v-for="(tool, index) in tools" :key="index" trigger="hover" placement="top">
              <template #trigger>
                <div :data-mode="tool.mode" class="toolbar-btn" :class="{ active: state.currentMode === tool.mode }"
                  @click="updateTool(tool.mode)">
                  <SvgIcon :icon="tool.icon" :localIcon="tool.localIcon" class="text-20px font-bold" />
                </div>
              </template>
              {{ tool.label }}
            </n-popover>
          </div>

          <!-- 操作工具栏 -->
          <div class="operate-controls" v-if="route.query.anoType !== 'audit' && route.query.anoType !== 'result'">
            <n-popover trigger="hover" placement="right">
              <template #trigger>
                <div class="operate-btn" @click="saveCurrentAnnotation()">
                  <SvgIcon icon="mingcute:save-line" local-icon="mingcute--save-line" class="text-24px" />
                </div>
              </template>
              保存当前标注
            </n-popover>
            <n-divider class="!my-4px" />
            <n-popover trigger="hover" placement="right">
              <template #trigger>
                <div class="operate-btn">
                  <SvgIcon icon="icon-park-outline:invalid-files" local-icon="icon-park-outline--invalid-files"
                    class="text-24px" />
                </div>
              </template>
              <div class="flex justify-start items-center gap-8px w-auto h-40px">
                <n-checkbox v-model:checked="imgInvalidValue" />
                <n-button strong secondary type="primary" @click="markInvalid">
                  保存为无效数据
                </n-button>
              </div>
            </n-popover>
            <n-divider class="!my-4px" />
            <n-popover trigger="hover" placement="right">
              <template #trigger>
                <div class="operate-btn" @click="clearCurrentAnnotation">
                  <SvgIcon icon="mdi:book-cancel-outline" local-icon="mdi--book-cancel-outline" class="text-24px" />
                </div>
              </template>
              取消当前标注
            </n-popover>
          </div>

          <!-- 审核 anoType: audit -->
          <div class="operate-controls" v-if="shouldShowButtons">
            <n-popconfirm @positive-click="handlePass('1')">
              <template #trigger>
                <n-popover trigger="hover" placement="right">
                  <template #trigger>
                    <div class="operate-btn">
                      <SvgIcon local-icon="codicon--pass" class="text-24px" />
                    </div>
                  </template>
                  通过
                </n-popover>
              </template>
              是否确认通过？
            </n-popconfirm>
            <n-popover trigger="hover" placement="right">
              <template #trigger>
                <div class="operate-btn" @click="handlePass('2')">
                  <SvgIcon local-icon="nonicons--not-found-16" class="text-22px" />
                </div>
              </template>
              不通过
            </n-popover>
          </div>

          <!-- 缩放控制 -->
          <div class="zoom-controls">
            <n-popover trigger="hover" placement="left">
              <template #trigger>
                <div class="zoom-btn" @click="zoomIn">
                  <SvgIcon icon="gg:zoom-in" local-icon="gg--zoom-in" class="text-24px" />
                </div>
              </template>
              放大
            </n-popover>
            <n-divider class="!my-4px" />
            <n-popover trigger="hover" placement="left">
              <template #trigger>
                <div class="zoom-btn" @click="zoomOut">
                  <SvgIcon icon="gg:zoom-out" local-icon="gg--zoom-out" class="text-24px" />
                </div>
              </template>
              缩小
            </n-popover>
          </div>
        </div>

        <!-- 标签列表 -->
        <!-- v-if="route.query.anoType !== 'audit' && route.query.anoType !== 'result'" -->
        <div class="label-list flex flex-col panel-border right-panel h-full overflow-hidden">
          <!-- 上部：标签列表区域 -->
          <div class="section tag-section flex flex-col h-[50%] border-b border-gray-200">
            <div class="panel-title w-full">
              <n-input-group class="w-full">
                <n-input-group-label>
                  <span class="text-sm">标签列表</span>
                </n-input-group-label>
                <n-input placeholder="请输入标签名" v-model:value="tagSearchVal" />
                <n-input-group-label @click="searchTag">
                  <span class="text-sm">搜索</span>
                </n-input-group-label>
              </n-input-group>
            </div>
            <div class="panel-content flex-1 overflow-y-auto p-2 space-y-2" style="height: calc(100% - 40px);">
              <!-- <div v-for="(tag, index) in pagTagList" :key="index" class="label-item h-36px bg-white"
                @click="tagClick(tag)">
                <n-popover trigger="hover" placement="top">
                  <template #trigger>
                    <span>{{ index + 1 === 10 ? '0' : index + 1 }} {{ tag.labelName }}</span>
                  </template>
                  <div>{{ tag.labelName }}</div>
                </n-popover>
                <div class="flex items-center gap-1">
                  <span class="w-3 h-3 rounded-full" :style="{ backgroundColor: tag.labelColor }"></span>
                  <n-popover trigger="hover" placement="right">
                    <template #trigger>
                      <div class="zoom-btn" @click="topTag(tag)">
                        <SvgIcon icon="lsicon:top-filled" local-icon="lsicon--top-filled" class="text-20px" />
                      </div>
                    </template>
                    置顶
                  </n-popover>
                </div>
              </div> -->
              <div v-for="(tag, index) in pagTagList" :key="index" class="label-item h-36px bg-white rounded-lg flex items-center justify-between px-3 py-2
         transition-all duration-200 hover:shadow-sm cursor-pointer border
         hover:border-gray-300" :class="{
          'border-primary ring-1 ring-primary/20 bg-primary/5': tag.labelId === selectedTagId,
          'border-transparent': tag.labelId !== selectedTagId
        }" @click="tagClick(tag)">
                <!-- 标签序号与名称 -->
                <n-popover trigger="hover" placement="top">
                  <template #trigger>
                    <span class="flex items-center gap-2">
                      <!-- 序号样式优化：带背景的小圆圈 -->
                      <span
                        class="w-5 h-5 rounded-full bg-gray-100 text-gray-700 text-xs flex items-center justify-center">
                        {{ (index + 1).toString().padStart(2, '0') }}
                      </span>
                      <span class="text-gray-800 font-medium">{{ tag.labelName }}</span>
                    </span>
                  </template>
                  <div class="px-2 py-1 text-sm">{{ tag.labelName }}</div>
                </n-popover>

                <!-- 颜色指示器与置顶按钮 -->
                <div class="flex items-center gap-2">
                  <!-- 颜色圆点：增加边框和轻微阴影 -->
                  <span class="w-3.5 h-3.5 rounded-full shadow-sm border border-white/50"
                    :style="{ backgroundColor: tag.labelColor }"></span>

                  <!-- 置顶按钮：hover 效果优化 -->
                  <n-popover trigger="hover" placement="right">
                    <template #trigger>
                      <div class="zoom-btn p-1.5 rounded-full hover:bg-gray-100 transition-colors" @click="topTag(tag)">
                        <SvgIcon icon="lsicon:top-filled" local-icon="lsicon--top-filled"
                          class="text-20px text-gray-600 hover:text-primary" />
                      </div>
                    </template>
                    <div class="px-2 py-1 text-sm">置顶</div>
                  </n-popover>
                </div>
              </div>
            </div>
            <div
              class="pagination-container__page w-full h-46px flex justify-center py-4px box-border boder-b-[#eee] border-t-1 z-666">
              <NPagination v-model:page="current" v-model:page-size="pageSize" :page-count="totalPage"
                v-if="pagTagList.length > 0" :page-slot="5" />
            </div>
          </div>

          <!-- 下部：标注框列表区域 -->
          <div class="section box-section flex flex-col h-[50%]">
            <div class="panel-title">
              <h3 class="text-sm">标注框列表</h3>
            </div>
            <div class="panel-content flex-1 overflow-y-auto grid grid-cols-2 gap-4px content-start"
              style="height: calc(100% - 40px);">
              <div v-for="(annotation, index) in operateAnnotations" :key="index"
                class="label-item h-36px bg-white border border-gray-200 rounded-lg flex items-center justify-between p-3 hover:bg-gray-50 transition-colors"
                :style="{
                  'background': annotation.id === activeLayerItem?.id ? '#91d5ff' : ''
                }" @click="selectFeature(annotation.id)">
                <n-popover trigger="hover" placement="top">
                  <template #trigger>
                    <div class="w-full flex justify-start items-center">
                      <span @click="updateTextTag(annotation)"
                        :class="['font-medium truncate max-w-[100%] flex-1', { 'text-red-500': annotation.props.name === undefined }]">{{
                          annotation.props.name || '标签未命名' }}</span>
                      <div v-if="route.query.anoType !== 'audit' && route.query.anoType !== 'result'"
                        class="delete cursor-pointer" @click="removeFeatureAndText(annotation.id)">
                        <SvgIcon icon="material-symbols:delete" local-icon="material-symbols--delete"
                          class="text-20px" />
                      </div>
                    </div>
                  </template>
                  <div class="flex-col items-end gap-4px">
                    <span>{{ annotation.props.name || '标签未命名' }}</span>
                    <div class="flex items-center gap-4px">
                      <div class="eye cursor-pointer" @click="toggleEye(annotation)">
                        <SvgIcon :local-icon="!annotation.isEye ? 'mdi--eye-off' : 'mdi--eye'" class="text-20px" />
                      </div>
                      <div v-if="route.query.anoType !== 'audit' && route.query.anoType !== 'result'"
                        class="delete cursor-pointer" @click="removeFeatureAndText(annotation.id)">
                        <SvgIcon icon="material-symbols:delete" local-icon="material-symbols--delete"
                          class="text-20px" />
                      </div>
                    </div>
                  </div>
                </n-popover>
              </div>
            </div>
          </div>
        </div>
      </div>
    </main>

    <!-- 标签选择 -->
    <TagOperateModal v-model:visible="tagVisible" v-model:rowData="tagRowData" @tagClick="tagClick" />

    <!--验收-->
    <NModal v-model:show="isShowValidate" :close-on-esc="false" class="wrap-tag-modal">
      <NCard :title="validateTitle" class="w-640px" :bordered="false" size="huge" role="dialog" aria-modal="true">
        <NForm ref="formRef" :model="validateModel">
          <div class="h-full w-full">
            <div v-if="validateStatus === '1'">
              <NFormItem label="保存类型" path="type">
                <NRadioGroup v-model:value="validateModel.verifyState">
                  <NRadio v-for="item in statusOptions" :key="item.value" :value="item.value" :label="item.label" />
                </NRadioGroup>
              </NFormItem>
              <div class="bg-[#fff4e6] p-8px py-8px">
                注:验收完成后任务就会结束 不能再进行操作
              </div>
            </div>
            <div v-if="validateStatus === '2'">
              <NFormItem label="打回类型" path="repulseType">
                <NRadioGroup v-model:value="validateModel.returnState">
                  <NRadio v-for="item in repulseOptions" :key="item.value" :value="item.value" :label="item.label" />
                </NRadioGroup>
              </NFormItem>
            </div>
          </div>
        </NForm>
        <template #footer>
          <div class="h-auto w-full flex items-center justify-end gap-24px">
            <NButton type="primary" @click="handleModalDefine('validate')">确定</NButton>
            <NButton @click="handleModalCancel('validate')">取消</NButton>
          </div>
        </template>
      </NCard>
    </NModal>

    <!-- 不通过意见modal -->
    <NModal v-model:show="noPassVisable" :close-on-esc="false" class="wrap-tag-modal">
      <NCard style="width: 520px" title="验收意见" :bordered="false" size="huge" role="dialog" aria-modal="true">
        <div class="w-full h-full">
          <n-input v-model:value="validateModel.message" type="textarea" placeholder="请输入验收意见" />
        </div>
        <template #footer>
          <div class="h-auto w-full flex items-center justify-end gap-24px">
            <NButton type="primary" @click="handleModalDefine('pass')">添加意见</NButton>
            <NButton @click="handleModalCancel('pass0')">无意见</NButton>
            <!-- 取消 -->
            <NButton @click="() => noPassVisable = false">取消</NButton>
          </div>
        </template>
      </NCard>
    </NModal>
  </div>
</template>

<script lang="ts" setup>
import { computed, onMounted, onUnmounted, reactive, ref, watch } from 'vue';
import type { ImageData, ImageList, Label, AnnotationItem, ValidationResult, SaveResult, CurrentImage } from './types';
import { useImageAnnotation } from './useImageAnnotation';
import TagOperateModal from "./modules/TagOperateModal.vue";
import AnoTypeTabs from "./modules/AnoTypeTabs.vue";
import ValidationButtons from "./modules/ValidationButtons.vue";

// 导入图片列表模拟数据
import mockImgJson from "./mockImg.json";
import mockTagJson from "./mockTag.json";

import _ from 'lodash';
import { nanoid } from "nanoid";
import { usePagination } from 'vue-request';
import { addDataMarkInfo, fileIsApprove, getDataDetailsNoMarkFilePath, getSelectGroupLabelPage, MarkFileUpload, returnTask, topUpLabel, verifyComplete } from '@/service/api/ano';
import { useBoolean } from '~/packages/hooks/src';

const router = useRouter();
const route = useRoute();

const currentTabVal = ref<string | undefined | number>(0);

// 组件引用
const annotationContainer = ref<HTMLElement | null>(null);

// 使用自定义hook
const {
  initAnnotation,
  state,
  tools,
  setMode,
  getModeName,
  zoomIn,
  zoomOut,
  loadImage,
  getAnnotations,
  clearAnnotations,
  createEyeFeature,
  addFeature,
  removeFeature,
  selectFeature,
  createText,
  addText,
  removeText,
  hasText,
  getActiveFeature,
  getLabelTextPosition,
  renderCustomAnnotations,
  getAnnotationFormData,
  calculateRatio,
  scaleShapeByRatio,
  refresh,
  resize,
  undo,
  redo,
  exportImage,
  _featureLayer,
  _gMap,
  _textLayer,
  _state,
} = useImageAnnotation({
  containerRef: annotationContainer,
  containerId: 'annotationContainer',
  // defaultImageUrl: mockImgJson[0].imgPath,
  defaultImageUrl: undefined,
});

// tools update
function updateTool(mode: string) {
  setMode(mode);
  if (mode === 'CIRCLE' || mode === 'POINT' || mode === 'RECT' || mode === 'POLYGON') {
    pagImgList.value[imgListCurrent.value].isInvalid = 1;
  }
}

// 图片数据
const activeImageId = ref<number | null>(null);
const currentImgIndex = ref<number>(0);
const imgInvalidValue = ref<Boolean>(false); // 是否为无效数据
const isBackendSave = ref<Boolean>(false); // 后台保存中
const curDefaultMarkInfo = ref<any>([]);
const prevMarkInfo = ref<any>([]);


const { data: imgData, run: runImgList, current: imgListCurrent, total: imgListTotal, totalPage: imgListTotalPage, pageSize: imgListPageSize } = usePagination(
  getDataDetailsNoMarkFilePath,
  {
    defaultParams: [
      {
        sign: route.query?.anoType,
        state: currentTabVal.value,
        limit: 10,
        sonId: route.query?.id,
        markUserId: route.query?.markUserId,
        taskId: route.query?.taskId,
      },
    ],
    pagination: {
      currentKey: "page",
      pageSizeKey: "limit",
      totalKey: "data.total",
    },
    manual: true, // 关键配置：不立即执行
  },
);

const pagImgList = computed<ImageList>(
  () =>
    imgData.value?.data?.records.map((val: any, idx: number) => {
      return {
        ...val
      }
    }) || [],
);

// watch pagImgList
watch(() => pagImgList.value, async (newVal) => {
  if (newVal.length === 0) {
    await loadImage(undefined);
    operateAnoMap.clear();
    operateAnnotations.value = [];
    clearAnnotations();
    return;
  }

  let imgIdx = currentImgIndex.value ?? 0;
  if (imgIdx >= newVal.length) {
    imgIdx = newVal.length - 1;
  }

  if (newVal[imgIdx]?.fileId) {
    if (imgIdx == 0 || imgIdx == 9) {
      await loadImage(newVal[imgIdx].imgPath);
    }
    activeImageId.value = newVal[imgIdx].fileId;
    currentImgIndex.value = imgIdx;
  }
  await nextTick();
  await initAnnotation();
  operateAnoMap.clear();
  operateAnnotations.value = [];
  clearAnnotations();

  if (newVal[imgIdx]?.markInfo) {
    let annotations: any[] = JSON.parse(newVal[imgIdx].markInfo);
    renderCustomAnnotations(annotations);
    await nextTick();
    curDefaultMarkInfo.value = operateAnnotations.value.map((item: any) => {
      return {
        id: item.id,
        textId: item.props.textId,
        isEye: item.isEye,
        operateIdx: item.operateIdx,
        props: item.props,
        shape: item.shape,
        style: item.style,
        type: item.type
      }
    });
  }
});

watch(() => activeImageId.value, async (newVal, oldVal) => {
  if (newVal) {
    const img = pagImgList.value.find((item: any) => item.fileId === newVal);
    if (img) {
      curDefaultMarkInfo.value = [];
      // await loadImage(img.imgPath);
    }
  }
})

// watch route
watchEffect(async () => {
  // 提取并验证路由参数
  const { imgIdx, id, markUserId, taskId } = route.query;

  // 安全处理 imgIdx，确保为有效数字
  const imgIdxNum = Number(imgIdx);
  const validImgIdx = isNaN(imgIdxNum) || imgIdxNum < 0 ? 0 : Math.floor(imgIdxNum);

  // 计算分页信息
  const pageSize = 10;
  const currentPage = Math.floor(validImgIdx / pageSize) + 1;
  const currentImageIndex = validImgIdx % pageSize;

  try {
    // 等待DOM更新
    await nextTick();

    // 加载图片列表，添加类型检查
    await runImgList({
      sign: route.query?.anoType,
      page: currentPage,
      limit: pageSize,
      state: currentTabVal.value,
      sonId: id?.toString(),  // 确保为字符串类型
      markUserId: markUserId?.toString(),
      taskId: taskId?.toString()
    });

    // 更新当前图片索引
    currentImgIndex.value = currentImageIndex;
  } catch (error) {
    console.error('图片列表加载失败:', error);
  }
});

const currentImageName = computed(() => {
  const image: ImageData | undefined = pagImgList.value.find((img: any) => img.fileId === activeImageId.value);
  // return getFileNameFromUrl(image?.imgPath || '');
  return image?.fileName;
});
const currentImageInvalid = computed(() => {
  const image: ImageData | undefined = pagImgList.value.find((img: any) => img.fileId === activeImageId.value);
  return image?.isInvalid === 0;
});
const currentImageMessage = computed(() => {
  const image: ImageData | undefined = pagImgList.value.find((img: any) => img.fileId === activeImageId.value);
  return image?.notPassMessage;
})
const currentImageResult = computed(() => {
  if (currentTabVal.value === "0") {
    return {
      status: currentImageMessage.value ? 'error' : 'success',
      message: undefined
    }
  }
  if (currentTabVal.value === "1") {
    if (shouldShowButtons.value) {
      return {
        status: currentImageMessage.value ? 'error' : 'success',
        message: !!currentImageMessage.value ? currentImageMessage.value : (route.query.anoType === 'audit' ? '审核通过' : "验收通过")
      }
    } else if (route.query.anoType === 'validateUser') {
      return {
        status: currentImageMessage.value ? 'error' : 'success',
        message: currentImageMessage.value
      }
    } else {
      return {
        status: 'success',
        message: route.query.anoType === 'audit' ? '审核通过' : "验收通过"
      }
    }
  }
  if (currentTabVal.value === "2") {
    if (shouldShowButtons.value) {
      return {
        status: "error",
        message: !!currentImageMessage.value ? currentImageMessage.value : (route.query.anoType === 'audit' ? '审核不通过' : "验收不通过")
      }
    } else if (route.query.anoType === 'validateUser') {
      return {
        status: "error",
        message: currentImageMessage.value
      }
    } else {
      return {
        status: 'success',
        message: route.query.anoType === 'audit' ? '审核通过' : "验收通过"
      }
    }
  }
})

// watch currentImageInvalid
watch(() => currentImageInvalid.value, (newVal) => {
  imgInvalidValue.value = newVal;
  state.isImgInvalid = newVal ? 0 : 1;
}, {
  immediate: true
});

watch(() => imgInvalidValue.value, (newVal) => {
  state.isImgInvalid = newVal ? 0 : 1;
})

// 标签列表
const selectedTagIndex = ref<number>(-1);
const selectedTagId = ref<number | string | undefined>(undefined);
const tags = reactive<Label[]>(mockTagJson);
const tagSearchVal = ref<string | undefined>(undefined);
const tagRowData = ref<any>(null);
const { bool: tagVisible, setTrue: openTagModal } = useBoolean();

const { data, run, current, totalPage, pageSize } = usePagination(
  getSelectGroupLabelPage,
  {
    defaultParams: [
      {
        limit: 10,
        sonId: route.query?.id,
      },
    ],
    pagination: {
      currentKey: "page",
      pageSizeKey: "limit",
      totalKey: "data.total",
    },
  },
);

const pagTagList = computed(
  () =>
    data.value?.data.records.map((val: any, idx: number) => {
      let newIdx = idx + 1;
      if (newIdx >= 10) {
        newIdx = parseInt(String(newIdx).slice(-1));
      }
      return {
        idx: newIdx,
        ...val,
      };
    }) || [],
);

function handlePageChange(page: number) {
  currentImgIndex.value = 0;
}

function searchTag() {
  run({
    page: 1,
    limit: 10,
    sonId: route.query?.id,
    labelName: tagSearchVal.value
  })
}

async function topTag(rowData: any) {
  const res = await topUpLabel({
    labelId: rowData.labelId,
    sonId: route.query?.id,
  });
  if (res.data) {
    tagSearchVal.value = "";
    window.$message?.success?.(`置顶成功`);
    run({
      page: 1,
      limit: 10,
      sonId: route.query?.id,
      labelName: tagSearchVal.value
    })
  }
};

function updateTextTag(feature: any) {
  tagRowData.value = {
    sonId: route.query?.id,
  };
  openTagModal();
}

function tagClick(tagData: any) {
  selectedTagId.value = tagData?.labelId;

  const activeFeature = getActiveFeature();
  activeFeature.props.name = tagData.labelName;
  const { id, props, type: mode, shape, isEye, operateIdx } = activeFeature;
  const drawingStyle = {
    fillStyle: tagData.labelColor,
    strokeStyle: tagData.labelColor,
    fill: true,
    globalAlpha: 0.3,
    lineWidth: 2,
  };

  removeFeature(id);
  removeText(id);
  let feature = createEyeFeature(mode, shape, { isEye, operateIdx }, id, drawingStyle);
  feature.props.textId = tagData.labelId;
  addFeature(feature);
  selectFeature(id);
  createTagTextForFeature(feature, tagData, props.name);

  tagVisible.value = false;
}

// 标注列表
const operateAnoMap: Map<string, any> = new Map();
const operateAnnotations = ref<any>([]);
const annotations = computed(() => getAnnotations());
const activeLayerItem = computed(() => getActiveFeature());

// watch annotations
watch(() => annotations.value, (newVal, oldVal) => {
  if (newVal instanceof Array && newVal.length > 0) {
    newVal = newVal.filter(item => item.type !== 'POINT');
    newVal.forEach((item: any) => {
      operateAnoMap.set(item.id, item);
    })

    let operateAnoList = Array.from(operateAnoMap.values());
    operateAnnotations.value = _.orderBy(operateAnoList, ['operateIdx'], ['asc']);
  } else {
    // operateAnoMap.clear();
    // operateAnnotations.value = [];
  }
});

watch(() => state.allFeatures, (newVal, oldVal) => {
  if (operateAnnotations.value.length == 0 || !activeLayerItem.value) return;
  if (newVal instanceof Array && newVal.length > 0) {
    newVal = newVal.filter(item => item.type !== 'POINT');
    newVal.forEach((item: any) => {
      // if 判断是否存在
      if (operateAnoMap.has(item.id)) {
        const existingItem = operateAnoMap.get(item.id);
        if (!_.isEqual(existingItem, item)) {
          operateAnoMap.set(item.id, item);
        }
      }
    })

    let operateAnoList = Array.from(operateAnoMap.values());
    operateAnnotations.value = _.orderBy(operateAnoList, ['operateIdx'], ['asc']);
  }
}, {
  deep: true
});

/**
 * 检查标注信息中是否存在未设置标签的项
 * @param {Array} markInfo - 标注信息数组
 * @param {string} [propName='name'] - 需要检查的属性名，默认为'name'
 * @param {string} [warningMessage='所有标注必须设置标签！'] - 警告提示信息
 * @returns {boolean} - 是否存在空标签项
 */
function checkEmptyMarkLabels(markInfo: any, propName = 'name') {
  const hasEmpty = markInfo.some((item: any) => {
    return !item.props || !item.props[propName];
  });

  return hasEmpty;
}

// 校验函数：判断标注信息是否有编辑过
function checkMarkInfo(markInfo: any) {
  // return _.isEqual(curDefaultMarkInfo.value, markInfo);
  return _state.isImgRender ? true : _.isEqual(curDefaultMarkInfo.value, markInfo);
}

// 校验函数：判断无效状态是否有变化
function checkInvalid() {
  return currentImageInvalid.value === imgInvalidValue.value;
}

/**
 * 校验当前标注状态
 * @returns 校验结果对象，包含是否通过校验、是否有变更、是否存在空标签等信息
 */
function validateCurrentAnnotation(): {
  isValid: boolean;
  hasChanges: boolean;
  hasEmptyLabels: boolean;
} {
  // 初始化校验结果
  let hasChanges = false;
  let hasEmptyLabels = false;

  // 判断当前是有效标注还是无效标注状态
  if (imgInvalidValue.value) {
    // 无效标注状态下的变更检查
    hasChanges = !checkInvalid();
  } else {
    // 处理标注信息用于校验
    const processedMarkInfo = operateAnnotations.value.map((item: any) => ({
      id: item.id,
      textId: item.props.textId,
      isEye: item.isEye,
      operateIdx: item.operateIdx,
      props: item.props,
      shape: item.shape,
      style: item.style,
      type: item.type
    }));

    // 检查是否有变更
    hasChanges = !checkMarkInfo(processedMarkInfo) || !checkInvalid();

    // 如果有变更，检查是否存在未设置标签的项
    if (hasChanges) {
      hasEmptyLabels = checkEmptyMarkLabels(processedMarkInfo);
    }
  }

  // 校验通过的条件：没有空标签
  const isValid = !hasEmptyLabels;

  return {
    isValid,
    hasChanges,
    hasEmptyLabels
  };
}

/**
 * 保存标注方法
 * 先执行校验，再处理接口调用操作
 */
async function saveAnnotation(isInvalid: boolean, isRefresh: boolean = true, showMessage: boolean = false): Promise<SaveResult> {
  try {
    await nextTick();
    // prev perm
    if (route.query.anoType === 'audit' || route.query.anoType === 'result') {
      return { success: true, hasChanges: false };
    }

    if (state.isImgRender) {
      return { success: true, hasChanges: false };
    }

    // 1. 执行核心校验逻辑
    isBackendSave.value = true;
    const { isValid, hasChanges, hasEmptyLabels } = validateCurrentAnnotation();

    // 1.1 检查是否有未设置标签的项
    if (hasEmptyLabels) {
      window.$message?.warning?.("所有标注必须设置标签！");
      isBackendSave.value = false;
      return { success: false, hasChanges: false };
    }

    // 1.2 检查是否无效且无变更
    if (!isValid || !hasChanges) {
      isBackendSave.value = false;

      if (!hasChanges) {
        showMessage && window.$message?.info?.("未检测到变更，无需保存");
        return { success: true, hasChanges: false };
      }

      return { success: false, hasChanges };
    }

    // 2. 校验通过后，执行文件上传等前置操作
    const currentImg: CurrentImage = pagImgList.value[currentImgIndex.value];
    if (!currentImg) {
      throw new Error("当前图片不存在");
    }

    // 获取文件后缀
    const fileName = getFileNameFromUrl(currentImg?.imgPath || '');
    const fileSuffix = fileName.split(".").pop() || '';

    // 准备并上传表单数据
    const formData = await getAnnotationFormData(
      route.query.id as string,
      currentImg.version,
      fileSuffix
    );

    const uploadResponse = await MarkFileUpload(formData);
    if (!uploadResponse?.data) {
      window.$message?.error?.("文件上传失败");
      return { success: false, hasChanges: true };
    }

    // 3. 处理标注信息
    let markInfo = "";
    let labels = "";

    if (!isInvalid) {
      // 处理标注信息，添加必要属性
      const processedMarkInfo: AnnotationItem[] = operateAnnotations.value.map((item: any) => ({
        id: item.id,
        textId: item.props.textId,
        isEye: item.isEye,
        operateIdx: item.operateIdx,
        props: Object.assign({}, item.props, {
          operateWidth: item.props.operateWidth ?? _state.currentViewWidth,
          operateHeight: item.props.operateHeight ?? _state.currentViewHeight,
        }),
        shape: item.shape,
        style: item.style,
        type: item.type
      }));


      markInfo = JSON.stringify(processedMarkInfo);

      // 提取唯一标签并拼接
      const uniqueTextIds = new Set(processedMarkInfo.map(item => item.textId));
      labels = Array.from(uniqueTextIds).join(",");
    }

    // 4. 准备保存参数并调用保存接口
    const params = {
      markInfo,
      labels,
      fileId: currentImg.fileId,
      isInvalid: imgInvalidValue.value ? 0 : 1,
      sonId: route.query.id,
      markUserId: route.query?.markUserId,
      taskId: route.query?.taskId,
      operateWidth: _state.currentViewWidth,
      operateHeight: _state.currentViewHeight,
    };

    const res = await addDataMarkInfo(params);
    if (!res?.data) {
      window.$message?.error?.("标注信息保存失败");
      return { success: false, hasChanges: true };
    }

    // 保存成功后的处理
    isBackendSave.value = false;
    clearCurrentAnnotation();

    // 需要刷新时更新列表
    if (isRefresh) {
      const commonParams = {
        sonId: route.query?.id,
        markUserId: route.query?.markUserId,
        taskId: route.query?.taskId,
      };

      // 刷新图片列表
      await runImgList({
        ...commonParams,
        state: currentTabVal.value,
        page: imgListCurrent.value,
        limit: 10,
        sign: route.query?.anoType,
      });

      // 刷新标签列表
      await run({
        ...commonParams,
        page: 1,
        limit: 10,
        labelName: tagSearchVal.value
      });
    }

    window.$message?.success?.(isInvalid ? "标注无效成功" : "标注保存成功");

    const markInfoLen = !!markInfo ? JSON.parse(markInfo).length : 0;
    return { success: true, hasChanges: true, markInfoLen };
  } catch (error) {
    console.error("保存标注时发生错误:", error);
    window.$message?.error?.(`保存失败: ${error instanceof Error ? error.message : '未知错误'}`);
    isBackendSave.value = false;
    return { success: false, hasChanges: true };
  }
}

// 保存当前标注并处理图片切换逻辑
async function saveCurrentAnnotation() {
  try {
    const listParams = {
      state: currentTabVal.value,
      page: imgListCurrent.value,
      limit: 10,
      sonId: route.query?.id,
      markUserId: route.query?.markUserId,
      taskId: route.query?.taskId,
      sign: route.query?.anoType,
    };

    const { success, hasChanges, markInfoLen } = await saveAnnotation(imgInvalidValue.value as unknown as boolean, false);

    if (success) {
      let increment = 0;
      if (currentTabVal.value == 1) {
        if (hasChanges) {
          increment = markInfoLen != 0 ? 1 : 0;
        }
        if (imgInvalidValue.value) {
          increment = 0;
        }
        if (!hasChanges) {
          increment = 1;
        }
      } else if (currentTabVal.value == 2) {
        if (success) {
          increment = 0;
        }
        if (!hasChanges) {
          increment = 1;
        }
      } else if (currentTabVal.value == 3) {
        if (!imgInvalidValue.value) {
          increment = 0;
        } else {
          increment = 1;
        }
      } else {
        increment = 1;
      }


      // 兜底
      if (currentImgIndex.value == pagImgList.value.length - 1 && imgListCurrent.value === imgListTotalPage.value) {
        increment--;
      }

      listParams.page = currentImgIndex.value + increment < 0 ? imgListCurrent.value - 1 : imgListCurrent.value;
      let newIndex = currentImgIndex.value + increment < 0 ? 9 : currentImgIndex.value + increment;

      if (newIndex < pagImgList.value.length) {
        await setupImage(pagImgList.value[newIndex], newIndex);
        if (hasChanges) {
          await runImgList(listParams);
          await anoTabRef.value?.getDataDetailsCount();
        }
      } else {
        if (imgListCurrent.value < imgListTotalPage.value) {
          await runImgList({ ...listParams, page: imgListCurrent.value + 1 });
          await anoTabRef.value?.getDataDetailsCount();
          currentImgIndex.value = 0;
          clearAnnotations();
        } else {
          currentImgIndex.value = 9;
          imgInvalidValue.value = currentImageInvalid.value;
          await runImgList(listParams);
          window.$message?.info('已经是最后一张图片');
          clearAnnotations();
        }
      }

      // await anoTabRef.value?.getDataDetailsCount();
      // await run({
      //   page: 1,
      //   limit: 10,
      //   sonId: route.query?.id,
      //   labelName: tagSearchVal.value
      // })
    } else {
      console.log("标注保存失败");
      // 可添加失败处理逻辑（如重试提示等）
    }
  } catch (error) {
    // 虽然函数内部已处理异常，但仍可捕获可能的意外错误
    console.error("调用保存函数时发生错误：", error);
  }
}

// 标记为无效标注
async function markInvalid() {
  imgInvalidValue.value = true;
  saveCurrentAnnotation();
}

function clearCurrentAnnotation() {
  clearAnnotations();
  operateAnoMap.clear();
  operateAnnotations.value = [];
}

/**
 * 加载图片并设置标注信息
 * @param image 图片信息对象
 * @param imageIndex 图片索引
 */
async function setupImage(image: any, imageIndex: number): Promise<void> {
  // 设置当前图片ID和索引
  if (image.fileId !== null) {
    await loadImage(image.imgPath);
    activeImageId.value = image.fileId;
    currentImgIndex.value = imageIndex;
    clearCurrentAnnotation();
    imgInvalidValue.value = image.isInvalid === 0;
  }

  operateAnoMap.clear();

  // 处理标注信息
  if (image.markInfo) {
    try {
      const annotations: any[] = JSON.parse(image.markInfo);
      renderCustomAnnotations(annotations);

      // 等待DOM更新后处理当前标注信息
      await nextTick();

      curDefaultMarkInfo.value = operateAnnotations.value.map((item: any) => ({
        id: item.id,
        textId: item.props.textId,
        isEye: item.isEye,
        operateIdx: item.operateIdx,
        props: item.props,
        shape: item.shape,
        style: item.style,
        type: item.type
      }));
    } catch (error) {
      console.error("处理标注信息时发生错误:", error);
      // 可以根据需要添加错误处理逻辑
    }
  }
}

const selectImage = _.throttle(async function selectImage(image: ImageData, imageIndex: number, isRefresh: boolean = true) {
  if (_state.isImgRender) {
    window.$message?.warning('图片加载中，请稍候再进行操作!');
    return;
  }

  try {
    // 1. 先保存当前标注
    const { success } = await saveAnnotation(imgInvalidValue.value as unknown as boolean, isRefresh, false);

    // 2. 如果保存失败，给出提示并终止执行
    if (!success) {
      console.error('保存当前标注失败，无法切换图像');
      return;
    }

    // 3. 保存成功，准备切换图像
    if (image.fileId !== null) {
      await loadImage(image.imgPath);
      activeImageId.value = image.fileId;
      currentImgIndex.value = imageIndex;
      clearCurrentAnnotation();
      imgInvalidValue.value = image.isInvalid === 0;
    }

    await nextTick();

    // 4. 清除旧标注并准备加载新标注
    operateAnoMap.clear();
    operateAnnotations.value = [];

    // 5. 如果有标注信息，加载并渲染
    if (image.markInfo) {
      try {
        let annotations: any[] = JSON.parse(image.markInfo);
        renderCustomAnnotations(annotations);
        await nextTick();

        // 7. 更新当前标注信息
        curDefaultMarkInfo.value = operateAnnotations.value.map((item: any) => ({
          id: item.id,
          textId: item.props.textId,
          isEye: item.isEye,
          operateIdx: item.operateIdx,
          props: item.props,
          shape: item.shape,
          style: item.style,
          type: item.type
        }));
      } catch (error) {
        console.error('解析或渲染标注失败:', error);
      }
    }
  } catch (error) {
    console.error('图片选择失败:', error);
  }
}, 350)

// 切换标注可见性
function toggleEye(annotation: any) {
  const { id, type: mode, shape, style: drawingStyle, isEye, operateIdx } = annotation;

  annotation.isEye = !isEye;
  operateAnoMap.set(id, annotation);

  let operateAnoList = Array.from(operateAnoMap.values());
  operateAnnotations.value = _.orderBy(operateAnoList, ['operateIdx'], ['asc']);

  if (isEye) {
    removeFeature(annotation.id);
    removeText(annotation.id);
  } else {
    let feature = createEyeFeature(mode, shape, { isEye: true, operateIdx }, id, drawingStyle);
    addFeature(feature);
    createTagTextForFeature(feature, pagTagList.value[selectedTagIndex.value], annotation.props.name);
  }
}

onMounted(() => {
  document.addEventListener('keydown', handleNumberKeyPress);
  document.addEventListener('keydown', handleDeleteKeyPress);
  document.addEventListener('keydown', handleSaveKeyPress);
  document.addEventListener('keydown', handleUndoRedoKeyPress);
  document.addEventListener('keydown', handleImageIndexNavigation);
  document.addEventListener('keydown', handleZoomKeys);
  document.addEventListener('keydown', handleEscapeKey);
  document.addEventListener('keydown', handleArrowKeys);
});

onUnmounted(() => {
  // 已有的事件移除
  document.removeEventListener('keydown', handleNumberKeyPress);
  document.removeEventListener('keydown', handleDeleteKeyPress);
  document.removeEventListener('keydown', handleSaveKeyPress);
  document.removeEventListener('keydown', handleUndoRedoKeyPress);
  document.removeEventListener('keydown', handleImageIndexNavigation);
  document.removeEventListener('keydown', handleZoomKeys);
  document.removeEventListener('keydown', handleEscapeKey);
  document.removeEventListener('keydown', handleArrowKeys);
});

/**
 * 为选中的标注框创建标签文本
 * @param feature 选中的标注框对象
 * @param tag 标签数据对象
 */
function createTagTextForFeature(feature: any, tag: Label, textName?: string) {
  if (hasText(`${feature.id}`)) {
    removeText(`${feature.id}`);
  };

  // 获取标注框位置用于文本定位
  const textPosition = getLabelTextPosition(feature.shape, feature.type);

  // 创建文本标注
  const text = createText(
    `${feature.id}`, // 使用标注框ID确保唯一性
    textName ? textName : tag.labelName,        // 标签名称
    textPosition          // 文本位置
  );

  // 添加文本到图层
  addText(text);

  // 更新标注框属性
  feature.props.name = textName ? textName : tag.labelName;
  operateAnoMap.set(feature.id, feature);

  // 刷新标注列表
  operateAnnotations.value = Array.from(operateAnoMap.values());
}

/**
 * 处理数字键按下事件，为选中的标注框添加对应标签文本
 * @param e 键盘事件对象
 */
function handleNumberKeyPress(e: KeyboardEvent) {
  // 忽略输入框等可编辑元素中的按键
  if (['INPUT', 'TEXTAREA', 'SELECT'].includes(document.activeElement?.tagName || '')) {
    return;
  }

  // 检查是否是数字键0-9
  if (/^[0-9]$/.test(e.key)) {
    const keyNumber = parseInt(e.key, 10);
    const tagIndex: number = keyNumber === 0 ? 9 : keyNumber - 1; // 0对应第10个标签(索引9)
    const activeFeature = getActiveFeature();

    // 验证条件：有选中的标注框且标签索引有效
    if (activeFeature && tagIndex >= 0 && tagIndex < tags.length) {
      selectedTagIndex.value = tagIndex;
      selectedTagId.value = pagTagList.value[selectedTagIndex.value]?.labelId;

      const activeFeature = getActiveFeature();
      activeFeature.props.name = pagTagList.value[selectedTagIndex.value].labelName;
      const { id, type: mode, shape, isEye, operateIdx } = activeFeature;
      const drawingStyle = {
        fillStyle: pagTagList.value[selectedTagIndex.value].labelColor,
        strokeStyle: pagTagList.value[selectedTagIndex.value].labelColor,
        fill: true,
        globalAlpha: 0.3,
        lineWidth: 2,
      };

      removeFeature(id);
      removeText(id);

      let feature = createEyeFeature(mode, shape, { isEye, operateIdx }, id, drawingStyle);
      feature.props.textId = pagTagList.value[selectedTagIndex.value].labelId;
      addFeature(feature);
      selectFeature(id);
      createTagTextForFeature(feature, pagTagList.value[selectedTagIndex.value]);
    } else {
      // 添加失败提示
      window.$message?.warning('请选择有效的标注框和标签');
    }
  }
}

// removeFeatureAndText
function removeFeatureAndText(id: string) {
  removeFeature(id);
  removeText(id);
  // map delete
  operateAnoMap.delete(id);
  let operateAnoList = Array.from(operateAnoMap.values());
  operateAnnotations.value = _.orderBy(operateAnoList, ['operateIdx'], ['asc']);
}

/**
 * 处理删除键事件
 * @param e 键盘事件对象
 */
function handleDeleteKeyPress(e: KeyboardEvent) {
  // 忽略输入框等可编辑元素中的按键
  if (['INPUT', 'TEXTAREA', 'SELECT'].includes(document.activeElement?.tagName || '')) {
    return;
  }

  // 检查是否是删除键
  if (e.key === 'Delete' || e.key === 'Backspace') {
    const activeFeature = getActiveFeature();
    if (activeFeature) {
      removeFeatureAndText(activeFeature.id);
      window.$message?.success('已删除选中标注');
    } else {
      window.$message?.warning('请先选择要删除的标注框');
    }
  }
}
/**
 * 保存快捷键处理 (Ctrl+S)
 */
function handleSaveKeyPress(e: KeyboardEvent) {
  // 审核
  if (route.query.anoType === 'audit') return;

  // 忽略输入框等可编辑元素中的按键
  if (['INPUT', 'TEXTAREA', 'SELECT'].includes(document.activeElement?.tagName || '')) {
    return;
  }

  if (e.ctrlKey && e.key === 's') {
    e.preventDefault(); // 阻止浏览器默认保存行为
    saveCurrentAnnotation();
  }
}

/**
 * 撤销/重做快捷键处理 (Ctrl+Z/Ctrl+Y)
 */
function handleUndoRedoKeyPress(e: KeyboardEvent) {
  if (route.query.anoType === 'audit') return;
  // 忽略输入框等可编辑元素中的按键
  if (['INPUT', 'TEXTAREA', 'SELECT'].includes(document.activeElement?.tagName || '')) {
    return;
  }

  operateAnoMap.clear();

  // 撤销 (Ctrl+Z)
  if (e.ctrlKey && e.key === 'z') {
    e.preventDefault();
    undo();
    getAnnotations();
  }
  // 重做 (Ctrl+Y)
  else if (e.ctrlKey && e.key === 'y') {
    e.preventDefault();
    redo();
    getAnnotations();
  }
}

//  + -快捷键
function handleZoomKeys(e: KeyboardEvent) {
  // 忽略输入框等可编辑元素中的按键
  if (['INPUT', 'TEXTAREA', 'SELECT'].includes(document.activeElement?.tagName || '')) {
    return;
  }

  // 处理+/-快捷键（支持数字键盘和主键盘）
  if (e.key === '+' || e.key === '=') {
    e.preventDefault();
    zoomIn();
  } else if (e.key === '-') {
    e.preventDefault();
    zoomOut();
  }
};

function handleEscapeKey(e: KeyboardEvent) {
  if (e.key === 'Escape') {
    setMode('PAN');
  }
}

//  A D 键 图片导航
async function handleImageIndexNavigation(e: KeyboardEvent) {
  // 忽略输入框等可编辑元素中的按键
  if (['INPUT', 'TEXTAREA', 'SELECT'].includes(document.activeElement?.tagName || '')) {
    return;
  }

  // if (_state.isImgRender) {
  //   window.$message?.warning('图片加载中，请稍候再进行操作!');
  //   return;
  // }

  const anoType = route.query.anoType;
  const isAudit = !anoType || ['validateUser', 'online', 'setOnline'].includes(anoType);
  const isAuditAndTab2 = isAudit && currentTabVal.value == 4; // 2
  // 仅处理A和D键，不与Ctrl/Shift等组合
  if (!e.ctrlKey && !e.shiftKey && !e.altKey) {
    const key = e.key.toLowerCase();
    // 确保图片列表有数据
    if (pagImgList.value.length === 0) return;

    // A键 - 上一张图片
    if (key === 'a') {
      if (_state.isImgRender) {
        window.$message?.warning('图片加载中，请稍候再进行操作!');
        return;
      }
      e.preventDefault();
      // const newIndex = currentImgIndex.value - 1;
      const newIndex = !isAuditAndTab2 ? currentImgIndex.value - 1 : currentImgIndex.value;
      if (newIndex >= 0) {
        // 普通导航到上一张
        await selectImage(pagImgList.value[newIndex], newIndex, false);
        await runImgList({
          state: currentTabVal.value,
          page: imgListCurrent.value,
          limit: 10,
          sonId: route.query?.id,
          markUserId: route.query?.markUserId,
          taskId: route.query?.taskId,
          sign: route.query?.anoType,
        });
      } else {
        // 尝试加载上一页
        if (imgListCurrent.value > 1) {
          const isRefresh = false;
          const { success } = await saveAnnotation(imgInvalidValue.value as unknown as boolean, isRefresh, false);
          if (success) {
            await runImgList({
              state: currentTabVal.value,
              page: imgListCurrent.value - 1,
              limit: 10,
              sonId: route.query?.id,
              markUserId: route.query?.markUserId,
              taskId: route.query?.taskId,
              sign: route.query?.anoType,
            });
            currentImgIndex.value = 9;
            clearAnnotations();
          }
        } else {
          await selectImage(pagImgList.value[0], 0, false);
          await runImgList({
            state: currentTabVal.value,
            page: imgListCurrent.value,
            limit: 10,
            sonId: route.query?.id,
            markUserId: route.query?.markUserId,
            taskId: route.query?.taskId,
            sign: route.query?.anoType,
          });
          window.$message?.info('已经是第一张图片');
        }
      }
    }
    // D键 - 下一张图片
    else if (key === 'd') {
      e.preventDefault();
      if (_state.isImgRender) {
        window.$message?.warning('图片加载中，请稍候再进行操作!');
        return;
      }
      saveCurrentAnnotation();
    }

    // 刷新
    // await anoTabRef.value?.getDataDetailsCount();
    // await run({
    //   page: 1,
    //   limit: 10,
    //   sonId: route.query?.id,
    //   labelName: tagSearchVal.value
    // })

  } else if (e.ctrlKey) {
    if (route.query.anoType === 'audit') return;
    // 保持原有的Ctrl+D复制逻辑
    const activeFeature = getActiveFeature();
    if (!activeFeature) return;

    const { props, type: mode, style: drawingStyle, shape } = activeFeature;
    const timestamp = Date.now();
    const randomId = nanoid();
    const id = `feature-${randomId}-${timestamp}`;
    let copyShape = getCopyShape(mode, shape);

    if (e.key === 'd') {
      e.preventDefault();
      let feature = createEyeFeature(mode, copyShape, { isEye: true, operateIdx: new Date().getTime() }, id, drawingStyle);
      feature.props.name = props.name;
      addFeature(feature);
      const textPosition = getLabelTextPosition(feature.shape, feature.type) as any;
      let text = createText(id, feature.props.name, { x: textPosition.x, y: textPosition.y });
      addText(text);
    }
  }
}

function handleArrowKeys(e: KeyboardEvent) {
  // 忽略输入框等可编辑元素中的按键
  if (['INPUT', 'TEXTAREA', 'SELECT'].includes(document.activeElement?.tagName || '')) {
    return;
  }

  // 只有键盘的上下左右才能执行下面的逻辑
  if (!['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight'].includes(e.key)) {
    return;
  }

  // feature
  const activeFeature = getActiveFeature();
  if (activeFeature !== null) {
    const { type, shape, props } = activeFeature;
    const opShape = getOpShape(type, shape, 1, e.key as opState);
    activeFeature.updateShape(opShape);
    // text
    const activeText = _textLayer.value.getTextById(activeFeature.id);
    const textShape = getTextShapePosition(type, opShape);
    activeText.updatePosition(textShape);
  }
}

// ------------------------------------------------------------------
// 定义标签数量类型
interface TabNumbers {
  all: number;
  haveAno: number;
  noAno: number;
  invalid: number;
}

// 定义标签配置类型
interface AnoTabsConfig {
  tabNum: TabNumbers;
}

const activeTab = ref('0');
const anoTabRef = ref<HTMLDivElement | null>(null);

// validate
const isShowValidate = ref<Boolean>(false);
const validateStatus = ref<String>("0");
const validateModel = ref<any>({
  verifyState: null,
  returnState: "",
  message: null,
});
const statusOptions = ref([
  { value: "1", label: "保存全部数据 " },
  { value: "2", label: "仅保存验收通过的数据 " },
]);
const repulseOptions = ref([
  { value: "1", label: "未验收的数据 " },
  { value: "2", label: "验收不通过的数据 " },
  { value: "3", label: "未验收+验收不通过的数据" },
]);

const tabConfig = ref<AnoTabsConfig>({
  tabNum: {
    all: 0,
    haveAno: 0,
    noAno: 0,
    invalid: 0
  }
});

const noPassVisable = ref<Boolean>(false);

const validateTitle = computed(() => {
  if (validateStatus.value === "0") {
    return "剩余验收通过";
  } else if (validateStatus.value === "1") {
    return "验收完成";
  } else if (validateStatus.value === "2") {
    return "打回任务";
  }
});

const curTabNum = ref<number>(0);

const shouldShowButtons = computed(() => {
  const anoType = route.query.anoType;
  const isValidAnoType = anoType === "validate" || anoType === "audit";
  return isValidAnoType && !!curTabNum.value;
});

// 验收结果
const shouldShowResult = computed(() => {
  const anoType = route.query.anoType;
  const showTypes: any[] = ['validate', 'validateUser', 'audit'];
  return showTypes.includes(anoType) && pagImgList.value.length;
});

const handleTabChange = async (tab: string) => {
  currentTabVal.value = tab;
  currentImgIndex.value = 0;
  operateAnnotations.value = [];
  operateAnoMap.clear();
  clearAnnotations();

  if (pagImgList.value.length === 0) {
    console.log('null');
  }

  await runImgList({
    state: tab,
    page: 1,
    limit: 10,
    sonId: route.query?.id,
    markUserId: route.query?.markUserId,
    taskId: route.query?.taskId,
    sign: route.query?.anoType,
  })
};

// 打开验证模态框
const openValidateModal = () => {
  // 显示模态框的逻辑
  isShowValidate.value = true;
};

// 刷新数据
const refreshAllData = async () => {
  // 调用相关数据刷新方法
  currentImgIndex.value = 0;
  operateAnnotations.value = [];
  operateAnoMap.clear();
  await runImgList({
    state: currentTabVal.value,
    page: 1,
    limit: 10,
    sonId: route.query?.id,
    markUserId: route.query?.markUserId,
    taskId: route.query?.taskId,
    sign: route.query?.anoType,
  });
  await anoTabRef.value?.getDataDetailsCount();
  await run({
    page: 1,
    limit: 10,
    sonId: route.query?.id,
    labelName: tagSearchVal.value
  })
};

const handleModalDefine = async (sign: string) => {
  switch (sign) {
    case "validate":
      await handleValidateSuccess();
      break;
    case "pass":
      await handlePassSuccess();
      break;
    default:
      break;
  }
};

const handleModalCancel = (sign: string) => {
  if (sign === "validate") {
    isShowValidate.value = false;
  }
  if (sign === "pass") {
    noPassVisable.value = false;
    handleApprove("2");
  }
  if (sign === "pass0") {
    noPassVisable.value = false;
    validateModel.value.message = undefined;
    handleApprove("2");
  }
};

const handleValidateSuccess = async () => {
  if (validateStatus.value === "1") {
    const res = await verifyComplete({
      taskId: route.query?.taskId,
      verifyState: validateModel.value.verifyState,
    });
    if (res.data) {
      window.$message?.success?.(`操作成功`);
      router.back();
    }
  }
  if (validateStatus.value === "2") {
    const res = await returnTask({
      taskId: route.query?.taskId,
      returnState: validateModel.value.returnState,
      id: route.query?.markUserId,
    });
    if (res.data) {
      window.$message?.success?.(`操作成功`);
      router.back();
    }
  }
};

const handlePassSuccess = async () => {
  noPassVisable.value = false;
  handleApprove("2", "验收意见保存成功！");
};

const handlePass = async (sign: string) => {
  switch (sign) {
    case "1":
      await handleApprove("1");
      break;
    case "2":
      noPassVisable.value = true;
      break;
    default:
      break;
  }
};

const handleApprove = async (
  sign: string | number,
  tooltipText: string = "审核完成!",
) => {
  try {
    const params = {
      taskId: route.query?.taskId,
      isApprove: sign,
      fileId: pagImgList.value[currentImgIndex.value]?.fileId,
      notPassMessage: validateModel.value.message,
    };

    const res = await fileIsApprove(params);

    if (res.data) {
      validateModel.value.message = undefined;
      if (tooltipText) {
        window.$message?.success?.(tooltipText);
      }

      const listParams = {
        state: currentTabVal.value,
        page: imgListCurrent.value,
        limit: 10,
        sonId: route.query?.id,
        markUserId: route.query?.markUserId,
        taskId: route.query?.taskId,
        sign: route.query?.anoType,
      };
      let increment =
        currentTabVal.value == 1 ? (sign == 2 ? 0 : 1) :
          currentTabVal.value == 2 ? (sign == 1 ? 0 : 1) :
            0;

      // 兜底
      if (currentImgIndex.value == pagImgList.value.length - 1 && imgListCurrent.value === imgListTotalPage.value) {
        increment--;
      }

      listParams.page = currentImgIndex.value + increment < 0 ? imgListCurrent.value - 1 : imgListCurrent.value;
      let newIndex = currentImgIndex.value + increment < 0 ? 9 : currentImgIndex.value + increment;
      if (newIndex < pagImgList.value.length) {
        await setupImage(pagImgList.value[newIndex], newIndex);
        await runImgList(listParams);
      } else {
        if (imgListCurrent.value < imgListTotalPage.value) {
          await runImgList({ ...listParams, page: imgListCurrent.value + 1 });
          currentImgIndex.value = 0;
          clearAnnotations();
        } else {
          currentImgIndex.value = 9;
          await runImgList(listParams);
          window.$message?.info('已经是最后一张图片');
          clearAnnotations();
        }
      }

      await anoTabRef.value?.getDataDetailsCount();
      await run({
        page: 1,
        limit: 10,
        sonId: route.query?.id,
        labelName: tagSearchVal.value
      })
    }
  } catch (error) {
    console.error("Error handling approve:", error);
  }
};

const handleBack = () => {
  router.back();
}

// utils.ts
type opState = "ArrowUp" | "ArrowDown" | "ArrowLeft" | "ArrowRight";
type shapeType = "POLYGON" | "RECT" | "CIRCLE";

function getStepX(state: opState, step: string | number = 1) {
  let opStep;
  if (state === "ArrowLeft") {
    opStep = -step;
  }
  if (state === "ArrowRight") {
    opStep = step;
  }
  return opStep;
}

function getStepY(state: opState, step: string | number = 1) {
  let opStep;
  if (state === "ArrowUp") {
    opStep = -step;
  }
  if (state === "ArrowDown") {
    opStep = step;
  }
  return opStep;
}

function getOpShape(
  type: shapeType,
  shape: any,
  step: string | number = 1,
  state: opState,
) {
  let stepX = getStepX(state, step) ?? 0;
  let stepY = getStepY(state, step) ?? 0;
  switch (type) {
    case "POLYGON":
      return {
        points: shape.points.map((val: any) => {
          return {
            x: val.x + stepX,
            y: val.y + stepY,
          };
        }),
      };
    case "RECT":
      return {
        width: shape.width,
        height: shape.height,
        x: shape.x + stepX,
        y: shape.y + stepY,
      };
    case "CIRCLE":
      return {
        cx: shape.cx + stepX,
        cy: shape.cy + stepY,
        r: shape.r,
      };
    default:
      return {};
  }
}

/**
 * 根据形状类型获取文本位置
 * @param {string} type - 形状类型 (CIRCLE, POLYGON, RECT等)
 * @param {Object} opShape - 形状对象
 * @returns {Object} 文本位置 {x, y}
 */
function getTextShapePosition(type: string, opShape: any) {
  const positionMap: any = {
    CIRCLE: { x: opShape.cx, y: opShape.cy - opShape.r },
    POLYGON: {
      x: opShape.points?.[0]?.x || '',
      y: opShape.points?.[0]?.y || ''
    },
    RECT: { x: opShape.x, y: opShape.y }
  };

  // 返回对应类型的位置，默认返回空值
  return positionMap[type] || { x: '', y: '' };
}

function getCopyShape(type: shapeType, shape: any, step: string | number = 50) {
  switch (type) {
    case "POLYGON":
      return {
        points: shape.points.map((val: any) => {
          return {
            x: val.x + step,
            y: val.y,
          };
        }),
      };
    case "RECT":
      return {
        width: shape.width,
        height: shape.height,
        x: shape.x + step,
        y: shape.y,
      };
    case "CIRCLE":

      return {
        cx: shape.cx + step,
        cy: shape.cy,
        r: shape.r,
      };
    default:
      return {};
  }
}

/**
 * 从URL中提取文件名及后缀
 * @param url 完整的资源URL
 * @returns 提取到的文件名+后缀，失败时返回空字符串
 */
function getFileNameFromUrl(url: string): string {
  try {
    // 创建URL对象解析路径
    const urlObj = new URL(url);
    const pathname = urlObj.pathname;

    // 分割路径并过滤空字符串
    const pathSegments = pathname.split('/').filter(segment => segment);

    // 返回最后一个有效路径段作为文件名
    return pathSegments.length > 0 ? pathSegments[pathSegments.length - 1] : '';
  } catch (error) {
    console.error('解析URL失败:', error);
    return '';
  }
}
</script>

<style type="text/tailwindcss">
@layer utilities {
  .toolbar-btn {
    @apply flex items-center justify-center p-3 rounded-full hover:bg-primary/10 text-gray-700 hover:text-primary transition-all duration-200 relative mx-0.5 transform hover:scale-110 active:scale-95;
  }

  .toolbar-btn.active {
    @apply bg-primary/10 text-primary shadow-inner;
  }

  .toolbar-btn span {
    @apply absolute bottom-full left-1/2 transform -translate-x-1/2 mb-2 bg-white shadow-md rounded-md px-2 py-1 text-xs text-gray-700 opacity-0 transition-all duration-200 whitespace-nowrap pointer-events-none;
  }

  .toolbar-btn:hover span {
    @apply opacity-100;
  }

  .annotation-area {
    @apply relative w-full h-full bg-gray-100 rounded-lg border border-gray-200 overflow-hidden grid grid-cols-[25%_50%_25%];
  }

  .annotation-toolbar {
    @apply z-50;
  }

  .ailabel-container {
    @apply absolute inset-0 z-10;
  }

  .zoom-controls {
    @apply absolute bottom-4 right-4 bg-white/90 backdrop-blur-sm rounded-lg shadow-md border border-gray-100 p-1.5 flex flex-col items-center transition-all duration-300 hover:shadow-lg z-20;
  }

  .zoom-btn {
    @apply w-8 h-8 flex items-center justify-center text-gray-700 hover:text-primary transition-all duration-200 rounded-md hover:bg-primary/10 relative;
  }

  .zoom-btn span {
    @apply absolute right-full top-1/2 transform -translate-y-1/2 mr-2 bg-white shadow-md rounded-md px-2 py-1 text-xs text-gray-700 opacity-0 transition-all duration-200 whitespace-nowrap pointer-events-none;
  }

  .zoom-btn:not(:last-child) {
    @apply mb-1 border-b border-gray-100;
  }

  .operate-controls {
    @apply absolute top-4 right-4 bg-white/90 backdrop-blur-sm rounded-lg shadow-md border border-gray-100 p-1.5 flex flex-col items-center transition-all duration-300 hover:shadow-lg z-20;
  }

  .operate-btn {
    @apply w-8 h-8 flex items-center justify-center text-gray-700 hover:text-primary transition-all duration-200 rounded-md hover:bg-primary/10 relative;
  }

  .operate-btn span {
    @apply absolute right-full top-1/2 transform -translate-y-1/2 mr-2 bg-white shadow-md rounded-md px-2 py-1 text-xs text-gray-700 opacity-0 transition-all duration-200 whitespace-nowrap pointer-events-none;
  }

  .operate-btn:not(:last-child) {
    @apply mb-1 border-b border-gray-100;
  }

  .panel-title {
    @apply bg-gray-50 text-gray-700 font-medium px-4 py-3 border-b border-gray-200 flex items-center justify-between sticky top-0 z-10;
  }

  .panel-content {
    @apply p-4 overflow-y-auto;
  }

  .image-grid {
    @apply grid grid-cols-2 gap-3;
  }

  .image-item {
    @apply rounded-md overflow-hidden cursor-pointer transition-all duration-200 hover:shadow-md border border-transparent hover:border-primary;
  }

  .label-item {
    @apply p-2 mb-2 rounded-md flex items-center justify-between cursor-pointer transition-all duration-200 hover:bg-gray-50 border border-gray-200;
  }

  .label-item span {
    @apply overflow-hidden text-ellipsis whitespace-nowrap;
  }

  .pagination-btn {
    @apply w-8 h-8 flex items-center justify-center rounded-md border border-gray-200 text-gray-600 hover:bg-primary hover:text-white hover:border-primary transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed;
  }

  .panel-border {
    @apply relative;
  }

  .panel-border::after {
    content: '';
    @apply absolute top-0 bottom-0 w-[1px] bg-gray-200;
  }

  .left-panel::after {
    @apply right-0;
  }

  /* .right-panel::after {
    @apply left-0;
  } */
}

.invalid-tip {
  position: absolute;
  z-index: 1;
  left: 0;
  top: 0;
  width: 160px;
  height: 160px;
  background-color: rgba(0, 0, 0, 0.4);
  color: #fff;
  font-size: 32px;
  padding: 40px;
  text-align: center;
  line-height: 40px;
  z-index: 9999;

  &:before {
    position: absolute;
    content: " ";
    width: 40px;
    height: 20px;
    bottom: -80px;
    overflow: hidden;
    left: 0;
    border-left: 0;
    border-right: 80px solid transparent;
    border-top: 40px solid rgba(0, 0, 0, 0.4);
    border-bottom: 40px solid transparent;
  }

  &:after {
    position: absolute;
    content: " ";
    width: 40px;
    height: 20px;
    bottom: -80px;
    overflow: hidden;
    right: 0;
    border-left: 80px solid transparent;
    border-right: 0;
    border-top: 40px solid rgba(0, 0, 0, 0.4);
    border-bottom: 40px solid transparent;
  }
}

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
</style>
