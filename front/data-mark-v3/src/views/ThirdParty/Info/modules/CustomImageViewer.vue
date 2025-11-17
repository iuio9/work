<template>
  <div>
    <!-- 自定义查看器 -->
    <transition name="fade">
      <div v-if="visible" class="custom-viewer" @click.self="hideViewer">
        <div class="viewer-content flex-col justify-center items-center gap-24px">
          <div class="dual-images flex items-center my-48px relative">
            <div class="absolute right-8px -top-36px" @click="hideViewer">
              <SvgIcon localIcon="ThirdParty_Close" class="text-icon text-36px" />
            </div>
            <div class="image-side original flex-col justify-center items-center gap-14px">
              <h3>原始图片</h3>
              <!--<img :src="currentImage.original" alt="Original Image" />-->
              <NImage
                width="480"
                :src="currentImage.original"
              />
            </div>
            <div class="item_arrow">
              <div class="flow-arrow"><span class="aibp-custom-icon aibp-custom-icon-arrow">
                  <svg width="24" height="24"><path fill="#B8BABF" d="m8.053 3 9.192 9.192L8 21.437v-5.253l3.79-3.79L8 8.603V3.052L8.053 3Z"></path></svg></span>
              </div>
            </div>
            <div class="image-side effect flex-col justify-center items-center gap-14px">
              <h3>效果图</h3>
              <!--<img :src="currentImage.effect" alt="Effect Image" />-->
              <NImage
                width="480"
                :src="currentImage.effect"
              />
            </div>
          </div>
          <!-- 插槽用于插入自定义内容 -->
          <slot :image="currentImage"></slot>
          <div class="w-auto flex justify-center items-center gap-24px">
            <NButton type="primary">
              上一条
            </NButton>
            <NButton type="primary">
              下一条
            </NButton>
          </div>
        </div>
      </div>
    </transition>
  </div>
</template>

<script setup>
import { ElImage } from 'element-plus';
import SvgIcon from "@/components/custom/svg-icon.vue";

// 定义 props
const props = defineProps({
  images: {
    type: Array,
    required: true,
    validator: (arr) => arr.every(item => typeof item === 'object' && 'original' in item && 'effect' in item),
  },
  currentImage: {
    type: Object,
    required: true,
  },
  visible: {
    type: Boolean,
    default: false,
  },
});

const emit = defineEmits();

// 隐藏查看器的方法
const hideViewer = () => {
  emit("close", {})
};
</script>

<style scoped>
.image-list {
  display: flex;
  flex-wrap: wrap;
}

.custom-viewer {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background-color: rgba(0, 0, 0, 0.7);
  display: flex;
  justify-content: center;
  align-items: center;
  z-index: 9998;
}

.viewer-content {
  background-color: white;
  padding: 20px;
  border-radius: 8px;
  max-width: 90%;
  max-height: 90%;
  overflow: hidden;
  width: auto; /* 确保有足够空间放置两张图片 */
}

.dual-images {
  display: flex;
  justify-content: space-between;
}

.image-side {
  flex: 1;
  text-align: center;
  padding: 0 20px;
}

.image-side img {
  max-width: 100%;
  height: auto;
  border: 1px solid #ddd;
  padding: 5px;
  background-color: #fff;
}

.fade-enter-active, .fade-leave-active {
  transition: opacity 0.3s;
}
.fade-enter, .fade-leave-to /* .fade-leave-active below version 2.1.8 */ {
  opacity: 0;
}

.image-slot {
  font-size: 14px;
  color: red;
  text-align: center;
  line-height: 100px;
}

:deep(.custom-viewer) {
  z-index: 666
}
</style>
