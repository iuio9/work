<template>
  <NModal v-model:show="visible" title="标签管理" preset="card" class="w-680px">
    <NScrollbar class="h-auto">
      <component :is="component" v-model:sonId="rowData.sonId" @tagClick="tagClick"></component>
    </NScrollbar>
    <template #footer>
      <NSpace justify="end" :size="16">
        <NButton type="default" @click="closeDrawer">关闭操作窗口</NButton>
      </NSpace>
    </template>
  </NModal>
</template>

<script setup lang="ts">
import TagConfig from "./tagConfig.vue";

defineOptions({
  name: 'TagOperateModal'
});

interface Emits {
  (e: 'submitted'): void;
  (e: 'tagClick', tagData: any): void;
}

type TagConfigInstance = InstanceType<typeof TagConfig>;

const emit = defineEmits<Emits>();

const visible = defineModel<boolean>('visible', {
  default: false
});

const rowData = defineModel<any>('rowData', {
  default: () => {}
});

const component: Ref<TagConfigInstance | null> = ref(TagConfig);

const closeDrawer: () => void = () => {
  visible.value = false;
}

function tagClick(tagData: any) {
  emit('tagClick', tagData)
}
</script>

<style scoped></style>
