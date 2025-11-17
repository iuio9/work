<template>
  <main>
    <n-pagination
      :page="pageCurrent"
      :item-count="pageTotal"
      :page-size="pageSize"
      size="medium"
      show-size-picker
      :page-slot="8"
      :page-sizes="pageSizes"
      :on-update:page="pageEvent"
      :on-update:page-size="pageSizeEvent">
      <template #prefix="{ itemCount }"> 共 {{ itemCount }} 项 </template>
    </n-pagination>
  </main>
</template>

<script setup lang="ts">
const props = withDefaults(defineProps<{
  pageCurrent: Number,
  pageSize: Number,
  pageTotal: Number,
}>(), {
  pageCurrent: 1,
  pageSize: 10,
});

const emit = defineEmits(['toPage'])
const pageSizes = ref([
  { label: '5 每页', value: 5 },
  { label: '10 每页', value: 10 },
  { label: '20 每页', value: 20 },
  { label: '50 每页', value: 50 },
  { label: '100 每页', value: 100  }
])

const pageEvent = (page: Number,) => {
  emit('toPage', {
    sign: "PAGE",
    page,
    pageSize: props.pageSize,
  });
}
const pageSizeEvent = (size: Number,) => {
  emit('toPage', {
    sign: "SIZE",
    page: 1,
    pageSize: size
  });
}

</script>
