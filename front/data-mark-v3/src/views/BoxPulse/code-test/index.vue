<script setup lang="ts">
import { useRouter } from 'vue-router'

const { VITE_IFRAME_URL } = import.meta.env;
const iframeUrl = ref<string>(VITE_IFRAME_URL);
</script>

<template>
  <div class="w-full h-full">
    <iframe :src="iframeUrl" frameborder="0" class="w-full h-full"
      sandbox="allow-same-origin allow-scripts allow-forms"></iframe>
  </div>
</template>

<style scoped></style>
