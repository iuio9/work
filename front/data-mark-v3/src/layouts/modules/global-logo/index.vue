<script setup lang="ts">
import { $t } from "@/locales";

defineOptions({
  name: "GlobalLogo",
});

interface Props {
  /** Whether to show the title */
  showTitle?: boolean;
}

withDefaults(defineProps<Props>(), {
  showTitle: true,
});
</script>

<template>
  <RouterLink to="/" class="w-full flex-center nowrap-hidden">
    <SystemLogo class="text-32px text-primary" />
    <h2
      v-show="showTitle"
      class="text-14px text-primary font-bold transition duration-300 ease-in-out line-clamp-2 whitespace-normal w-64%"  
    >
      {{ $t("system.title") }}
    </h2>
    <!-- <div v-show="showTitle" class="text-12px text-primary font-bold transition duration-300 ease-in-out line-clamp-2 w-[102px] h-auto">{{ $t("system.title") }}</div> -->
  </RouterLink>
</template>

<style scoped></style>
