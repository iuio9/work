<script setup lang="ts">
import { computed } from "vue";
import { useAppStore } from "@/store/modules/app";
import HeaderBanner from "./modules/header-banner.vue";
import CardData from "./modules/card-data.vue";
import CardData1 from "./modules/card-data1.vue";
import LineChart from "./modules/line-chart.vue";
import LineChart1 from "./modules/line-chart1.vue";
import PieChart from "./modules/pie-chart.vue";
import PieChart1 from "./modules/pie-chart1.vue";

const appStore = useAppStore();

const gap = computed(() => (appStore.isMobile ? 0 : 16));
</script>

<template>
  <NSpace vertical :size="16">
    <HeaderBanner />
    <div class="w-full h-auto flex justify-between items-center gap-16px">
      <div class="w-1/2">
        <CardData title="存储服务器"/>
      </div>
      <div class="w-1/2">
        <CardData1 title="算法服务器"/>
      </div>
    </div>
    <div class="w-full h-auto flex justify-between items-center gap-16px">
      <div class="w-1/2">
        <NGrid :x-gap="gap" :y-gap="16" responsive="screen" item-responsive>
          <NGi span="24 s:24 m:24">
            <NCard :bordered="false" class="card-wrapper">
              <LineChart title="存储服务器"/>
            </NCard>
          </NGi>
          <NGi span="24 s:24 m:24">
            <NCard :bordered="false" class="card-wrapper">
              <PieChart title="存储服务器"/>
            </NCard>
          </NGi>
        </NGrid>
      </div>
      <div class="w-1/2">
        <NGrid :x-gap="gap" :y-gap="16" responsive="screen" item-responsive>
          <NGi span="24 s:24 m:24">
            <NCard :bordered="false" class="card-wrapper">
              <LineChart1 title="算法服务器"/>
            </NCard>
          </NGi>
          <NGi span="24 s:24 m:24">
            <NCard :bordered="false" class="card-wrapper">
              <PieChart1 title="算法服务器" />
            </NCard>
          </NGi>
        </NGrid>
      </div>
    </div>
  </NSpace>
</template>

<style scoped lang="scss"></style>
