<script setup lang="ts">
import { computed } from "vue";
import { createReusableTemplate } from "@vueuse/core";
import axios from "axios";
import { getComputerInfo } from "@/service/api/home";

defineOptions({
  name: "CardData",
});

interface CardData {
  key: string;
  title: string;
  value: number;
  unit: string;
  color: {
    start: string;
    end: string;
  };
  icon: string;
}

interface Props {
  title: string
}
const props = withDefaults(defineProps<Props>(), {
  title: ""
})

const cardData = ref<any>([
  {
    key: "visitCount",
    title: "CPU",
    value: 0,
    unit: "",
    color: {
      start: "#ec4786",
      end: "#b955a4",
    },
    icon: "ph:cpu-bold",
  },
  {
    key: "turnover",
    title: "GPU",
    value: 0,
    unit: "",
    color: {
      start: "#865ec0",
      end: "#5144b4",
    },
    icon: "oui:storage",
  },
  {
    key: "downloadCount",
    title: "内存",
    value: 0,
    unit: "",
    color: {
      start: "#56cdf3",
      end: "#719de3",
    },
    icon: "fluent-mdl2:server-processes",
  },
  {
    key: "dealCount",
    title: "磁盘",
    value: 0,
    unit: "",
    color: {
      start: "#fcbc25",
      end: "#f68057",
    },
    icon: "emojione-monotone:optical-disk",
  },
]);
const cardData1 = ref<any>([
  {
    key: "visitCount",
    title: "CPU",
    value: 0,
    unit: "",
    color: {
      start: "#ec4786",
      end: "#b955a4",
    },
    icon: "ph:cpu-bold",
  },
  {
    key: "turnover",
    title: "GPU",
    value: 0,
    unit: "",
    color: {
      start: "#865ec0",
      end: "#5144b4",
    },
    icon: "oui:storage",
  },
  {
    key: "downloadCount",
    title: "0",
    value: 970925,
    unit: "",
    color: {
      start: "#56cdf3",
      end: "#719de3",
    },
    icon: "fluent-mdl2:server-processes",
  },
  {
    key: "dealCount",
    title: "磁盘",
    value: 0,
    unit: "",
    color: {
      start: "#fcbc25",
      end: "#f68057",
    },
    icon: "emojione-monotone:optical-disk",
  },
]);

const getMonitorServer = async () => {
  const res = await getComputerInfo();
  if (res.data) {
    if (props.title === "存储服务器") {
      const { cpu, gpu, mem, sysFileinfo } = res.data.storeage;
      cardData.value[0] = Object.assign({}, cardData.value[0], cpu, {
        unit1: "HZ",
      });
      cardData.value[1] = Object.assign({}, cardData.value[1], gpu, {
        unit1: "HZ",
      });
      cardData.value[2] = Object.assign({}, cardData.value[2], mem, {
        unit1: "KB",
      });
      cardData.value[3] = Object.assign({}, cardData.value[3], sysFileinfo, {
        unit1: "MB",
      });
    }
    if (props.title === "算法服务器") {
      const { cpu, gpu, mem, sysFileinfo } = res.data.algorithm;
      cardData1.value[0] = Object.assign({}, cardData1.value[0], cpu, {
        unit1: "HZ",
      });
      cardData1.value[1] = Object.assign({}, cardData1.value[1], gpu, {
        unit1: "HZ",
      });
      cardData1.value[2] = Object.assign({}, cardData1.value[2], mem, {
        unit1: "KB",
      });
      cardData1.value[3] = Object.assign({}, cardData1.value[3], sysFileinfo, {
        unit1: "MB",
      });
    }
  }
};

interface GradientBgProps {
  gradientColor: string;
}

const [DefineGradientBg, GradientBg] =
  createReusableTemplate<GradientBgProps>();

function getGradientColor(color: CardData["color"]) {
  return `linear-gradient(to bottom right, ${color.start}, ${color.end})`;
}

let timer = null;
onMounted(() => {
  getMonitorServer();
  timer = setInterval(() => {
    getMonitorServer();
  }, 5000);
});

onBeforeUnmount(() => {
  clearInterval(timer);
});
</script>

<template>
  <NCard :bordered="false" size="small" class="card-wrapper">
    <!-- define component start: GradientBg -->
    <DefineGradientBg v-slot="{ $slots, gradientColor }">
      <div class="rd-8px px-16px pb-4px pt-8px text-white" :style="{ backgroundImage: gradientColor }">
        <component :is="$slots.default" />
      </div>
    </DefineGradientBg>
    <!-- define component end: GradientBg -->

    <div class="w-full h-auto flex items-center justify-center font-bold text-18px mb-16px">{{ props.title }}</div>
    <NGrid responsive="screen" :x-gap="16" :y-gap="16">
      <NGi
        v-for="item in cardData"
        :key="item.key"
        :span="item.title === 'CPU' ? '24 s:12 m:12': '12 s:6 m:6'"
        v-show="item.title !== 'GPU'"
      >
        <GradientBg :gradient-color="getGradientColor(item.color)" class="flex-1">
          <div class="w-full flex justify-between items-center">
            <div class="w-30% h-auto flex flex-col justify-around items-center">
              <div class="flex items-center justify-start">
                <span class="text-16px">{{ item.title }}</span>
                <!-- <span v-if="item.unit1">({{ item.unit1 }})</span> -->
              </div>
              <div class="flex justify-between">
                <SvgIcon :icon="item.icon" class="text-32px" />
              </div>
            </div>
            <div class="w-70% h-auto flex flex-col justify-between items-end">
              <div class="flex items-center">
                <span class="mr-4px">已使用:</span>
                <span>{{ item.used }}</span>
              </div>
              <div class="flex items-center">
                <span class="mr-4px">未使用:</span>
                <span>{{ item.free }}</span>
              </div>
            </div>
          </div>
        </GradientBg>
      </NGi>
    </NGrid>
  </NCard>
</template>

<style scoped></style>
