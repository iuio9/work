<script setup lang="ts">
import { watch } from "vue";
import { useAppStore } from "@/store/modules/app";
import { useEcharts } from "@/hooks/common/echarts";
import { getComputerInfo, getDaysComputerInfo } from "@/service/api/home";

defineOptions({
  name: "PieChart",
});

interface Props {
  title: string;
}
const props = withDefaults(defineProps<Props>(), {
  title: "",
});

const appStore = useAppStore();

const { domRef, updateOptions } = useEcharts(() => ({
  animation: false, // 取消动画效果
  tooltip: {
    trigger: "item",
  },
  legend: {
    bottom: "1%",
    left: "center",
    itemStyle: {
      borderWidth: 0,
    },
  },
  series: [
    {
      color: ["#5da8ff", "#8e9dff", "#fedc69", "#26deca"],
      name: "数据监控",
      type: "pie",
      radius: ["45%", "75%"],
      avoidLabelOverlap: false,
      itemStyle: {
        borderRadius: 10,
        borderColor: "#fff",
        borderWidth: 1,
      },
      label: {
        show: true,
        position: "center",
      },
      emphasis: {
        label: {
          show: true,
          fontSize: "12",
        },
      },
      labelLine: {
        show: false,
      },
      data: [] as { name: string; value: number }[],
    },
  ],
}));

async function mockData() {
  await new Promise((resolve) => {
    setTimeout(resolve, 1000);
  });

  updateOptions((opts) => {
    opts.series[0].data = [
      { name: "CPU利用率", value: 0 },
      { name: "GPU利用率", value: 0 },
      { name: "内存", value: 0 },
      { name: "磁盘", value: 0 },
    ];

    return opts;
  });
}

const getMonitorServer = async () => {
  // const res = await getComputerInfo();
  const res = await getDaysComputerInfo();
  if (res.data) {
    if (props.title === "存储服务器" && res.data.storagePie && res.data.storagePie[0]) {
      const { avg_cpu_usage, avg_gpu_usage, avg_mem_usage, avg_sys_fileinfo_usage } = res.data.storagePie[0];
      updateOptions((opts) => {
        opts.series[0].data = [
          { name: "CPU利用率", value: avg_cpu_usage || 0 },
          { name: "GPU利用率", value: avg_gpu_usage || 0 },
          { name: "内存", value: avg_mem_usage || 0 },
          { name: "磁盘", value: avg_sys_fileinfo_usage || 0 },
        ];
        return opts;
      });
    }
    if (props.title === "算法服务器" && res.data.algorithmPie && res.data.algorithmPie[0]) {
      const { avg_cpu_usage, avg_gpu_usage, avg_mem_usage, avg_sys_fileinfo_usage } = res.data.algorithmPie[0];
      updateOptions((opts) => {
        opts.series[0].data = [
          { name: "CPU利用率", value: avg_cpu_usage || 0 },
          { name: "GPU利用率", value: avg_gpu_usage || 0 },
          { name: "内存", value: avg_mem_usage || 0 },
          { name: "磁盘", value: avg_sys_fileinfo_usage || 0 },
        ];
        return opts;
      });
    }
  } else {
    // 使用模拟数据
    // mockData();
  }
};

function updateLocale() {
  updateOptions((opts, factory) => {
    const originOpts = factory();

    opts.series[0].name = originOpts.series[0].name;

    opts.series[0].data = [
      { name: "内存利用率", value: 20 },
      { name: "CPU利用率", value: 10 },
      { name: "接口状态", value: 40 },
      { name: "ping连通性", value: 30 },
    ];

    return opts;
  });
}

async function init() {
  mockData();
}

watch(
  () => appStore.locale,
  () => {
    updateLocale();
  }
);

let timer = null;
onMounted(() => {
  getMonitorServer();
  // mockData();
  timer = setInterval(() => {
    getMonitorServer();
  }, 6000);
});
onBeforeUnmount(() => {
  clearInterval(timer);
});
// init
// init();
</script>

<template>
  <NCard :bordered="false" class="card-wrapper">
    <div ref="domRef" class="h-360px overflow-hidden"></div>
  </NCard>
</template>

<style scoped></style>
