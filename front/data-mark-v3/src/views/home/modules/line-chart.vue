<script setup lang="ts">
import { watch } from "vue";
import { useAppStore } from "@/store/modules/app";
import { useEcharts } from "@/hooks/common/echarts";
import { getDaysComputerInfo } from "@/service/api/home";

defineOptions({
  name: "LineChart",
});

interface Props {
  title: string;
}
const props = withDefaults(defineProps<Props>(), {
  title: "",
});

const appStore = useAppStore();

const { domRef, updateOptions } = useEcharts(() => ({
  tooltip: {
    trigger: "axis",
    axisPointer: {
      type: "cross",
      label: {
        backgroundColor: "#6a7985",
      },
    },
  },
  legend: {
    data: ["本机CPU使用率"],
  },
  grid: {
    left: "3%",
    right: "4%",
    bottom: "3%",
    containLabel: true,
  },
  xAxis: {
    type: "category",
    boundaryGap: false,
    data: [] as string[],
    axisLabel: {
      formatter: function (value) {
        let strs = value.split("");
        let rows = [];
        let row = "";
        for (let i = 0; i < strs.length; i++) {
          if (row.length > 10) {
            rows.push(row);
            row = "";
          }
          row += strs[i];
        }
        if (row) {
          rows.push(row);
        }
        return rows.join("\n");
      },
      rich: {
        // 可以在这里定义样式，比如字体大小、颜色等
      },
    },
  },
  yAxis: {
    type: "value",
  },
  series: [
    {
      color: "#8e9dff",
      name: "本机CPU使用率",
      type: "line",
      smooth: true,
      stack: "Total",
      areaStyle: {
        color: {
          type: "linear",
          x: 0,
          y: 0,
          x2: 0,
          y2: 1,
          colorStops: [
            {
              offset: 0.25,
              color: "#8e9dff",
            },
            {
              offset: 1,
              color: "#fff",
            },
          ],
        },
      },
      emphasis: {
        focus: "series",
      },
      data: [] as number[],
    },
  ],
}));

async function mockData() {
  await new Promise((resolve) => {
    setTimeout(resolve, 1000);
  });

  updateOptions((opts) => {
    opts.xAxis.data = [
      "06:00",
      "08:00",
      "10:00",
      "12:00",
      "14:00",
      "16:00",
      "18:00",
      "20:00",
      "22:00",
      "24:00",
    ];
    opts.series[0].data = [
      4623, 6145, 6268, 6411, 1890, 4251, 2978, 3880, 3606, 4311,
    ];
    opts.series[1].data = [
      2208, 2016, 2916, 4512, 8281, 2008, 1963, 2367, 2956, 678,
    ];

    return opts;
  });
}

function updateLocale() {
  updateOptions((opts, factory) => {
    const originOpts = factory();

    opts.legend.data = originOpts.legend.data;
    opts.series[0].name = originOpts.series[0].name;
    opts.series[1].name = originOpts.series[1].name;

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

const getLineData = async () => {
  const res = await getDaysComputerInfo();
  if (res.data) {
    if (props.title === "存储服务器") {
      const timeList = res.data.storageBar.map((val) => {
        return val.hour;
      });
      const cpuList = res.data.storageBar.map((val) => {
        return (val.avg_cpu_usage = val.avg_cpu_usage ?? 0);
      });
      const gpuList = res.data.storageBar.map((val) => {
        return (val.avg_gpu_usage = val.avg_gpu_usage ?? 0);
      });
      updateOptions((opts) => {
        opts.xAxis.data = timeList;
        opts.series[0].data = cpuList;
        // opts.series[1].data = gpuList;
        return opts;
      });
    }
    if (props.title === "算法服务器") {
      const timeList = res.data.algorithmBar.map((val) => {
        return val.hour;
      });
      const cpuList = res.data.algorithmBar.map((val) => {
        return (val.avg_cpu_usage = val.avg_cpu_usage
          ? val.avg_cpu_usage
          : "0");
      });
      const gpuList = res.data.algorithmBar.map((val) => {
        return (val.avg_gpu_usage = val.avg_gpu_usage
          ? val.avg_gpu_usage
          : "0");
      });
      updateOptions((opts) => {
        opts.xAxis.data = timeList;
        opts.series[0].data = cpuList;
        opts.series[1].data = gpuList;
        return opts;
      });
    }
  }
};

let timer = null;
onMounted(() => {
  getLineData();
  // timer = setInterval(() => {
  //   getLineData();
  // }, 6000);
  // getLineData();
});

onBeforeUnmount(() => {
  clearInterval(timer);
});
// init
// init();
</script>

<template>
  <NCard :bordered="false" class="card-wrapper">
    <!--<div ref="domRef" class="h-360px overflow-hidden"></div>-->
    <div ref="domRef" class="h-360px overflow-hidden"></div>
  </NCard>
</template>

<style scoped></style>
