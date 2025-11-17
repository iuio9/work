<script setup lang="ts">
import { useAppStore } from '@/store/modules/app';
import { useEcharts } from '@/hooks/common/echarts';

defineOptions({
  name: 'ConfusionMatrix'
});

var predictedCategories = ['insulator', 'damper', 'windbirdrepellent', 'foreignmatter', 'crane_noworking', 'pumptruck_noworking', 'piledriver', 'tippertruck', 'crane_working', 'pumptruck_working','mixer', 'towercrane', 'bulldozer', 'excavator', 'background FN'];
var trueCategories = ['insulator', 'damper', 'windbirdrepellent', 'foreignmatter', 'crane_noworking', 'pumptruck_noworking', 'piledriver', 'tippertruck', 'crane_working', 'pumptruck_working','mixer', 'towercrane', 'bulldozer', 'excavator', 'background FP'];
var data = [
  [0.87, 0.03, 0.09, 0.02, 0.01, 0.01, 0.04, 0.04, 0.14, 0.10, 0.03, 0.02, 0.02, 0.13],
  [0.03, 0.83, 0.03, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.02, 0.01, 0.01, 0.01, 0.17],
  [0.09, 0.03, 0.96, 0.03, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.03],
  [0.02, 0.01, 0.03, 0.91, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.09],
  [0.01, 0.01, 0.01, 0.01, 0.99, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.02],
  [0.01, 0.01, 0.01, 0.01, 0.01, 0.98, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01],
  [0.04, 0.01, 0.01, 0.01, 0.01, 0.01, 0.98, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.04],
  [0.04, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.98, 0.01, 0.01, 0.01, 0.01, 0.01, 0.04],
  [0.14, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.95, 0.01, 0.01, 0.01, 0.01, 0.14],
  [0.10, 0.02, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.96, 0.01, 0.01, 0.01, 0.10],
  [0.03, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.86, 0.07, 0.06, 0.03],
  [0.02, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.07, 0.89, 0.06, 0.02],
  [0.02, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.06, 0.06, 0.97, 0.02],
  [0.13, 0.17, 0.03, 0.09, 0.02, 0.01, 0.04, 0.04, 0.14, 0.10, 0.03, 0.02, 0.02, 0.13]
];

var option = {
  tooltip: {
    formatter: function (params) {
      return params.value;
    }
  },
  grid: {
    left: '0%',
    right: '4%',
    bottom: '0%',
    top: '3%',
    containLabel: true
  },
  xAxis: {
    type: 'category',
    data: trueCategories,
    axisTick: {
      show: false
    },
    axisLabel: {
      interval: 0,
      rotate: 45
    }
  },
  yAxis: {
    type: 'category',
    data: predictedCategories,
    axisTick: {
      show: false
    },
    axisLabel: {
      interval: 0,
      rotate: 0
    }
  },
  visualMap: {
    min: 0,
    max: 1,
    calculable: true,
    inRange: {
      color: ['#d9d9d9', '#003366']
    },
    orient: 'horizontal',
    left: 'right',
    bottom: '10%'
  },
  series: [
    {
      name: 'Confusion Matrix',
      type: 'heatmap',
      data: data,
      label: {
        show: true,
        formatter: function (params) {
          return params.value.toFixed(2);
        }
      },
      emphasis: {
        itemStyle: {
          shadowBlur: 10,
          shadowColor: 'rgba(0, 0, 0, 0.5)'
        }
      }
    }
  ]
};

const { domRef } = useEcharts(() => option);

</script>

<template>
  <div ref="domRef" class="h-360px overflow-hidden"></div>
</template>

<style scoped></style>
