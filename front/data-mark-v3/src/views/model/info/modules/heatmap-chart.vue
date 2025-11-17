<script setup lang="ts">
import { useAppStore } from '@/store/modules/app';
import { useEcharts } from '@/hooks/common/echarts';

defineOptions({
  name: 'PieChart'
});

const { domRef, updateOptions } = useEcharts(() => ({
  tooltip: {
    trigger: 'axis',
    axisPointer: {
      type: 'shadow'
    }
  },
  visualMap: {
    show: true,
    min: 0,
    max: 1,
    calculable: true,
    inRange: {
      color: ['#d9d9d9', '#0080ff']
    }
  },
  xAxis: {
    type: 'category',
    data: [
      'insulator', 'damper', 'windbirdrepellent', 'foreignmatter',
      'crane_nowworking', 'pumptruck_nowworking', 'piledriver', 'tippertruck',
      'crane_working', 'pumptruck_working', 'mixer', 'towercrane', 'bulldozer',
      'excavator', 'background_FN'
    ],
    axisLabel: {
      rotate: 45
    }
  },
  yAxis: {
    type: 'category',
    data: [
      'insulator', 'damper', 'windbirdrepellent', 'foreignmatter',
      'crane_nowworking', 'pumptruck_nowworking', 'piledriver', 'tippertruck',
      'crane_working', 'pumptruck_working', 'mixer', 'towercrane', 'bulldozer',
      'excavator', 'background_FP'
    ],
  },
  grid: {
    left: '0%',
    right: '4%',
    bottom: '3%',
    top: '3%',
    containLabel: true
  },
  series: [
    {
      name: '评估结果',
      type: 'heatmap',
      data: [
        { x: 'insulator', y: 'insulator', value: 0.87 },
        { x: 'damper', y: 'damper', value: 0.83 },
        { x: 'windbirdrepellent', y: 'windbirdrepellent', value: 0.96 },
        { x: 'foreignmatter', y: 'foreignmatter', value: 0.91 },
        { x: 'crane_nowworking', y: 'crane_nowworking', value: 0.99 },
        { x: 'pumptruck_nowworking', y: 'pumptruck_nowworking', value: 0.98 },
        { x: 'piledriver', y: 'piledriver', value: 0.98 },
        { x: 'tippertruck', y: 'tippertruck', value: 0.98 },
        { x: 'crane_working', y: 'crane_working', value: 0.95 },
        { x: 'pumptruck_working', y: 'pumptruck_working', value: 0.96 },
        { x: 'mixer', y: 'mixer', value: 0.86 },
        { x: 'towercrane', y: 'towercrane', value: 0.89 },
        { x: 'bulldozer', y: 'bulldozer', value: 0.97 },
        { x: 'excavator', y: 'excavator', value: 0.97 },
        { x: 'background_FN', y: 'background_FN', value: 0.01 }
      ],
      label: {
        show: true,
        formatter: '{b}',
        position: 'top'
      }
    }
  ]
}));

</script>

<template>
  <div ref="domRef" class="h-360px overflow-hidden"></div>
</template>

<style scoped></style>
