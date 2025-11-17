<script setup lang="ts">
import { useAppStore } from '@/store/modules/app';
import { useEcharts } from '@/hooks/common/echarts';

defineOptions({
  name: 'PieChart'
});

const appStore = useAppStore();

const { domRef, updateOptions } = useEcharts(() => ({
  tooltip: {
    trigger: 'axis',
    axisPointer: {
      type: 'cross',
      crossStyle: {
        color: '#999'
      }
    }
  },
  legend: {
    orient: 'vertical', // 设置图例为垂直布局
    right: '5%', // 设置图例在右侧的位置
    top: 'center' // 设置图例在垂直方向上的居中位置
  },
  radar: {
    indicator: [
      { name: 'F1分数', max: 100 },
      { name: '格式遵从性', max: 100 },
      { name: 'ROUGE-1', max: 100 },
      { name: 'BLEU-4', max: 100 },
      { name: 'ROUGE-2', max: 100 },
      { name: 'ROUGE-L', max: 100 }
    ]
  },
  series: [
    {
      name: '评估结果',
      type: 'radar',
      data: [
        { value: 50.08, name: 'F1分数' },
        { value: 0, name: '格式遵从性' },
        { value: 46.24, name: 'ROUGE-1' },
        { value: 17.66, name: 'BLEU-4' },
        { value: 27.45, name: 'ROUGE-2' },
        { value: 40.93, name: 'ROUGE-L' }
      ],
      areaStyle: {}
    }
  ]
}));
</script>

<template>
  <div ref="domRef" class="h-360px overflow-hidden"></div>
</template>

<style scoped></style>
