<script setup lang="ts">
import { useAppStore } from '@/store/modules/app';
import { useEcharts } from '@/hooks/common/echarts';

defineOptions({
  name: 'PieChart'
});

const appStore = useAppStore();

const data = [
  { recall: 0.0, precision: 0.878, label: 'insulator' },
  { recall: 0.1, precision: 0.806, label: 'damper' },
  { recall: 0.2, precision: 0.971, label: 'windturbinepillar' },
  { recall: 0.3, precision: 0.907, label: 'foreignmatter' },
  { recall: 0.4, precision: 0.995, label: 'crane_noworking' },
  { recall: 0.5, precision: 0.982, label: 'pumptruck_noworking' },
  { recall: 0.6, precision: 0.982, label: 'pledder' },
  { recall: 0.7, precision: 0.967, label: 'tippertruck' },
  { recall: 0.8, precision: 0.965, label: 'crane_working' },
  { recall: 0.9, precision: 0.991, label: 'pumptruck_working' },
  { recall: 1.0, precision: 0.868, label: 'mixer' },
  { recall: 0.0, precision: 0.917, label: 'towercrane' },
  { recall: 0.1, precision: 0.978, label: 'bulldozer' },
  { recall: 0.2, precision: 0.969, label: 'excavator' },
  { recall: 0.3, precision: 0.943, label: 'all classes' }
];

// 指定图表的配置项和数据
var option = {
  tooltip: {
    trigger: 'axis',
    axisPointer: {
      type: 'cross',
      crossStyle: {
        color: '#999'
      }
    }
  },
  xAxis: {
    type: 'value',
    name: 'Recall',
    min: 0,
    max: 1,
    interval: 0.2,
    boundaryGap: [0, '100%']
  },
  yAxis: {
    type: 'value',
    name: 'Precision',
    min: 0,
    max: 1,
    interval: 0.2,
    boundaryGap: [0, '100%']
  },
  series: []
};

// 准备数据
data.forEach(item => {
  const existingSeries = option.series.find(series => series.name === item.label);
  if (existingSeries) {
    existingSeries.data.push([item.recall, item.precision]);
  } else {
    option.series.push({
      name: item.label,
      type: 'line',
      smooth: true,
      data: [[item.recall, item.precision]]
    });
  }
});

const { domRef, updateOptions } = useEcharts(() => option);

</script>

<template>
  <div ref="domRef" class="h-360px overflow-hidden"></div>
</template>

<style scoped></style>
