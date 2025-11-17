<script setup lang="ts">
import { useAppStore } from '@/store/modules/app';
import { useEcharts } from '@/hooks/common/echarts';

defineOptions({
  name: 'PieChart'
});
const indicatorData = [
  {name: '平均精度(mPrecision)', max: 1},
  {name: '平均召回率(mRecall)', max: 1},
  {name: '均值平均精度(mAP@0.5)', max: 1},
  {name: '漏检率(MissRate)', max: 1},
  {name: '虚警率(FalseAlarmRate)', max: 1},
  {name: '平均正确率(mAccuracy)', max: 1}
];

const valueData = [0.940, 0.914, 0.943, 0.086, 0.060, 0.869];

const option = {
  tooltip: {},
  radar: {
    indicator: indicatorData,
    radius: '80%',
    center: ['50%', '50%'],
    startAngle: 90,
    splitNumber: 5,
    shape: 'circle',
    name: {
      formatter: '{value}',
      textStyle: {
        color: '#000'
      }
    },
    axisLine: {
      lineStyle: {
        color: 'rgba(0, 0, 0, 0.2)'
      }
    },
    splitLine: {
      lineStyle: {
        color: 'rgba(0, 0, 0, 0.2)'
      }
    },
    splitArea: {
      show: false
    }
  },
  series: [
    {
      name: '指标得分',
      type: 'radar',
      data: [
        {
          value: valueData,
          symbol: 'none',
          itemStyle: {
            color: 'rgb(255, 99, 71)'
          },
          areaStyle: {
            color: 'rgba(255, 99, 71, 0.3)'
          }
        }
      ]
    }
  ]
};

const { domRef } = useEcharts(() => option);
</script>

<template>
  <div ref="domRef" class="h-360px overflow-hidden"></div>
</template>

<style scoped></style>
