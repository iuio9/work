<template>
  <div class="ano-type-tabs box-border w-full flex items-center">
    <NTabs type="line" animated class="wrap_tabs" @update:value="handleTabChange" v-model:value="activeTab"
      v-if="tabPanes.length > 0">
      <template v-for="pane in tabPanes" :key="pane.name">
        <NTabPane :name="pane.name" :tab="pane.tab">
          <template #tab>
            {{ pane.label }}({{ getTabCount(pane.name) }})
          </template>
        </NTabPane>
      </template>
    </NTabs>
  </div>
</template>

<script setup lang="ts">
import { computed, ref, watch, onMounted } from 'vue';
import { useRoute } from 'vue-router';
import { NTabs, NTabPane } from 'naive-ui';
import { DataDetailsCount } from '@/service/api/ano';

// 定义标签数量类型
interface TabNumbers {
  all: number;
  haveAno: number;
  noAno: number;
  invalid: number;
}

// 定义标签配置类型
export interface AnoTabsConfig {
  tabNum: TabNumbers;
}

// 定义标签页项类型
interface TabPane {
  name: string;
  tab: string;
  label: string;
}

// 定义路由参数中anoType的可能值
type AnoType = 'validateUser' | 'online' | 'setOnline' | 'validate' | 'audit' | 'result' | undefined;

defineOptions({
  name: 'AnoTypeTabs'
})

// 组件属性定义
const props = defineProps<{
  modelValue?: string;
}>();

// 组件事件定义
const emit = defineEmits<{
  (e: 'update:modelValue', value: string): void;
  (e: 'tabChange', value: string): void;
  (e: 'beforeLeave', leave: string, active: string): void;
  (e: 'countLoaded', data: AnoTabsConfig): void; // 新增数据加载完成事件
}>();

// 活跃标签
const activeTab = defineModel<string>('activeTab', {
  default: '0'
});

const curTabNum = defineModel<number>('curTabNum', {
  default: 0
});

// 标签配置数据
const tabConfig = defineModel<AnoTabsConfig>('tabConfig', {
  default: {
    tabNum: {
      all: 0,
      haveAno: 0,
      noAno: 0,
      invalid: 0
    }
  }
});

// 监听外部值变化
watch(
  () => props.modelValue,
  (newVal) => {
    if (newVal) {
      activeTab.value = newVal;
    }
  }
);

// 监听内部值变化并同步到外部
watch(
  () => activeTab.value,
  (newVal) => {
    emit('update:modelValue', newVal);
  }
);

// 获取路由信息
const route = useRoute();

// 获取数据详情计数
const getDataDetailsCount = async () => {
  try {
    const res = await DataDetailsCount({
      sonId: route.query.id as string, // 类型断言确保类型正确
      markUserId: route.query?.markUserId as string | undefined,
      taskId: route.query.anoType === 'result' ? undefined : route.query?.taskId as string | undefined,
      sign: route.query.anoType
    });

    if (res.data) {
      // 更新标签计数
      tabConfig.value.tabNum = {
        all: res.data.all || 0,
        haveAno: res.data.haveAno || 0,
        noAno: res.data.noAno || 0,
        invalid: res.data.invalid ?? 0
      };
      setCurTabNum(activeTab.value);
      emit('countLoaded', tabConfig.value); // 触发数据加载完成事件
    }
  } catch (error) {
    console.error('获取数据详情计数失败:', error);
    // 可以添加错误处理逻辑，如显示错误提示
  }
};

const setCurTabNum = (val: string | number) => {
  const strVal = String(val); // 将 val 转换为字符串
  switch (strVal) {
    case "1":
      curTabNum.value = tabConfig.value.tabNum.haveAno;
      break;
    case "2":
      curTabNum.value = tabConfig.value.tabNum.noAno;
      break;
    case "0":
      curTabNum.value = tabConfig.value.tabNum.all;
      break;
    case "3":
      curTabNum.value = tabConfig.value.tabNum.invalid;
      break;
    default:
      // 可以根据需要处理其他情况，这里不做处理
      break;
  }
};

// 处理标签页切换
const handleTabChange = (value: string) => {
  activeTab.value = value;
  setCurTabNum(value);
  emit('tabChange', value);
};

// 处理标签页切换前的逻辑
const handleTabBefore = (leave: string, active: string) => {
  emit('beforeLeave', leave, active);
};

// 生成标签页配置
const tabPanes = computed<TabPane[]>(() => {
  const anoType = route.query.anoType as AnoType;

  // 配置映射表
  const configMap: Record<string, TabPane[]> = {
    default: [
      { name: '0', tab: '全部', label: '全部' },
      { name: '1', tab: '有标注信息', label: '有标注信息' },
      { name: '2', tab: '无标注信息', label: '无标注信息' },
      { name: '3', tab: '无效数据信息', label: '无效数据信息' }
    ],
    validate: [
      { name: '0', tab: '全部', label: '未经过验收' },
      { name: '1', tab: '有标注信息', label: '验收通过' },
      { name: '2', tab: '无标注信息', label: '验收不通过' },
      { name: '3', tab: '无效数据信息', label: '无效数据信息' }
    ],
    audit: [
      { name: '0', tab: '全部', label: '未审核信息' },
      { name: '1', tab: '有标注信息', label: '审核通过' },
      { name: '2', tab: '无标注信息', label: '审核未通过' },
      { name: '3', tab: '无效数据信息', label: '无效数据信息' }
    ],
    result: [
      { name: '0', tab: '全部', label: '全部' }
    ]
  };

  // 确定使用哪种配置
  if (!anoType || ['validateUser', 'online', 'setOnline'].includes(anoType)) {
    return configMap.default;
  }

  if (Object.prototype.hasOwnProperty.call(configMap, anoType)) {
    return configMap[anoType];
  }

  return [];
});

// 获取标签页计数
const getTabCount = (name: string): number => {
  const countKeyMap: Record<string, keyof TabNumbers> = {
    '0': 'all',
    '1': 'haveAno',
    '2': 'noAno',
    '3': 'invalid'
  };

  const key = countKeyMap[name];
  return key ? tabConfig.value.tabNum[key] : 0;
};

// 监听路由参数变化，重新获取数据
watch(
  () => [route.query.id, route.query.markUserId, route.query.taskId, route.query.anoType],
  () => {
    getDataDetailsCount();
  },
  { immediate: true } // 初始加载时执行
);

// 组件挂载时获取数据
onMounted(() => {
  // getDataDetailsCount();
});

defineExpose({
  getDataDetailsCount
});
</script>

<style scoped></style>
