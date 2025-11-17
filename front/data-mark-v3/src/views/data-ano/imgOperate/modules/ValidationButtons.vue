<template>
  <div class="w-auto flex items-center justify-end gap-[24px]" v-if="
    route.query.anoType === 'validate' ||
    route.query.anoType === 'audit'
  ">
    <n-button type="primary" @click="handleValidate('0')" :loading="loadingStatus === '0'">
      剩余验收通过
    </n-button>
    <n-button type="primary" @click="handleValidate('1')" v-show="!isAuditType" :loading="loadingStatus === '1'">
      验收完成
    </n-button>
    <n-button type="primary" @click="handleValidate('2')" :loading="loadingStatus === '2'">
      打回任务
    </n-button>
  </div>
</template>

<script setup lang="ts">
import { ref, computed } from 'vue';
import { useRoute } from 'vue-router';
import { remainingApprove } from '@/service/api/ano'; // 请替换为实际API路径

// 定义组件的Emits
const emit = defineEmits<{
  (e: 'validate', statusCode: string): void;
  (e: 'show-validate-modal'): void;
  (e: 'refresh-data'): void;
}>();

// 路由信息
const route = useRoute();

// 状态管理
const validateStatus = defineModel('validateStatus', {
  type: String,
  default: ''
});
const loadingStatus = ref('');

// 计算属性：是否为审核类型
const isAuditType = computed(() => {
  return route.query.anoType === 'audit';
});

// 处理验收操作
const handleValidate = async (statusCode: string) => {
  validateStatus.value = statusCode;
  loadingStatus.value = statusCode;

  try {
    switch (statusCode) {
      case "0":
        const res = await remainingApprove({
          taskId: route.query?.taskId,
          id: route.query?.markUserId,
        });
        console.log(res);

        if (res.data) {
          window.$message?.success?.("操作成功！");
          emit('refresh-data'); // 触发父组件刷新数据
        }
        break;

      case "1":
      case "2":
        emit('show-validate-modal'); // 显示验证模态框
        break;
    }

  } catch (error) {
    console.error('操作失败:', error);
    window.$message?.error?.("操作失败，请重试！");
  } finally {
    loadingStatus.value = ''; // 重置加载状态
  }
};

// 暴露给父组件的属性和方法
defineExpose({
  validateStatus,
  handleValidate
});
</script>

<style scoped>
/* 可以根据需要添加组件特定样式 */
</style>
