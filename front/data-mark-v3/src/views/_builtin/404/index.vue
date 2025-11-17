<script lang="ts" setup>
import { onMounted } from 'vue';
import { useMessage, useDialog } from 'naive-ui';
import { useRouter } from 'vue-router';

const router = useRouter();
const message = useMessage();
const dialog = useDialog();

onMounted(() => {
  let countdown = 3;
  const dialogRef = dialog.warning({
    title: '提示',
    content: `将在 ${countdown} 秒后跳转到首页`,
    positiveText: '立即跳转',
    negativeText: '取消',
    onPositiveClick: () => {
      clearInterval(timer);
      dialogRef.destroy();
      router.push('/');
    },
    onNegativeClick: () => {
      clearInterval(timer);
      dialogRef.destroy();
    }
  });

  const timer = setInterval(() => {
    countdown--;
    if (countdown >= 0) {
      dialogRef.content = `将在 ${countdown} 秒后跳转到首页`;
      dialogRef.update({
        content: dialogRef.content
      });
    } else {
      clearInterval(timer);
      dialogRef.destroy();
      router.push('/');
    }
  }, 1000);
});
</script>


<template>
  <ExceptionBase type="404" />
</template>

<style scoped></style>
