export const useAnoStore = defineStore('ano', () => {
  // 使用 ref 创建响应式状态
  const anoType = ref<string>("1");

  // 定义 actions (methods) 来改变状态
  function updateAno(ano: string) {
    anoType.value = ano;
  }

  return { anoType, updateAno };
});