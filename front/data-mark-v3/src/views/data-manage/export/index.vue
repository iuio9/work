<template>
  <div
    style="padding: 0 !important"
    class="wrap_main w-full h-full p-0 bg-[#f7f7f9] flex flex-col justify-start items-center"
  >
    <div
      class="header w-full h-[48px] box-border bg-[#fff] box-border px-16px py-0 flex items-center"
    >
      <div
        class="item_return w-auto h-full flex items-center cursor-pointer"
        @click="handleBack()"
      >
        <!--<SvgIcon
          icon="oui:return-key"
          class="inline-block align-text-bottom text-18px color-[#000]"
        ></SvgIcon>-->
        <SvgIcon
          local-icon="oui--return-key"
          class="inline-block align-text-bottom text-18px text-[#000]"
        ></SvgIcon>
        <span
          class="block ml-[4px] w-auto h-full flex items-center text-[12px] text-[#84868c]"
          >返回</span
        >
      </div>
      <div
        class="item_name w-auto h-full flex items-center text-16px text-[#151b26] font-[500] ml-[12px]"
      >
        name
      </div>
    </div>
    <!-- content -->
    <div class="content w-full flex-1 p-[16px]">
      <n-card title="" class="w-full h-full">
        <n-form
          class="w-70% h-auto"
          ref="formRef"
          :model="formModel"
          :rules="formRules"
          label-width="100"
          label-align="left"
          label-placement="left"
        >
          <n-form-item label="选择导出位置" path="export" class="w-[44%]">
            <n-tree-select
              v-model:value="formModel.export"
              default-value="选择导出位置"
            />
          </n-form-item>
          <!--导出数据-->
          <n-form-item label="导出数据" path="exportData" class="w-[70%]">
            <n-radio-group
              v-model:value="formModel.exportData"
              name="radiogroup"
            >
              <n-space>
                <n-radio value="" class="flex items-center">
                  <span> 导出全部数据，包含源文件及已有的标注文件 </span>
                </n-radio>
                <n-radio value="" class="flex items-center">
                  <span> 仅导出源文件 </span>
                </n-radio>
              </n-space>
            </n-radio-group>
          </n-form-item>
          <!--标注格式-->
          <n-form-item label="标注格式" path="anoType" class="w-[70%]">
            <n-radio-group v-model:value="formModel.anoType" name="radiogroup">
              <n-space>
                <n-radio value="" class="flex items-center">
                  <span> 平台默认格式 </span>
                </n-radio>
                <n-radio value="" class="flex items-center">
                  <span> json </span>
                </n-radio>
              </n-space>
            </n-radio-group>
          </n-form-item>
          <!--导出记录-->
          <n-form-item label="导出记录" path="anoType" class="w-[70%]">
          </n-form-item>
        </n-form>
      </n-card>
    </div>
    <!--footer-->
    <div class="w-full h-[72px] bg-[#fff] flex items-center px-48px">
      <div class="item_btnCon w-auto h-full flex items-center gap-24px">
        <n-button type="primary" @click="handleExport()">开始导出</n-button>
        <n-button @click="handleBack()">返回</n-button>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
interface FormModel {
  export: string;
  exportData: string;
  anoType: string;
  exportRecord: string;
}

// data
const router = useRouter();
const formModel = ref<FormModel>({
  export: "",
  exportData: "",
  anoType: "",
  exportRecord: "",
});
const formRules = {};
// methods
const handleBack = () => {
  router.back();
};
const handleExport = () => {};
</script>

<style scoped>
.wrap_main {
  padding: 0 !important;
}
</style>
