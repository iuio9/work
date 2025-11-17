<script setup lang="ts">
import {NButton} from "naive-ui";
import {updateDataSetRemark} from "@/service/api/dataManage";

defineOptions({
  name: "VersionInfo"
});

// props
interface Props {
  rowData?: any
}
const props = withDefaults(defineProps<Props>(), {});

// emits
interface Emits {
  (e: 'update'): void;
  (e: 'remark'): () => any;
}
const emit = defineEmits<Emits>();

function createDefaultData(): any {
  return {
    groupName: "",
    sonId: "",
    version: "",
    createTime: "",
    importRecord: "查看导入记录",
    remark: "",
    anoType: "",
    anoTemplate: "",
    dataTotal: "",
    markInfo: "",
    tagNumber: "",
    beconfirm: "",
    anoSize: ""
  }
}
const model: any = reactive(createDefaultData());
const operateObj: any = ref({
  isRemark: false,
})


type operateType = "remark" | "detail"
const handleOperate = (type: operateType) => {
  switch (type) {
    case "remark":
      operateObj.value.isRemark = true;
      break;
    default:
      operateObj.value.isRemark = false;
  }
}

const handleDefine = async () => {
  // request
  const res = await updateDataSetRemark({
    sonId: props.rowData.sonId,
    remark: model.remark,
  });
  if(res.data == 1) {
    window.$message?.success("备注修改成功！");
    operateObj.value.isRemark = false;
    emit("remark", {
      remark: model.remark,
    })
  }
}
const handleCancel = () => {
  operateObj.value.isRemark = false;
}

const handleInitModel = () => {
  Object.assign(model, createDefaultData(), props.rowData);
}

const handleImpRecords = () => {
  // isImportModal.value = true;
  emit('import', model)
}

onMounted(() => {
  handleInitModel();
})


</script>

<template>
  <div class="wrap-v-info w-600px h-auto p-24px box-border flex-col justify-start items-center">
    <div class="items-con w-full h-auto">
      <div class="title text-[#151b26] font-[500] text-[16px] mb-24px">基本信息</div>
      <div class="main w-full h-auto">
        <div class="w-full flex items-center mb-24px">
          <div class="w-1/2 flex justify-start items-center">
            <span class="inline-block text-[12px] text-[#5c5f66] font-400">数据集名称：</span>
            <span class="inline-block text-[12px] text-[#151b26] font-400">{{ model.groupName }}</span>
          </div>
          <div class="w-1/2 flex justify-start items-center">
            <span class="inline-block text-[12px] text-[#5c5f66] font-400 mr-24px">数据集ID：</span>
            <span class="inline-block text-[12px] text-[#151b26] font-400">{{ model.sonId }}</span>
          </div>
        </div>
        <div class="w-full flex items-center mb-24px">
          <div class="w-1/2 flex justify-start items-center">
            <span class="inline-block text-[12px] text-[#5c5f66] font-400 mr-24px">版本号：</span>
            <span class="inline-block text-[12px] text-[#151b26] font-400">V{{ model.version }}</span>
          </div>
          <div class="w-1/2 flex justify-start items-center">
            <span class="inline-block text-[12px] text-[#5c5f66] font-400 mr-24px">创建时间：</span>
            <span class="inline-block text-[12px] text-[#151b26] font-400">{{ model.createTime }}</span>
          </div>
        </div>
        <div class="w-full flex items-center mb-24px">
          <div class="w-1/2 flex justify-start items-center">
            <span class="inline-block text-[12px] text-[#5c5f66] font-400 mr-24px">导入记录：</span>
            <span
              @click="handleImpRecords()"
              class="inline-block text-[12px] text-[#2468f2] font-400 cursor-pointer"
              v-if="model.dataImportCount">查看导入记录</span>
            <span class="inline-block text-[12px] text-[#84868c] font-400" v-else>暂无导入记录</span>
          </div>
        </div>
        <div class="w-full flex items-center">
          <div class="w-full flex justify-start items-center">
            <span class="inline-block text-[12px] text-[#5c5f66] font-400 mr-24px">备注：</span>
            <div class="flex items-center" v-show="!operateObj.isRemark">
              <span class="inline-block text-[12px] text-[#2468f2] font-400">{{ model.remark }}</span>
              <div class="svg-icon ml-4px" @click="handleOperate('remark')">
                <svg-icon icon="lucide:edit" class="text-[16px]"></svg-icon>
              </div>
            </div>
            <div class="flex items-center" v-show="operateObj.isRemark">
              <n-input
                v-model:value="model.remark"
                placeholder="请输入备注信息"
              />
              <div class="ml-14px flex items-center gap-4px">
                <n-button quaternary type="info" @click="handleDefine()" size="small">
                  确定
                </n-button>
                <n-button quaternary @click="handleCancel()" size="small">
                  取消
                </n-button>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
    <div class="items-con w-full h-auto mt-24px" v-if="false">
      <div class="title text-[#151b26] font-[500] text-[16px] mb-24px">标注信息</div>
      <div class="main w-full h-auto">
        <div class="w-full flex items-center mb-24px">
          <div class="w-1/2 flex justify-start items-center">
            <span class="inline-block text-[12px] text-[#5c5f66] font-400">标注类型：</span>
            <span class="inline-block text-[12px] text-[#151b26] font-400">{{ model.anoType }}</span>
          </div>
          <div class="w-1/2 flex justify-start items-center">
            <span class="inline-block text-[12px] text-[#5c5f66] font-400">标注模板：</span>
            <span class="inline-block text-[12px] text-[#151b26] font-400">{{ model.anoTemplate }}</span>
          </div>
        </div>
        <div class="w-full flex items-center mb-24px">
          <div class="w-1/2 flex justify-start items-center">
            <span class="inline-block text-[12px] text-[#5c5f66] font-400">数据总量：</span>
            <span class="inline-block text-[12px] text-[#151b26] font-400">{{ model.dataTotal }}</span>
          </div>
          <div class="w-1/2 flex justify-start items-center">
            <span class="inline-block text-[12px] text-[#5c5f66] font-400">已标注：</span>
            <span class="inline-block text-[12px] text-[#151b26] font-400">{{ model.markInfo }}</span>
          </div>
        </div>
        <div class="w-full flex items-center mb-24px">
          <div class="w-1/2 flex justify-start items-center">
            <span class="inline-block text-[12px] text-[#5c5f66] font-400">标签个数：</span>
            <span class="inline-block text-[12px] text-[#151b26] font-400">{{ model.tagNumber }}</span>
          </div>
          <div class="w-1/2 flex justify-start items-center">
            <span class="inline-block text-[12px] text-[#5c5f66] font-400">待确认：</span>
            <span class="inline-block text-[12px] text-[#151b26] font-400">{{ model.beconfirm }}</span>
          </div>
        </div>
        <div class="w-full flex items-center">
          <div class="w-1/2 flex justify-start items-center">
            <span class="inline-block text-[12px] text-[#5c5f66] font-400">大小：</span>
            <span class="inline-block text-[12px] text-[#151b26] font-400">{{ model.size }}</span>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>

</style>
