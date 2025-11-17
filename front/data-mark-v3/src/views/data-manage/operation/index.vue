<script setup lang="ts">
import aType from "@/assets/imgs/aType.png";

import { NCascader, StepsProps } from "naive-ui";
import { getDicts } from "@/service/api";
import { useFormRules, useNaiveForm } from "@/hooks/common/form";
import { fetchDataSetAdd, getTreeLevelDict } from "@/service/api/dataManage";
import { getDictDataTree } from "@/service/api/expansion";
const infoObj = ref({
  icon: "fluent:ios-arrow-24-filled",
  localIcon: "fluent--ios-arrow-24-filled",
  title: "创建数据集",
});
const formInfo = ref({
  model: {
    inputValue: null,
    dataType: "1",
    annotationType: "",
    anomarkTemp: "",
  },
  formRef: null,
  rules: {
    inputValue: {
      required: true,
      trigger: ["blur", "input"],
      message: "请输入 inputValue",
    },
  },
  imgPaths: {
    aType,
  },
});

const router = useRouter();
const route = useRoute();

const currentStatus = ref<StepsProps["status"]>("process");
const current = ref<number | null>(1);

// data
const mapObj = ref<any>({
  typeList: [],
  verList: [],
  anoList: [],
  temList: [],
});
const model = reactive<any>({
  groupName: null,
  // dataTypeId: null,
  // dataType: "",
  markType: "",
  markTemp: "",
  version: "1",
});
const { formRef, validate, restoreValidation } = useNaiveForm();
const { defaultRequiredRule } = useFormRules();
const rules: Record<string, any> = {
  groupName: defaultRequiredRule,
  dataTypeId: defaultRequiredRule,
};

watch(
  () => model.dataType,
  (val) => {
    mapObj.value.anoList = mapObj.value.typeList.find((item: any) => {
      return `${item.id}` === `${val}`;
    }).children;
    model.markType = mapObj.value.anoList[0].id;
  },
  {
    deep: true,
  },
);

watch(
  () => model.markType,
  (val) => {
    mapObj.value.temList = mapObj.value.anoList.find((item: any) => {
      return `${item.id}` === `${val}`;
    }).children;
    model.markTemp = mapObj.value.temList[0].id;
  },
  {
    deep: true,
  },
);

// methods
const handleBack = () => {
  router.back();
};
// cancel
const handleCancel = () => {
  router.back();
};
const handleCreImp = async () => {
  await validate();
  router.push({
    path: "/data-manage/import",
    query: {
      ...model,
      dataTypeId: route.query.dataTypeId,
    },
  });
};
const handleSuccess = async () => {
  const res = await fetchDataSetAdd({
    ...model,
    dataTypeId: route.query.dataTypeId,
  });
  if (res.data?.status == 1) {
    window.$message?.success("创建数据集成功！");
    router.back();
  }
};

// 获取对应字典信息
const getDictData = async (typeId = "6") => {
  const params = { typeId };
  const res = await getDicts(params);
  return res.data;
};

// radio click
const handleRadioClick = (sign: string, rowData: any) => {
  if (sign === "dataType") {
    const { id } = rowData;
    model.dataType = id;
    mapObj.value.anoList = mapObj.value.typeList.find((item: any) => {
      return `${item.id}` === `${id}`;
    }).children;
    model.markType = mapObj.value.anoList[0].id;
    mapObj.value.temList = mapObj.value.anoList.find((item: any) => {
      return `${item.id}` === `${model.markType}`;
    }).children;
    model.markTemp = mapObj.value.temList[0].id;
  }
  if (sign === "markTemp") {
    const { id } = rowData;
    model.markTemp = id;
  }
  if (sign === "markType") {
    const { id } = rowData;
    model.markType = id;
    mapObj.value.temList = mapObj.value.anoList.find((item: any) => {
      return `${item.id}` === `${id}`;
    }).children;
    model.markTemp = mapObj.value.temList[0].id;
  }
};

// 数据集类型dataList
const recursionData = (data: any) => {
  // eslint-disable-next-line no-param-reassign
  data = data.map((item: any, index: string | number) => {
    if (item.children) {
      if (item.children.length > 0) recursionData(item.children);
      if (item.children.length === 0) delete item.children;
    }
    item.label = item.dictLabel;
    item.value = item.id;
    return item;
  });
  return data;
};

// taskType datalist
const mapOptions = ref([]);
const getMapList = async () => {
  const params = {
    typeId: "6",
  };
  const res = await getDictDataTree(params);
  mapOptions.value = recursionData(res.data);
};

onMounted(async () => {
  await getMapList();
  mapObj.value.typeList = await getDictData(route.query.dataTypeId);
  getTreeLevelDict({ dataTypeId: route.query.dataTypeId}).then(res => {
    model.groupName = res.data;
  });
  // model.dataType = mapObj.value.typeList[0].id;
});
</script>

<template>
  <div class="wrap_container flex_col_start">
    <div class="header">
      <div class="h_back flex_start" @click="handleBack()">
        <!--<SvgIcon
          :icon="infoObj.icon"
          class="inline-block align-text-bottom text-16px"
        />-->
        <SvgIcon local-icon="oui--return-key" class="inline-block align-text-bottom text-16px text-[#000000]" />
        <span>返回</span>
      </div>
      <div class="h_title">{{ infoObj.title }}</div>
    </div>
    <div
      class="content flex flex-col justify-start items-center min-h-500px flex-col-stretch gap-16px overflow-hidden lt-sm:overflow-auto">
      <!--<div class="wrap_timeline w-auto flex justify-center items-center">
        <div class="w-auto flex justify-start items-center">
          <span
            class="block flex justify-center items-center w-22px h-22px rounded-[50%] bg-[#2468f2] border-[1px] border-[#e8e9eb] text-[12px] text-[#ffffff]">1</span>
          <span class="ml-[8px] text-[14px]">创建数据集</span>
        </div>
        <div class="line w-110px h-1px bg-[#e8e9eb] mx-8px"></div>
        <div class="w-auto flex justify-start items-center">
          <span
            class="block flex justify-center items-center w-22px h-22px rounded-[50%] bg-[#fff] border-[1px] border-[#e8e9eb] text-[12px] text-[#b8babf]">2</span>
          <span class="ml-[8px] text-[14px]">导入数据</span>
        </div>
      </div>-->
      <div class="wrap_timeline w-full flex justify-center items-center box-border py-12px">
        <n-steps :current="current as number" :status="currentStatus" class="w-50% ml-180px">
          <n-step title="创建数据集" />
          <n-step title="导入数据" />
        </n-steps>
      </div>
      <div class="item_card">
        <n-card title="基本信息">
          <n-form class="w-70% h-auto" ref="formRef" :model="model" :rules="rules" label-width="100" label-align="left"
            label-placement="left">
            <n-form-item label="数据集名称" path="groupName" class="w-[70%]">
              <n-input v-model:value="model.groupName" placeholder="请输入数据集名称" />
            </n-form-item>
            <n-form-item label="数据集类型" path="dataTypeId" class="w-[70%]" v-if="false">
              <n-cascader v-model:value="model.dataTypeId" clearable placeholder="请选择数据集类型" :options="mapOptions"
                check-strategy="child" />
            </n-form-item>
            <n-form-item label="数据类型" path="type" class="mt-[4px]" v-show="false">
              <n-radio-group v-model:value="model.dataType" name="type">
                <n-radio-button v-for="(val, idx) of mapObj.typeList" :value="val.id"
                  @click="handleRadioClick('dataType', val)">{{ val.dictLabel }}</n-radio-button>
              </n-radio-group>
            </n-form-item>
            <n-form-item label="数据集版本" path="version" class="w-[70%]">
              V1
            </n-form-item>
            <n-form-item label="标注类型" path="type" v-show="false">
              <!--<div class="flex flex-row  flex-wrap justify-start items-center gap-[24px]">
                <div
                  v-for="(item, index) of markTypeObj.list" :key="index"
                  class="form_annotationType flex flex-col justify-around items-center"
                  @click="handleAnoClick(item)"
                >
                  <span>{{ item.dictLabel }}</span>
                  <img :src="item.imgPath" alt="" />
                </div>
              </div>-->
              <n-radio-group v-model:value="model.markType" name="type">
                <n-radio-button v-for="val of mapObj.anoList" :value="val.id"
                  @click="handleRadioClick('markType', val)">{{ val.dictLabel }}</n-radio-button>
              </n-radio-group>
            </n-form-item>
            <n-form-item label="标注模板" path="version" v-show="false">
              <n-radio-group v-model:value="model.markTemp" name="radiogroup">
                <n-space>
                  <n-radio v-for="item in mapObj.temList" :key="item.id" :value="item.id" class="flex items-center"
                    @click="handleRadioClick('markTemp', item)">
                    <span>{{ item.dictLabel }}</span>
                  </n-radio>
                </n-space>
              </n-radio-group>
            </n-form-item>
            <n-form-item label="保存位置" path="version" v-show="false">
              <n-radio-group v-model:value="formInfo.model.anomarkTemp" name="radiogroup">
                <n-space>
                  <n-radio value="" class="flex items-center">
                    <span>平台存储</span>
                  </n-radio>
                  <n-radio value="" class="flex items-center">
                    <span>BOS存储</span>
                    <n-tooltip trigger="hover">
                      <template #trigger>
                        <!--<SvgIcon icon="ri:question-line" class="inline-block align-text-bottom text-18px ml-4px" />-->
                        <SvgIcon local-icon="ri--question-line"
                          class="inline-block align-text-bottom text-18px ml-4px" />
                      </template>
                      指定新创建数据集后续的存储方式。非平台存储的数据集，
                      在进行数据管理、标注、处理时需用户自行保证数据地址有效。
                    </n-tooltip>
                  </n-radio>
                </n-space>
              </n-radio-group>
            </n-form-item>
          </n-form>
        </n-card>
      </div>
    </div>
    <div class="footer flex_start">
      <n-button type="info" @click="handleCreImp()">创建并导入</n-button>
      <n-button @click="handleSuccess()">完成创建</n-button>
      <n-button @click="handleCancel()">取消</n-button>
    </div>
  </div>
</template>

<style scoped lang="scss">
.flex_start {
  display: flex;
  justify-content: flex-start;
  align-items: center;
}

.flex_center {
  display: flex;
  justify-content: center;
  align-items: center;
}

.flex_between {
  display: flex;
  justify-content: space-around;
  align-items: center;
}

.flex_around {
  display: flex;
  justify-content: space-around;
  align-items: center;
}

.flex_col_start {
  display: flex;
  flex-direction: column;
  justify-content: flex-start;
  align-items: center;
}

.flex_col_center {
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
}

.wrap_container {
  padding: 0;
  width: 100%;
  height: 100%;
  background-color: #f7f7f9;

  .header {
    width: 100%;
    padding: 0 16px;
    box-sizing: border-box;
    height: 48px;
    background-color: #fff;
    display: flex;
    justify-content: flex-start;
    align-items: center;

    .h_back {
      span {
        color: #303540;
        font-size: 12px;
        cursor: pointer;
      }
    }

    .h_title {
      margin-left: 16px;
      font-size: 16px;
      color: #151b26;
      font-weight: 500;
    }
  }

  .content {
    padding: 16px 24px;
    box-sizing: border-box;
    width: 100%;
    flex: 1;

    .item_card {
      width: 100%;
      min-height: 240px;

      .form_annotationType {
        padding: 8px;
        box-sizing: border-box;
        width: 128px;
        height: 128px;
        border: 1px solid #ddd;
        border-radius: 4px;
      }
    }
  }

  .footer {
    width: 100%;
    height: 60px;
    background-color: #fff;
    padding: 0 24px;
    box-sizing: border-box;

    .n-button {
      margin-right: 24px;
    }
  }
}
</style>
