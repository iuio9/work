<script setup lang="ts">
import { NButton, NForm, NInput, TreeOption } from "naive-ui";
import type { CascaderOption } from "naive-ui";
import { useFormRules, useNaiveForm } from "@/hooks/common/form";
import { useBoolean } from "~/packages/hooks";
import { selectLabelList } from "@/service/api/ano";
import {
  dataSetImport,
  fetchDataSetAdd,
  getSonIdByLabelGroupIds,
} from "@/service/api/dataManage";
import UploadOperateDrawer from "./modules/upload-operate-drawer.vue";

import _ from "lodash";

const route = useRoute();
const router = useRouter();
const { bool: drawerVisible, setTrue: openDrawer } = useBoolean();

const currentStep = ref<number | null>(1);

const uploadRef = ref(null);
const { formRef, validate, restoreValidation } = useNaiveForm();
const model = ref<any | null>({
  markStatus: "0",
  anoType: "0",
  fileList: [],
  imgList: [],
  uploadList: [],
  tagSelectionMode: "group",
  groupIds: [],
  tagIds: [],
});
const { defaultRequiredRule, createRequiredRule } = useFormRules();
const rules: Record<string, any> = {
  groupName: createRequiredRule("数据集名称不能为空！"),
  markStatus: createRequiredRule("请选择数据标注状态！"),
  dataTypeId: defaultRequiredRule,
};
const anoOptions = ref<any>([
  { value: "0", label: "图像分割" },
  { value: "1", label: "物体检测" },
]);
const isUpSuccess = ref<boolean>(false);
// cascader
const checkStrategy = ref<"all" | "parent" | "child">("parent");
const showPath = ref<Boolean>(true);
const cascade = ref<Boolean>(true);
const responsiveMaxTagCount = ref<Boolean>(false);
const filterable = ref<Boolean>(false);
const hoverTrigger = ref<Boolean>(false);
const clearFilterAfterSelect = ref<Boolean>(true);
const tagOptions = getOptions();

// 导入方式
const importOptions = ref<any>([
  {
    value: "0",
    label: "本地导入",
    children: [
      // 上传图片 上传压缩包
      { value: "0-0", label: "上传图片" },
      { value: "0-1", label: "上传压缩包" },
    ],
  },
]);
const importCheckStrategyIsChild = ref(true);
const importShowPath = ref(true);
const importHoverTrigger = ref(false);
const importFilterable = ref(false);

function getOptions(depth = 2, iterator = 1, prefix = "") {
  const length = 12;
  const options: CascaderOption[] = [];
  for (let i = 1; i <= length; ++i) {
    if (iterator === 1) {
      options.push({
        value: `v-${i}`,
        label: `l-${i}`,
        disabled: i % 5 === 0,
        children: getOptions(depth, iterator + 1, `${String(i)}`),
      });
    } else if (iterator === depth) {
      options.push({
        value: `v-${prefix}-${i}`,
        label: `l-${prefix}-${i}`,
        disabled: i % 5 === 0,
      });
    } else {
      options.push({
        value: `v-${prefix}-${i}`,
        label: `l-${prefix}-${i}`,
        disabled: i % 5 === 0,
        children: getOptions(depth, iterator + 1, `${prefix}-${i}`),
      });
    }
  }
  return options;
}

function handleBack() {
  router.back();
  // router.replace({
  //   name: 'data-manage_map'
  // });
}

async function handleOperate(sign: "next" | "prev" | "create") {
  if (sign === "next" && currentStep.value < 4) {
    await validate();
    currentStep.value += 1;
  } else if (sign === "prev" && currentStep.value > 1) {
    currentStep.value -= 1;
  } else if (sign === "create") {
    await validate();
    const params = {
      tagSelectionMode: model.value.tagSelectionMode ?? 'group',
      version: 1,
      dataTypeId: route.query.dataTypeId ? route.query.dataTypeId : undefined,
      groupName: model.value.groupName,
      anoType: model.value.anoType,
      markStatus: model.value.markStatus,
      importMode: model.value.importMode
        ? model.value.importMode.split("-")[1]
        : undefined,
      fileIds: (model.value.imgList as any[]).map((item) => item.id).join(","),
      groupIds: model.value.tagSelectionMode === 'group' ?
        Array.isArray(model.value.groupIds) && model.value.groupIds.length !== 0
          ? model.value.groupIds.join(",")
          : undefined : undefined,
      tagIds: model.value.tagSelectionMode === 'single' ?
        Array.isArray(model.value.groupIds) && model.value.groupIds.length !== 0
          ? model.value.groupIds.join(",")
          : undefined : undefined,
      sonId: route.query.sonId,
    };
    if (route.query.sign === "mapToImport") {
      const res = await dataSetImport(params);
      if (res.data) {
        window.$message?.success("导入数据集成功！");
        router.replace({
          name: "data-manage_map",
          query: {
            dataTypeId: route.query.dataTypeId,
            sonId: res.data.sonId,
            groupId: res.data.groupId,
          },
        });
      }
    } else {
      const res = await fetchDataSetAdd(params);
      if (res.data) {
        window.$message?.success("创建数据集成功！");
        router.replace({
          name: "data-manage_map",
          query: {
            dataTypeId: route.query.dataTypeId,
            sonId: res.data.sonId,
            groupId: res.data.groupId,
          },
        });
      }
    }
  }
}

function uploadSubmit(data: any) {
  model.value.imgList = data.fileList;
}

function handleRadioChange() {
  uploadRef.value.resetFiles();
  model.value.imgList = [];
  model.value.importMode = "0-1";
}

function handleRadioChange1() {
  model.value.groupIds = [];
  model.value.tagIds = [];
}

function handleUpdateValue(value: any, type: string) {
  if (type === "import") {
    model.value.imgList = [];
  }
}

// 标签组列表
type LabelValueChildrenItem = {
  label: string;
  value: number;
  disabled?: boolean; // 可选的禁用属性
  color?: string; // 可选的颜色属性
  children?: LabelValueChildrenItem[];
};

const groupOptions: any = ref<any>([]);
const singleTagOptions: any = ref<any>([]);

function transformData(data: any[], type: string = 'group'): LabelValueChildrenItem[] {
  return data.map((group) => ({
    label: group.labelGroupName,
    value: group.id.toString(), // 确保 value 是字符串类型
    disabled: false, // 禁用状态
    children:
      group.list.length > 0
        ? group.list.map((item) => ({
          label: item.labelName,
          value: type === 'group' ? `${group.id}-${item.id}` : item.id.toString(), // 组合父级和子级的 id
          color: item.labelColor, // 添加颜色属性
          disabled: type === 'group' ? true : false,
        }))
        : [],
  }));
}
async function getTagGroupList() {
  const res = await selectLabelList();
  let dataList: any;
  let singleTagList: any;
  if (res.data) {
    dataList = transformData(res.data, 'group');
    singleTagList = transformData(res.data, 'single');
  } else {
    dataList = [];
    singleTagList = [];
  }
  groupOptions.value = [...dataList];
  singleTagOptions.value = [...singleTagList];
}

async function findGroupById(sonId: string) {
  const res = await getSonIdByLabelGroupIds({ sonId });
  if (res.data) {
    model.value.groupIds = res.data.map((val) => `${val}`);
  }
}

function findGroupNameById(id: string) {
  const findGroup = groupOptions.value.find((group) => group.value === id);
  return findGroup?.label ?? "";
}

function findTagGroupNameById(id: string) {
  const mergedChildren = singleTagOptions.value.flatMap(item => item.children);
  const findGroup = mergedChildren.find((group) => group.value === id);
  return findGroup?.label ?? "";
}

onMounted(() => {
  const { operateStep, groupName, anoType, sonId, tagSelectionMode } = route.query as any;
  getTagGroupList();
  findGroupById(sonId);
  if (operateStep) {
    currentStep.value = Number(operateStep);
    model.value.groupName = groupName as string;
    model.value.anoType = (anoType as string) ?? "0";
    model.value.tagSelectionMode = (tagSelectionMode as string) ?? 'group';
  }
});
</script>

<template>
  <div class="wrap_container h-full w-full">
    <NCard class="h-full" title="创建数据集" :segmented="{ content: true, footer: 'soft' }">
      <template #header-extra>
        <NButton type="primary" ghost size="small" @click="handleBack()">
          <template #icon>
            <SvgIcon local-icon="oui--return-key" class="text-[24px]"></SvgIcon>
          </template>
          返回数据集列表
        </NButton>
      </template>
      <div class="wrap_content h-full w-full flex items-center justify-around">
        <div class="item_l_stepMain border-[] h-full w-15% flex items-center justify-center border-r-2">
          <NSteps :current="currentStep" status="process" vertical class="ml-24px box-border h-full pt-25%">
            <NStep title="基本信息" />
            <NStep title="导入文件" />
            <NStep title="关联标签组" />
            <NStep title="确认提交" />
          </NSteps>
        </div>
        <div class="ml-24px box-border h-full w-85% overflow-y-auto pt-16px">
          <!--基本信息-->
          <div v-if="currentStep == 1" class="item_step h-auto w-50% flex-col justify-start">
            <div class="item_step_title text-[20px] font-550">基本信息</div>
            <div class="mt-24px">
              <NForm ref="formRef" :model="model" :rules="rules" label-placement="left" :label-width="100">
                <NGrid responsive="screen" item-responsive :cols="1">
                  <!--数据集名称-->
                  <NFormItemGi span="24 m:24" label="数据集名称:" path="groupName">
                    <NInput v-model:value="model.groupName" placeholder="请输入数据集名称" />
                  </NFormItemGi>
                  <!--数据集版本-->
                  <NFormItemGi span="24 m:24" label="数据集版本:" path="setName">V1</NFormItemGi>
                  <!--标注类型-->
                  <NFormItemGi span="24 m:24" label="标注类型:" path="setName">
                    <NRadioGroup v-model:value="model.anoType" name="radiobuttongroup1">
                      <NRadioButton v-for="item in anoOptions" :key="item.value" :value="item.value"
                        :label="item.label" />
                    </NRadioGroup>
                  </NFormItemGi>
                </NGrid>
              </NForm>
            </div>
          </div>
          <!--导入文件-->
          <div v-if="currentStep == 2" class="item_step h-auto w-50% flex-col justify-start">
            <div class="item_step_title text-[20px] font-550">导入文件</div>
            <div class="mt-24px">
              <NForm ref="formRef" :model="model" :rules="rules" label-placement="left" :label-width="120">
                <NGrid responsive="screen" item-responsive :cols="1">
                  <!--数据标注状态-->
                  <NFormItemGi span="24 m:24" label="数据标注状态:" path="markStatus">
                    <NRadioGroup v-model:value="model.markStatus" name="radiogroup" @update:value="handleRadioChange">
                      <NSpace>
                        <NRadio value="0" class="flex items-center">
                          <span>无标注信息</span>
                        </NRadio>
                        <NRadio value="1" class="flex items-center">
                          <span>有标注信息</span>
                        </NRadio>
                      </NSpace>
                    </NRadioGroup>
                  </NFormItemGi>
                  <!--导入方式-->
                  <NFormItemGi span="24 m:24" label="导入方式:" path="setName">
                    <!-- 级联选择器 naive ui -->
                    <NCascader v-model:value="model.importMode" placeholder="请选择导入方式" :disabled="model.markStatus == 1"
                      :expand-trigger="importHoverTrigger ? 'hover' : 'click'" :options="importOptions" :check-strategy="importCheckStrategyIsChild ? 'child' : 'all'
                        " :show-path="importShowPath" :filterable="importFilterable"
                      @update:value="handleUpdateValue($event, 'import')" />
                  </NFormItemGi>
                  <!--上传文件-->
                  <NFormItemGi v-if="model.importMode" span="24 m:24" :label="model.importMode === '0-0' ? '上传图片' : '上传压缩包'
                    " path="fileUpload">
                    <NButton type="primary" ghost @click="() => openDrawer()">
                      <template #icon>
                        <SvgIcon local-icon="ep--upload-filled" class="text-[24px]"></SvgIcon>
                      </template>
                      {{
                        model.importMode === "0-0" ? "上传图片" : "上传压缩包"
                      }}
                    </NButton>
                    <div v-if="isUpSuccess" class="ml-16px flex items-center justify-start">
                      <NSpin size="small" />
                      <div class="ml-8px text-14px">
                        文件异步上传中... 请稍等!!!
                      </div>
                    </div>
                    <div v-show="model.imgList.length !== 0" class="ml-16px flex items-center justify-start">
                      <span>已上传</span>
                      <span>{{ model.imgList.length }}</span>
                      <span>{{
                        importMode === "0-0" ? "张图片" : "个文件"
                      }}</span>
                    </div>
                  </NFormItemGi>
                </NGrid>
              </NForm>
            </div>
            <UploadOperateDrawer ref="uploadRef" v-model:visible="drawerVisible" v-model:isUpSuccess="isUpSuccess"
              v-model:markStatus="model.markStatus" v-model:importMode="model.importMode" @submitted="uploadSubmit" />
          </div>
          <!--关联标签组-->
          <div v-if="currentStep == 3" class="item_step h-full w-60% w-full flex-col justify-start">
            <div class="item_step_title text-[20px] font-550">关联标签组</div>

            <div class="mt-24px min-h-0 flex-1 overflow-y-auto">
              <NForm ref="formRef" :model="model" :rules="rules" label-placement="left" :label-width="100">
                <NGrid responsive="screen" item-responsive :cols="1">
                  <!-- 添加单选按钮组用于选择标签选择方式 -->
                  <NFormItemGi span="24 m:24" label="关联标签方式:" path="">
                    <n-radio-group v-model:value="model.tagSelectionMode" name="tag-selection-mode"
                      @update:value="handleRadioChange1">
                      <n-space>
                        <n-radio value="group">按标签组勾选</n-radio>
                        <n-radio value="single">按单标签勾选</n-radio>
                      </n-space>
                    </n-radio-group>
                  </NFormItemGi>
                  <!--标签组选择-->
                  <NFormItemGi span="24 m:24" :label="model.tagSelectionMode === 'group' ? '标签组选择:' : '单标签选择:'" path="">
                    <NCascader v-model:value="model.groupIds" multiple clearable placeholder="选择指定标签组下的标签（*可多选）"
                      :max-tag-count="responsiveMaxTagCount ? 'responsive' : undefined"
                      :expand-trigger="hoverTrigger ? 'hover' : 'click'"
                      :options="model.tagSelectionMode === 'group' ? groupOptions : singleTagOptions" :cascade="cascade"
                      :check-strategy="model.tagSelectionMode === 'group' ? 'all' : 'child'" :show-path="showPath"
                      :filterable="filterable" :clear-filter-after-select="clearFilterAfterSelect" />
                  </NFormItemGi>
                </NGrid>
              </NForm>
            </div>
          </div>
          <!--确认提交-->
          <div v-if="currentStep == 4" class="item_step h-auto w-full flex-col justify-start">
            <div class="item_step_title text-[20px] font-550">确认提交</div>
            <div class="mt-24px h-auto w-full">
              <NCollapse class="w-full" default-expanded-names="1" accordion>
                <NCollapseItem title="基本信息" name="1">
                  <div class="wrap_collapse ml-24px w-60% flex-col items-start justify-start gap-12px">
                    <div class="flex items-center justify-start gap-8px">
                      <span>数据集组名称:</span>
                      <span>{{ model.groupName }}</span>
                    </div>
                    <div class="flex items-center justify-start gap-8px">
                      <span>数据集版本:</span>
                      <span>V1</span>
                    </div>
                    <div class="flex items-center justify-start gap-8px">
                      <span>标注类型:</span>
                      <span>{{ anoOptions[model.anoType].label }}</span>
                    </div>
                  </div>
                </NCollapseItem>
                <NCollapseItem title="导入文件" name="2">
                  <div class="wrap_collapse ml-24px w-80% flex-col items-start justify-start gap-12px">
                    <div class="flex items-center justify-start gap-8px">
                      <span>数据标注状态:</span>
                      <span>{{
                        model.markStatus == 0 ? "无标注信息" : "有标注信息"
                      }}</span>
                    </div>
                    <div class="flex items-center justify-start gap-8px">
                      <span>导入方式:</span>
                      <span>本地上传</span>
                    </div>
                    <div class="w-full flex items-start justify-start gap-8px">
                      <span>文件列表:</span>
                      <NVirtualList style="width: 80%; max-height: 340px" :item-size="42" :items="model.imgList">
                        <template #default="{ item }">
                          <div class="h-auto w-full flex flex-col items-start justify-start">
                            <div :key="item.key"
                              class="item h-44px w-full flex items-center justify-between py-8px hover:bg-[#ebf7ed]">
                              <div class="w-90% flex items-center justify-start">
                                <span class="ml-4px w-90% truncate">{{
                                  item.name
                                }}</span>
                                <NButton v-show="item.status && item.status === 'finished'
                                  " quaternary type="primary">
                                  已完成
                                </NButton>
                              </div>
                            </div>
                          </div>
                        </template>
                      </NVirtualList>
                    </div>
                  </div>
                </NCollapseItem>
                <NCollapseItem title="关联标签" name="3">
                  <div class="wrap_collapse ml-24px w-60% flex-col items-start justify-start gap-12px">
                    <div class="flex items-start justify-start gap-8px">
                      <span class="block w-128px" v-if="model.tagSelectionMode === 'group'">标签组列表:</span>
                      <span class="block w-128px" v-else>单标签列表:</span>
                      <div class="w-full flex items-center justify-start gap-14px"
                        v-if="model.tagSelectionMode === 'group'">
                        <NTag v-for="(item, index) of model.groupIds" :key="index" type="success">
                          {{ findGroupNameById(item) }}
                        </NTag>
                      </div>
                      <div class="w-full flex items-center justify-start gap-14px flex-wrap" v-else>
                        <NTag v-for="(item, index) of model.groupIds" :key="index" type="success">
                          {{ findTagGroupNameById(item) }}
                        </NTag>
                      </div>
                    </div>
                  </div>
                </NCollapseItem>
              </NCollapse>
            </div>
          </div>
        </div>
      </div>
      <template #footer>
        <div class="wrap_footer h-auto w-full flex items-center justify-center gap-24px">
          <NButton type="primary" size="small" :disabled="isUpSuccess" @click="handleOperate('create')">
            <template #icon>
              <SvgIcon local-icon="icon-park-outline--file-success" class="text-[24px]"></SvgIcon>
            </template>
            完成创建
          </NButton>
          <NButton type="primary" ghost size="small" :disabled="currentStep == 4 || (currentStep == 2 && isUpSuccess)"
            @click="handleOperate('next')">
            <template #icon>
              <SvgIcon local-icon="ic--round-next-plan" class="text-[24px]"></SvgIcon>
            </template>
            下一步
          </NButton>
          <NButton type="primary" ghost size="small" :disabled="currentStep == 1" @click="handleOperate('prev')">
            <template #icon>
              <SvgIcon local-icon="ph--key-return-bold" class="text-[24px]"></SvgIcon>
            </template>
            返回上一步
          </NButton>
        </div>
      </template>
    </NCard>
  </div>
</template>

<style scoped></style>
