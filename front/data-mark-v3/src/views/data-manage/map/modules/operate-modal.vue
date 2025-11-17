<script setup lang="tsx">
import {reactive, watch} from 'vue';
import {useFormRules, useNaiveForm} from '@/hooks/common/form';
import {getDicts} from "@/service/api";
import {fetchDataSetAddDataVersion} from "@/service/api/dataManage";

defineOptions({
  name: 'MenuOperateModal'
});

export type OperateType = NaiveUI.TableOperateType | 'addChild';

interface Props {
  /** the type of operation */
  operateType: OperateType;
  /** the edit menu data or the parent menu data when adding a child menu */
  rowData?: Api.SystemManage.Menu | null;
  /** all pages */
  allPages: string[];
}

const props = defineProps<Props>();

interface Emits {
  (e: 'success'): void;
}

const emit = defineEmits<Emits>();

const visible = defineModel<boolean>('visible', {
  default: false
});

const {formRef, validate, restoreValidation} = useNaiveForm();
const {defaultRequiredRule} = useFormRules();

type Model = Pick<
  Api.SystemManage.Menu,
  | 'menuType'
  | 'menuName'
  | 'routeName'
  | 'routePath'
  | 'component'
  | 'order'
  | 'i18nKey'
  | 'icon'
  | 'iconType'
  | 'status'
  | 'parentId'
  | 'keepAlive'
  | 'constant'
  | 'href'
  | 'hideInMenu'
  | 'activeMenu'
  | 'multiTab'
  | 'fixedIndexInTab'
> & {
  query: NonNullable<Api.SystemManage.Menu['query']>;
  buttons: NonNullable<Api.SystemManage.Menu['buttons']>;
  layout: string;
  page: string;
  pathParam: string;
};

const model: Model = reactive(createDefaultModel());

function createDefaultModel(): Model {
  return {
    newVersion: "1",
    isExtendHisVersion: true,
    groupName: null,
    dataType: '',
    markType: '',
    markTemp: "",
    version: "",
  };
}

type RuleKey = Extract<keyof Model, 'menuName' | 'status' | 'routeName' | 'routePath'>;

const rules: Record<RuleKey, App.Global.FormRule> = {
  menuName: defaultRequiredRule,
  status: defaultRequiredRule,
  routeName: defaultRequiredRule,
  routePath: defaultRequiredRule
};

function closeDrawer() {
  visible.value = false;
}

async function handleSuccess() {
  // await validate();
  // request
  const params = {
    version: model.version,
    newVersion: model.newVersion.slice(1,),
    remark: model.remark,
    groupId: model.groupId,
    dataType: model.isExtendHisVersion ? 0 : Number(model.dataType),
    markType: model.isExtendHisVersion ? 0 : Number(model.markType),
    markTemp: model.isExtendHisVersion ? 0 : Number(model.markTemp),
    isInherit: Number(model.isExtendHisVersion)
  };
  const res = await fetchDataSetAddDataVersion(params);
  if(res.data?.status == 1) {
    window.$message?.success("新增数据集成功！");
    // storage
    // const mapArrStr = JSON.parse(localStorage.getItem('mapArr'));
    // let mapArr = [];
    // if(res.data.sonId && res.data.groupId) {
    //   if(mapArrStr) {
    //     mapArr.push({
    //       sonId: res.data.sonId,
    //       groupId: res.data.groupId,
    //     })
    //   } else {
    //     mapArr = [{
    //       sonId: res.data.sonId,
    //       groupId: res.data.groupId,
    //     }];
    //   }
    //   localStorage.setItem("mapArr", JSON.stringify(mapArr));
    // }
    // ----------------------------------------------------------------
    closeDrawer();
    emit('success');
  }
}

let historyOpts = reactive([]);

function handleInitModel() {
  Object.assign(model, createDefaultModel(), props.rowData);
}

watch(visible, () => {
  if (visible.value) {
    handleInitModel();
    restoreValidation();
    historyOpts = (model.dataSonResponseList).map((item, index) => {
      return {
        value: `${item.version}`,
        label: `V${item.version}`
      }
    });
    model.remark = "";
    model.newVersion = `V${((model.dataSonResponseList)[0]).version + 1}`;
    model.version = historyOpts[0].value;
  }
});

// data
const mapObj = ref<any>({
  typeList: [],
  verList: [],
  anoList: [],
  temList: []
});

watch(() => model.isExtendHisVersion, (val) => {
  if (!val) {
    const dataType = (mapObj.value.typeList)[0].id;
    model.dataType = dataType;
    mapObj.value.anoList = (mapObj.value.typeList).find((item: any) => {
      return `${item.id}` === `${dataType}`;
    }).children;
    // model.markType = (mapObj.value.anoList)[0].id;
  }
},{
  deep: true
});

watch(() => model.markType, (val) => {
  mapObj.value.temList = (mapObj.value.anoList).find((item: any) => {
    return `${item.id}` === `${val}`;
  }).children;
  model.markTemp = (mapObj.value.temList)[0].id;
}, {
  deep: true,
})

// 获取对应字典信息
const getDictData = async (typeId = "6") => {
  const params = { typeId };
  const res = await getDicts(params);
  return res.data;
}

// radio click
const handleRadioClick = (sign: string, rowData: any) => {
  if(sign === "dataType") {
    const { id } = rowData;
    model.dataType = id;
    mapObj.value.anoList = (mapObj.value.typeList).find((item: any) => {
      return `${item.id}` === `${id}`;
    }).children;
    model.markType = (mapObj.value.anoList)[0].id;
    mapObj.value.temList = (mapObj.value.anoList).find((item: any) => {
      return `${item.id}` === `${model.markType}`;
    }).children;
    model.markTemp = (mapObj.value.temList)[0].id;
  }
  if(sign === "markTemp") {
    const { id } = rowData;
    model.markTemp = id;
  }
  if(sign === "markType") {
    const { id } = rowData;
    model.markType = id;
    mapObj.value.temList = (mapObj.value.anoList).find((item: any) => {
      return `${item.id}` === `${id}`;
    }).children;
    model.markTemp = (mapObj.value.temList)[0].id;
  }
}

onMounted(async () => {
  mapObj.value.typeList = await getDictData("6");
  model.dataType = (mapObj.value.typeList)[0].id;
});


</script>

<template>
  <NModal v-model:show="visible" title="新增数据集版本" preset="card" class="w-800px">
    <NScrollbar class="h-480px pr-20px">
      <NForm ref="formRef" :model="model" :rules="rules" label-placement="left" :label-width="100">
        <NGrid responsive="screen" item-responsive>
          <NFormItemGi span="24 m:24" label="数据集版本" path="menuType">
            {{ model.newVersion }}
          </NFormItemGi>
          <NFormItemGi span="24 m:24" label="备注信息" path="icon">
            <n-input
              v-model:value="model.remark"
              type="textarea"
              placeholder="请输入备注信息"
            />
          </NFormItemGi>
          <NFormItemGi span="24 m:24" label="继承历史版本" path="icon">
            <n-switch v-model:value="model.isExtendHisVersion"/>
          </NFormItemGi>
          <!-- <NFormItemGi span="24 m:24" label="标注类型" path="type" v-if="!model.isExtendHisVersion"> -->
          <NFormItemGi span="24 m:24" label="标注类型" path="type" v-show="false">
            <n-radio-group v-model:value="model.markType" name="type">
              <n-radio-button
                @click="handleRadioClick('markType', val)"
                v-for="(val) of mapObj.anoList" :value="val.id">{{ val.dictLabel }}</n-radio-button>
            </n-radio-group>
          </NFormItemGi>
          <!-- <NFormItemGi span="24 m:24" label="标注模板" path="type" v-if="!model.isExtendHisVersion"> -->
          <NFormItemGi span="24 m:24" label="标注模板" path="type" v-show="false">
            <n-radio-group v-model:value="model.markTemp" name="radiogroup">
              <n-space>
                <n-radio
                  @click="handleRadioClick('markTemp', item)"
                  v-for="item in mapObj.temList" :key="item.id" :value="item.id" class="flex items-center">
                  <span>{{ item.dictLabel }}</span>
                </n-radio>
              </n-space>
            </n-radio-group>
          </NFormItemGi>
          <NFormItemGi span="24 m:24" label="历史版本" path="icon" v-if="model.isExtendHisVersion">
            <n-select v-model:value="model.version" :options="historyOpts"/>
          </NFormItemGi>
        </NGrid>
      </NForm>
    </NScrollbar>
    <template #footer>
      <NSpace justify="end" :size="16">
        <NButton @click="closeDrawer">取消</NButton>
        <NButton type="primary" @click="handleSuccess">完成</NButton>
      </NSpace>
    </template>
  </NModal>
</template>

<style scoped></style>
