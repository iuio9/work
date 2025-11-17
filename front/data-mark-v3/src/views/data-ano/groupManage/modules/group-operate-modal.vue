<script setup lang="ts">
import { computed, reactive, watch } from 'vue';
import { useFormRules, useNaiveForm } from '@/hooks/common/form';
import { $t } from '@/locales';
import { teamAdd, teamEdit, getDeptByUserList } from '@/service/api/ano';
import { NCascader, NInput, NTag, NButton, NPopconfirm } from "naive-ui";

defineOptions({
  name: 'GroupOperateModal'
});

export type OperateType = NaiveUI.TableOperateType | 'addChild';

interface Props {
  /** the type of operation */
  operateType: OperateType;
  /** the edit menu data or the parent menu data when adding a child menu */
  rowData?: any;
  /** all pages */
  allPages: string[];
}

const props = defineProps<Props>();

interface Emits {
  (e: 'submitted'): void;
}

const emit = defineEmits<Emits>();

const visible = defineModel<boolean>('visible', {
  default: false
});

const { formRef, validate, restoreValidation } = useNaiveForm();
const { defaultRequiredRule } = useFormRules();

const title = computed(() => {
  const titles: Record<OperateType, string> = {
    add: "创建团队",
    edit: "编辑团队"
  };
  return titles[props.operateType];
});

const model: any = reactive(createDefaultModel());

function createDefaultModel(): any {
  return {
    teamName: null,
    teamDec: null,
    teamType: "1",
  };
}


const rules: Record<string, App.Global.FormRule> = {
  teamName: defaultRequiredRule,
  url: defaultRequiredRule,
  requestType: defaultRequiredRule,
  teamType: defaultRequiredRule,
};

/** the enabled role options */
function handleInitModel() {
  Object.assign(model, createDefaultModel());

  if (!props.rowData) return;

  if (props.operateType === 'add') {
    groupData.value = [];
  }

  if (props.operateType === 'edit') {
    Object.assign(model, props.rowData, {
      teamType: String(props.rowData.teamType),
    });
    const { userList } = model;
    if (userList && userList.length > 0) {
      groupData.value = userList.map(val => {
        return {
          userId: val.userId,
          remark: val.remark,
          status: "启用"
        }
      });
    }
  }
}

function closeDrawer() {
  visible.value = false;
}

interface User {
  userId: number | string;
  remark: string;
}

interface TeamParams {
  teamName: string;
  teamDec: string;
  userList: User[];
  [key: string]: any;
}

async function handleSubmit() {
  try {
    // 先进行验证
    await validate();

    // 提取公共的数据处理逻辑
    const userList = groupData.value.map((val) => ({
      userId: val.userId,
      remark: val.remark
    }));

    const params: TeamParams = {
      teamName: model.teamName,
      teamDec: model.teamDec,
      userList,
      id: props?.rowData.id,
      teamType: model.teamType,
    };

    let res;
    if (props.operateType === "add") {
      res = await teamAdd(params);
    } else if (props.operateType === "edit") {
      res = await teamEdit(params);
    } else {
      // 处理未知的操作类型
      throw new Error(`未知的操作类型: ${props.operateType}`);
    }

    // 检查响应结果
    if (res && res.data > 0) {
      const successTip = props.operateType === 'add' ? '创建成功' : '编辑成功';
      window.$message?.success(successTip);
      closeDrawer();
      emit('submitted');
    } else {
      // 处理操作失败的情况
      const errorTip = props.operateType === 'add' ? '创建失败' : '编辑失败';
      // window.$message?.error(errorTip);
    }
  } catch (error) {
    // 统一处理异常
    console.error('提交表单时发生错误:', error);
    window.$message?.error('提交表单时发生错误，请稍后重试');
  }
}

watch(visible, () => {
  if (visible.value) {
    handleInitModel();
    restoreValidation();
  }
});

// newCode

const groupData = ref<any>([]);
const groupColumns = ref<any>([
  {
    title: '所属成员',
    key: 'userName',
    width: 200,
    render: (row: any, index: string | number) => {
      return [
        h(
          NCascader,
          {
            value: row.userId,
            multiple: false,
            cascade: true,
            clearable: true,
            checkStrategy: "child",
            options: options.value,
            onUpdateValue(v) {
              groupData.value[index].userId = v
            }
          }
        )
      ]
    }
  },
  {
    title: '备注',
    key: 'remark',
    render: (row: any, index: string | number) => {
      return h(NInput, {
        value: row.remark,
        placeholder: '长度不超过20字',
        clearable: true, // 可选属性，是否显示清除按钮
        maxlength: 20,   // 可选属性，最大字符长度
        showCount: true,  // 可选属性，是否显示字符计数
        onUpdateValue(v) {
          groupData.value[index].remark = v
        }
      })
    }
  },
  {
    title: '状态',
    key: 'status',
    render: (row: any) => {
      return [
        h(NTag, { type: 'success' }, '启用')
      ]
    }
  },
  {
    title: '操作',
    key: 'action',
    render: (row: any) => {
      return [
        h("div", {
          class: "flex-center gap-8px"
        }, [
          h(NPopconfirm, {
            onPositiveClick: (event) => handleOperate(event, row, 'delete')
          }, {
            default: () => h("span", {}, $t("common.confirmDelete")),
            trigger: () => h(NButton, {
              type: "error",
              ghost: true,
              size: "small",
            }, $t("common.delete"))
          })
        ])
      ]
    }
  }
]);
const options = ref<any>([]);
const teamTypes = ref<any>([
  { label: "标注团队", value: "1" },
  { label: "审核团队", value: "2" },
]);

function convertToCascaderOptions(data) {
  return data.map(item => {
    const option = {
      label: item.deptName,
      value: `d_${item.deptId}`,
      children: item.userList.map(user => {
        // const label = `${user.userName}_${user.nickName}`
        const label = `${user.nickName}`
        return {
          label: label,
          value: user.userId
        }
      })
    };
    return option;
  });
}

const getData = async () => {
  const res = await getDeptByUserList();
  options.value = convertToCascaderOptions(res.data);
};

const addPerson = () => {
  const newPerson = {
    userId: undefined,
    remark: '',
    status: '启用'
  };
  groupData.value.push(newPerson);
};

const handleOperate = (event: any, row: any, type: string) => {
  if (type === 'delete') {
    groupData.value.splice(groupData.value.indexOf(row), 1);
  }
}

onMounted(() => {
  getData();
});
</script>

<template>
  <NModal v-model:show="visible" :title="title" preset="card" class="w-900px">
    <NScrollbar class="h-480px pr-20px">
      <NForm ref="formRef" :model="model" :rules="rules" label-placement="left" :label-width="120">
        <NFormItem label="团队名称" path="teamName">
          <NInput v-model:value="model.teamName" placeholder="请输入团队名称" />
        </NFormItem>
        <NFormItem label="团队类型" path="teamType">
          <NRadioGroup v-model:value="model.teamType" name="teamType">
            <NSpace>
              <NRadio v-for="val in teamTypes" :key="val.label" :value="val.value">
                {{ val.label }}
              </NRadio>
            </NSpace>
          </NRadioGroup>
        </NFormItem>
        <NFormItem label="团队描述" path="modelDesc">
          <NInput type="textarea" v-model:value="model.teamDec" placeholder="请输入团队描述" />
        </NFormItem>
        <NFormItem span="24" label="团队成员(必填)">
          <!-- <n-cascader multiple cascade :check-strategy="checkStrategy" :options="options"
            @update:value="handleUpdateValue" /> -->
          <div class="w-full h-auto flex-col justify-around items-start gap-4px">
            <n-data-table :columns="groupColumns" :data="groupData" :pagination="false" :bordered="false" />
            <NButton type="primary" quaternary size="small" @click="addPerson">
              <template #icon>
                <svg-icon local-icon="add_person" class="text-[24px] text-[#2d7a67]"></svg-icon>
              </template>
              添加成员
            </NButton>
          </div>
        </NFormItem>
      </NForm>
    </NScrollbar>
    <template #footer>
      <NSpace justify="end" :size="16">
        <NButton @click="closeDrawer">{{ $t('common.cancel') }}</NButton>
        <NButton type="primary" @click="handleSubmit">{{ $t('common.confirm') }}</NButton>
      </NSpace>
    </template>
  </NModal>
</template>

<style scoped></style>
