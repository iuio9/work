<script setup lang="ts">
import { computed, shallowRef, watch } from 'vue';
import { $t } from '@/locales';
import { fetchGetAllPages, fetchGetMenuButtonTree, fetchAddRoleButton, fetchGetRoleButton } from '@/service/api';
import { NButton, NTag } from 'naive-ui';

defineOptions({
  name: 'MenuAuthModal'
});

interface Props {
  /** the roleId */
  roleId: number;
}

interface Emits {
  (e: 'treeSubmit'): any;
}

const emit = defineEmits<Emits>();


const props = defineProps<Props>();

const visible = defineModel<boolean>('visible', {
  default: false
});

function closeModal() {
  emit("treeSubmit", 111)
  visible.value = false;
}
const title = computed(() => $t('common.edit') + $t('page.manage.role.buttonAuth'));

const home = shallowRef('');

async function getHome() {
  home.value = 'home';
}

const pages = shallowRef<string[]>([]);
const tree = shallowRef<TreeNode[]>([]);

async function getTree() {
  const { error, data } = await fetchGetMenuButtonTree();

  if (!error) {
    const treeData: TreeNode[] = convertToTreeData(data);
    tree.value = treeData;
    console.log('tree.value: ', tree.value);
  }
}

const checks = shallowRef<number[]>([]);

async function getChecks() {
  const res: any = await fetchGetRoleButton({
    roleId: props.roleId
  });
  checks.value = res.data.menuIds.map(val => `${val}`) || [];
}

async function handleSubmit() {
  console.log(checks.value);
  // request
  const res = await fetchAddRoleButton({
    menuIds: checks.value,
    roleId: props.roleId,
  })
  if (res.data >= 1) {
    window.$message?.success?.($t('common.modifySuccess'));
    closeModal();
  }
}

// 原始数据类型
interface ButtonPermission {
  id: number;
  menuName: string;
  parentId: number;
  buttonName: string;
  permission: string;
  sort: number;
}

interface MenuData {
  menuName: string;
  buttonPermissions: ButtonPermission[];
}

// Naive UI 树形组件节点类型
interface TreeNode {
  key: string;
  label: string;
  children?: TreeNode[];
  [key: string]: any; // 允许额外字段
}

/**
 * 将权限数据转换为 Naive UI 树形组件所需格式
 * @param permissions - 原始权限数据
 * @returns 树形组件数据
 */
function convertToTreeData(permissions: MenuData[]): TreeNode[] {
  return permissions.map(menu => ({
    key: menu.menuName,
    label: menu.menuName,
    checkboxDisabled: true,
    children: menu.buttonPermissions.map(button => ({
      key: button.id.toString(),
      label: button.buttonName,
      permission: button.permission,
      sort: button.sort
    }))
  }));
}

function init() {
  getHome();
  getTree();
  getChecks();
}

watch(visible, val => {
  if (val) {
    init();
  }
});
</script>

<template>
  <NModal v-model:show="visible" :title="title" preset="card" class="w-480px">
    <NTree v-model:checked-keys="checks" :data="tree" checkable expand-on-click virtual-scroll block-line
      class="h-280px" />
    <template #footer>
      <NSpace justify="end">
        <NButton size="small" class="mt-16px" @click="closeModal">
          {{ $t('common.cancel') }}
        </NButton>
        <NButton type="primary" size="small" class="mt-16px" @click="handleSubmit">
          {{ $t('common.confirm') }}
        </NButton>
      </NSpace>
    </template>
  </NModal>
</template>

<style scoped></style>
