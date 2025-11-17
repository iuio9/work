<script setup lang="ts">
import { computed, shallowRef, watch } from 'vue';
import { $t } from '@/locales';
import { fetchGetAllPages, fetchGetMenuTree, fetchAddRoleMenu, fetchGetRoleMenus } from '@/service/api';
import { NTag } from 'naive-ui';

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

const title = computed(() => $t('common.edit') + $t('page.manage.role.menuAuth'));

const home = shallowRef('');

async function getHome() {
  console.log(props.roleId);

  home.value = 'home';
}

async function updateHome(val: string) {
  // request

  home.value = val;
}

const pages = shallowRef<string[]>([]);

// async function getPages() {
//   const { error, data } = await fetchGetAllPages();
//
//   if (!error) {
//     pages.value = data;
//   }
// }

const pageSelectOptions = computed(() => {
  const opts: CommonType.Option[] = pages.value.map(page => ({
    label: page,
    value: page
  }));

  return opts;
});

const tree = shallowRef<Api.SystemManage.MenuTree[]>([]);

function addDisabledProperty(items) {
    return items.map(item => ({
        ...item,
        disabled: item.label === 'home'
    }));
}

async function getTree() {
  const { error, data } = await fetchGetMenuTree();

  if (!error) {
    tree.value = addDisabledProperty(removeEmptyChildren(deepMapData(data)));
  }
}

// function deepMapData<T extends { meta?: { hideInMenu?: boolean }; children?: T[]; label?: string; zhLabel?: string }[]>(
//   data: T
// ): T {
//   return data.filter((item) => {
//     if (item.meta && item.meta.hideInMenu) {
//       return false;
//     }
//     if (item.children && item.children.length > 0) {
//       item.children = deepMapData(item.children);
//     } else {
//       delete item.children;
//     }
//     if (item.label) {
//       const label = `route.${item.label}`;
//       item.zhLabel = $t(label);
//     }
//     return true;
//   }) as T;
// }

function deepMapData<T extends {
  meta?: { hideInMenu?: boolean | number };
  children?: T[];
  label?: string;
  zhLabel?: string;
  isText?: boolean;
  buttonPermission?: any[]
}[]>(
  data: T
): T {
  return data.filter((item) => {
    if (item.hideInMenu === true || item.hideInMenu === 1) {
      return false;
    }

    if (item.children && item.children.length > 0) {
      item.children = deepMapData(item.children);
    } else {
      delete item.children;
    }

    if (item.label) {
      const label = `route.${item.label}`;
      item.zhLabel = item.isText ? item.label : $t(label);
    }

    if (item.buttonPermission) {
      item.children = item.buttonPermission.map(val => {
        return {
          prefix: () =>
            h(
              NTag,
              { type: 'success' },
              `${val.buttonName}`
            ),
          isText: true,
          label: "",
          ...val
        };
      });
    }

    return true;
  }) as T;
}

function removeEmptyChildren(data) {
  // 如果 data 是数组，则遍历数组中的每个元素
  if (Array.isArray(data)) {
    return data.map(item => removeEmptyChildren(item));
  }

  // 如果 data 是对象
  if (typeof data === 'object' && data !== null) {
    // 检查 children 属性是否存在且为空数组
    if (data.children && data.children.length === 0) {
      delete data.children;
    }

    // 递归处理子节点
    if (data.children) {
      data.children = removeEmptyChildren(data.children);
    }
  }

  return data;
}

const checks = shallowRef<number[]>([]);

async function getChecks() {
  // request
  const res: any = await fetchGetRoleMenus({
    roleId: props.roleId
  });
  checks.value = res.data.menuIds;
}

async function handleSubmit() {
  console.log(checks.value, props.roleId);
  // request
  const res = await fetchAddRoleMenu({
    menuIds: checks.value,
    roleId: props.roleId,
  })
  if (res.data >= 1) {
    window.$message?.success?.($t('common.modifySuccess'));
    closeModal();
  }
}

function init() {
  getHome();
  // getPages();
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
    <NTree v-model:checked-keys="checks" :data="tree" key-field="id" label-field="zhLabel" checkable expand-on-click
      virtual-scroll block-line class="h-280px" />
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
