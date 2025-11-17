import { useAuthStore } from "@/store/modules/auth";

export default {
  mounted(el, binding, vnode) {
    const { value } = binding;
    // 字符串转数组
    const valueArr = value.split(',');
    const all_permission = "*:*:*";

    const authStore = useAuthStore();
    const permissions = authStore.userInfo.buttons;

    if (valueArr && valueArr instanceof Array && valueArr.length > 0) {
      const permissionFlag = valueArr

      const hasPermissions = permissions.some(permission => {
        return all_permission === permission || permissionFlag.includes(permission)
      })

      if (!hasPermissions) {
        el.parentNode && el.parentNode.removeChild(el)
      }
    } else {
      throw new Error(`请设置操作权限标签值`)
    }
  }
}
