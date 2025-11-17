import { request } from '../request';

/** get role list */
export function fetchGetRoleList(params?: Api.SystemManage.RoleSearchParams) {
  return request<Api.SystemManage.RoleList>({
    url: '/role/list',
    method: 'get',
    params
  });
}

/**
 * get all roles
 *
 * these roles are all enabled
 */
export function fetchGetAllRoles() {
  return request<Api.SystemManage.AllRole[]>({
    url: '/systemManage/getAllRoles',
    method: 'get'
  });
}

/** get user list */
export function fetchGetUserList(params?: Api.SystemManage.UserSearchParams) {
  return request<Api.SystemManage.UserList>({
    url: '/user/list',
    method: 'get',
    params
  });
}

/** get menu list */
export function fetchGetMenuList() {
  return request<Api.SystemManage.MenuList>({
    url: '/systemManage/getMenuList/v2',
    method: 'get'
  });
}

/** get all pages */
export function fetchGetAllPages() {
  return request<string[]>({
    url: '/systemManage/getAllPages',
    method: 'get'
  });
}

/** get menu tree */
export function fetchGetMenuTree() {
  return request<Api.SystemManage.MenuTree[]>({
    url: '/menu/getMenuTree',
    method: 'get'
  });
}

// /menu/getMenuButtonTree
export function fetchGetMenuButtonTree() {
  return request<Api.SystemManage.MenuTree[]>({
    url: '/menu/getMenuButtonTree',
    method: 'get'
  });
}

// /role/getRoleButton
export function fetchGetRoleButton(params: any) {
  return request<Api.SystemManage.MenuTree[]>({
    url: '/role/getRoleButton',
    method: 'get',
    params
  });
}

// role/addRoleButton post
export function fetchAddRoleButton(data: any) {
  return request<any>({
    url: '/role/addRoleButton',
    method: 'post',
    data
  });
}


// post /role/addRoleMenu
export function fetchAddRoleMenu(data: any) {
  return request<any>({
    url: '/role/addRoleMenu',
    method: 'post',
    data
  });
}

// ---------------------用户管理--------------------
export function fetchUserDel(ids: any[]) {
  return request<any>({
    url: '/user/remove',
    method: 'delete',
    data: {
      ids
    }
  });
}

export function fetchUserAdd(data: any) {
  return request<any>({
    url: '/user/add',
    method: 'post',
    data
  });
}

export function fetchUserEdit(data: any) {
  return request<any>({
    url: '/user/edit',
    method: 'post',
    data
  });
}

// 角色管理
export function fetchGetRoles() {
  return request<Api.SystemManage.AllRole[]>({
    url: '/role/getRoleSelect',
    method: 'get'
  });
}

export function fetchGetRoleMenus(params: any) {
  return request<Api.SystemManage.AllRole[]>({
    url: '/role/getRoleMenu',
    method: 'get',
    params
  });
}

export function fetchRoleDel(ids: any[]) {
  return request<any>({
    url: '/role/remove',
    method: 'delete',
    data: {
      ids
    }
  });
}

export function fetchRoleEdit(data: any) {
  return request<any>({
    url: '/role/edit',
    method: 'post',
    data
  });
}

export function fetchRoleAdd(data: any) {
  return request<any>({
    url: '/role/add',
    method: 'post',
    data
  });
}

// /user/resetPassword post
export function fetchResetPassword(data: any) {
  return request<any>({
    url: '/user/resetPassword',
    method: 'post',
    data
  });
}

// /user/resetDefaultPassword get
export function resetDefaultPassword(params: any) {
  return request<Api.SystemManage.AllRole[]>({
    url: '/user/resetDefaultPassword',
    method: 'get',
    params
  });
}
