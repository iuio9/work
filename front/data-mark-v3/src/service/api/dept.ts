import { request } from '../request';

// one menu
export function fetchGetDeptList(params?: any) {
  return request<any>({
    url: '/dept/list',
    method: 'get',
    params
  });
}

// get /dept/getDeptSelect
export function fetchGetDeptSelect(params?: any) {
  return request<any>({
    url: '/dept/getDeptSelect',
    method: 'get',
    params
  });
}

export function fetchDeptAdd(data: any) {
  return request<any>({
    url: '/dept/add',
    method: 'post',
    data
  });
}

export function fetchDeptEdit(data: any) {
  return request<any>({
    url: '/dept/edit',
    method: 'post',
    data
  });
}

export function fetchDeptRemove(data: any) {
  return request<any>({
    url: '/dept/remove',
    method: 'delete',
    data: {
      ids: data
    }
  });
}
