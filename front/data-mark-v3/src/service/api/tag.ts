import { request } from '../request';

export function fetchLabelAdd(data?: any) {
  return request<any>({
    url: '/label/add',
    method: 'post',
    data
  });
}

export function fetchLabelEdit(data?: any) {
  return request<any>({
    url: '/label/edit',
    method: 'post',
    data
  });
}

export function fetchLabelRemove(data?: any) {
  return request<any>({
    url: '/label/remove',
    method: 'delete',
    data
  });
}

export function fetchLabelList(params?: any) {
  return request<any>({
    url: '/label/list',
    method: 'get',
    params
  });
}

// -------------标签组管理------------
export function fetchLabelGroupAdd(data?: any) {
  return request<any>({
    url: '/label/group/add',
    method: 'post',
    data
  });
}

export function fetchLabelGroupEdit(data?: any) {
  return request<any>({
    url: '/label/group/edit',
    method: 'post',
    data
  });
}

export function fetchLabelGroupRemove(data?: any) {
  return request<any>({
    url: '/label/group/remove',
    method: 'delete',
    data
  });
}

export function fetchLabelGroupList(params?: any) {
  return request<any>({
    url: '/label/group/list',
    method: 'get',
    params
  });
}

export function getSelectImportFileList(params?: any) {
  return request<any>({
    url: '/selectImportFileList',
    method: 'get',
    params
  });
}
