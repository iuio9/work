import {request} from '../request';

// one menu
export function fetchGetDictList(params?: any) {
  return request<any>({
    url: '/dict/type/list',
    method: 'get',
    params
  });
}

export function fetchDictAdd(data: any) {
  return request<any>({
    url: '/dict/type/add',
    method: 'post',
    data
  });
}

export function fetchDictEdit(data: any) {
  return request<any>({
    url: '/dict/type/edit',
    method: 'post',
    data
  });
}

export function fetchDictRmove(data: any) {
  return request<any>({
    url: '/dict/type/remove',
    method: 'delete',
    data: {
      ids: data
    }
  });
}

// two menu
export function fetchGetTwoDictList(params?: any) {
  return request<any>({
    url: '/dict/data/list',
    method: 'get',
    params
  });
}

export function fetchGetTreeData(params?: any) {
  return request<any>({
    url: '/dict/data/getDictDataTree',
    method: 'get',
    params
  });
}

export function fetchTwoDictAdd(data: any) {
  return request<any>({
    url: '/dict/data/add',
    method: 'post',
    data
  });
}

export function fetchTwoDictEdit(data: any) {
  return request<any>({
    url: '/dict/data/edit',
    method: 'post',
    data
  });
}

export function fetchTwoDictRmove(data: any) {
  return request<any>({
    url: '/dict/data/remove',
    method: 'delete',
    data: {
      ids: data
    }
  });
}

// 根据字典类型查询字典数据信息
export function getDicts(params?: any) {
  return request({
    url: `/dict/data/getDictDataTree`,
    method: 'get',
    params
  });
}
