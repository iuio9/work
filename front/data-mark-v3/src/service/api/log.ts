import { request } from '../request';

export function fetchGetLogList(params?: any) {
  return request<any>({
    url: '/log/oper/list',
    method: 'get',
    params
  });
}

export function fetchGetLogDetail(params?: any) {
  return request<any>({
    url: '/log/oper/getOperLog',
    method: 'get',
    params
  });
}

export function fetchGetLoginList(params?: any) {
  return request<any>({
    url: '/log/login/list',
    method: 'get',
    params
  });
}
