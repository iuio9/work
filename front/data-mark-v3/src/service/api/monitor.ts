import { request } from '../request';

// ano查看
export function getDataDetails(params?: any) {
  return request<any>({
    url: '/data/set/getDataDetails',
    method: 'get',
    params
  });
}
