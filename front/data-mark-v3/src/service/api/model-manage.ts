import { request } from "../request";

export function getModelList(data?: any) {
  return request<any>({
    url: `/algorithm/model/getModelList`,
    method: "post",
    data,
  });
}

// algorithm/example/getExampleList 获取模型对应功能
export function getExampleList(data?: any) {
  return request<any>({
    url: `/algorithm/example/getExampleList`,
    method: "post",
    data,
  });
}

// /algorithm/model/trainStart
export function trainStart(data?: any) {
  return request<any>({
    url: `/algorithm/model/trainStart`,
    method: "post",
    data,
  });
}


// 评估列表
export function getAssessLst(data?: any) {
  const { page, limit, id, ...rest } = data;
  return request<any>({
    url: `/algorithm/model/getAssessLst?page=${page}&limit=${limit}`,
    method: "post",
    data: rest,
  });
}


// 评估删除
export function delAssessTask(data?: any) {
  const { id, ...rest } = data;
  return request<any>({
    url: `/algorithm/model/delAssessTask`,
    method: "post",
    data: rest,
  });
}

// 评估详情 
export function modelLastAssess(data?: any) {
  return request<any>({
    url: `/algorithm/model/modelLastAssess`,
    method: "post",
    data,
  });
}