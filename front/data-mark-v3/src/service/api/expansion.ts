import { request } from "../request";

export function getTaskPage(params?: any) {
  return request<any>({
    url: `/algorithm/task/getTaskPage`,
    method: "post",
    params,
  });
}

export function getTaskDetail(params?: any) {
  return request<any>({
    url: "/algorithm/task/getTaskDetail",
    method: "post",
    data: params,
  });
}

export function addTask(data?: any) {
  return request<any>({
    url: "/algorithm/task/addTask",
    method: "post",
    data,
  });
}

export function delTask(data?: any) {
  return request<any>({
    url: "/algorithm/task/delTask",
    method: "post",
    data,
  });
}

export function submitTask(data?: any) {
  return request<any>({
    url: "/algorithm/task/submitTask",
    method: "post",
    data,
  });
}

// 字典
export function getDictDataTree(params?: any) {
  return request<any>({
    url: "/dict/data/getDictDataTree",
    method: "get",
    params,
  });
}

// 数据集 无分页 /data/mark/getDataSetNoPage
export function getDataSetListNoPage(params?: any) {
  return request<any>({
    url: "/data/set/getDataSetListNoPage",
    method: "get",
    params,
  });
}

// 训练模型 无分页
export function getExampleList(data?: any) {
  return request<any>({
    url: "/algorithm/example/getExampleList",
    method: "post",
    data,
  });
}


// 复制标签组 /label/group/copyLabelGroup
export function copyLabelGroup(data?: any) {
  return request<any>({
    url: "/label/group/copyLabelGroup",
    method: "post",
    data,
  });
}

// 关联数据集 /label/group/assocDataSet
export function assocDataSet(data?: any) {
  return request<any>({
    url: "/label/group/assocDataSet",
    method: "post",
    data,
  });
}

// 数据组信息 /label/group/getDataSonLabelStatus
export function getDataSonLabelStatus(params?: any) {
  return request<any>({
    url: "/label/group/getDataSonLabelStatus",
    method: "get",
    params,
  });
}


export function getTeamList(params?: any) {
  return request<any>({
    url: "/team/getTeamList",
    method: "get",
    params,
  });
}

// /manyMark/addManyMarkTask  post
export function addManyMarkTask(data?: any) {
  return request<any>({
    url: "/manyMark/addManyMarkTask",
    method: "post",
    data,
  });
}

// /manyMark/getMyCreateTaskList get
export function getMyCreateTaskList(params?: any) {
  return request<any>({
    url: "/manyMark/getMyCreateTaskList",
    method: "get",
    params,
  });
}

// /label/group/importLabel post
export function importLabel(data?: any) {
  return request<any>({
    url: "/label/group/importLabel",
    method: "post",
    data,
  });
}

// /file/TemDownload post
export function TemDownload(data?: any) {
  return request<any>({
    url: "/file/TemDownload",
    method: "post",
    data,
    responseType: 'blob'
  });
}
