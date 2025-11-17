import { request } from '../request';

// -----------数据集------------
export function fetchGetDataSetList(params?: any) {
  return request<any>({
    url: '/data/set/getDataSetList',
    method: 'get',
    params
  });
}

// 导入记录
export function fetchImportList(params?: any) {
  return request<any>({
    url: '/selectImportList',
    method: 'get',
    params
  });
}

export function getTreeLevelDict(params?: any) {
  const url = `/dict/data/getTreeLevelDict/${params.dataTypeId}`;
  return request<any>({
    url,
    method: 'get'
    // params,
  });
}

export function getTreeLevelDictIds(params?: any) {
  const url = `/dict/data/getTreeLevelDictIds/${params.dataTypeId}`;
  return request<any>({
    url,
    method: 'get'
    // params,
  });
}

export function fetchDataSetAdd(data?: any) {
  return request<Api.Auth.LoginToken>({
    url: '/data/set/add',
    method: 'post',
    data
  });
}

export function dataSetImport(data?: any) {
  return request<Api.Auth.LoginToken>({
    url: '/data/set/dataSetImport',
    method: 'post',
    data
  });
}

export function deleteDataGroup(data?: any) {
  const url = `/data/set/deleteDataGroup?groupId=${data.groupId}`;
  return request<Api.Auth.LoginToken>({
    url,
    method: 'delete'
  });
}

export function deleteDataSet(data?: any) {
  const url = `/data/set/deleteDataSet?sonId=${data.sonId}`;
  return request<Api.Auth.LoginToken>({
    url,
    method: 'delete'
  });
}

export function fetchDataSetAddDataVersion(data?: any) {
  return request<any>({
    url: '/data/set/addDataVersion',
    method: 'post',
    data
  });
}

export function fileUpload(data?: any) {
  return request<any>({
    url: '/temp/anyUpload',
    method: 'post',
    data,
    headers: {
      'Content-Type': 'multipart/form-data'
    }
  });
}

export function temDownload(data?: any) {
  return request<any>({
    url: `/tem/download?sign=${data.sign}`,
    method: 'post',
    data,
    responseType: 'blob' // 表明返回服务器返回的数据类型
  });
}

export function fileUploadDel(data?: any) {
  return request<any>({
    url: '/temp/deleteTempFile',
    method: 'post',
    data
  });
}

// 修改备注 /data/set/updateDataSetRemark
export function updateDataSetRemark(data?: any) {
  return request<any>({
    url: '/data/set/updateDataSetRemark',
    method: 'post',
    data
  });
}

export function updateDataSetName(data?: any) {
  return request<any>({
    url: '/data/set/updateDataSetName',
    method: 'post',
    data
  });
}

// 模型管理
export function getModelPage(params?: any) {
  return request<any>({
    url: '/algorithm/model/getModelPage',
    method: 'post',
    params
  });
}

export function getModelList(params?: any) {
  return request<any>({
    url: '/algorithm/model/getModelList',
    method: 'post',
    data: params
  });
}

export function getModelDetail(params?: any) {
  return request<any>({
    url: '/algorithm/model/getModelDetail',
    method: 'get',
    params
  });
}
// 模型管理 add
export function addModel(data?: any) {
  return request<any>({
    url: '/algorithm/model/addModel',
    method: 'post',
    data
  });
}

// ---------算法--------
export function getExamplePage(data?: any) {
  return request<any>({
    url: `/algorithm/example/getExamplePage`,
    method: 'post',
    params: data
  });
}

// // algorithm/model/getModelList
// export function getModelPage(data?: any) {
//   return request<any>({
//     url: `/algorithm/model/getModelPage`,
//     method: "post",
//     params: data,
//   });
// }

// /algorithm/task/getTaskPage
export function getTaskPage(data?: any) {
  return request<any>({
    url: `/algorithm/task/getTaskPage`,
    method: 'post',
    params: data
  });
}

export function getExampleDetail(params?: any) {
  return request<any>({
    url: '/algorithm/example/getExampleDetail',
    method: 'get',
    params
  });
}

export function addExample(data?: any) {
  return request<any>({
    url: '/algorithm/example/addExample',
    method: 'post',
    data
  });
}
export function updateExample(data?: any) {
  return request<any>({
    url: '/algorithm/example/updateExample',
    method: 'post',
    data
  });
}
export function delExample(data?: any) {
  return request<any>({
    url: '/algorithm/example/delExample',
    method: 'post',
    data: {
      ids: data
    }
  });
}

// 导出 /file/download
export function fileDownload(data?: any) {
  const { sonId } = data;
  return request<any>({
    url: `/file/download?sonId=${sonId}`,
    method: 'post',
    data,
    responseType: 'blob' // 表明返回服务器返回的数据类型
  });
}

export function getSelectDataSetDictList(params?: any) {
  return request<any>({
    url: '/data/set/type/selectDataSetDictList',
    method: 'get',
    params
  });
}

// 结束
export function trainStop(data?: any) {
  return request<any>({
    url: `/algorithm/model/trainStop/${data.taskId}`,
    method: 'post',
    data
  });
}

// 开始
export function trainStart(data?: any) {
  return request<any>({
    url: `/algorithm/model/trainStart`,
    method: 'post',
    data
  });
}
// 评估
export function trainAssess(data?: any) {
  return request<any>({
    url: `/algorithm/model/trainAssess/${data.taskId}`,
    method: 'post',
    data
  });
}

export function trainAssessStart(data?: any) {
  return request<any>({
    url: `/algorithm/model/trainAssess/start`,
    method: 'post',
    data
  });
}

export function updateModel(data?: any) {
  return request<any>({
    url: '/algorithm/model/updateModel',
    method: 'post',
    data
  });
}
export function delModel(data?: any) {
  return request<any>({
    url: '/algorithm/model/delModel',
    method: 'post',
    data: {
      ids: data
    }
  });
}

// 压缩包chunk 进度条
export function uploadZipChunk(data?: any) {
  return request<any>({
    url: '/bigFileUpload/uploadChunk',
    method: 'post',
    data,
    headers: {
      'Content-Type': 'multipart/form-data'
    }
  });
}

// 根据sonId查找关联标签组
export function getSonIdByLabelGroupIds(params?: any) {
  return request<any>({
    url: '/label/group/getSonIdByLabelGroupIds',
    method: 'get',
    params
  });
}

// /algorithm/dataEnhancementTask/submitTask post
export function imgSubmitTask(data?: any) {
  return request<any>({
    url: '/algorithm/dataEnhancementTask/submitTask',
    method: 'post',
    data
  });
}

// /label/selectDataSetLabelPage get
export function selectDataSetLabelPage(params?: any) {
  return request<any>({
    url: '/label/selectDataSetLabelPage',
    method: 'get',
    params
  });
}
