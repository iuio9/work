import { request } from '../request';

// /model/evaluation/list get
export function fetchModelEvaluationList(params?: any) {
  return request<any>({
    url: '/model/evaluation/list',
    method: 'get',
    params
  });
}

// /model/evaluation/delModel post
export function delModelEvaluation(params?: any) {
  return request<any>({
    url: '/model/evaluation/delModel',
    method: 'post',
    data: params
  });
}

// /model/assess/delTask post
export function delModelAssess(params?: any) {
  return request<any>({
    url: '/model/assess/delTask',
    method: 'post',
    data: params
  });
}


// /model/evaluation/addModel post
export function addModelEvaluation(params?: any) {
  return request<any>({
    url: '/model/evaluation/addModel',
    method: 'post',
    data: params
  });
}

// /model/evaluation/modelDetails get
export function fetchModelDetails(params?: any) {
  return request<any>({
    url: '/model/evaluation/modelDetails',
    method: 'get',
    params
  });
}


// /model/evaluation/editModel post
export function editModelEvaluation(params?: any) {
  return request<any>({
    url: '/model/evaluation/editModel',
    method: 'post',
    data: params
  });
}

// /model/evaluation/submitApprove get
export function submitApproveModelEvaluation(params?: any) {
  return request<any>({
    url: '/model/evaluation/submitApprove',
    method: 'get',
    params
  });
}


// /model/approve/list get
export function fetchModelApproveList(params?: any) {
  return request<any>({
    url: '/model/approve/list',
    method: 'get',
    params
  });
}

// 审核通过 /model/approve/pass get
export function passModelApprove(params?: any) {
  return request<any>({
    url: '/model/approve/pass',
    method: 'get',
    params
  });
}

// 审核不通过 /model/approve/notPass get
export function notPassModelApprove(params?: any) {
  return request<any>({
    url: '/model/approve/notPass',
    method: 'get',
    params
  });
}

// /model/evaluation/generatePad post
export function generatePadModelEvaluation(params?: any) {
  return request<any>({
    url: '/model/evaluation/generatePad',
    method: 'post',
    data: params
  });
}

// upload post 文件上传
export function upload(params?: any) {
  return request<any>({
    url: '/upload',
    method: 'post',
    data: params,
    headers: {
      'Content-Type': 'multipart/form-data',
      'type': "3"
    },
  });
}

// /model/evaluation/modelBackFill post
export function modelBackFillModelEvaluation(params?: any) {
  return request<any>({
    url: '/model/evaluation/modelBackFill',
    method: 'post',
    data: params
  });
}


// /api/debugModel post
export function debugModel(params?: any) {
  return request<any>({
    url: '/api/debugModel',
    method: 'post',
    data: params,
    headers: {
      'Content-Type': 'multipart/form-data',
    }
  });
}

// /api/savaDebugLog post
export function savaDebugLog(params?: any) {
  return request<any>({
    url: '/api/savaDebugLog',
    method: 'post',
    data: params
  });
}

// /api/oneClickDebugging get
export function oneClickDebugging(params?: any) {
  return request<any>({
    url: '/api/oneClickDebugging',
    method: 'get',
    params
  });
}

// /model/approve/getModelTypeList get
export function getModelTypeList(params?: any) {
  return request<any>({
    url: '/model/approve/getModelTypeList',
    method: 'get',
    params
  });
}

// /data/set/getDataSetListNoPage get
export function getDataSetListNoPage(params?: any) {
  return request<any>({
    url: '/data/set/getDataSetListNoPage',
    method: 'get',
    params
  });
}

// /model/assess/createAssessTask post
export function createAssessTask(params?: any) {
  return request<any>({
    url: '/model/assess/createAssessTask',
    method: 'post',
    data: params,
    headers: {
      'Content-Type': 'multipart/form-data',
    }
  });
}

// /model/assess/modelDetail get
export function assessModelDetail(params?: any) {
  return request<any>({
    url: '/model/assess/modelDetail',
    method: 'get',
    params
  });
}

// /model/assess/taskDetail get
export function assessTaskDetail(params?: any) {
  return request<any>({
    url: '/model/assess/taskDetail',
    method: 'get',
    params
  });
}

export function assessModelEdit(params?: any) {
  return request<any>({
    url: '/model/assess/edit',
    method: 'post',
    data: params,
    headers: {
      'Content-Type': 'multipart/form-data',
    }
  });
}

// /model/assess/editTask post
export function editAssessTask(params?: any) {
  return request<any>({
    url: '/model/assess/editTask',
    method: 'post',
    data: params,
    headers: {
      'Content-Type': 'multipart/form-data',
    }
  });
}


// /model/assess/listPage get
export function getAssessTaskList(params?: any) {
  return request<any>({
    url: '/model/assess/listPage',
    method: 'get',
    params
  });
}

// /model/report/listPage get
export function getReportTaskList(params?: any) {
  return request<any>({
    url: '/model/report/listPage',
    method: 'get',
    params
  });
}

// /thirdModelAssess/start get
export function startAssessTask(params?: any) {
  return request<any>({
    url: '/thirdModelAssess/start',
    method: 'get',
    params
  });
}

// 重新开始
export function restartAssessTask(params?: any) {
  return request<any>({
    url: '/thirdModelAssess/restart',
    method: 'get',
    params
  });
}

// /thirdModelAssess/stop get
export function pauseAssessTask(params?: any) {
  return request<any>({
    url: '/thirdModelAssess/pause',
    method: 'get',
    params
  });
}

// /thirdModelAssess/continue get
export function continueAssessTask(params?: any) {
  return request<any>({
    url: '/thirdModelAssess/continue',
    method: 'get',
    params
  });
}

// termination get
export function terminationAssessTask(params?: any) {
  return request<any>({
    url: '/thirdModelAssess/termination',
    method: 'get',
    params
  });
}

// viewResult get
export function viewResultAssessTask(params?: any) {
  return request<any>({
    url: '/thirdModelAssess/viewResult',
    method: 'get',
    params
  });
}

// 完成对接 post
export function finishContactAssessTask(params?: any) {
  return request<any>({
    url: '/model/assess/finishContact',
    method: 'post',
    data: params
  });
}

// 选择自增id post
export function selectIncrementIdAssessTask(params?: any) {
  return request<any>({
    url: '/model/assess/selectIncrementId',
    method: 'post',
    data: params
  });
}

// 获取模型调试信息 post
export function getModelDebugInfo(params?: any) {
  return request<any>({
    url: '/model/assess/getModelDebugInfo',
    method: 'post',
    data: params
  });
}

// getAlgorithmList get
export function getAlgorithmList(params?: any) {
  return request<any>({
    url: '/model/code/getAlgorithmList',
    method: 'get',
    params
  });
}

// /api/savaDebugLog get
export function savaDebugLogGet(params?: any) {
  return request<any>({
    url: '/api/savaDebugLog',
    method: 'get',
    params
  });
}

// /api/isSavaDebugLog get
export function isSavaDebugLogGet(params?: any) {
  return request<any>({
    url: '/api/isSavaDebugLog',
    method: 'get',
    params
  });
}

// /algorithm/blackAndWhiteTest/start post
export function startBlackWhiteTest(params?: any) {
  return request<any>({
    url: '/algorithm/blackAndWhiteTest/start',
    method: 'post',
    data: params,
    headers: {
      'Content-Type': 'multipart/form-data',
    }
  });
}

// /algorithm/blackAndWhiteTest/result post
export function getBlackWhiteTestResult(params?: any) {
  return request<any>({
    url: '/algorithm/blackAndWhiteTest/result',
    method: 'post',
    data: params
  });
}

// /model/evaluation/deleteFile post
export function deleteFile(params?: any) {
  return request<any>({
    url: '/model/evaluation/deleteFile',
    method: 'post',
    data: params
  });
}

// /model/assess/deleteFile post
export function assessDeleteFile(params?: any) {
  return request<any>({
    url: '/model/assess/deleteFile',
    method: 'post',
    data: params
  });
}


// restartTask post
export function restartTask(params?: any) {
  return request<any>({
    url: '/model/assess/restartManufacturer',
    method: 'post',
    data: params
  });
}

// 获取参数名
export function getParamName(params?: any) {
  return request<any>({
    url: '/model/assess/getParamName',
    method: 'post',
    data: params
  });
}
