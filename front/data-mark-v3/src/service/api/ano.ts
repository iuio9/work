import { request } from "../request";

// ano查看
export function getDataDetails(params?: any) {
  return request<any>({
    url: "/data/set/getDataDetails",
    method: "get",
    params,
  });
}
export function getDataDetailsNoMarkFilePath(params?: any) {
  return request<any>({
    url: "/data/set/getDataDetailsNoMarkFilePath",
    method: "get",
    params,
  });
}

export function getImgDataDetails(params?: any) {
  return request<any>({
    url: "/data/mark/getDataDetails",
    method: "get",
    params,
  });
}

// /label/selectGroupLabel
export function getSelectGroupLabel(params?: any) {
  return request<any>({
    url: "/label/selectGroupLabel",
    method: "get",
    params,
  });
}

// /label/selectGroupLabelPage get
export function getSelectGroupLabelPage(params?: any) {
  return request<any>({
    url: "/label/selectDataSetLabelPage",
    method: "get",
    params,
  });
}

// /label/addDataSetAndLabel
export function addDataSetAndLabel(data?: any) {
  return request<any>({
    url: "/label/addDataSetAndLabel",
    method: "post",
    data,
  });
}

// /label/selectDataSetLabel
export function selectDataSetLabel(params?: any) {
  return request<any>({
    url: "/label/selectDataSetLabel",
    method: "get",
    params,
  });
}

export function selectLabelList(params?: any) {
  return request<any>({
    url: "/label/group/selectLabelList",
    method: "get",
    params,
  });
}

// /label/deleteDataSetLabel
export function deleteDataSetLabel(data?: any) {
  return request<any>({
    url: "/label/deleteDataSetLabel",
    method: "delete",
    data,
  });
}

// /mark/MarkFileUpload
export function MarkFileUpload(data?: any) {
  return request<any>({
    url: "/mark/MarkFileUpload",
    method: "post",
    data,
    headers: {
      "Content-Type": "multipart/form-data",
    },
  });
}

// /data/mark/addDataMarkInfo
export function addDataMarkInfo(data?: any) {
  return request<any>({
    url: "/data/mark/addDataMarkInfo",
    method: "post",
    data,
  });
}

// 在线标注列表 /data/mark/getDataSetMarkList
export function getDataSetMarkList(params?: any) {
  return request<any>({
    url: "/data/mark/getDataSetMarkList",
    method: "get",
    params,
  });
}

// 获取tab数量 /data/set/DataDetailsCount
export function DataDetailsCount(params?: any) {
  return request<any>({
    url: "/data/set/DataDetailsCount",
    method: "get",
    params,
  });
}

// /label/addSaveLabel 添加标签
export function addSaveLabel(data?: any) {
  return request<any>({
    url: "/label/addSaveLabel",
    method: "post",
    data,
  });
}

// bind 绑定标签
export function bindLabel(data?: any) {
  return request<any>({
    url: "/label/addBatchLabel",
    method: "post",
    data,
  });
}

export function deleteResultFile(data?: any) {
  return request<any>({
    url: "/mark/deleteResultFile",
    method: "delete",
    data,
  });
}

export function deleteFile(data?: any) {
  return request<any>({
    url: "/mark/deleteFile",
    method: "delete",
    data,
  });
}

// /make/revokeFile post
export function revokeFile(data?: any) {
  return request<any>({
    url: "/make/revokeFile",
    method: "post",
    data,
  });
}


// /algorithm/segment/start
export function segmentStart(data?: any) {
  return request<any>({
    url: "/algorithm/segment/start",
    method: "post",
    data,
  });
}

export function getTaskResult(data?: any) {
  const { page, limit, taskId } = data;
  return request<any>({
    url: `/algorithm/task/getTaskResult?page=${page}&limit=${limit}`,
    method: "post",
    data: {
      taskId,
    },
  });
}

// data/set/getResultDataSetSave

export function getResultDataSetSave(data?: any) {
  return request<any>({
    url: `/data/set/getResultDataSetSave`,
    method: "post",
    data,
  });
}

// group团队管理
export function getTeamList(params?: any) {
  return request<any>({
    url: `/team/list`,
    method: "get",
    params,
  });
}

export function teamAdd(data?: any) {
  const { page, limit, taskId } = data;
  return request<any>({
    url: `/team/add`,
    method: "post",
    data,
  });
}

export function teamEdit(data?: any) {
  return request<any>({
    url: `/team/edit`,
    method: "post",
    data,
  });
}

export function teamRemove(data?: any) {
  return request<any>({
    url: `/team/remove`,
    method: "delete",
    data,
  });
}

// 部门人员级联 /dept/getDeptByUserList
export function getDeptByUserList(params?: any) {
  return request<any>({
    url: `/dept/getDeptByUserList`,
    method: "get",
    params,
  });
}

// /manyMark/getMyCreateTaskList  get
export function getMyCreateTaskList(params?: any) {
  return request<any>({
    url: `/manyMark/getMyCreateTaskList`,
    method: "get",
    params,
  });
}

// /manyMark/viewProgress
export function viewProgress(params?: any) {
  return request<any>({
    url: `/manyMark/viewProgress`,
    method: "get",
    params,
  });
}

// /manyMark/endTask get
export function endTask(params?: any) {
  return request<any>({
    url: `/manyMark/endTask`,
    method: "get",
    params,
  });
}

// /manyMark/deleteTask delete
export function deleteTask(data?: any) {
  return request<any>({
    url: `/manyMark/deleteTask`,
    method: "delete",
    params: data,
  });
}

// /manyMark/getMyReceiveList get
export function getMyReceiveList(params?: any) {
  return request<any>({
    url: `/manyMark/getMyReceiveList`,
    method: "get",
    params,
  });
}

// /manyMark/taskShift post
export function taskShift(data?: any) {
  return request<any>({
    url: `/manyMark/taskShift`,
    method: "post",
    data,
  });
}

// /manyMark/getByTaskIdTeamList get
export function getByTaskIdTeamList(params?: any) {
  return request<any>({
    url: `/manyMark/getByTaskIdTeamList`,
    method: "get",
    params,
  });
}

// /manyMark/endUserTask get
export function endUserTask(params?: any) {
  return request<any>({
    url: `/manyMark/endUserTask`,
    method: "get",
    params,
  });
}

// /manyMark/withdraw delete
export function withdraw(data?: any) {
  return request<any>({
    url: `/manyMark/withdraw`,
    method: "delete",
    data,
  });
}

// manyMark/allocationNum get
export function allocationNum(params?: any) {
  return request<any>({
    url: `/manyMark/allocationNum`,
    method: "get",
    params,
  });
}

// /role/isManyTask get
export function isManyTask(params?: any) {
  return request<any>({
    url: `/role/isManyTask`,
    method: "get",
    params,
  });
}

// /many/ToExamine/fileIsApprove post
export function fileIsApprove(data?: any) {
  return request<any>({
    url: `/many/ToExamine/fileIsApprove`,
    method: "post",
    data,
  });
}

// /many/ToExamine/verifyComplete get
export function verifyComplete(params?: any) {
  return request<any>({
    url: `/many/ToExamine/verifyComplete`,
    method: "get",
    params,
  });
}

// /many/ToExamine/returnTask get
export function returnTask(params?: any) {
  return request<any>({
    url: `/many/ToExamine/returnTask`,
    method: "get",
    params,
  });
}

// /many/ToExamine/remainingApprove get
export function remainingApprove(params?: any) {
  return request<any>({
    url: `/many/ToExamine/remainingApprove`,
    method: "get",
    params,
  });
}

// /many/ToExamine/submitTask get
export function submitTask(params?: any) {
  return request<any>({
    url: `/many/ToExamine/submitTask`,
    method: "get",
    params,
  });
}

// /many/ToExamine/submitTaskPrompt get
export function submitTaskPrompt(params?: any) {
  return request<any>({
    url: `/many/ToExamine/submitTaskPrompt`,
    method: "get",
    params,
  });
}

// /many/allocation/distributionExamine get
export function distributionExamine(params?: any) {
  return request<any>({
    url: `/many/allocation/distributionExamine`,
    method: "get",
    params,
  });
}

// /many/allocation/examineDetails get
export function examineDetails(params?: any) {
  return request<any>({
    url: `/many/allocation/examineDetails`,
    method: "get",
    params,
  });
}

// /many/allocation/myExamineTaskList get
export function myExamineTaskList(params?: any) {
  return request<any>({
    url: `/many/allocation/myExamineTaskList`,
    method: "get",
    params,
  });
}

// /many/ToExamine/submitTaskPrompt get
export function submitTaskPrompt2(params?: any) {
  return request<any>({
    url: `/many/allocation/submitExamineTaskPrompt`,
    method: "get",
    params,
  });
}

// /many/allocation/submitExamineTask get
export function submitExamineTask2(params?: any) {
  return request<any>({
    url: `/many/allocation/submitExamineTask`,
    method: "get",
    params,
  })
}

// /many/allocation/examineTeamInfo get
export function examineTeamInfo(params?: any) {
  return request<any>({
    url: `/many/allocation/examineTeamInfo`,
    method: "get",
    params,
  })
}

// /many/allocation/examineReturn post
export function examineReturn(data?: any) {
  return request<any>({
    url: `/many/allocation/examineReturn`,
    method: "post",
    data,
  })
}


// /many/allocation/confirmAudit get
export function confirmAudit(params?: any) {
  return request<any>({
    url: `/many/allocation/confirmAudit`,
    method: "get",
    params,
  })
}

// /many/allocation/examineTaskShift post
export function examineTaskShift(data?: any) {
  return request<any>({
    url: `/many/allocation/examineTaskShift`,
    method: "post",
    data,
  })
}

// /data/set/type/selectDataSetDictList post
export function selectDataSetDictList(data?: any) {
  return request<any>({
    url: `/data/set/type/selectDataSetDictList`,
    method: "post",
    data,
  })
}

// /data/result/saveResult post
export function saveResult(data?: any) {
  return request<any>({
    url: `/data/result/saveResult`,
    method: "post",
    data,
  })
}

// /label/topUpLabel get
export function topUpLabel(params?: any) {
  return request<any>({
    url: `/label/topUpLabel`,
    method: "get",
    params,
  })
}

// /many/allocation/approved get
export function approved(params?: any) {
  return request<any>({
    url: `/many/allocation/approved`,
    method: "get",
    params,
  })
}

// /data/set/getDataSonType get
export function getDataSonType(params?: any) {
  return request<any>({
    url: `/data/set/getDataSonType`,
    method: "get",
    params,
  })
}
