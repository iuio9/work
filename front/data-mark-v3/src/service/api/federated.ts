import { request } from '../request';

/** 获取全部联邦节点 */
export function fetchFederatedNodes() {
  return request<any>({
    url: '/federated/nodes',
    method: 'get'
  });
}

/** 注册节点 */
export function registerFederatedNode(data: {
  nodeId: string;
  host: string;
  port: number;
  metadata?: Record<string, any>;
}) {
  return request<any>({
    url: '/federated/register',
    method: 'post',
    data
  });
}

/** 主动上报心跳 */
export function sendFederatedHeartbeat(nodeId: string, metadata?: Record<string, any>) {
  return request<any>({
    url: `/federated/heartbeat/${nodeId}`,
    method: 'post',
    data: metadata || {}
  });
}

/** 获取任务列表 */
export function fetchFederatedJobs() {
  return request<any>({
    url: '/federated/jobs',
    method: 'get'
  });
}

interface CreateJobPayload {
  modelType: string;
  hyperParameters: Record<string, any>;
  participantNodeIds: string[];
  baselineAccuracy?: number | null;
  allowedDropPercent?: number;
}

/** 创建联邦训练任务 */
export function createFederatedJob(payload: CreateJobPayload) {
  const { modelType, ...body } = payload;
  return request<any>({
    url: '/federated/jobs',
    method: 'post',
    params: { modelType },
    data: body
  });
}

/** 启动任务 */
export function startFederatedJob(jobId: string) {
  return request<any>({
    url: `/federated/jobs/${jobId}/start`,
    method: 'post'
  });
}

/** 停止任务 */
export function stopFederatedJob(jobId: string) {
  return request<any>({
    url: `/federated/jobs/${jobId}/stop`,
    method: 'post'
  });
}


