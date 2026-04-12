import { request } from '../request';

/** 获取所有训练任务 */
export function fetchDistillationTasks() {
  return request<any>({
    url: '/model-distillation/tasks',
    method: 'get'
  });
}

/** 扫描后端 datasets.root 目录，返回可用数据集列表（供创建任务时下拉选择） */
export function fetchDatasetList() {
  return request<
    Array<{
      label: string;
      value: string;
      type: 'yolo' | 'classification' | 'unknown' | string;
      path?: string;
      /** 来自数据集目录下可选的 meta.json */
      description?: string;
    }>
  >({
    url: '/model-distillation/datasets',
    method: 'get'
  });
}

/** 获取已完成的训练任务（用于自动标注） */
export function fetchCompletedDistillationModels(params?: {
  minAccuracy?: number;
  teacherModel?: string;
  studentModel?: string;
}) {
  return request<any>({
    url: '/model-distillation/completed-models',
    method: 'get',
    params
  });
}

/** 创建训练任务 */
export function createDistillationTask(data: {
  taskName: string;
  teacherModel: string;
  studentModel: string;
  totalEpochs: number;
  batchSize: number;
  learningRate: number;
  temperature: number;
  alpha: number;
  loraRank: number;
}) {
  return request<any>({
    url: '/model-distillation/tasks',
    method: 'post',
    data
  });
}

/** 启动训练任务 */
export function startDistillationTask(taskId: string) {
  return request<any>({
    url: `/model-distillation/tasks/${taskId}/start`,
    method: 'post'
  });
}

/** 停止训练任务 */
export function stopDistillationTask(taskId: string) {
  return request<any>({
    url: `/model-distillation/tasks/${taskId}/stop`,
    method: 'post'
  });
}

/** 获取任务详情 */
export function getDistillationTaskDetail(taskId: string) {
  return request<any>({
    url: `/model-distillation/tasks/${taskId}`,
    method: 'get'
  });
}

/** 删除训练任务 */
export function deleteDistillationTask(taskId: string) {
  return request<any>({
    url: `/model-distillation/tasks/${taskId}`,
    method: 'delete'
  });
}

// ========== 模型推理（自动标注） ==========

/** 提交推理任务 */
export function submitInferenceTask(data: {
  taskId: string;
  inputDir: string;
  outputDir: string;
  batchSize?: number;
  datasetId?: string;
  autoImport?: boolean;
}) {
  return request<any>({
    url: '/model-distillation/inference/submit',
    method: 'post',
    data
  });
}

/** 获取推理任务状态 */
export function getInferenceStatus(inferenceId: string) {
  return request<any>({
    url: `/model-distillation/inference/${inferenceId}`,
    method: 'get'
  });
}

/** 获取所有推理任务 */
export function getAllInferenceTasks() {
  return request<any>({
    url: '/model-distillation/inference/list',
    method: 'get'
  });
}

/** 删除推理任务记录 */
export function deleteInferenceTask(inferenceId: string) {
  return request<any>({
    url: `/model-distillation/inference/${inferenceId}`,
    method: 'delete'
  });
}

/**
 * 绑定外部模型权重到已有任务
 * 将本地离线训练好的权重文件（如 best.pt）的绝对路径写入任务记录，
 * 并将任务状态自动置为 COMPLETED，使其可以被"用于标注"功能使用。
 */
export function bindExternalModel(taskId: string, modelPath: string) {
  return request<any>({
    url: `/model-distillation/tasks/${taskId}/bind-external-model`,
    method: 'patch',
    params: { modelPath }
  });
}

// Force module reload - timestamp: 2025-01-13T12:00:00Z