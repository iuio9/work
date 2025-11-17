<script setup lang="ts">
import { computed, h, onMounted, reactive, ref } from 'vue';
import {
  NButton,
  NCard,
  NDataTable,
  NForm,
  NFormItem,
  NInput,
  NInputNumber,
  NModal,
  NSelect,
  NSpace,
  NTag,
  type DataTableColumns,
  type FormInst,
  type FormRules,
  useMessage
} from 'naive-ui';
import dayjs from 'dayjs';
import {
  createFederatedJob,
  fetchFederatedJobs,
  fetchFederatedNodes,
  registerFederatedNode,
  sendFederatedHeartbeat,
  startFederatedJob,
  stopFederatedJob
} from '@/service/api/federated';

interface FederatedNodeItem {
  nodeId: string;
  host: string;
  port: number;
  active: boolean;
  lastHeartbeatAt?: string;
  metadata?: Record<string, any> | null;
  metadataText?: string;
}

interface TrainingJobItem {
  jobId: string;
  modelType: string;
  participantNodeIds: string[];
  status: string;
  createdAt?: string;
  updatedAt?: string;
  baselineAccuracy?: number | null;
  allowedDropPercent?: number | null;
  currentRound?: number;
  lastGlobalAccuracy?: number | null;
  hyperParameters?: Record<string, any>;
  participantsText?: string;
  hyperParamsText?: string;
}

const message = useMessage();

const nodeList = ref<FederatedNodeItem[]>([]);
const jobList = ref<TrainingJobItem[]>([]);
const nodeLoading = ref(false);
const jobLoading = ref(false);
const heartbeatBusyNode = ref<string | null>(null);
const jobOperating = ref<{ id: string; type: 'start' | 'stop' } | null>(null);

const modelTypeOptions = [
  { label: 'YOLOv8', value: 'YOLO_V8' },
  { label: 'LSTM', value: 'LSTM' },
  { label: 'UNet', value: 'UNET' },
  { label: 'ResNet', value: 'RESNET' },
  { label: 'Vision Transformer', value: 'VISION_TRANSFORMER' }
];

const nodeOptions = computed(() =>
  nodeList.value.map(item => ({
    label: `${item.nodeId} (${item.host}:${item.port})`,
    value: item.nodeId
  }))
);

const nodeModalVisible = ref(false);
const jobModalVisible = ref(false);

const nodeFormRef = ref<FormInst | null>(null);
const nodeForm = reactive({
  nodeId: '',
  host: '',
  port: 8080,
  metadata: ''
});
const nodeRules: FormRules = {
  nodeId: [{ required: true, message: '请输入节点ID' }],
  host: [{ required: true, message: '请输入节点地址' }],
  port: [{ required: true, type: 'number', message: '请输入端口号' }]
};

const jobFormRef = ref<FormInst | null>(null);
const jobForm = reactive({
  modelType: 'YOLO_V8',
  participantNodeIds: [] as string[],
  baselineAccuracy: null as null | number,
  allowedDropPercent: 5,
  hyperParameters: {
    numRounds: 10,
    learningRate: 0.001,
    batchSize: 32
  }
});
const jobRules: FormRules = {
  modelType: [{ required: true, message: '请选择模型类型' }],
  participantNodeIds: [{ required: true, type: 'array', message: '请选择参与节点' }],
  'hyperParameters.numRounds': [{ required: true, type: 'number', message: '请输入训练轮数' }]
};

function resetNodeForm() {
  nodeForm.nodeId = '';
  nodeForm.host = '';
  nodeForm.port = 8080;
  nodeForm.metadata = '';
}

function resetJobForm() {
  jobForm.modelType = 'YOLO_V8';
  jobForm.participantNodeIds = [];
  jobForm.baselineAccuracy = null;
  jobForm.allowedDropPercent = 5;
  jobForm.hyperParameters.numRounds = 10;
  jobForm.hyperParameters.learningRate = 0.001;
  jobForm.hyperParameters.batchSize = 32;
}

const nodeColumns: DataTableColumns<FederatedNodeItem> = [
  {
    title: '节点ID',
    key: 'nodeId',
    width: 160
  },
  {
    title: '主机',
    key: 'host',
    width: 160
  },
  {
    title: '端口',
    key: 'port',
    width: 80
  },
  {
    title: '状态',
    key: 'active',
    width: 100,
    render: row =>
      h(
        NTag,
        { type: row.active ? 'success' : 'error', size: 'small' },
        { default: () => (row.active ? '在线' : '离线') }
      )
  },
  {
    title: '上次心跳',
    key: 'lastHeartbeatAt',
    width: 200,
    render: row => row.lastHeartbeatAt ? dayjs(row.lastHeartbeatAt).format('YYYY-MM-DD HH:mm:ss') : '-'
  },
  {
    title: '节点元数据',
    key: 'metadataText',
    minWidth: 220,
    ellipsis: {
      tooltip: true
    }
  },
  {
    title: '操作',
    key: 'actions',
    width: 160,
    align: 'center',
    render: row =>
      h(
        NSpace,
        { justify: 'center' },
        {
          default: () => [
            h(
              NButton,
              {
                size: 'small',
                type: 'primary',
                text: true,
                loading: heartbeatBusyNode.value === row.nodeId,
                onClick: () => handleHeartbeat(row)
              },
              { default: () => '发送心跳' }
            )
          ]
        }
      )
  }
];

const jobColumns: DataTableColumns<TrainingJobItem> = [
  {
    title: '任务ID',
    key: 'jobId',
    width: 180,
    ellipsis: {
      tooltip: true
    }
  },
  {
    title: '模型类型',
    key: 'modelType',
    width: 140
  },
  {
    title: '状态',
    key: 'status',
    width: 120,
    render: row => {
      const statusMap: Record<string, { label: string; type: 'success' | 'error' | 'warning' | 'default' }> = {
        RUNNING: { label: '运行中', type: 'success' },
        CREATED: { label: '未启动', type: 'default' },
        STOPPED: { label: '已停止', type: 'warning' },
        DEGRADED: { label: '精度告警', type: 'error' },
        FAILED: { label: '失败', type: 'error' }
      };
      const current = statusMap[row.status] || { label: row.status, type: 'default' };
      return h(
        NTag,
        { type: current.type, size: 'small' },
        { default: () => current.label }
      );
    }
  },
  {
    title: '参与节点',
    key: 'participantsText',
    minWidth: 200,
    ellipsis: { tooltip: true }
  },
  {
    title: '轮次',
    key: 'currentRound',
    width: 80,
    render: row => row.currentRound ?? '-'
  },
  {
    title: '当前精度',
    key: 'lastGlobalAccuracy',
    width: 120,
    render: row => row.lastGlobalAccuracy != null ? `${(row.lastGlobalAccuracy * 100).toFixed(2)}%` : '-'
  },
  {
    title: '基线精度',
    key: 'baselineAccuracy',
    width: 120,
    render: row => row.baselineAccuracy != null ? `${(row.baselineAccuracy * 100).toFixed(2)}%` : '-'
  },
  {
    title: '允许下降',
    key: 'allowedDropPercent',
    width: 120,
    render: row => row.allowedDropPercent != null ? `${row.allowedDropPercent}%` : '-'
  },
  {
    title: '超参数',
    key: 'hyperParamsText',
    minWidth: 220,
    ellipsis: { tooltip: true }
  },
  {
    title: '操作',
    key: 'operate',
    width: 220,
    align: 'center',
    render: row => {
      const startDisabled = row.status === 'RUNNING';
      const stopDisabled = row.status !== 'RUNNING';
      return h(
        NSpace,
        { justify: 'center' },
        {
          default: () => [
            h(
              NButton,
              {
                size: 'small',
                type: 'primary',
                text: true,
                disabled: startDisabled,
                loading: jobOperating.value?.id === row.jobId && jobOperating.value?.type === 'start',
                onClick: () => handleStartJob(row.jobId)
              },
              { default: () => '启动' }
            ),
            h(
              NButton,
              {
                size: 'small',
                type: 'warning',
                text: true,
                disabled: stopDisabled,
                loading: jobOperating.value?.id === row.jobId && jobOperating.value?.type === 'stop',
                onClick: () => handleStopJob(row.jobId)
              },
              { default: () => '停止' }
            )
          ]
        }
      );
    }
  }
];

function serializeMetadata(metadata?: Record<string, any> | null) {
  if (!metadata || Object.keys(metadata).length === 0) return '-';
  try {
    return JSON.stringify(metadata);
  } catch {
    return '-';
  }
}

function serializeHyperParams(params?: Record<string, any>) {
  if (!params) return '-';
  try {
    return Object.entries(params)
      .map(([key, value]) => `${key}: ${value}`)
      .join(', ');
  } catch {
    return '-';
  }
}

async function getNodes() {
  nodeLoading.value = true;
  try {
    const res = await fetchFederatedNodes();
    const list = (res?.data || []) as FederatedNodeItem[];
    nodeList.value = list.map(item => ({
      ...item,
      metadataText: serializeMetadata(item.metadata)
    }));
  } finally {
    nodeLoading.value = false;
  }
}

async function getJobs() {
  jobLoading.value = true;
  try {
    const res = await fetchFederatedJobs();
    const list = (res?.data || []) as TrainingJobItem[];
    jobList.value = list.map(item => ({
      ...item,
      participantsText: (item.participantNodeIds || []).join(', ') || '-',
      hyperParamsText: serializeHyperParams(item.hyperParameters)
    }));
  } finally {
    jobLoading.value = false;
  }
}

onMounted(() => {
  getNodes();
  getJobs();
});

async function handleHeartbeat(row: FederatedNodeItem) {
  heartbeatBusyNode.value = row.nodeId;
  try {
    await sendFederatedHeartbeat(row.nodeId, row.metadata || {});
    message.success('已发送心跳');
    await getNodes();
  } finally {
    heartbeatBusyNode.value = null;
  }
}

async function submitNodeForm() {
  await nodeFormRef.value?.validate();
  let metadata: Record<string, any> | undefined;
  if (nodeForm.metadata) {
    try {
      metadata = JSON.parse(nodeForm.metadata);
    } catch {
      message.error('元数据需为合法 JSON');
      return;
    }
  }
  await registerFederatedNode({
    nodeId: nodeForm.nodeId,
    host: nodeForm.host,
    port: Number(nodeForm.port),
    metadata
  });
  message.success('节点注册成功');
  nodeModalVisible.value = false;
  resetNodeForm();
  getNodes();
}

async function submitJobForm() {
  await jobFormRef.value?.validate();
  if (!jobForm.participantNodeIds.length) {
    message.error('请至少选择一个参与节点');
    return;
  }
  await createFederatedJob({
    modelType: jobForm.modelType,
    participantNodeIds: jobForm.participantNodeIds,
    baselineAccuracy: jobForm.baselineAccuracy ?? undefined,
    allowedDropPercent: jobForm.allowedDropPercent ?? undefined,
    hyperParameters: { ...jobForm.hyperParameters }
  });
  message.success('任务创建成功');
  jobModalVisible.value = false;
  resetJobForm();
  getJobs();
}

async function handleStartJob(jobId: string) {
  jobOperating.value = { id: jobId, type: 'start' };
  try {
    await startFederatedJob(jobId);
    message.success('任务已启动');
    getJobs();
  } finally {
    jobOperating.value = null;
  }
}

async function handleStopJob(jobId: string) {
  jobOperating.value = { id: jobId, type: 'stop' };
  try {
    await stopFederatedJob(jobId);
    message.success('任务已停止');
    getJobs();
  } finally {
    jobOperating.value = null;
  }
}
</script>

<template>
  <div class="federated-page flex-col-stretch gap-16px">
    <NCard title="节点管理" :bordered="false" size="small">
      <template #header-extra>
        <NButton type="primary" @click="() => { nodeModalVisible = true; resetNodeForm(); }">
          注册节点
        </NButton>
      </template>
      <NDataTable
        :columns="nodeColumns"
        :data="nodeList"
        :loading="nodeLoading"
        :scroll-x="960"
        :pagination="false"
      />
    </NCard>

    <NCard title="联邦训练任务" :bordered="false" size="small">
      <template #header-extra>
        <NButton type="primary" @click="() => { jobModalVisible = true; resetJobForm(); }">
          创建任务
        </NButton>
      </template>
      <NDataTable
        :columns="jobColumns"
        :data="jobList"
        :loading="jobLoading"
        :scroll-x="1200"
        :pagination="false"
      />
    </NCard>

    <NModal v-model:show="nodeModalVisible">
      <NCard title="注册节点" style="width: 520px" :bordered="false" size="huge">
        <NForm ref="nodeFormRef" :model="nodeForm" :rules="nodeRules" label-placement="left" label-width="85">
          <NFormItem label="节点ID" path="nodeId">
            <NInput v-model:value="nodeForm.nodeId" placeholder="例如：node-1" />
          </NFormItem>
          <NFormItem label="主机地址" path="host">
            <NInput v-model:value="nodeForm.host" placeholder="例如：10.0.0.1" />
          </NFormItem>
          <NFormItem label="端口" path="port">
            <NInputNumber v-model:value="nodeForm.port" :min="1" :max="65535" />
          </NFormItem>
          <NFormItem label="元数据(JSON)" path="metadata">
            <NInput
              v-model:value="nodeForm.metadata"
              type="textarea"
              placeholder='{"gpu":"A100"}'
              :autosize="{ minRows: 3, maxRows: 5 }"
            />
          </NFormItem>
        </NForm>
        <template #footer>
          <NSpace justify="end">
            <NButton @click="() => (nodeModalVisible = false)">取消</NButton>
            <NButton type="primary" @click="submitNodeForm">提交</NButton>
          </NSpace>
        </template>
      </NCard>
    </NModal>

    <NModal v-model:show="jobModalVisible">
      <NCard title="创建联邦任务" style="width: 560px" :bordered="false" size="huge">
        <NForm ref="jobFormRef" :model="jobForm" :rules="jobRules" label-placement="left" label-width="120">
          <NFormItem label="模型类型" path="modelType">
            <NSelect v-model:value="jobForm.modelType" :options="modelTypeOptions" />
          </NFormItem>
          <NFormItem label="参与节点" path="participantNodeIds">
            <NSelect
              v-model:value="jobForm.participantNodeIds"
              multiple
              :options="nodeOptions"
              placeholder="请选择至少一个节点"
            />
          </NFormItem>
          <NFormItem label="训练轮数" path="hyperParameters.numRounds">
            <NInputNumber v-model:value="jobForm.hyperParameters.numRounds" :min="1" />
          </NFormItem>
          <NFormItem label="学习率">
            <NInputNumber v-model:value="jobForm.hyperParameters.learningRate" :step="0.0001" />
          </NFormItem>
          <NFormItem label="批次大小">
            <NInputNumber v-model:value="jobForm.hyperParameters.batchSize" :min="1" />
          </NFormItem>
          <NFormItem label="基线精度">
            <NInputNumber v-model:value="jobForm.baselineAccuracy" :min="0" :max="1" :precision="3" placeholder="0~1 之间，可选" />
          </NFormItem>
          <NFormItem label="允许精度下降(%)">
            <NInputNumber v-model:value="jobForm.allowedDropPercent" :min="0" :max="100" />
          </NFormItem>
        </NForm>
        <template #footer>
          <NSpace justify="end">
            <NButton @click="() => (jobModalVisible = false)">取消</NButton>
            <NButton type="primary" @click="submitJobForm">提交</NButton>
          </NSpace>
        </template>
      </NCard>
    </NModal>
  </div>
</template>

<style scoped>
.federated-page {
  min-height: 600px;
}
</style>

