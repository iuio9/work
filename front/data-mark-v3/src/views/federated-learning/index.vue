<template>
  <div class="federated-learning-container">
    <n-card title="联邦学习管理平台" :bordered="false" class="h-full rounded-8px shadow-sm">
      <n-tabs v-model:value="activeTab" type="line" animated>
        <!-- 节点管理标签页 -->
        <n-tab-pane name="nodes" tab="节点管理">
          <div class="p-4">
            <n-space vertical :size="16">
              <!-- 节点统计卡片 -->
              <n-grid :cols="4" :x-gap="16">
                <n-gi>
                  <n-statistic label="总节点数" :value="nodeStats.total">
                    <template #prefix>
                      <n-icon :component="ServerOutline" />
                    </template>
                  </n-statistic>
                </n-gi>
                <n-gi>
                  <n-statistic label="活跃节点" :value="nodeStats.active" class="text-green-500">
                    <template #prefix>
                      <n-icon :component="CheckmarkCircleOutline" />
                    </template>
                  </n-statistic>
                </n-gi>
                <n-gi>
                  <n-statistic label="离线节点" :value="nodeStats.inactive" class="text-red-500">
                    <template #prefix>
                      <n-icon :component="CloseCircleOutline" />
                    </template>
                  </n-statistic>
                </n-gi>
                <n-gi>
                  <n-statistic label="GPU总数" :value="nodeStats.totalGpus">
                    <template #prefix>
                      <n-icon :component="HardwareChipOutline" />
                    </template>
                  </n-statistic>
                </n-gi>
              </n-grid>

              <!-- 节点列表 -->
              <n-data-table
                :columns="nodeColumns"
                :data="nodes"
                :loading="nodesLoading"
                :bordered="false"
                :single-line="false"
              />
            </n-space>
          </div>
        </n-tab-pane>

        <!-- 训练任务标签页 -->
        <n-tab-pane name="jobs" tab="训练任务">
          <div class="p-4">
            <n-space vertical :size="16">
              <!-- 操作按钮 -->
              <n-space>
                <n-button type="primary" @click="showCreateJobModal = true">
                  <template #icon>
                    <n-icon :component="AddOutline" />
                  </template>
                  创建训练任务
                </n-button>
                <n-button @click="refreshJobs">
                  <template #icon>
                    <n-icon :component="RefreshOutline" />
                  </template>
                  刷新
                </n-button>
              </n-space>

              <!-- 任务列表 -->
              <n-data-table
                :columns="jobColumns"
                :data="jobs"
                :loading="jobsLoading"
                :bordered="false"
                :single-line="false"
              />
            </n-space>
          </div>
        </n-tab-pane>

        <!-- 训练监控标签页 -->
        <n-tab-pane name="monitor" tab="训练监控">
          <div class="p-4">
            <n-empty v-if="!selectedJob" description="请选择一个训练任务">
              <template #extra>
                <n-button size="small" @click="activeTab = 'jobs'">
                  前往任务列表
                </n-button>
              </template>
            </n-empty>

            <div v-else>
              <n-space vertical :size="16">
                <!-- 任务信息卡片 -->
                <n-card title="任务信息" :bordered="false">
                  <n-descriptions :column="3" bordered>
                    <n-descriptions-item label="任务ID">
                      {{ selectedJob.jobId }}
                    </n-descriptions-item>
                    <n-descriptions-item label="模型类型">
                      <n-tag :type="getModelTypeColor(selectedJob.modelType)">
                        {{ selectedJob.modelType }}
                      </n-tag>
                    </n-descriptions-item>
                    <n-descriptions-item label="状态">
                      <n-tag :type="getStatusType(selectedJob.status)">
                        {{ selectedJob.status }}
                      </n-tag>
                    </n-descriptions-item>
                    <n-descriptions-item label="当前轮次">
                      {{ selectedJob.currentRound || 0 }} / {{ selectedJob.numRounds || 10 }}
                    </n-descriptions-item>
                    <n-descriptions-item label="参与节点">
                      {{ selectedJob.participantCount || 0 }}
                    </n-descriptions-item>
                    <n-descriptions-item label="当前精度">
                      {{ (selectedJob.currentAccuracy * 100).toFixed(2) }}%
                    </n-descriptions-item>
                  </n-descriptions>
                </n-card>

                <!-- 精度曲线图 -->
                <n-card title="精度变化曲线" :bordered="false">
                  <div ref="chartRef" style="height: 400px"></div>
                </n-card>
              </n-space>
            </div>
          </div>
        </n-tab-pane>
      </n-tabs>
    </n-card>

    <!-- 创建训练任务对话框 -->
    <n-modal
      v-model:show="showCreateJobModal"
      preset="card"
      title="创建训练任务"
      style="width: 600px"
      :bordered="false"
      :segmented="{ content: true }"
    >
      <n-form
        ref="jobFormRef"
        :model="jobForm"
        :rules="jobFormRules"
        label-placement="left"
        label-width="120px"
      >
        <n-form-item label="模型类型" path="modelType">
          <n-select
            v-model:value="jobForm.modelType"
            :options="modelTypeOptions"
            placeholder="请选择模型类型"
          />
        </n-form-item>

        <n-form-item label="训练轮数" path="numRounds">
          <n-input-number
            v-model:value="jobForm.numRounds"
            :min="1"
            :max="1000"
            placeholder="输入训练轮数"
            style="width: 100%"
          />
        </n-form-item>

        <n-form-item label="学习率" path="learningRate">
          <n-input-number
            v-model:value="jobForm.learningRate"
            :min="0.0001"
            :max="1"
            :step="0.0001"
            placeholder="输入学习率"
            style="width: 100%"
          />
        </n-form-item>

        <n-form-item label="批次大小" path="batchSize">
          <n-input-number
            v-model:value="jobForm.batchSize"
            :min="1"
            :max="512"
            placeholder="输入批次大小"
            style="width: 100%"
          />
        </n-form-item>

        <n-form-item label="本地训练轮数" path="localEpochs">
          <n-input-number
            v-model:value="jobForm.localEpochs"
            :min="1"
            :max="100"
            placeholder="输入本地训练轮数"
            style="width: 100%"
          />
        </n-form-item>

        <n-form-item label="精度下降阈值" path="allowedDropPercent">
          <n-input-number
            v-model:value="jobForm.allowedDropPercent"
            :min="0"
            :max="100"
            :step="0.1"
            placeholder="允许的精度下降百分比"
            style="width: 100%"
          >
            <template #suffix>%</template>
          </n-input-number>
        </n-form-item>

        <n-form-item label="参与节点" path="participantNodeIds">
          <n-select
            v-model:value="jobForm.participantNodeIds"
            multiple
            :options="activeNodeOptions"
            placeholder="选择参与训练的节点"
          />
        </n-form-item>
      </n-form>

      <template #footer>
        <n-space justify="end">
          <n-button @click="showCreateJobModal = false">取消</n-button>
          <n-button type="primary" @click="handleCreateJob" :loading="creating">
            创建
          </n-button>
        </n-space>
      </template>
    </n-modal>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted, h } from 'vue';
import { NButton, NTag, NSpace, NIcon, useMessage, useDialog } from 'naive-ui';
import {
  ServerOutline,
  CheckmarkCircleOutline,
  CloseCircleOutline,
  HardwareChipOutline,
  AddOutline,
  RefreshOutline,
  PlayOutline,
  StopOutline,
  EyeOutline
} from '@vicons/ionicons5';
import {
  fetchFederatedNodes,
  fetchFederatedJobs,
  createFederatedJob,
  startFederatedJob,
  stopFederatedJob
} from '@/service/api';
import * as echarts from 'echarts';

// ==================== 响应式数据 ====================
const message = useMessage();
const dialog = useDialog();

const activeTab = ref('nodes');
const nodes = ref<any[]>([]);
const jobs = ref<any[]>([]);
const nodesLoading = ref(false);
const jobsLoading = ref(false);
const showCreateJobModal = ref(false);
const creating = ref(false);
const selectedJob = ref<any>(null);
const chartRef = ref<HTMLElement | null>(null);

// ==================== 表单数据 ====================
const jobForm = ref({
  modelType: 'RESNET',
  numRounds: 10,
  learningRate: 0.001,
  batchSize: 32,
  localEpochs: 5,
  allowedDropPercent: 5.0,
  participantNodeIds: []
});

const jobFormRules = {
  modelType: { required: true, message: '请选择模型类型', trigger: 'change' },
  numRounds: { required: true, type: 'number', message: '请输入训练轮数', trigger: 'blur' },
  participantNodeIds: {
    required: true,
    type: 'array',
    min: 1,
    message: '请至少选择一个节点',
    trigger: 'change'
  }
};

// ==================== 计算属性 ====================
const nodeStats = computed(() => {
  const total = nodes.value.length;
  const active = nodes.value.filter((n: any) => n.active).length;
  const inactive = total - active;
  const totalGpus = nodes.value.reduce((sum: number, n: any) => sum + (n.gpuCount || 0), 0);
  return { total, active, inactive, totalGpus };
});

const activeNodeOptions = computed(() => {
  return nodes.value
    .filter((n: any) => n.active)
    .map((n: any) => ({
      label: `${n.nodeId} (${n.host}:${n.port})`,
      value: n.nodeId
    }));
});

const modelTypeOptions = [
  { label: 'YOLOv8 - 目标检测', value: 'YOLO_V8' },
  { label: 'LSTM - 序列模型', value: 'LSTM' },
  { label: 'UNet - 图像分割', value: 'UNET' },
  { label: 'ResNet - 图像分类', value: 'RESNET' },
  { label: 'Vision Transformer - ViT', value: 'VISION_TRANSFORMER' }
];

// ==================== 表格列定义 ====================
const nodeColumns = [
  { title: '节点ID', key: 'nodeId', width: 200, ellipsis: { tooltip: true } },
  { title: '主机地址', key: 'host', width: 150 },
  { title: '端口', key: 'port', width: 80 },
  {
    title: '状态',
    key: 'active',
    width: 100,
    render: (row: any) =>
      h(
        NTag,
        { type: row.active ? 'success' : 'error' },
        { default: () => (row.active ? '在线' : '离线') }
      )
  },
  { title: 'GPU数量', key: 'gpuCount', width: 100 },
  { title: 'GPU型号', key: 'gpuModel', width: 150, ellipsis: { tooltip: true } },
  { title: '数据集大小', key: 'datasetSize', width: 120 },
  {
    title: '最后心跳',
    key: 'lastHeartbeatAt',
    width: 180,
    render: (row: any) => (row.lastHeartbeatAt ? new Date(row.lastHeartbeatAt).toLocaleString() : '-')
  }
];

const jobColumns = [
  { title: '任务ID', key: 'jobId', width: 200, ellipsis: { tooltip: true } },
  {
    title: '模型类型',
    key: 'modelType',
    width: 150,
    render: (row: any) => h(NTag, { type: getModelTypeColor(row.modelType) }, { default: () => row.modelType })
  },
  {
    title: '状态',
    key: 'status',
    width: 120,
    render: (row: any) => h(NTag, { type: getStatusType(row.status) }, { default: () => row.status })
  },
  {
    title: '进度',
    key: 'progress',
    width: 120,
    render: (row: any) => `${row.currentRound || 0}/${row.numRounds || 10}`
  },
  {
    title: '精度',
    key: 'currentAccuracy',
    width: 100,
    render: (row: any) => (row.currentAccuracy ? `${(row.currentAccuracy * 100).toFixed(2)}%` : '-')
  },
  { title: '节点数', key: 'participantCount', width: 80 },
  {
    title: '创建时间',
    key: 'createdAt',
    width: 180,
    render: (row: any) => (row.createdAt ? new Date(row.createdAt).toLocaleString() : '-')
  },
  {
    title: '操作',
    key: 'actions',
    width: 200,
    fixed: 'right',
    render: (row: any) =>
      h(
        NSpace,
        {},
        {
          default: () => [
            h(
              NButton,
              {
                size: 'small',
                type: 'primary',
                disabled: row.status !== 'CREATED',
                onClick: () => handleStartJob(row.jobId)
              },
              { default: () => '启动', icon: () => h(NIcon, { component: PlayOutline }) }
            ),
            h(
              NButton,
              {
                size: 'small',
                type: 'warning',
                disabled: row.status !== 'RUNNING',
                onClick: () => handleStopJob(row.jobId)
              },
              { default: () => '停止', icon: () => h(NIcon, { component: StopOutline }) }
            ),
            h(
              NButton,
              {
                size: 'small',
                onClick: () => viewJobDetails(row)
              },
              { default: () => '查看', icon: () => h(NIcon, { component: EyeOutline }) }
            )
          ]
        }
      )
  }
];

// ==================== 工具函数 ====================
function getStatusType(status: string): 'success' | 'warning' | 'error' | 'info' | 'default' {
  const statusMap: Record<string, any> = {
    CREATED: 'info',
    RUNNING: 'success',
    COMPLETED: 'success',
    FAILED: 'error',
    STOPPED: 'warning',
    DEGRADED: 'warning'
  };
  return statusMap[status] || 'default';
}

function getModelTypeColor(modelType: string): 'primary' | 'success' | 'warning' | 'error' | 'info' {
  const colorMap: Record<string, any> = {
    YOLO_V8: 'primary',
    LSTM: 'success',
    UNET: 'warning',
    RESNET: 'info',
    VISION_TRANSFORMER: 'error'
  };
  return colorMap[modelType] || 'default';
}

// ==================== API 调用方法 ====================
async function refreshNodes() {
  nodesLoading.value = true;
  try {
    const res = await fetchFederatedNodes();
    nodes.value = res.data || [];
  } catch (error) {
    message.error('获取节点列表失败');
    console.error(error);
  } finally {
    nodesLoading.value = false;
  }
}

async function refreshJobs() {
  jobsLoading.value = true;
  try {
    const res = await fetchFederatedJobs();
    jobs.value = res.data || [];
  } catch (error) {
    message.error('获取任务列表失败');
    console.error(error);
  } finally {
    jobsLoading.value = false;
  }
}

async function handleCreateJob() {
  creating.value = true;
  try {
    await createFederatedJob({
      modelType: jobForm.value.modelType,
      hyperParameters: {
        numRounds: jobForm.value.numRounds,
        learningRate: jobForm.value.learningRate,
        batchSize: jobForm.value.batchSize,
        localEpochs: jobForm.value.localEpochs
      },
      participantNodeIds: jobForm.value.participantNodeIds,
      allowedDropPercent: jobForm.value.allowedDropPercent
    });
    message.success('创建任务成功');
    showCreateJobModal.value = false;
    refreshJobs();
  } catch (error) {
    message.error('创建任务失败');
    console.error(error);
  } finally {
    creating.value = false;
  }
}

async function handleStartJob(jobId: string) {
  dialog.warning({
    title: '确认启动',
    content: '确定要启动此训练任务吗？',
    positiveText: '确定',
    negativeText: '取消',
    onPositiveClick: async () => {
      try {
        await startFederatedJob(jobId);
        message.success('任务已启动');
        refreshJobs();
      } catch (error) {
        message.error('启动任务失败');
        console.error(error);
      }
    }
  });
}

async function handleStopJob(jobId: string) {
  dialog.warning({
    title: '确认停止',
    content: '确定要停止此训练任务吗？',
    positiveText: '确定',
    negativeText: '取消',
    onPositiveClick: async () => {
      try {
        await stopFederatedJob(jobId);
        message.success('任务已停止');
        refreshJobs();
      } catch (error) {
        message.error('停止任务失败');
        console.error(error);
      }
    }
  });
}

function viewJobDetails(job: any) {
  selectedJob.value = job;
  activeTab.value = 'monitor';
  // 初始化图表
  setTimeout(() => {
    initChart();
  }, 100);
}

function initChart() {
  if (!chartRef.value) return;

  const chart = echarts.init(chartRef.value);
  const option = {
    tooltip: {
      trigger: 'axis'
    },
    legend: {
      data: ['训练精度']
    },
    grid: {
      left: '3%',
      right: '4%',
      bottom: '3%',
      containLabel: true
    },
    xAxis: {
      type: 'category',
      boundaryGap: false,
      data: Array.from({ length: 10 }, (_, i) => `轮次 ${i + 1}`)
    },
    yAxis: {
      type: 'value',
      min: 0,
      max: 1,
      axisLabel: {
        formatter: (value: number) => `${(value * 100).toFixed(0)}%`
      }
    },
    series: [
      {
        name: '训练精度',
        type: 'line',
        data: [0.6, 0.65, 0.7, 0.72, 0.75, 0.77, 0.79, 0.81, 0.82, 0.83],
        smooth: true,
        lineStyle: {
          color: '#18a058'
        },
        itemStyle: {
          color: '#18a058'
        },
        areaStyle: {
          color: {
            type: 'linear',
            x: 0,
            y: 0,
            x2: 0,
            y2: 1,
            colorStops: [
              { offset: 0, color: 'rgba(24, 160, 88, 0.3)' },
              { offset: 1, color: 'rgba(24, 160, 88, 0)' }
            ]
          }
        }
      }
    ]
  };

  chart.setOption(option);
}

// ==================== 生命周期 ====================
onMounted(() => {
  refreshNodes();
  refreshJobs();

  // 定时刷新节点状态（每10秒）
  setInterval(() => {
    refreshNodes();
  }, 10000);

  // 定时刷新任务状态（每5秒）
  setInterval(() => {
    refreshJobs();
  }, 5000);
});
</script>

<style scoped lang="scss">
.federated-learning-container {
  height: 100%;
  padding: 16px;
}
</style>
