<script setup lang="tsx">
import { NButton, NPopconfirm, NProgress, NDynamicInput } from 'naive-ui';
import { useAppStore } from '@/store/modules/app';
import { useTable, useTableOperate } from '@/hooks/common/table';
import { delExample, getExamplePage, getTaskPage, trainAssess, trainStop } from '@/service/api/dataManage';
import { useBoolean } from '~/packages/hooks';
import { $t } from '@/locales';
import { useWebSocketStore } from '@/store/modules/websocket';
import { delTask, getDataSetListNoPage } from '@/service/api/expansion';
import UserSearch from './modules/user-search.vue';
import assReport from '@/assets/imgs/assReport.png';
import assessDetail from '@/assets/imgs/assessDetail.webp';
import assess0 from "@/assets/imgs/assess0.png";
import assess1 from "@/assets/imgs/assess1.png";
import assess2 from "@/assets/imgs/assess2.png";
import assess3 from "@/assets/imgs/assess3.png";
import ReportDetailModal from "./modules/ReportDetailModal.vue";
import NetWorkLogDrawer from "./modules/netWork-Log-drawer.vue";
import VueJsonPretty from 'vue-json-pretty'
import 'vue-json-pretty/lib/styles.css'
import { resolveDirective, withDirectives } from 'vue';
import {
  getAssessTaskList,
  startAssessTask, pauseAssessTask,
  continueAssessTask, terminationAssessTask,
  viewResultAssessTask, restartAssessTask, finishContactAssessTask,
  delModelAssess,
  restartTask
} from '@/service/api/third';

import LogViewer from "@/components/custom/LogViewer.vue"
import RealtimeLog from "@/components/custom/RealtimeLog.vue"
import RealtimeLogList from "./modules/RealtimeLogList.vue"
import { useMultiTaskWebSocket } from './useMultiTaskWebSocket';
import { downloadFile1 } from '@/utils/util';
import { getToken } from '@/store/modules/auth/shared';

enum TaskStatusKey {
  Testing = '1',       // 正在测试
  TestFinished = '2',  // 完成测试
  AssessFinished = '3',// 完成评估
  Ended = '4'          // 已结束
}

enum ModelEvaluationStatus {
  /** 评估中 */
  IN_PROGRESS = 'IN_PROGRESS',
  /** 评估完成 */
  COMPLETED = 'COMPLETED',
  /** 评估失败 */
  FAILED = 'FAILED',
  /** 待评估 */
  PENDING = 'PENDING',
}

enum OperateType {
  Pause = 'pause',
  Terminate = 'terminate',
  Restart = 'restart',
  StartAssess = 'startAssess',
  ViewAssessDetail = 'viewAssessDetail',
  Delete = 'delete',
  Log = 'log',
  Generate = 'generate',
  Script = 'Script',
  Start = 'start',
  Continue = 'continue',
  Contact = 'contact',
  Edit = 'edit',
  Download = 'download',
}

// 操作按钮配置类型
type ButtonConfig = {
  text: string;
  type?: 'primary' | 'success' | 'warning' | 'error';
  onClick: (row: any) => void;
  confirmText?: string;
  [prop: string]: any;
};

/**
 * 创建普通按钮
 */
const createButton = (props: any, content: string, permission?: string) => {
  const authDir = resolveDirective('hasPermi')
  return withDirectives(
    h(NButton, {
      ...props,
      size: 'small',
      type: 'primary',
      ghost: false,
    }, content),
    [
      [
        authDir,
        permission
      ]
    ]
  )
};

/**
 * 创建确认按钮
 */
const createConfirmButton = (
  text: string,
  confirmText: string,
  onClick: (event: Event) => void,
  permission?: string
) => {
  return h(
    NPopconfirm,
    {
      onPositiveClick: (event: Event) => {
        event.stopPropagation();
        onClick(event);
      },
    },
    {
      trigger: () => createButton({ type: 'error' }, text, permission),
      default: () => h("span", {}, confirmText),
    },
  );
};

const appStore = useAppStore();
const router = useRouter();
const route = useRoute();
const { bool: visible, setTrue: openModal } = useBoolean();
const {
  columns,
  columnChecks,
  data,
  getData,
  getDataByPage,
  loading,
  mobilePagination,
  searchParams,
  resetSearchParams
} = useTable({
  sign: 'id',
  apiFn: getAssessTaskList,
  showTotal: true,
  apiParams: {
    isTrain: 1,
    current: 1,
    size: 10,
    taskName: null,
  },
  columns: () => [
    {
      type: 'selection'
    },
    {
      key: 'index',
      title: '编号',
      align: 'center',
      width: 60,
      fixed: 'left',
    },
    {
      key: 'taskName',
      title: '任务名称',
      align: 'left',
      width: 160,
      fixed: 'left',
      ellipsis: { tooltip: true },
    },
    {
      key: 'taskType',
      title: '任务类型',
      align: 'center',
      width: 120,
      ellipsis: { tooltip: true },
      render: (row: any) => {
        const taskTypeMap: any = {
          1: '分类任务',
          2: '目标检测',
        };
        return taskTypeMap[row.taskType] || '未知';
      }
    },
    {
      key: 'modelName',
      title: '模型名称',
      align: 'left',
      width: 160,
      fixed: 'left',
      ellipsis: { tooltip: true },
    },
    {
      title: '任务进度',
      key: 'taskProgress',
      align: 'center',
      width: 180,
      render: (row: any) => {
        const progress = row.taskProgress ? row.taskProgress.split("%")[0] : 0;
        return h("div", { style: { width: '100%', height: '20px' } }, [
          h(NProgress, {
            type: "line",
            "indicator-placement": "inside",
            processing: true,
            percentage: Number(progress),
            style: { height: '4px', borderRadius: '2px' } // 进度条样式优化
          })
        ]);
      }
    },
    {
      title: '任务状态',
      key: 'taskStatus',
      align: 'center',
      width: 120,
      ellipsis: { tooltip: true },
      render: (row: any) => {
        const TaskStatusText = {
          [TaskStatus.PENDING]: '待评估',
          [TaskStatus.RUNNING]: '执行中',
          [TaskStatus.COMPLETED]: '已完成',
          [TaskStatus.FAILED]: '任务失败',
          [TaskStatus.TERMINATED]: '终止',
          [TaskStatus.CONTINUED]: '继续',
          // 待处理
          [TaskStatus.PROCESSING]: '待处理',
        };
        return TaskStatusText[row.taskStatus] || '未知状态';
      }
    },
    {
      title: '关联数据集',
      key: 'sonName',
      align: 'left',
      width: 160,
      ellipsis: { tooltip: true }
    },
    {
      title: '错误原因',
      key: 'errorMessage',
      align: 'left',
      width: 240,
      ellipsis: { tooltip: true } // 长文本自动省略并显示tooltip
    },
    {
      title: '创建时间',
      key: 'createTime',
      align: 'center',
      width: 160,
      ellipsis: { tooltip: true }
    },
    // {
    //   key: 'operate',
    //   title: '操作',
    //   align: 'center',
    //   width: 260,
    //   fixed: 'right',
    //   render: (row: any, index: number) => {
    //     const { taskStatus, taskId, taskType } = row;

    //     // 定义需要过滤的操作类型（taskType为1时）
    //     const filteredOperationsForType1 = [
    //       OperateType.Start,      // 开始评估
    //       OperateType.Terminate,  // 终止
    //       OperateType.Restart,    // 重新开始
    //       OperateType.Continue    // 继续
    //     ];

    //     // 根据状态配置不同的操作按钮
    //     const getButtonConfigs = (): ButtonConfig[] => {
    //       // 保持原有状态逻辑不变
    //       let buttonConfigs = [];

    //       switch (taskStatus) {
    //         case TaskStatus.RUNNING: // 执行中
    //           buttonConfigs = [
    //             // { text: '查看日志', onClick: () => handleOperate(OperateType.Log, row), permission: 'thirdparty:assess:log' },
    //             { text: '暂停', onClick: () => handleOperate(OperateType.Pause, row), permission: 'thirdparty:assess:pause' },
    //             { text: '终止', onClick: () => handleOperate(OperateType.Terminate, row), permission: 'thirdparty:assess:stop' },
    //             { text: '重新开始', onClick: () => handleOperate(OperateType.Restart, row), permission: 'thirdparty:assess:restart' },
    //             // { text: '继续', onClick: () => handleOperate(OperateType.Continue, row), permission: 'thirdparty:assess:continue' },
    //           ];
    //           break;
    //         case TaskStatus.COMPLETED: // 已完成
    //           buttonConfigs = [
    //             // { text: '查看日志', onClick: () => handleOperate(OperateType.Log, row), permission: 'thirdparty:assess:log' },
    //             // {
    //             //   text: '生成评估报告',
    //             //   onClick: () => handleOperate(OperateType.Generate, row),
    //             //   type: 'primary',
    //             //   permission: 'thirdparty:assess:generate'
    //             // },
    //             { text: '查看评估报告', onClick: () => handleOperate(OperateType.ViewAssessDetail, row), permission: 'thirdparty:assess:report' },
    //             { text: '重新开始', onClick: () => handleOperate(OperateType.Restart, row), permission: 'thirdparty:assess:restart' },
    //             {
    //               text: '删除',
    //               onClick: () => handleOperate(OperateType.Delete, row),
    //               confirmText: `确定删除任务 ${row.id}？`,
    //               permission: 'thirdparty:assess:delete'
    //             },
    //           ];
    //           break;
    //         case TaskStatus.FAILED: // 任务失败
    //           buttonConfigs = [
    //             // { text: '查看日志', onClick: () => handleOperate(OperateType.Log, row), permission: 'thirdparty:assess:log' },
    //             { text: '重新开始', onClick: () => handleOperate(OperateType.Restart, row), permission: 'thirdparty:assess:restart' },
    //             {
    //               text: '删除',
    //               onClick: () => handleOperate(OperateType.Delete, row),
    //               confirmText: `确定删除任务 ${row.id}？`,
    //               permission: 'thirdparty:assess:delete'
    //             },
    //           ];
    //           break;
    //         case TaskStatus.PENDING: // 待评估
    //           buttonConfigs = [
    //             { text: '开始评估', onClick: () => handleOperate(OperateType.StartAssess, row), permission: 'thirdparty:assess:start' },
    //             // { text: '查看日志', onClick: () => handleOperate(OperateType.Log, row), permission: 'thirdparty:assess:log' },
    //             {
    //               text: '删除',
    //               onClick: () => handleOperate(OperateType.Delete, row),
    //               confirmText: `确定删除任务 ${row.id}？`,
    //               permission: 'thirdparty:assess:delete'
    //             },
    //           ];
    //           break;
    //         case TaskStatus.TERMINATED: // 终止
    //           buttonConfigs = [
    //             // { text: '查看日志', onClick: () => handleOperate(OperateType.Log, row), permission: 'thirdparty:assess:log' },
    //             // { text: '继续', onClick: () => handleOperate(OperateType.Continue, row), permission: 'thirdparty:assess:continue' },
    //             { text: '重新开始', onClick: () => handleOperate(OperateType.Restart, row), permission: 'thirdparty:assess:restart' },
    //             {
    //               text: '删除',
    //               onClick: () => handleOperate(OperateType.Delete, row),
    //               confirmText: `确定删除任务 ${row.id}？`,
    //               permission: 'thirdparty:assess:delete'
    //             },
    //           ];
    //           break;
    //         case TaskStatus.CONTINUED: // 继续
    //           buttonConfigs = [
    //             // { text: '查看日志', onClick: () => handleOperate(OperateType.Log, row), permission: 'thirdparty:assess:log' },
    //             { text: '暂停', onClick: () => handleOperate(OperateType.Pause, row), permission: 'thirdparty:assess:pause' },
    //             { text: '终止', onClick: () => handleOperate(OperateType.Terminate, row), permission: 'thirdparty:assess:stop' },
    //             { text: '重新开始', onClick: () => handleOperate(OperateType.Restart, row), permission: 'thirdparty:assess:restart' },
    //           ];
    //           break;
    //         case TaskStatus.PROCESSING: // 待处理
    //           buttonConfigs = [
    //             { text: '对接厂商', onClick: () => handleOperate(OperateType.Contact, row), permission: 'thirdparty:assess:contact' },
    //             { text: '编辑', onClick: () => handleOperate(OperateType.Edit, row), permission: 'thirdparty:assess:edit' },
    //             {
    //               text: '删除',
    //               onClick: () => handleOperate(OperateType.Delete, row),
    //               confirmText: `确定删除任务 ${row.id}？`,
    //               permission: 'thirdparty:assess:delete'
    //             },
    //           ];
    //           break;
    //         default:
    //           buttonConfigs = [
    //             {
    //               text: '删除',
    //               onClick: () => handleOperate(OperateType.Delete, row),
    //               confirmText: `确定删除任务 ${row.id}？`,
    //               permission: 'thirdparty:assess:delete'
    //             },
    //           ];
    //       }

    //       // 当taskType为1时，过滤特定操作按钮
    //       if (taskType === 1) {
    //         return buttonConfigs.filter(config => {
    //           // 提取按钮对应的操作类型
    //           const buttonOperateType = getOperateTypeFromButtonConfig(config);
    //           // 排除filteredOperationsForType1中定义的操作
    //           return !filteredOperationsForType1.includes(buttonOperateType);
    //         });
    //       }

    //       return buttonConfigs;
    //     };

    //     // 根据按钮配置获取对应的操作类型
    //     const getOperateTypeFromButtonConfig = (config: ButtonConfig) => {
    //       // 通过onClick回调函数中的OperateType映射到具体操作类型
    //       // 这里需要根据实际的handleOperate实现来匹配
    //       if (config.onClick.toString().includes('OperateType.Start')) {
    //         return OperateType.Start;
    //       } else if (config.onClick.toString().includes('OperateType.Terminate')) {
    //         return OperateType.Terminate;
    //       } else if (config.onClick.toString().includes('OperateType.Restart')) {
    //         return OperateType.Restart;
    //       } else if (config.onClick.toString().includes('OperateType.Continue')) {
    //         return OperateType.Continue;
    //       }
    //       // 其他操作类型...
    //       return null;
    //     };

    //     // 生成操作按钮
    //     const buttons = getButtonConfigs().map((config, index) => {
    //       if (config.confirmText) {
    //         return createConfirmButton(
    //           config.text,
    //           config.confirmText,
    //           () => config.onClick(row),
    //           config.permission
    //         );
    //       }

    //       return createButton(
    //         {
    //           type: config.type || 'primary',
    //           onClick: () => {
    //             const event = window.event as MouseEvent;
    //             event.stopPropagation();
    //             config.onClick(row);
    //           }
    //         },
    //         config.text,
    //         config.permission
    //       );
    //     });

    //     return h('div', { class: 'flex items-center gap-2 flex-wrap' }, buttons);
    //   }
    // }
    {
      key: 'operate',
      title: '操作',
      align: 'center',
      width: 260,
      fixed: 'right',
      render: (row: any, index: number) => {
        const { taskStatus, taskId, taskType } = row;

        // 操作类型枚举
        enum OperateType {
          Start = 'start',
          Pause = 'pause',
          Terminate = 'terminate',
          Restart = 'restart',
          Continue = 'continue',
          Delete = 'delete',
          ViewAssessDetail = 'viewAssessDetail',
          Contact = 'contact',
          Edit = 'edit',
          StartAssess = 'startAssess',
          Download = 'download',
        }

        // 定义需要过滤的操作类型（taskType为1时）
        const filteredOperationsForType1 = [
          OperateType.Start,
          OperateType.Pause,
          OperateType.Terminate,
          OperateType.Restart,
          OperateType.Continue,
          // OperateType.Contact,
          OperateType.Edit,
          OperateType.StartAssess,
          OperateType.ViewAssessDetail,
        ];

        const filteredOperationsForType2 = [
          OperateType.Download,
        ];

        // 按钮配置映射表
        const BUTTONS_MAP: Record<number, Array<{
          text: string;
          operateType: OperateType;
          permission?: string;
          confirmText?: string;
        }>> = {
          [TaskStatus.RUNNING]: [
            { text: '编辑', operateType: OperateType.Edit, permission: 'thirdparty:assess:edit' },
            { text: '暂停', operateType: OperateType.Pause, permission: 'thirdparty:assess:pause' },
            { text: '终止', operateType: OperateType.Terminate, permission: 'thirdparty:assess:stop' },
            { text: '重新开始', operateType: OperateType.Restart, permission: 'thirdparty:assess:restart' }
          ],
          [TaskStatus.COMPLETED]: [
            { text: '编辑', operateType: OperateType.Edit, permission: 'thirdparty:assess:edit' },
            { text: '重新开始', operateType: OperateType.Restart, permission: 'thirdparty:assess:restart' },
            { text: '删除', operateType: OperateType.Delete, permission: 'thirdparty:assess:delete', confirmText: `确定删除任务 ${row.id}？` },
            { text: '查看评估报告', operateType: OperateType.ViewAssessDetail, permission: 'thirdparty:assess:report' },
            // 下载压缩包
            { text: '下载压缩包', operateType: OperateType.Download, permission: 'thirdparty:assess:download' },
          ],
          [TaskStatus.FAILED]: [
            { text: '编辑', operateType: OperateType.Edit, permission: 'thirdparty:assess:edit' },
            { text: '重新开始', operateType: OperateType.Restart, permission: 'thirdparty:assess:restart' },
            { text: '删除', operateType: OperateType.Delete, permission: 'thirdparty:assess:delete', confirmText: `确定删除任务 ${row.id}？` }
          ],
          [TaskStatus.PENDING]: [
            { text: '编辑', operateType: OperateType.Edit, permission: 'thirdparty:assess:edit' },
            { text: '开始评估', operateType: OperateType.StartAssess, permission: 'thirdparty:assess:start' },
            { text: '删除', operateType: OperateType.Delete, permission: 'thirdparty:assess:delete', confirmText: `确定删除任务 ${row.id}？` }
          ],
          [TaskStatus.TERMINATED]: [
            { text: '编辑', operateType: OperateType.Edit, permission: 'thirdparty:assess:edit' },
            { text: '重新开始', operateType: OperateType.Restart, permission: 'thirdparty:assess:restart' },
            { text: '删除', operateType: OperateType.Delete, permission: 'thirdparty:assess:delete', confirmText: `确定删除任务 ${row.id}？` }
          ],
          [TaskStatus.CONTINUED]: [
            { text: '编辑', operateType: OperateType.Edit, permission: 'thirdparty:assess:edit' },
            { text: '暂停', operateType: OperateType.Pause, permission: 'thirdparty:assess:pause' },
            { text: '终止', operateType: OperateType.Terminate, permission: 'thirdparty:assess:stop' },
            { text: '重新开始', operateType: OperateType.Restart, permission: 'thirdparty:assess:restart' }
          ],
          [TaskStatus.PROCESSING]: [
            { text: '对接厂商', operateType: OperateType.Contact, permission: 'thirdparty:assess:contact' },
            { text: '编辑', operateType: OperateType.Edit, permission: 'thirdparty:assess:edit' },
            { text: '删除', operateType: OperateType.Delete, permission: 'thirdparty:assess:delete', confirmText: `确定删除任务 ${row.id}？` }
          ]
        };

        // 获取当前状态下的按钮配置
        const getButtonConfigs = (): ButtonConfig[] => {
          const baseButtons = BUTTONS_MAP[taskStatus] || [];

          return baseButtons
            .filter(btn => {
              if (taskType === 1 && filteredOperationsForType2.includes(btn.operateType)) {
                return false;
              }
              if (taskType === 2 && filteredOperationsForType2.includes(btn.operateType)) {
                return false;
              }
              return true;
            })
            .map(btn => ({
              ...btn,
              onClick: () => handleOperate(btn.operateType, row)
            }));
        };

        // 按钮渲染
        const buttons = getButtonConfigs().map((config, index) => {
          if (config.confirmText) {
            return createConfirmButton(
              config.text,
              config.confirmText,
              () => config.onClick(row),
              config.permission
            );
          }

          return createButton(
            {
              type: config.type || 'primary',
              onClick: () => {
                const event = window.event as MouseEvent;
                event.stopPropagation();
                config.onClick(row);
              }
            },
            config.text,
            config.permission
          );
        });

        return h('div', { class: 'flex items-center gap-2 flex-wrap' }, buttons);
      }
    }
  ],
  immediate: true
});

const mockData = [
  {
    taskId: 1,
    taskInputName: "NLP模型优化",
    modelName: "BERT-base",
    taskName: "模型优化",
    taskProgress: "0%",
    taskStatus: ModelEvaluationStatus.PENDING,
    groupVName: "GLUE基准测试集",
    taskException: "",
    createTime: "2025-05-13 08:45:00",
    taskType: 1 // 数字类型：测试
  },
  {
    taskId: 6,
    taskInputName: "强化学习模型训练",
    modelName: "PPO",
    taskName: "模型训练",
    taskProgress: "45%",
    taskStatus: ModelEvaluationStatus.IN_PROGRESS,
    groupVName: "OpenAI Gym环境",
    taskException: "",
    createTime: "2025-05-13 13:00:00",
    taskType: 2 // 数字类型：评估
  },
  {
    taskId: 2,
    taskInputName: "目标检测模型评估",
    modelName: "YOLOv8",
    taskName: "模型评估",
    taskProgress: "100%",
    taskStatus: ModelEvaluationStatus.COMPLETED,
    groupVName: "COCO评估集",
    taskException: "",
    createTime: "2025-05-11 14:30:00",
    taskType: 2 // 数字类型：评估
  },
  {
    taskId: 5,
    taskInputName: "推荐系统评估",
    modelName: "Wide&Deep",
    taskName: "模型评估",
    taskProgress: "100%",
    taskStatus: ModelEvaluationStatus.COMPLETED,
    groupVName: "MovieLens数据集",
    taskException: "",
    createTime: "2025-05-13 11:20:00",
    taskType: 2 // 数字类型：评估
  },
  {
    taskId: 3,
    taskInputName: "语义分割模型测试",
    modelName: "UNet++",
    taskName: "模型测试",
    taskProgress: "65%",
    taskStatus: ModelEvaluationStatus.FAILED,
    groupVName: "Cityscapes测试集",
    taskException: "GPU内存不足，训练中断",
    createTime: "2025-05-12 10:15:00",
    taskType: 1 // 数字类型：测试
  },
  {
    taskId: 7,
    taskInputName: "异常检测模型测试",
    modelName: "Isolation Forest",
    taskName: "模型测试",
    taskProgress: "20%",
    taskStatus: ModelEvaluationStatus.FAILED,
    taskException: "数据格式不兼容，需要重新预处理",
    groupVName: "KDD Cup 99数据集",
    createTime: "2025-05-13 14:15:00",
    taskType: 1 // 数字类型：测试
  },
  {
    taskId: 4,
    taskInputName: "图像分类模型训练",
    modelName: "ResNet50",
    taskName: "模型训练",
    taskProgress: "30%",
    taskStatus: ModelEvaluationStatus.IN_PROGRESS,
    groupVName: "ImageNet训练集",
    taskException: "",
    createTime: "2025-05-10 09:00:00",
    taskType: 1 // 数字类型：测试
  },
];

const taskId = ref();
const trainType = ref(0);

const {
  drawerVisible,
  operateType,
  editingData,
  checkedRowKeys,
  onBatchDeleted,
  onDeleted
  // closeDrawer
} = useTableOperate(data, getData);

const reportShowModal = ref<any>(false);
const detailShowModal = ref<any>(false);
const processedData = ref<any>({});

// 使用流程
const assessTitle = ref<string>("使用流程");
const aInfoList = ref<any>([
  {
    name: '评估数据准备',
    info: '准备用于评估模型能力的数据集,并在 数据集管理 中导入和发布',
    btns: [],
    icon: 'data-collect',
    imgSrc: assess0
  },
  {
    name: '模型结果生成',
    info: '使用所选数据集，批量生成模型推理结果，以便进行下一步打分',
    btns: [],
    icon: 'data-qc',
    imgSrc: assess1
  },
  // {
  //   name: '人工在线评估',
  //   info: '点击评估任务操作栏「在线评估」按钮，对模型结果进行多维度人工评估',
  //   btns: [],
  //   icon: 'data-intellect',
  //   imgSrc: assess2
  // },
  {
    name: '评估指标计算',
    info: '根据所选自动评估方法，自动对推理结果进行评分，并汇总计算评估指标、产出评估报告',
    btns: [],
    icon: 'data-annotation',
    imgSrc: assess3
  }
]);

const isModalVisible = ref(false);

const selectedTask = {
  name: '任务一',
  description: '这是一个示例任务的描述信息。',
  versionName: 'v1.0',
  versionDescription: '这是第一个版本的描述。',
  evaluationDataset: '数据集A',
  evaluationModel: '模型X',
  indicators: ["1", "2", "3", "4", "5", "6"]
};

// 脚本运行
const scriptModalVisible = ref<Boolean>(false);
const configList = ref<any>([
  {
    formName: "脚本文件",
    type: "upload",
    value: undefined,
    placeholder: "",
    width: "60%",
    serverKey: "scriptFileId",
    isShow: true
  },
  {
    formName: "参数配置",
    type: "dynamicInput",
    value: null,
    width: "80%",
    serverKey: "",
    isShow: true
  }
]);
const scriptModel = ref<any>({
  scriptFileId: '',
  scriptParams: []
});

function handleAdd() {
  operateType.value = 'add';
  openModal();
}

async function handleBatchDelete() {
  // request
  const res = await delExample(checkedRowKeys.value);
  if (res.data >= 1 || !res.data) {
    onBatchDeleted();
  }
}

const handleDelete = async row => {
  const res = await delModelAssess({ id: row.id });
  if (res.data) {
    window.$message?.success?.('删除成功！');
    getDataByPage();
  }
};

const mapEditingData = (data: any) => {
  return { ...data, modelId: route.query.modelId };
};

// 模型训练
const handleCreateTask = () => {
  router.push({
    name: 'thirdparty_createtask',
    query: {
      sign: 'create'
    }
  });
};

const assessDetailList = ref<any>([]);

const handleOperate = async (operateType: OperateType, row: any) => {
  const { id } = row;
  const params = {
    taskId: id,
  }
  try {
    switch (operateType) {
      case OperateType.Pause:
        // 暂停任务逻辑
        await pauseAssessTask(params);
        await getDataByPage()
        break;
      case OperateType.Terminate:
        // 终止任务逻辑（通常为强制停止）
        await terminationAssessTask(params);
        await getDataByPage()
        break;
      case OperateType.Continue:
        // 继续任务（可能需要校验前置条件）
        await continueAssessTask(params);
        break;
      case OperateType.Restart:
        // 重新开始任务（可能需要重置任务状态）
        await restartAssessTask(params);
        await getDataByPage()
        break;
      case OperateType.StartAssess:
        // 开始评估任务（可能需要校验前置条件）
        await startAssessTask(params);
        break;
      case OperateType.ViewAssessDetail:
        const res = await viewResultAssessTask(params);
        if (res.data) {
          assessDetailList.value = transformMetrics(res.data);
          processedData.value = processEvaluationData(assessDetailList.value);
          detailShowModal.value = true;
        }
        break;
      case OperateType.Delete:
        // 删除任务（异步操作）
        await handleDelete(row);
        await getDataByPage()
        break;
      case OperateType.Log:
        openModal();
        break;
      case OperateType.Generate:
        router.push({
          name: 'thirdparty_info',
          query: {
            taskId: row.taskId,
            modelId: '3'
          }
        });
        break;
      case OperateType.Script:
        // 打开脚本运行模态框
        scriptModalVisible.value = true;
        break;
      case OperateType.Contact:
        curTaskId.value = row.id;
        // 联系厂商
        const { VITE_WS_BASE_URL } = import.meta.env;
        socketUrl.value = `${VITE_WS_BASE_URL}/ws/task/${row.id}`
        contactModalVisible.value = true;
        break;
      case OperateType.Edit:
        router.push({
          name: 'thirdparty_createtask',
          query: {
            sign: 'edit',
            id: row.id
          }
        });
        break;
      case OperateType.Download:
        // 下载评估报告
        await downloadZip('task', row);
        break;
    }

  } catch (error: any) {
    window.$message?.error?.('操作失败：' + error.message);
  }
};

// 根据表单字段类型返回对应的组件
type FormFieldConfig = {
  type: 'input' | 'textarea' | 'select' | 'dynamicInput' | 'text' | 'radioGroup' | 'cascader' | 'upload' | 'datetime'
}
const getFormComponent = (type: FormFieldConfig['type'], value: unknown, rowData: any) => {
  switch (type) {
    case 'input':
      return NInput;
    case 'textarea':
      return h(NInput, { type: 'textarea' });
    case 'select':
      return NSelect;
    case 'text':
      return {
        setup() {
          return () => h('span', field.value);
        }
      };
    case 'radioGroup':
      return {
        setup(props) {
          return () => h(NRadioGroup, {
            'v-model:value': props.value,
            name: 'anoType',
            size: 'large'
          }, props.modelList.map((item: any) =>
            h(NRadioButton, {
              value: item.value,
              label: item.label,
              onChange: (e: any) => handleFieldChange(field, e)
            })
          ));
        }
      };
    case 'cascader':
      return NCascader;
    case 'datetime':
      return h(NDatePicker, {
        'v-model:value': value,
        type: 'datetime',
        clearable: true,
        class: '!w-full'
      });
    case 'upload':
      return h(NUpload, {
        'v-model:file-list': fileList.value,
        action: 'https://www.mocky.io/v2/5e4bafc63100007100d8b70f',
        onChange: handleUploadChange,
        onRemove: handleRemove,
        'onUpdate:file-list': handleFileListChange
      }, [
        h(NButton, {}, '上传文件')
      ]);
    case 'dynamicInput':
      return h(NDynamicInput, {
        value,
        class: "w-full",
        'on-create': () => {
          return {
            formName: "",
            key: "",
            value: undefined,
          }
        }
      }, {
        default: ({ value: item }) => h('div', {
          class: "w-full flex-center gap-8px"
        }, [
          h(NInput, {
            value: item.formName,
            'onUpdate:modelValue': (val) => item.formName = val,
            type: 'text',
            placeholder: "请输入指标名称"
          }),
          h(NInput, {
            value: item.key,
            'onUpdate:modelValue': (val) => item.key = val,
            type: 'text',
            placeholder: "请输入参数键名"
          }),
          h(NInput, {
            modelValue: item.value,
            'onUpdate:modelValue': (val) => item.value = val,
            type: 'text',
            placeholder: "请输入参数值"
          })
        ]),
        'create-button-default': () => '添加脚本配置参数'
      });
    case "checkbox":
      return h(NCheckboxGroup, {
        'v-model:value': value,
      }, () => rowData.checkboxList.map((item: any) => h(NCheckbox, {
        value: item.value,
        label: item.formName,
      })))
    default:
      return null;
  }
};

const scriptJson = {
  "data": {
    "train_data_path": "/data/train.csv",
    "test_data_path": "/data/test.csv",
    "data_format": "csv",
    "preprocessing": {
      "normalize": true,
      "augmentation": {
        "rotation": 10,
        "scale": 0.1
      }
    }
  },
  "model": {
    "architecture": "CNN",
    "params": {
      "num_layers": 5,
      "num_neurons": 128,
      "kernel_size": 3,
      "stride": 1
    },
    "pretrained_model_path": "/models/pretrained_model.pth"
  },
  "training": {
    "optimizer": {
      "name": "Adam",
      "learning_rate": 0.001,
      "momentum": 0.9
    },
    "loss_function": "Cross-Entropy Loss",
    "epochs": 100,
    "batch_size": 32
  },
  "hardware": {
    "device": "GPU",
    "gpu_id": 0
  },
  "misc": {
    "log_path": "/logs/train.log",
    "model_save_path": "/models/trained_model.pth",
    "random_seed": 42
  }
};


// --------------------------------------------
enum TaskStatus {
  PENDING,     // 0
  RUNNING,     // 1
  COMPLETED,   // 2
  FAILED,      // 3
  TERMINATED,  // 4
  CONTINUED,    // 5
  PROCESSING
}

const mockAssessData = ref<any>(
  [
    {
      "taskId": 1,
      "taskInputName": "图像分类测试任务",
      "taskType": 1,
      "modelName": "ResNet50",
      "taskProgress": "0%",
      "taskStatus": 0,
      "groupVName": "COCO数据集",
      "taskException": "",
      "createTime": "2025-05-20"
    },
    {
      "taskId": 2,
      "taskInputName": "目标检测评估任务",
      "taskType": 2,
      "modelName": "YOLOv8",
      "taskProgress": "45%",
      "taskStatus": 1,
      "groupVName": "PASCAL VOC",
      "taskException": "",
      "createTime": "2025-05-21"
    },
    {
      "taskId": 3,
      "taskInputName": "语义分割测试任务",
      "taskType": 1,
      "modelName": "UNet",
      "taskProgress": "100%",
      "taskStatus": 2,
      "groupVName": "Cityscapes",
      "taskException": "",
      "createTime": "2025-05-18"
    },
    {
      "taskId": 4,
      "taskInputName": "实例分割评估任务",
      "taskType": 2,
      "modelName": "Mask R-CNN",
      "taskProgress": "30%",
      "taskStatus": 3,
      "groupVName": "COCO数据集",
      "taskException": "标注准确率低于阈值",
      "createTime": "2025-05-19"
    },
    {
      "taskId": 5,
      "taskInputName": "关键点检测测试任务",
      "taskType": 1,
      "modelName": "HRNet",
      "taskProgress": "60%",
      "taskStatus": 4,
      "groupVName": "MNIST",
      "taskException": "人工干预终止",
      "createTime": "2025-05-22"
    },
    {
      "taskId": 6,
      "taskInputName": "图像分类评估任务",
      "taskType": 2,
      "modelName": "EfficientNet",
      "taskProgress": "75%",
      "taskStatus": 5,
      "groupVName": "CIFAR-10",
      "taskException": "",
      "createTime": "2025-05-23"
    }
  ]
)

// 对接厂商
const contactModalVisible = ref(false);
const socketUrl = ref('ws://your-api-server/log-stream')
const logs = ref([])
const isConnected = ref(false)

const curTaskId = ref<string | undefined>(undefined);

const finishContact = async () => {
  console.log('完成对接')
  const res = await finishContactAssessTask({
    id: curTaskId.value
  });
  if (res.data) {
    window?.$message?.success('完成对接成功');
    contactModalVisible.value = false
    getData();
  }
}

// 使用Map管理所有WebSocket连接，键为taskId，值为WebSocket实例
const socketMap = new Map();

// 封装WebSocket创建逻辑，支持重连机制
function createWebSocket(taskId) {
  const socketUrl = `${import.meta.env.VITE_WS_BASE_URL}/websocket/assessProgress/${taskId}`;
  const socket = new WebSocket(socketUrl);

  socket.onopen = () => {
    console.log(`WebSocket连接 ${taskId} 已打开`);
  };

  socket.onmessage = (event) => {
    try {
      const obj = JSON.parse(event.data);
      updateTaskProgress(obj);

      // 任务完成时关闭连接
      if (isTaskComplete(obj)) {
        closeSocket(taskId);
        getDataByPage(); // 刷新数据
      }
    } catch (error) {
      console.error('处理WebSocket消息出错:', error);
    }
  };

  socket.onerror = (error) => {
    console.error(`WebSocket错误 ${taskId}:`, error);
    // 可以在这里添加重连逻辑
  };

  socket.onclose = (event) => {
    console.log(`WebSocket连接 ${taskId} 已关闭，代码: ${event.code}`);
    socketMap.delete(taskId);

    // 非主动关闭时可以尝试重连
    if (event.code !== 1000) {
      setTimeout(() => initSocket(taskId), 3000);
    }
  };

  return socket;
}

// 初始化或重用WebSocket连接
function initSocket(taskId) {
  // 如果连接已存在且处于连接状态，则不创建新连接
  if (socketMap.has(taskId)) {
    const existingSocket = socketMap.get(taskId);
    if (existingSocket.readyState === WebSocket.OPEN || existingSocket.readyState === WebSocket.CONNECTING) {
      return existingSocket;
    }
  }

  const socket = createWebSocket(taskId);
  socketMap.set(taskId, socket);
  return socket;
}

// 更新任务进度
function updateTaskProgress(obj) {
  const index = data.value.findIndex((item) => item.id === obj.id);
  if (index !== -1) {
    const progress = parseProgress(obj.taskProgress);
    data.value[index].taskProgress = progress + '%';
    if (progress >= 100) {
      getDataByPage();
    }
  }
}

// 解析进度值
function parseProgress(progressStr) {
  if (!progressStr) return 0;
  const progress = parseFloat(progressStr);
  return isNaN(progress) ? 0 : progress;
}

// 判断任务是否完成
function isTaskComplete(obj) {
  const progress = parseProgress(obj.taskProgress);
  return progress >= 100;
}

// 关闭指定WebSocket连接
function closeSocket(taskId) {
  if (socketMap.has(taskId)) {
    const socket = socketMap.get(taskId);
    socket.close(1000, '任务完成');
    socketMap.delete(taskId);
  }
}

// 关闭所有WebSocket连接
function closeAllSockets() {
  socketMap.forEach((socket, taskId) => {
    socket.close(1000, '页面卸载');
  });
  socketMap.clear();
}

// 监听数据变化，初始化未完成任务的WebSocket连接
watch(() => data.value, (list) => {
  if (!Array.isArray(list) || list.length === 0) return;

  // 过滤出未完成的任务
  const incompleteTasks = list.filter(task => {
    const progress = parseProgress(task.taskProgress);
    return progress < 100;
  });

  // 为每个未完成任务创建WebSocket连接
  incompleteTasks.forEach(task => {
    initSocket(task.id);
  });

  // 关闭已完成任务的WebSocket连接
  const completedTaskIds = new Set(list
    .filter(task => parseProgress(task.taskProgress) >= 100)
    .map(task => task.id));

  socketMap.forEach((_, taskId) => {
    if (completedTaskIds.has(taskId)) {
      closeSocket(taskId);
    }
  });
});

// 在组件卸载时关闭所有连接
onUnmounted(() => {
  closeAllSockets();
});


// ------------------------------------
// 定义原始输入对象的类型
interface EvaluationMetrics {
  mPrecision?: number | null;
  PR_curve?: string;
  mFalseAlarmRate?: number | null;
  mAP_0_5?: number | null; // 替代 mAP@0.5
  mAccuracy?: number | null;
  mRecall?: number | null;
  mMissRate?: number | null;
  confusion_matrix?: string;
}

// 定义输出数组项的类型
interface TransformedItem {
  key: string;
  label: string;
  value: string;
}

// 中文 label 映射表
const labelMap: Record<string, string> = {
  'mPrecision': "平均精度 (mPrecision)",
  'mRecall': "平均召回率 (mRecall)",
  'mAP@0.5': "均值平均精度 (mAP@0.5)",
  'mMissRate': "漏检率 (MissRate)",
  'mFalseAlarmRate': "虚警率 (FalseAlarmRate)",
  'mAccuracy': "平均正确率 (mAccuracy)",
  PRCurve: "P-R曲线",
  ConfusionMatrix: "混淆矩阵图"
};

/**
 * 将原始评估数据转换为带 key、label 和 value 的数组对象
 * @param originalData 原始评估数据
 */
function transformMetrics(originalData: EvaluationMetrics): TransformedItem[] {
  return Object.keys(labelMap).map(key => {
    let originalKey = key;

    if (key === 'PRCurve') originalKey = 'PR_curve';
    if (key === 'ConfusionMatrix') originalKey = 'confusion_matrix';

    const value = originalData[originalKey as keyof EvaluationMetrics];

    return {
      key,
      label: labelMap[key],
      value: value === null || value === undefined ? "null" : String(value)
    };
  });
}

// 输入类型定义
interface TransformedItem {
  key: string;
  label: string;
  value: string;
}

// 输出类型定义
interface ProcessedMetrics {
  metrics: Array<{
    key: string;
    label: string;
    value: string;
  }>;
  prCurveSrc: string;
  confusionMatrixSrc: string;
}

/**
 * 将原始的数组对象转换为结构化的 metrics + 图片路径对象
 * @param data 原始数组对象
 */
function processEvaluationData(data: TransformedItem[]): ProcessedMetrics {
  const result: ProcessedMetrics = {
    metrics: [],
    prCurveSrc: '',
    confusionMatrixSrc: ''
  };

  data.forEach(item => {
    switch (item.key) {
      case 'PRCurve':
        result.prCurveSrc = item.value;
        break;
      case 'ConfusionMatrix':
        result.confusionMatrixSrc = item.value;
        break;
      default:
        // 只保留数值型指标，跳过值为 "null" 的项？
        result.metrics.push({ ...item });
        break;
    }
  });

  return result;
}


// 下载压缩包
const downloadZip = async (sign: 'task' | 'apply', row: any) => {
  const baseUrl = import.meta.env.VITE_SERVICE_BASE_URL;
  const url = new URL('/download/zip', baseUrl).toString();

  // 根据 sign 提取对应 ID
  const id = row.id;

  if (id === undefined || id === null) {
    throw new Error('无效的导出 ID');
  }

  const headers = {
    Authorization: `Bearer ${getToken()}`
  };

  try {
    await downloadFile1({
      url,
      params: { id },
      headers
    });
  } catch (error) {
    console.error(`文件导出失败（${sign}）:`, error);
  }
}

const logListRef = ref<any>(null);

const restartRun = async (row: any) => {
  const res = await restartTask({
    id: curTaskId.value
  });
  if (res.data >= 1) {
    logListRef.value?.clearLogs();
    window.$message?.success?.("重启成功！");
  }
}
</script>

<template>
  <div class="min-h-500px flex-col-stretch gap-16px overflow-hidden lt-sm:overflow-auto">
    <NCard :bordered="false" size="small" class="card-wrapper">
      <NCollapse default-expanded-names="user-search">
        <NCollapseItem :title="assessTitle" name="user-search">
          <NFlex justify="space-between" class="wrap-container">
            <div v-for="(item, index) of aInfoList" :key="index" class="item-manage flex justify-center items-center">
              <div class="item_main w-full">
                <div class="item-manage_icon">
                  <img :src="item.imgSrc" alt="" class="w-35%">
                  <div class="iconName">{{ item.name }}</div>
                </div>
                <div class="item-manage_info w-full flex justify-center items-center">{{ item.info }}</div>
                <div class="item-manage_btnC">
                  <NButton v-for="(val, idx) of item.btns" :key="idx" quaternary type="info" @click="navTo(val)">
                    {{ val.name }}
                  </NButton>
                </div>
              </div>
              <div class="item_arrow" v-if="index !== aInfoList.length - 1">
                <div class="flow-arrow"><span class="aibp-custom-icon aibp-custom-icon-arrow">
                    <svg width="24" height="24">
                      <path fill="#B8BABF" d="m8.053 3 9.192 9.192L8 21.437v-5.253l3.79-3.79L8 8.603V3.052L8.053 3Z">
                      </path>
                    </svg></span>
                </div>
              </div>
            </div>
          </NFlex>
        </NCollapseItem>
      </NCollapse>
    </NCard>
    <UserSearch v-model:model="searchParams" @reset="resetSearchParams" @search="getDataByPage" />
    <NCard title="测试评估任务" :bordered="false" size="small" class="sm:flex-1-hidden card-wrapper">
      <template #header-extra>
        <TableHeaderOperation v-model:columns="columnChecks" :disabled-delete="checkedRowKeys.length === 0"
          :loading="loading" :is-add="false" :is-del="false" @add="handleAdd" @delete="handleBatchDelete"
          @refresh="getData">
          <template #prefix>
            <NButton size="small" type="primary" class="-mr-24px" @click="handleCreateTask()"
              v-hasPermi="'thirdparty:assess:createTask'">创建评估任务</NButton>
          </template>
        </TableHeaderOperation>
      </template>
      <NDataTable v-model:checked-row-keys="checkedRowKeys" :columns="columns" :data="data" size="small"
        :flex-height="!appStore.isMobile" :scroll-x="1800" :loading="loading" remote :row-key="row => row.id"
        :pagination="mobilePagination" class="sm:h-full" />
      <UserOperateDrawer v-model:visible="drawerVisible" :operate-type="operateType"
        :row-data="mapEditingData(editingData)" @submitted="getDataByPage" />
    </NCard>
    <!-- 评估报告 modal -->
    <NModal v-model:show="reportShowModal">
      <NCard style="width: 600px" title="评估报告" :bordered="false" size="huge" role="dialog" aria-modal="true">
        <div class="wrap_content">
          <img :src="assReport" alt="" class="w-full h-660px" />
        </div>
        <template #footer>
          <NSpace justify="end" :size="16">
            <NButton type="primary">下载报告</NButton>
            <NButton @click="() => (reportShowModal = false)">关闭窗口</NButton>
          </NSpace>
        </template>
      </NCard>
    </NModal>
    <!-- 评估详情 modal -->
    <NModal v-model:show="detailShowModal">
      <NCard style="width: 800px" title="评估详情" :bordered="false" size="huge" role="dialog" aria-modal="true">
        <!-- 指标信息 -->
        <div class="metrics-container">
          <n-grid :cols="2" :x-gap="24" :y-gap="16">
            <n-gi v-for="item in processedData.metrics" :key="item.key">
              <n-form-item :label="item.label" label-placement="top">
                <n-input disabled :value="item.value" placeholder="暂无数据" />
              </n-form-item>
            </n-gi>
          </n-grid>
        </div>

        <!-- 图片区域 -->
        <div class="image-container" style="margin-top: 32px;">
          <h3 style="text-align: left; margin-bottom: 16px;">可视化图表</h3>
          <div class="image-row flex justify-between flex-wrap">
            <div class="image-item w-1/2">
              <strong>P-R曲线</strong>
              <n-image width="400" :src="processedData.prCurveSrc" />
            </div>
            <div class="image-item w-1/2">
              <strong>混淆矩阵图</strong>
              <n-image width="400" :src="processedData.confusionMatrixSrc" />
            </div>
            <div class="image-item w-1/2">
              <strong>ROC曲线</strong>
              <n-image width="400" :src="processedData.rocCurveSrc" />
            </div>
          </div>
        </div>
        <template #footer>
          <NSpace justify="end" :size="16">
            <NButton @click="() => (detailShowModal = false)">关闭窗口</NButton>
          </NSpace>
        </template>
      </NCard>
    </NModal>
    <!-- 脚本运行 -->
    <NModal v-model:show="scriptModalVisible">
      <NCard style="width: 800px" title="脚本运行" :bordered="false" size="huge" role="dialog" aria-modal="true">
        <div class="wrap_content">
          <NForm ref="formRef" :model="model" :rules="rules" label-placement="left" label-width="88px">
            <NFormItem label="数据路径" path="paramConfig">
              <n-input v-model:value="value" type="text" placeholder="请输入数据路径" />
            </NFormItem>
            <NFormItem label="脚本文件" path="deptName">
              <n-upload action="https://www.mocky.io/v2/5e4bafc63100007100d8b70f" :headers="{
                'naive-info': 'hello!',
              }" :data="{
                'naive-data': 'cool! naive!',
              }">
                <n-button>上传文件</n-button>
              </n-upload>
            </NFormItem>
            <NFormItem label="预处理参数" path="paramConfig">
              <n-dynamic-input preset="pair" key-placeholder="请输入参数键名" value-placeholder="请输入参数值" />
            </NFormItem>
            <NFormItem label="脚本命令" path="paramConfig">
              <n-input v-model:value="value" type="textarea" placeholder="请输入脚本命令" />
            </NFormItem>
            <NFormItem label="脚本示例" path="paramConfig">
              <n-popover trigger="hover" placement="right">
                <template #trigger>
                  <n-button quaternary type="info">
                    脚本示例（参考）
                  </n-button>
                </template>
                <div class="w-600px h-600px overflow-auto">
                  <VueJsonPretty path="res" :data="scriptJson" :show-length="true" />
                </div>
              </n-popover>
            </NFormItem>
          </NForm>
        </div>
        <template #footer>
          <NSpace justify="end" :size="16">
            <NButton type="primary">运行</NButton>
            <NButton @click="() => (scriptModalVisible = false)">关闭窗口</NButton>
          </NSpace>
        </template>
      </NCard>
    </NModal>
    <!--生成评估报告-->
    <ReportDetailModal v-model:show="isModalVisible" :task="selectedTask" />
    <!-- 网络日志 -->
    <NetWorkLogDrawer v-model:visible="visible" />
    <!-- 对接厂商modal -->
    <NModal v-model:show="contactModalVisible">
      <NCard style="width: 800px" title="对接厂商" :bordered="false" size="huge" role="dialog" aria-modal="true">
        <div class="wrap_content w-full h-440px -mt-16px">
          <!-- <LogViewer :socket-url="socketUrl" :container-height="500" @connected="onConnected"
            @disconnected="onDisconnected" @message="onMessage" @error="onError" /> -->
          <!-- <RealtimeLog :socketUrl="socketUrl" :maxLogs="2000" :autoConnect="true" /> -->
          <RealtimeLogList ref="logListRef" :socket-url="socketUrl" :autoConnect="true" />
        </div>
        <template #footer>
          <NSpace justify="right" :size="16">
            <!-- 开始  -->
            <!-- <NButton type="primary" @click="startRun">开始</NButton> -->
            <!-- 暂停 -->
            <!-- <NButton type="primary" @click="pauseRun">暂停</NButton> -->
            <!-- 继续   -->
            <!-- <NButton type="primary" @click="continueRun">继续</NButton> -->
            <!-- 结束 -->
            <!-- <NButton type="primary" @click="endRun">结束</NButton> -->
            <!-- 重新开始 -->
            <NButton type="primary" @click="restartRun">重新开始</NButton>
            <!-- 完成对接 -->
            <NButton type="primary" @click="finishContact">完成对接</NButton>
            <NButton @click="() => (contactModalVisible = false)">关闭窗口</NButton>
          </NSpace>
        </template>
      </NCard>
    </NModal>
  </div>
</template>

<style scoped lang="scss">
::-webkit-scrollbar-button {
  background-color: #ccc;
}

.card-wrapper {
  border-radius: 8px;
}

.item-manage {
  flex: 1;

  .item-manage_icon {
    display: flex;
    flex-direction: column;
    justify-content: center;
    flex-wrap: wrap;
    align-items: center;

    .iconName {
      font-size: 14px;
      color: #151b26;
      line-height: 22px;
      margin: 10px 0 8px;
      text-align: center;
    }
  }

  .item-manage_info {
    font-size: 12px;
    color: #84868c;
    line-height: 20px;
    margin-bottom: 8px;
  }

  .item-manage_btnC {
    display: flex;
    justify-content: center;
    align-items: center;

    .btn {
      color: #2468f2;
      font-size: 12px;
    }
  }
}
</style>
