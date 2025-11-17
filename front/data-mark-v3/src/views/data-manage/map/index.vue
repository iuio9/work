<script setup lang="tsx">
import { NButton, NCascader, NPopover, NProgress, NSpin, NTag, useDialog, useMessage } from 'naive-ui';
import axios from 'axios';
import { useBoolean } from '~/packages/hooks';
import {
  deleteDataGroup,
  deleteDataSet,
  fetchGetDataSetList,
  fetchImportList,
  fileDownload,
  getSelectDataSetDictList,
  updateDataSetName
} from '@/service/api/dataManage';
import { SignType } from '@/views/data-manage/interface/map';
import SvgIcon from '@/components/custom/svg-icon.vue';
import VersionInfo from '@/components/custom/version-info.vue';
import Pagination from '@/components/custom/Pagination.vue';
import { downloadByData } from '@/utils/common';
import noData from '@/assets/imgs/noData.png';
import { useTable } from '@/hooks/common/table';
import { getSelectImportFileList } from '@/service/api/tag';
import { $t } from '@/locales';
import { useAppStore } from '@/store/modules/app';
import OperateModal from './modules/operate-modal.vue';
import ImgInfo from './modules/ImgInfo.vue';
import { useAnoStore } from '@/store/modules/ano';
import { useAuthStore } from "@/store/modules/auth";
import ExportMapModal from './modules/exportMapModal.vue';

interface DataManageObj {
  title: string;
  infoList: any[];
  operationBtns: any[];
}

const dataManageObj = ref<DataManageObj>({
  title: '介绍',
  infoList: [
    {
      name: '数据采集',
      info: '支持采集图片类数据，可以从本地接入视频抽帧图片或通过接入云服务调用数据接入图片',
      btns: [],
      icon: 'data-collect'
    },
    {
      name: '数据质检',
      info: '支持对图像数据进行质检，质检报告中的指标可作为数据处理（标注、清洗）的重要参考',
      btns: [],
      icon: 'data-qc'
    },
    {
      name: '数据智能处理',
      info: '⽀持对图⽚和⽂本数据进⾏清洗，以及对图⽚数据进⾏增强处理，您可按需选择数据智能处理功能',
      btns: [],
      icon: 'data-intellect'
    },
    {
      name: '数据标注',
      info: '支持图片、文本、音频、视频数据标注，并支持丰富标注模板，个人在线标注及智能标注等多种方式',
      btns: [
        {
          name: '在线标注',
          routeName: 'annotation'
        }
      ],
      icon: 'data-annotation'
    }
  ],
  operationBtns: [
    { icon: 'material-symbols:add', name: '新增版本', perm: "system:map:addVersion" },
    { icon: 'material-symbols-light:border-all', name: '所有版本', perm: "system:map:allVersion" },
    // { icon: 'icon-park-outline:merge', name: '合并版本', perm: "system:map:concatVersion" },
    { icon: 'material-symbols-light:delete', name: '删除', perm: "system:map:mapDelete" }
  ]
});
const tableObj = ref<any>({
  columns: [
    {
      title: '版本',
      key: 'version',
      width: 60,
      render: row => {
        return [
          h(
            'div',
            {
              class: 'flex items-center'
            },
            [
              h(
                'span',
                {
                  class: 'text-[12px] text-[#151b26]'
                },
                `V${row.version}`
              ),
              h(
                NPopover,
                { trigger: 'hover', placement: 'right' },
                {
                  trigger: () =>
                    h('span', { class: 'block ml-4px' }, [
                      h(SvgIcon, {
                        icon: 'fluent:info-24-regular',
                        class: 'text-[14px]',
                        localIcon: 'fluent--info-24-regular'
                      })
                    ]),
                  default: () =>
                    h(VersionInfo, {
                      rowData: row,
                      onImport: e => handleImport(e),
                      onRemark: e => refresh(e)
                    })
                }
              )
            ]
          )
        ];
      },
      fixed: 'left'
    },
    {
      width: 180,
      title: '数据集ID',
      key: 'sonId',
      render: row => {
        return <span class="text-12px text-[#151b26]">{row.sonId}</span>;
      },
      fixed: 'left'
    },
    {
      width: 120,
      title: '数据集类型',
      key: 'dataTypeName'
    },
    {
      width: 100,
      title: '数据量',
      key: 'count',
      // minWidth: 200,
      render: row => {
        return <span class="text-12px text-[#151b26]">{row.count ? row.count : 0}</span>;
      }
    },
    {
      width: 120,
      title: '最近导入状态',
      key: 'importStatus',
      render: row => {
        if (row.importStatus === null) {
          return null;
        }

        const tagMap: Record<Api.Common.EnableStatus, NaiveUI.ThemeColor> = {
          1: 'success',
          2: 'warning',
          0: 'default'
        };
        const statusInfo = {
          0: '暂无导入记录',
          1: '已完成',
          2: '导入失败'
        };

        const label = statusInfo[row.importStatus];
        // return <NTag type={tagMap[row.importStatus]}>{label}</NTag>;
        return [
          h(
            NTag,
            {
              type: tagMap[row.importStatus],
              style: {
                display: Number(row.importStatus) == 0 ? '' : 'none'
              }
            },
            `${label}`
          ),
          h(
            'div',
            {
              style: {
                width: '80%'
              }
            },
            [
              h(NProgress, {
                type: 'line',
                'indicator-placement': 'inside',
                processing: false,
                percentage: row.progress,
                style: {
                  display: row.progress ? 'block' : 'none'
                }
              })
            ]
          ),
          h(
            NTag,
            {
              type: tagMap[row.importStatus],
              style: {
                display: Number(row.importStatus) == 1 ? '' : 'none'
              },
              class: 'cursor-pointer',
            },
            `${label}`
          ),
          h(
            NTag,
            {
              type: tagMap[row.importStatus],
              style: {
                display: Number(row.importStatus) == 2 ? '' : 'none'
              },
              class: 'cursor-pointer',
              onClick: () => handleImport(row)
            },
            `${label}`
          )
        ];
      }
    },
    {
      width: 120,
      title: "标注类型",
      key: "markType",
      render: (row) => {
        return (
          <div class="flex items-center">
            <span>{row.anoType == 0 ? '图像分割' : '物体检测'}</span>
          </div>
        );
      },
    },
    {
      width: 140,
      title: '标注状态',
      align: 'left',
      key: 'status',
    },
    {
      key: 'createTime',
      title: '创建时间',
      align: 'center',
      width: 200
    },
    {
      title: '操作',
      align: 'left',
      width: 340,
      key: '操作',
      render(row: any) {
        const isHideBtn = +row.count == 0;
        // isMany
        const isMany = row.isMany == 1;
        return [
          h(
            NButton,
            {
              type: "primary",
              ghost: true,
              size: "small",
              onClick: () => handleOperation(SignType.detail, row),
              class: 'mr-8px',
              size: 'small',
              disabled: isHideBtn,
            },
            '查看'
          ),
          // h(
          //   NButton,
          //   {
          //     type: 'info',
          //     quaternary: true,
          //     onClick: () => handleOperation(SignType.sceneMange, row),
          //     class: `h-16px`,
          //     size: 'small'
          //   },
          //   '场景管理'
          // ),
          h(
            NButton,
            {
              type: "primary",
              ghost: true,
              size: "small",
              disabled: isHideBtn || isMany,
              onClick: () => handleOperation(SignType.annotation, row),
              class: 'mr-8px',
              size: 'small',
            },
            '标注'
          ),
          h(
            NButton,
            {
              type: "primary",
              ghost: true,
              size: "small",
              onClick: () => handleOperation(SignType.import, row),
              class: 'mr-8px',
              disabled: isMany,
              size: 'small',
            },
            '导入文件'
          ),
          h(
            NButton,
            {
              type: "primary",
              ghost: true,
              size: "small",
              disabled: isHideBtn || isMany,
              class: 'mr-8px',
              onClick: () => handleOperation(SignType.export, row),
            },
            '导出数据集',
          ),
          h(
            NButton,
            {
              type: "primary",
              ghost: true,
              size: "small",
              onClick: () => handleOperation(SignType.delete, row),
              disabled: isMany,
            },
            '删除'
          )
        ];
      },
      fixed: 'right'
    }
  ],
  pagination: false,
  data: [{}]
});
const router = useRouter();
const dialog = useDialog();
const message = useMessage();
const { bool: visible, setTrue: openModal } = useBoolean();
const { bool: exportVisible, setTrue: openExportModal } = useBoolean();

const searchParams = reactive<any>({
  page: 1,
  limit: 10,
  groupName: undefined,
  dataTypeId: undefined
});
const pageConfig = reactive<any>({
  total: null,
  page: 1,
  limit: 10
});
const dataList = ref<any>([]);
const rowData = ref<any>({});

const handleBack = () => {
  router.replace({
    name: 'data-manage_maplist',
    query: {
      dataTypeId: route.query.dataTypeId
    }
  });
};

const isImportModal = ref<Boolean>(false);
const isExport = ref<Boolean>(false);
const isImgInfo = ref<Boolean>(false);
let importList = reactive([]);
const importCols = reactive<any[]>([
  { key: 'id', title: 'ID' },
  {
    key: 'message',
    title: '详情',
    render(row: any) {
      return [
        h(
          'span',
          {
            class: 'block ml-4px',
            onClick: () => handleImgInfo(row)
          },
          [
            h(SvgIcon, {
              icon: 'fluent:info-24-regular',
              class: 'text-[14px]',
              localIcon: 'fluent--info-24-regular'
            })
          ]
        )
      ];
    }
  },
  { key: 'fileSize', title: '文件大小' },
  { key: 'count', title: '数据量' },
  { key: 'nickName', title: '创建人' },
  { key: 'importStartTime', title: '导入开始时间' },
  { key: 'importEndTime', title: '导入完成时间' },
  {
    key: 'status',
    title: '导入状态',
    render: (row: any) => {
      if (row.status === null) {
        return null;
      }

      const tagMap: Record<Api.Common.EnableStatus, NaiveUI.ThemeColor> = {
        1: 'success',
        2: 'warning'
      };
      const statusObj = {
        1: '导入完成',
        2: '导入失败'
      };

      const label = statusObj[row.status];

      return <NTag type={tagMap[row.status]}>{label}</NTag>;
    }
  }
]);
const imgConfig = ref<any>({});
const appStore = useAppStore();

const handleImgInfo = async (row: any) => {
  localStorage.setItem('fileId', row.id);
  isImgInfo.value = true;
};

// methods
const route = useRoute();
const handleCreteMap = () => {
  const dataTypeId = route.query.dataTypeId;
  router.push({
    name: 'data-manage_operation',
    query: {
      dataTypeId
    }
  });
  // localStorage.setItem("dataTypeId", route.query.dataTypeId);
};

const handleCreteMap1 = () => {
  const dataTypeId = route.query.dataTypeId;
  router.push({
    name: 'dataset_operate',
    query: {
      dataTypeId
    }
  });
  // localStorage.setItem("dataTypeId", route.query.dataTypeId);
};

function downloadPost(config) {
  return new Promise((resolve, reject) => {
    axios({
      url: config.url, // 请求地址
      method: 'post',
      data: config.data, // 参数
      responseType: 'blob' // 表明返回服务器返回的数据类型
    })
      .then(res => {
        resolve(res);
      })
      .catch(err => {
        reject(err);
      });
  });
}

const handleOperation = async (sign: SignType, row) => {
  switch (sign) {
    case SignType.import:
      router.push({
        // path: '/data-manage/import',
        name: 'dataset_operate',
        query: {
          sonId: row.sonId,
          sign: 'mapToImport',
          dataTypeId: route.query.dataTypeId,
          operateStep: 2,
          groupName: row.groupName,
          anoType: row.anoType,
          tagSelectionMode: row.tagSelectionMode
        }
      });
      break;
    case SignType.sceneMange:
      router.push({
        name: 'dataset_taggroupmanager'
      });
      break;
    case SignType.export:
      // router.push({
      //   name: 'data-manage_export',
      //   params: {
      //     sign
      //   }
      // });
      // const res = await fileDownload({ sonId: row.sonId });
      // const blob = new Blob([res]); // 处理文档流

      // isExport.value = true;
      // const baseUrl = import.meta.env.VITE_SERVICE_BASE_URL;
      // const config = {
      //   url: `${baseUrl}/file/download?sonId=${row.sonId}`,
      //   data: {
      //     sonId: row.sonId
      //   }
      // };
      // const fileName = `数据集${row.sonId}.zip`;
      // const res = await downloadPost(config);
      // if (res.data) {
      //   isExport.value = false;
      // }
      // downloadByData(res.data, fileName);

      rowData.value = row;
      openExportModal();
      break;
    case SignType.annotation:
      const anoStore = useAnoStore();
      anoStore.updateAno(row.anoType ?? 1);
      router.push({
        // name: 'data-ano_operation',
        // name: 'data-ano_imgoperate',
        name: import.meta.env.VITE_TOGGLE_OPERATE === 'Y' ? 'data-ano_imgoperate' : 'data-ano_operation',
        params: {
          sign,
          id: row.sonId
        },
        query: {
          id: row.sonId,
          anoType: "setOnline",
          markType: row.anoType,  // 标注类型
        }
      });
      break;
    case SignType.detail:
      router.push({
        name: 'data-ano_detail',
        params: {
          sign
        },
        query: {
          id: row.sonId,
          isMany: row.isMany,
          markType: row.anoType,  // 标注类型
        }
      });
      break;
    case SignType.delete:
      dialog.warning({
        title: '删除数据集',
        content: '操作删除后，数据集及全部版本数据都将会被删除且不可恢复，确认要删除吗？',
        positiveText: '确定',
        negativeText: '取消',
        onPositiveClick: () => {
          deleteDataSet({ sonId: row.sonId }).then(res => {
            if (res.data == 1) {
              message.success('删除成功！');
              getMapData();
            }
          });
        },
        onNegativeClick: () => {
          message.error('取消');
        }
      });
      break;
    default:
      throw new Error('wrong operator');
  }
};
const handleHOperation = ({ name }: any, row) => {
  if (name === '新增版本') {
    rowData.value = row;
    openModal();
  } else if (name === '所有版本') {
    router.push({
      name: 'data-manage_detail'
    });
    localStorage.setItem('rowData', JSON.stringify(row));
  } else if (name === '合并版本') {
    isMerageModal.value = true;
  } else {
    dialog.warning({
      title: '删除数据集',
      content: '操作删除后，数据集及全部版本数据都将会被删除且不可恢复，确认要删除吗？',
      positiveText: '确定',
      negativeText: '取消',
      onPositiveClick: () => {
        deleteDataGroup({
          groupId: row.groupId
        }).then(res => {
          if (res.data == 1) {
            window.$message?.success?.('删除成功！');
            getMapData();
          }
        });
      },
      onNegativeClick: () => { }
    });
  }
};

const navTo = row => {
  const obj: any = {
    annotation: 'data-ano_online'
  };
  router.push({
    name: obj[row.routeName]
  });
};

// get data
const getMapData = async (params: any) => {
  // eslint-disable-next-line no-param-reassign
  params = {
    ...searchParams,
    ...params,
    dataTypeId: params?.dataTypeId ?? route.query.dataTypeId,
    dataTypeId: searchParams.dataTypeId
  };
  const res = await fetchGetDataSetList(params);
  let { records, total, current, size } = res.data;
  records = records.map((item, index) => {
    item.dataSonResponseList = item.dataSonResponseList.map(val => {
      return {
        ...val,
        progress: 0,
        groupName: item.groupName,
        groupId: item.groupId,
        isGroup: false
      };
    });
    return item;
  });
  dataList.value = [...records];
  // socket
  dataList.value.forEach(item => {
    item.dataSonResponseList.forEach(item1 => {
      if (item1.count >= 0 && Number(item1.importStatus) == 0) {
        initSocket1({ groupId: item.groupId, sonId: item1.sonId });
      }
    });
  });
  pageConfig.total = total;
};

const refresh = async () => {
  // searchParams.groupName = '';
  // searchParams.dataTypeId = null;
  getMapData({
    // page: 1, limit: 10,
    page: pageConfig.page, limit: pageConfig.limit,
    dataTypeId: route.query.dataTypeId,
  });
};

const refresh1 = async () => {
  // searchParams.groupName = "";
  // searchParams.dataTypeId = null;
  getMapData({ page: 1, limit: 10 });
};

const handleSearch = async () => {
  getMapData({ page: 1, limit: 10 });
};

const handleToPage = e => {
  const { sign, page, pageSize } = e;
  if (sign === 'SIZE') {
    pageConfig.page = 1;
    pageConfig.limit = pageSize;
    getMapData({
      page: 1,
      limit: pageSize
    });
  }
  if (sign === 'PAGE') {
    pageConfig.page = page;
    pageConfig.limit = pageSize;
    getMapData({
      page,
      limit: pageSize
    });
  }
};

const handleImport = async e => {
  const res = await fetchImportList({
    sonId: e.sonId
  });
  importList = [...res.data];
  isImportModal.value = true;
};

// update groupName
const handleDefine = async (item: any) => {
  // request
  const res = await updateDataSetName({
    groupId: item.groupId,
    groupName: item.groupName,
  });
  if (res.data == 1) {
    window.$message?.success('数据集名称修改成功！');
    item.isGroup = false;
    refresh();
  }
};
const handleCancel = (item: any) => {
  item.isGroup = false;
};

const handleUpdateGroup = (item: any) => {
  item.isGroup = true;
};

const handleIptBlur = () => {
  refresh();
};

// -------------socket---------------
const socket = ref(null);
function initSocket(sonId: any) {
  const mapArr = JSON.parse(localStorage.getItem('mapArr'));
  mapArr.forEach((val, idx) => {
    const socketUrl = `${import.meta.env.VITE_WS_BASE_URL}/websocket/dataSetProgress/${val.sonId}`;
    socket.value = new WebSocket(socketUrl);
    socket.value.onopen = () => {
      console.log('WebSocket连接已打开');
    };
    socket.value.onmessage = event => {
      console.log('WebSocket');
      // 根据val.groupId找到dataList中的dataSonResponseList
      // 根据val.groupId找到dataList中的索引
      const dataSonResponseList = dataList.value.find((data: any) => data.groupId === val.groupId)?.dataSonResponseList;
      if (dataSonResponseList) {
        const groupIdx = dataList.value.findIndex((data: any) => data.groupId === val.groupId);
        const msg = JSON.parse(event.data);
        // 根据val.sonId查找dataSonResponseList索引
        const sonIdx = dataSonResponseList.findIndex((data: any) => data.sonId === val.sonId);
        if (sonIdx !== -1) {
          if (msg.data) {
            dataList.value[groupIdx].dataSonResponseList[sonIdx].importStatus = 3;
            dataList.value[groupIdx].dataSonResponseList[sonIdx].progress = msg.data;
          }
          if (msg.data == 100) {
            getMapData();
            // 获取storage中的mapArr
            const mapArrStr = localStorage.getItem('mapArr');
            let mapArr = [];
            if (mapArrStr) {
              mapArr = JSON.parse(mapArrStr);
            }
            mapArr = mapArr.filter((item: any) => item.sonId !== val.sonId);
            localStorage.setItem('mapArr', JSON.stringify(mapArr));
            socket.value?.close();
          }
        }
      }
    };
    socket.value.onerror = error => {
      console.error('WebSocket错误:', error);
    };
    socket.value.onclose = () => {
      console.log('WebSocket连接已关闭');
    };
  });
}
let socketTimer = null;

function initSocket1(val: any) {
  const socket = {
    value: null
  };
  const socketUrl = `${import.meta.env.VITE_WS_BASE_URL}/websocket/dataSetProgress/${val.sonId}`;
  socket.value = new WebSocket(socketUrl);
  socket.value.onopen = () => {
    console.log('WebSocket连接已打开');
  };
  socket.value.onmessage = event => {
    const msg = JSON.parse(event.data);
    console.log('msg: ', msg);
    const dataSonResponseList = dataList.value.find((data: any) => data.groupId === val.groupId)?.dataSonResponseList;
    if (dataSonResponseList) {
      const groupIdx = dataList.value.findIndex((data: any) => data.groupId === val.groupId);
      // 根据val.sonId查找dataSonResponseList索引
      const sonIdx = dataSonResponseList.findIndex((data: any) => data.sonId === val.sonId);
      if (sonIdx !== -1) {
        if (msg.data) {
          dataList.value[groupIdx].dataSonResponseList[sonIdx].importStatus = 3;
          dataList.value[groupIdx].dataSonResponseList[sonIdx].progress = msg.data;
        }
        if (msg.data == 100) {
          socketTimer = setTimeout(() => {
            getMapData();
            localStorage.removeItem('isImport');
            socket.value?.close();
          }, 1000);
        }
      }
    }
  };
  socket.value.onerror = error => {
    console.error('WebSocket错误:', error);
  };
  socket.value.onclose = () => {
    console.log('WebSocket连接已关闭');
  };
}

// 级联筛选
const mapOptions = ref<any>([]);
const recursionData = (data: any, label: any) => {
  // eslint-disable-next-line no-param-reassign
  data = data.map((item: any, index: string | number) => {
    if (item.children) {
      if (item.children.length > 0) recursionData(item.children, item.dictLabel);
      if (item.children.length === 0) delete item.children;
    }
    item.label = label ? `${item.dictLabel}` : item.dictLabel;
    item.value = item.id;
    return item;
  });
  return data;
};
async function getMapClassifyList() {
  const res = await getSelectDataSetDictList({ page: 1, limit: 10 });
  mapOptions.value = recursionData(res.data);
}

// 合并版本
const isMerageModal = ref<Boolean>(false);
const merageModel = ref<any>({});
// cascader
const checkStrategy = ref<'all' | 'parent' | 'child'>('all');
const showPath = ref<Boolean>(true);
const cascade = ref<Boolean>(true);
const responsiveMaxTagCount = ref<Boolean>(true);
const filterable = ref<Boolean>(false);
const hoverTrigger = ref<Boolean>(false);
const clearFilterAfterSelect = ref<Boolean>(true);
const tagOptions = getOptions();
const saveOptions = ref<any>([
  { value: '1', label: '是' },
  { value: '0', label: '否' }
]);

function getOptions(depth = 2, iterator = 1, prefix = '') {
  const length = 12;
  const options: CascaderOption[] = [];
  for (let i = 1; i <= length; ++i) {
    if (iterator === 1) {
      options.push({
        value: `v-${i}`,
        label: `l-${i}`,
        disabled: i % 5 === 0,
        children: getOptions(depth, iterator + 1, `${String(i)}`)
      });
    } else if (iterator === depth) {
      options.push({
        value: `v-${prefix}-${i}`,
        label: `l-${prefix}-${i}`,
        disabled: i % 5 === 0
      });
    } else {
      options.push({
        value: `v-${prefix}-${i}`,
        label: `l-${prefix}-${i}`,
        disabled: i % 5 === 0,
        children: getOptions(depth, iterator + 1, `${prefix}-${i}`)
      });
    }
  }
  return options;
}

function handleCascaderUpdate() {
  const params = {
    page: 1,
    limit: 10,
    dataTypeId: searchParams.dataTypeId
  };
  getMapData(params);
}

function handleMerage() { }

onMounted(async () => {
  searchParams.dataTypeId = +route.query.dataTypeId;
  await getMapData();
  await getMapClassifyList();
  dataList.value.forEach(item => {
    item.dataSonResponseList.forEach(item1 => {
      if (item1.count >= 0 && Number(item1.importStatus) == 0) {
        initSocket1({ groupId: item.groupId, sonId: item1.sonId });
      }
      if (item1?.isSocket == 1) {
        console.log("isSocket");
        console.log(item1.isSocket);
        initSocket1({ groupId: item.groupId, sonId: item1.sonId });
      }
      const isImport = localStorage.getItem('isImport');
      if (isImport) {
        initSocket1({ groupId: item.groupId, sonId: item1.sonId });
      }
    });
  });
});
onUnmounted(() => {
  clearTimeout(socketTimer);
  socketTimer = null;
  // const mapArrStr = JSON.parse(localStorage.getItem('mapArr'));
  // let mapArr = [];
  // if (mapArrStr) {
  //   mapArr.push({
  //     sonId: route.query.sonId,
  //     groupId: route.query.groupId,
  //   })
  // } else {
  //   mapArr = [{
  //     sonId: route.query.sonId,
  //     groupId: route.query.groupId,
  //   }];
  // }
  // localStorage.setItem("mapArr", JSON.stringify(mapArr));
});

// newCode
const isPermission = (value: string) => {
  const authStore = useAuthStore();
  const permissions = authStore.userInfo.buttons;
  // return permissions.includes(value);
  return true;
}
</script>

<template>
  <div class="flex-col items-center justify-start gap-16px">
    <div v-show="isExport" class="mask-layer">
      <div class="loading-spinner">
        <!-- 这里可以放置任何你想要的加载动画 -->
        <NSpin size="large" description="数据集导出中... 请稍等" />
      </div>
    </div>
    <NCard v-if="false" :title="dataManageObj.title" :bordered="false" size="small" class="card-wrapper">
      <NFlex justify="space-between" class="wrap-container">
        <div v-for="(item, index) of dataManageObj.infoList" :key="index" class="item-manage">
          <div class="item-manage_icon">
            <SvgIcon :local-icon="item.icon" class="inline-block align-text-bottom text-46px" />
            <div class="iconName">{{ item.name }}</div>
          </div>
          <!--<div class="item-manage_info">{{ item.info }}</div>-->
          <div class="item-manage_btnC">
            <NButton v-for="(val, idx) of item.btns" :key="idx" quaternary type="info" @click="navTo(val)">
              {{ val.name }}
            </NButton>
          </div>
        </div>
      </NFlex>
    </NCard>
    <NCard size="small" class="relative box-border flex-1 px-8px py-8px">
      <div class="h-full w-full flex-col items-center justify-start">
        <div class="table_header h-[36px] w-full">
          <div class="header_r gap-4px">
            <NButton type="primary" ghost @click="handleCreteMap1()">
              <template #icon>
                <SvgIcon local-icon="mdi--tag-add" class="text-[24px]"></SvgIcon>
              </template>
              创建数据集
            </NButton>
            <div class="w-200px">
              <!--@keyup.enter.native="handleIptBlur()"-->
              <NInput v-model:value="searchParams.groupName" placeholder="按照数据集名称查询" class="header_r_ipt" clearable />
            </div>
            <div class="min-w-200px w-auto">
              <!--@update:value="handleCascaderUpdate()"-->
              <NCascader v-model:value="searchParams.dataTypeId" clearable placeholder="类型筛选" :options="mapOptions"
                check-strategy="all"></NCascader>
            </div>
            <NButton size="small" class="header_r_btn" @click="handleSearch">
              <template #icon>
                <SvgIcon local-icon="ic--round-search"></SvgIcon>
              </template>
            </NButton>
            <NButton size="small" class="header_r_btn" @click="refresh">
              <template #icon>
                <icon-mdi-refresh class="text-icon" />
              </template>
            </NButton>
          </div>
          <div class="header_l">
            <NButton size="small" ghost type="primary" class="mr-4px" @click="handleBack()">
              <template #icon>
                <SvgIcon local-icon="carbon--return" class="text-icon" />
              </template>
              <span>返回上一级</span>
            </NButton>
          </div>
        </div>
        <div v-if="dataList.length !== 0" class="table_content w-full flex-1">
          <div v-for="(item, index) of dataList" :key="item.id" class="item-table mb-16px">
            <div class="item-table_header">
              <div class="left">
                <span v-if="!item.isGroup" class="block cursor-pointer text-[#252933]" @click="handleUpdateGroup(item)">
                  {{ item.groupName }}
                </span>
                <div v-else="item.isGroup" class="flex items-center">
                  <NInput v-model:value="item.groupName" placeholder="请输入数据集名称" />
                  <div class="flex items-center gap-4px">
                    <NButton quaternary type="info" size="tiny" @click="handleDefine(item)">确定</NButton>
                    <NButton quaternary size="tiny" @click="handleCancel(item)">取消</NButton>
                  </div>
                </div>
                <span>数据集组ID: {{ item.groupId }}</span>
              </div>
              <div class="right">
                <div v-for="(val, index) of dataManageObj.operationBtns" :key="index" class="item_btn cursor-pointer"
                  @click="handleHOperation(val, item)" v-show="isPermission(val.perm)">
                  <SvgIcon :icon="val.icon" class="inline-block align-text-bottom text-18px" />
                  <span>{{ val.name }}</span>
                </div>
              </div>
            </div>
            <NDataTable size="small" :columns="tableObj.columns" :data="item.dataSonResponseList"
              :pagination="tableObj.pagination" :bordered="false" :row-key="(row) => row.sonId" />
          </div>
        </div>
        <div v-else class="table_content w-full flex-1">
          <div class="h-full w-full flex flex-col items-center justify-center">
            <img :src="noData" alt="" class="h-auto w-200px" />
            <div class="mt-4px text-[#848484]">暂无数据</div>
          </div>
        </div>
        <div class="mt-12px h-[24px] w-full flex items-center justify-end">
          <!--
<n-pagination
            v-model:page="pageConfig.page"
            v-model:page-size="pageConfig.limit"
            :page-count="pageConfig.total"
            show-size-picker
            :page-sizes="[10, 20, 30, 40]"
          >
          </n-pagination>
-->
          <Pagination :page-current="pageConfig.page" :page-total="pageConfig.total" :page-size="pageConfig.limit"
            @to-page="handleToPage"></Pagination>
        </div>
      </div>
    </NCard>
    <OperateModal v-model:visible="visible" :row-data="rowData" @success="refresh1" />
    <NModal v-model:show="isImportModal">
      <NCard style="width: 900px" title="导入记录" :bordered="false" size="huge" role="dialog" aria-modal="true">
        <div class="content">
          <NDataTable :columns="importCols" :data="importList" :bordered="false" />
        </div>
        <template #footer>
          <NSpace justify="end" :size="16">
            <NButton type="primary" @click="() => (isImportModal = false)">我知道了</NButton>
          </NSpace>
        </template>
      </NCard>
    </NModal>
    <!-- 上传图片详情 record-->
    <NModal v-model:show="isImgInfo">
      <ImgInfo></ImgInfo>
    </NModal>
    <!--合并版本-->
    <NModal v-model:show="isMerageModal" title="合并版本" preset="card" class="wrap_modal w-600px">
      <NScrollbar class="h-auto pr-20px">
        <NForm ref="formRef" :model="merageModel" label-placement="left" :label-width="140">
          <NGrid responsive="screen" item-responsive>
            <NFormItemGi span="24 m:24" label="选择数据集版本" path="groupVList">
              <NCascader v-model:value="merageModel.tags" multiple clearable placeholder="选择对应数据集下的版本进行合并操作（*可多选）"
                :max-tag-count="responsiveMaxTagCount ? 'responsive' : undefined
                  " :expand-trigger="hoverTrigger ? 'hover' : 'click'" :options="tagOptions" :cascade="cascade"
                :check-strategy="checkStrategy" :show-path="showPath" :filterable="filterable"
                :clear-filter-after-select="clearFilterAfterSelect" />
            </NFormItemGi>
            <NFormItemGi span="24 m:24" label="是否保留历史版本" path="historyVersion">
              <NRadioGroup v-model:value="merageModel.isHistorySave" name="radiogroup">
                <NSpace>
                  <NRadio v-for="item in saveOptions" :key="item.value" :value="item.value">
                    {{ item.label }}
                  </NRadio>
                </NSpace>
              </NRadioGroup>
            </NFormItemGi>
          </NGrid>
        </NForm>
      </NScrollbar>
      <template #footer>
        <NSpace justify="end" :size="16">
          <NButton @click="() => (isMerageModal = false)">关闭窗口</NButton>
          <NButton type="primary" @click="handleMerage()">开始合并</NButton>
        </NSpace>
      </template>
    </NModal>

    <!-- 导出数据集 -->
    <ExportMapModal v-model:visible="exportVisible" :sonId="rowData.sonId" />
  </div>
</template>

<style lang="scss" scoped>
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
    overflow: hidden;
    text-overflow: ellipsis;
    display: -webkit-box;
    -webkit-line-clamp: 3;
    -webkit-box-orient: vertical;
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

.table_header {
  display: flex;
  justify-content: space-between;
  align-items: center;

  .header_l {}

  .header_r {
    display: flex;
    justify-content: flex-start;
    align-items: center;

    .header_r_ipt {
      width: 200px;
    }

    .header_r_btn {
      margin-left: 4px;
      height: 34px;
    }
  }
}

.table_content {
  margin-top: 16px;

  .item-table_header {
    height: 52px;
    padding: 15px;
    background-color: #eeeeee !important;
    border-bottom: 1px solid #ededed;
    display: flex;
    justify-content: space-between;
    align-items: center;

    .left {
      display: flex;
      justify-content: space-between;
      align-items: center;

      span {
        font-size: 12px;
      }

      span:nth-of-type(2) {
        margin-left: 8px;
      }
    }

    .right {
      display: flex;
      justify-content: flex-start;
      align-items: center;

      .item_btn {
        margin-right: 24px;
        display: flex;
        justify-content: center;
        align-items: center;

        span {
          font-size: 12px;
          margin-left: 3px;
        }
      }
    }
  }

  .item-operation {
    .n-button {
      margin-right: 14px;
    }
  }
}

.mask-layer {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  background-color: rgba(255, 255, 255, 0.7);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 9999;
}
</style>
