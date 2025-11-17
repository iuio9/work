<script setup lang="ts">
import _ from "lodash";
import { nanoid } from "nanoid";
import AILabel from "ailabel";
import { NButton, useDialog, useMessage } from "naive-ui";

import noTag from "@/assets/imgs/noTag.png";
import {
  DataDetailsCount,
  MarkFileUpload,
  addDataMarkInfo,
  addDataSetAndLabel,
  addSaveLabel,
  deleteDataSetLabel,
  getDataDetailsNoMarkFilePath,
  getSelectGroupLabel,
  segmentStart,
  selectDataSetLabel,
  fileIsApprove,
  verifyComplete,
  returnTask,
  remainingApprove,
  submitTask,
  submitTaskPrompt,
  getSelectGroupLabelPage,
  topUpLabel
} from "@/service/api/ano";
import { fetchLabelEdit } from "@/service/api/tag";
import { useAnoStore } from "@/store/modules/ano";
import { $t } from "@/locales";
import { usePagination } from "vue-request";
import { useWebWorker } from '@vueuse/core';

interface IconInfo {
  name: string;
  icon: string;
  mode: string;
  [key: string]: string;
}

// data
const router = useRouter();
const route = useRoute();
const iconList: IconInfo[] = reactive([
  {
    name: "前景点标注",
    icon: "mdi:auto-fix",
    localIcon: "mdi--auto-fix",
    mode: "POINT",
    sign: "foreground_points",
    isOperate: false,
    permList: ["1"],
  },
  {
    name: "背景点标注",
    icon: "ic:sharp-auto-fix-off",
    localIcon: "ic--sharp-auto-fix-off",
    mode: "POINT",
    sign: "background_points",
    isOperate: false,
    permList: ["1"],
  },
  // {name: '图像居中', icon: 'carbon:center-to-fit', mode: 'Center'},
  {
    name: "多边形",
    icon: "uil:polygon",
    mode: "POLYGON",
    localIcon: "uil--polygon",
    isOperate: false,
    permList: ["1"],
  },
  {
    name: "矩形",
    icon: "ph:rectangle-bold",
    mode: "RECT",
    localIcon: "ph--rectangle-bold",
    isOperate: true,
    permList: ["0", "1"],
  },
  {
    name: "圆形",
    icon: "material-symbols:circle-outline",
    mode: "CIRCLE",
    localIcon: "material-symbols--circle-outline",
    isOperate: false,
    permList: ["1"],
  },
  {
    name: "撤销",
    icon: "fluent:arrow-undo-16-filled",
    mode: "Revoke",
    localIcon: "fluent--arrow-undo-16-filled",
    isOperate: false,
    permList: ["0", "1"],
  },
  {
    name: "移动",
    localIcon: "ic--baseline-pan-tool",
    mode: "PAN",
    isOperate: false,
    permList: ["0", "1"],
  },
  // { name: '重做', icon: 'fluent:arrow-redo-32-filled', mode: 'Back' },
  {
    name: "放大",
    icon: "gg:add",
    mode: "zoomIn",
    localIcon: "gg--add",
    isOperate: false,
    permList: ["0", "1"],
  },
  {
    name: "缩小",
    icon: "icon-park-outline:reduce-one",
    mode: "zoomOut",
    localIcon: "icon-park-outline--reduce-one",
    isOperate: false,
    permList: ["0", "1"],
  },
  // { name: "删除", icon: "material-symbols:delete-outline", mode: "delete" },
  // {
  //   name: "导入图片到该数据集",
  //   localIcon: "fa6-solid--file-import",
  //   mode: "import",
  //   isOperate: false,
  //   permList: ["0", "1"],
  // },
]);
const ctxMenuConfig = ref<any>({
  show: false,
  optionsComponent: {
    zIndex: 2000,
    minWidth: 250,
    x: 500,
    y: 200,
    getContainer: () => main_map.value,
  },
});
const tagModalVisible = ref(false); // 标签modal
const revokeList = ref<any>([]); // 撤销列表-记录用户撤销操作operate

const anoType = ref<string>("1"); // anoType 0图像分割 1物体检测
// -------------------------------------------------------------------

const navTo = (name: string) => {
  router.push({ name });
  router.push({
    name,
    query: {
      sonId: route.query.id,
      sign: "mapToImport",
    },
  });
};

const handleTagMoute = (sign: string, row: any) => {
  if (sign === "enter") {
    row.isHover = true;
  }
  if (sign === "leave") {
    row.isHover = false;
  }
};

// ------------------ailabel----------------------
//  挣得少 懒得抽象 想抽象自己弄去 谢谢
// 接手的那位哥们 好好干！
const main_map = ref<HTMLDivElement>(null);
const map = ref<HTMLDivElement>(null);
const GetWindowInfo = ref<any>({
  width: "",
  height: "",
});
const state = ref<any>({
  pointSign: "foreground_points",
  imgUrl: "",
  divId: "map", // main_map
  drawingStyle: {
    fillStyle: "#3CB371",
    strokeStyle: "#3CB371", // #3CB371
    fill: true, // 是否填充
    globalAlpha: 0.3,
    lineWidth: 2,
  },
  mode: "",
  itemName: "",
  editId: "", // 待填充图形id
  deleteIconId: "delete01",
  gMap: null, // AILabel实例
  gFirstFeatureLayer: null, // 矢量图层实例(矩形，多边形等矢量)
  allFeatures: [], // 所有features
  oldAllFeatures: [], // 所有features
  gFirstImageLayer: null, // 图层
  nHeight: "",
  nWidth: "",
  sHeight: "",
  sWidth: "",
  operateWidth: "",
  operateHeight: "",
  centerObj: {},
  gFirstTextLayer: null, // 文本
  gFirstMaskLayer: null, // 涂抹层
  zoom: 800,
  isLoadImgSuccess: true,
  foreground_points: [],
  background_points: [],
  isInvalid: false,
  isLocalAno: false, // 是否本地标注
});
const feaDefaultCon = ref<any>({
  id: "",
  name: "",
  color: "#3CB371",
  relatedDeleteMarkerId: "",
});

// 双击选中id checkId
const checkId = ref<any>("");
const operateId = ref<string>("");
const checkFeature = ref<any>({});
const activeFId = ref<any>("");

// 初始化实例
const initMap = () => {
  const gMap = new AILabel.Map(state.value.divId, {
    center: state.value.centerObj, // 为了让图片居中
    zoom: state.value.zoom, // 初始缩放级别
    mode: "RECT", // 绘制线段
    refreshDelayWhenZooming: true, // 缩放时是否允许刷新延时，性能更优
    zoomWhenDrawing: false, // 绘制时可滑轮缩放
    panWhenDrawing: true, // 绘制时可到边界外自动平移
    zoomWheelRatio: 2, // 控制滑轮缩放缩率[0, 10), 值越小，则缩放越快，反之越慢
    withHotKeys: false, // 关闭快捷键
  });
  state.value.gMap = gMap;
  if (route.query.anoType === 'audit' || route.query.anoType === 'result') {
    state.value.gMap.setMode("PAN");
  }
};
// 添加 text 文本图层，用于展示文本
const setGFirstTextLayer = () => {
  const gFirstTextLayer = new AILabel.Layer.Text(
    "first-layer-text", // id
    { name: "第一个文本图层" }, // props
    { zIndex: 12, opacity: 1 }, // style
  );
  state.value.gFirstTextLayer = gFirstTextLayer;
  state.value.gMap && state.value.gMap.addLayer(gFirstTextLayer);
};
// 添加矢量图层
const setGFirstFeatureLayer = (id = "first-layer-feature") => {
  // 添加矢量图层
  const gFirstFeatureLayer = new AILabel.Layer.Feature(
    id, // id
    { name: "第一个矢量图层" }, // props
    { zIndex: 10 }, // style
  );
  state.value.gFirstFeatureLayer = gFirstFeatureLayer;
  state.value.gMap && state.value.gMap.addLayer(gFirstFeatureLayer);
};
// 图片层添加
const setGFirstImageLayer = () => {
  // 图片层添加
  const gFirstImageLayer = new AILabel.Layer.Image(
    "first-layer-image", // id
    {
      src: state.value.imgUrl,
      width: state.value.nWidth,
      height: state.value.nHeight,
      crossOrigin: true, // 如果跨域图片，需要设置为true
      position: {
        // 左上角相对中心点偏移量
        x: 0,
        y: 0,
      },
      // 网格
      grid: {},
    }, // imageInfo
    { name: "第一个图片图层" }, // props
    { zIndex: 5 }, // style
  );
  // 添加到gMap对象
  state.value.gFirstImageLayer = gFirstImageLayer;
  state.value.gMap && state.value.gMap.addLayer(gFirstImageLayer);
  state.value.gMap &&
    state.value.gMap.centerAndZoom({
      center: state.value.centerObj,
      zoom: state.value.zoom,
    });
};

// 使用Image对象获取图片真实宽高
function getImageSizeByUrl(imgUrl: any) {
  return new Promise((resolve, reject) => {
    const containerWidth = main_map.value.offsetWidth;
    const containerHeight = main_map.value.offsetHeight;
    const image = new Image();
    image.crossOrigin = "Anonymous"; // 这行很重要，它允许跨域图片资源加载
    image.onload = () => {
      state.value.isLoadImgSuccess = true;
      let imgWidth = image.width;
      let imgHeight = image.height;
      const imgRatio = imgWidth / imgHeight;
      const containerRatio = containerWidth / containerHeight;
      if (imgRatio > containerRatio) {
        imgWidth = containerWidth;
        imgHeight = containerWidth / imgRatio;
      } else if (imgRatio < containerRatio) {
        imgHeight = containerHeight;
        imgWidth = containerHeight * imgRatio;
      } else {
        imgWidth = containerWidth;
        imgHeight = containerHeight;
      }
      resolve({
        width: imgWidth,
        height: imgHeight,
        centerObj: { x: imgWidth / 2, y: imgHeight / 2 },
        // centerObj: { x: 0, y: 0 },
      });
    };
    image.onerror = () => {
      state.value.isLoadImgSuccess = false;
      reject(new Error("error"));
    };
    image.src = imgUrl ?? state.value.imgUrl;
  });
}

function setImgSize(row: any) {
  const { width, height, centerObj } = row;
  state.value.nWidth = width;
  state.value.nHeight = height;
  // ------------newCode GetWindowInfo--------------
  GetWindowInfo.value.width = `${width}px`;
  GetWindowInfo.value.height = `${height}px`;
  // ---------------------------------
  if (!state.value.operateWidth) {
    state.value.operateWidth = width;
  }
  if (!state.value.operateHeight) {
    state.value.operateHeight = height;
  }
  state.value.centerObj = centerObj;
  // 重新设置map宽高
  // map.value.style.width = width + "px";
  // map.value.style.height = height + "px";
  return { nWidth: width, nHeight: height, centerObj };
}

// 获取所有features
const getFeatures = () => {
  // state.value.allFeatures = state.value.gFirstFeatureLayer?.getAllFeatures() ?? [];

  let featureList = state.value.gFirstFeatureLayer?.getAllFeatures() ?? [];
  featureList = featureList.map((item, index) => {
    item.isEye = item.isEye === 'undefined' ? true : item.isEye;
    item.operateIdx = index;
    return item;
  });
  state.value.allFeatures = featureList;
};
// 添加文本
const addLayerText = (
  textId: string | number,
  textName: any,
  textPosition: any,
) => {
  // 添加一个文本
  const gFirstText = new AILabel.Text(
    textId, // id
    { text: textName, position: textPosition, offset: { x: 0, y: 0 } },
    { name: textName }, // props
    // 文本显示 style
    {
      fillStyle: "#F4A460",
      strokeStyle: "#D2691E",
      background: true,
      globalAlpha: 1,
      fontColor: "white",
    },
  );
  state.value.gFirstTextLayer.addText(gFirstText);
};
// 添加图形
const addFeature = (data: any, type: string, id: any) => {
  const drawingStyle = state.value.drawingStyle;
  // 线
  if (type === "LINE") {
    const scale = state.value.gMap.getScale();
    const width = drawingStyle.lineWidth / scale;
    const lineFeature = new AILabel.Feature.Line(
      feaDefaultCon.value.id,
      { ...data, width },
      { name: feaDefaultCon.value.name, textId: id },
      drawingStyle,
    );
    state.value.gFirstFeatureLayer.addFeature(lineFeature);
    // 多边形 添加 文本
    const textPosition = setText(data, type);
    addLayerText(id, name, textPosition);
  }
  // 线段
  else if (type === "POLYLINE") {
    const scale = state.value.gMap.getScale();
    const width = drawingStyle.lineWidth / scale;
    const polylineFeature = new AILabel.Feature.Polyline(
      id, // id
      { points: data, width }, // shape
      { name, textId: id }, // props
      drawingStyle, // style
    );
    state.value.gFirstFeatureLayer.addFeature(polylineFeature);
    // 多边形 添加 文本
    const textPosition = setText(data, type);
    addLayerText(id, name, textPosition);
  }
  // 矩形
  else if (type === "RECT") {
    const drawingStyle = {
      fillStyle: feaDefaultCon.value.color,
      strokeStyle: feaDefaultCon.value.color, // #3CB371
      fill: true, // 是否填充
      globalAlpha: 0.3,
      lineWidth: 2,
    };
    const rectFeature = new AILabel.Feature.Rect(
      feaDefaultCon.value.id || id, // id
      data, // shape
      {
        name: feaDefaultCon.value.name,
        textId: feaDefaultCon.value.id,
        deleteMarkerId: feaDefaultCon.value.relatedDeleteMarkerId,
        checkId: nanoid(),
      }, // props
      drawingStyle, // style
    );
    state.value.gFirstFeatureLayer.addFeature(rectFeature);
    // 多边形 添加 文本
    const textPosition = setText(data, type);
    addLayerText(
      feaDefaultCon.value.id,
      feaDefaultCon.value.name,
      textPosition,
    );
  }
  // 多边形
  else if (type === "POLYGON") {
    const drawingStyle = {
      fillStyle: feaDefaultCon.value.color,
      strokeStyle: feaDefaultCon.value.color, // #3CB371
      fill: true, // 是否填充
      globalAlpha: 0.3,
      lineWidth: 2,
    };
    const polygonFeature = new AILabel.Feature.Polygon(
      feaDefaultCon.value.id || id, // id
      { points: data }, // shape
      {
        name: feaDefaultCon.value.name,
        textId: feaDefaultCon.value.id,
        deleteMarkerId: feaDefaultCon.value.relatedDeleteMarkerId,
        checkId: nanoid(),
      }, // props
      drawingStyle, // style
    );
    state.value.gFirstFeatureLayer.addFeature(polygonFeature);
    // 多边形 添加 文本
    const textPosition = setText(data, type);
    addLayerText(
      feaDefaultCon.value.id,
      feaDefaultCon.value.name,
      textPosition,
    );
  }
  // 圆
  else if (type === "CIRCLE") {
    const drawingStyle = {
      fillStyle: feaDefaultCon.value.color,
      strokeStyle: feaDefaultCon.value.color, // #3CB371
      fill: true, // 是否填充
      globalAlpha: 0.3,
      lineWidth: 2,
    };
    const gFirstFeatureCircle = new AILabel.Feature.Circle(
      feaDefaultCon.value.id || id, // id
      { cx: data.cx, cy: data.cy, r: data.r }, // shape
      {
        name: feaDefaultCon.value.id,
        textId: feaDefaultCon.value.id,
        deleteMarkerId: feaDefaultCon.value.relatedDeleteMarkerId,
        checkId: nanoid(),
      }, // props
      drawingStyle, // style
    );
    state.value.gFirstFeatureLayer.addFeature(gFirstFeatureCircle);
    // 多边形 添加 文本
    const textPosition = setText(data, type); // {x: 31.1640696608616, y: 48.89036653397688}
    addLayerText(
      feaDefaultCon.value.id,
      feaDefaultCon.value.name,
      textPosition,
    );
  }
  // 点
  else if (type === "POINT") {
    const randomId = `${new Date().getTime()}-${nanoid()}`;
    let color;
    if (state.value.pointSign === "background_points") {
      color = "#2468f2";
    }
    if (state.value.pointSign === "foreground_points") {
      color = "#FF8C00";
    }
    const gFirstFeaturePoint = new AILabel.Feature.Point(
      randomId, // id
      { x: data.x, y: data.y, r: 5 }, // shape
      { name, textId: id, sign: state.value.pointSign }, // props
      { fillStyle: color, zIndex: 5, lineWidth: 2 }, // style
    );
    state.value.gFirstFeatureLayer.addFeature(gFirstFeaturePoint);
    // addEvent();
  }
  // 注记
  else if (type === "MARKER") {
    const gFirstMarker = new AILabel.Marker(
      id, // id
      {
        src: "http://ailabel.com.cn/public/ailabel/demo/marker.png",
        position: {
          // marker坐标位置
          x: data.x,
          y: data.y,
        },
        offset: {
          x: -16,
          y: 32,
        },
      }, // markerInfo
      { name: "第一个marker注记", textId: id }, // props
    );
    state.value.gFirstFeatureLayer.addFeature(gFirstMarker);
    // 多边形 添加 文本
    const textPosition = setText(data, type);
    addLayerText(id, name, textPosition);
  }

  updateMarkInfo();
  getFeatures();
};

// newCode 重新渲染feature
const againAddFeature = (type: string, data: any, config: any) => {
  if (type === "RECT") {
    const { id, name, textId, deleteMarkerId, drawingStyle } = config;
    const rectFeature = new AILabel.Feature.Rect(
      id,
      data,
      {
        name,
        textId,
        deleteMarkerId,
      },
      drawingStyle,
    );
    rectFeature.operateIdx = config?.operateIdx;
    rectFeature.isEye = `${config.isEye}` === 'undefined' ? true : config.isEye;
    state.value.gFirstFeatureLayer.addFeature(rectFeature);
    // 多边形 添加 文本
    const textPosition = setText(data, type);
    addLayerText(id, name, textPosition);
  }
  if (type === "CIRCLE") {
    const { id, name, textId, deleteMarkerId, drawingStyle } = config;
    const gFirstFeatureCircle = new AILabel.Feature.Circle(
      id, // id
      { cx: data.cx, cy: data.cy, r: data.r }, // shape
      {
        name,
        textId,
        deleteMarkerId,
      }, // props
      drawingStyle, // style
    );
    gFirstFeatureCircle.operateIdx = config?.operateIdx;
    gFirstFeatureCircle.isEye = `${config.isEye}` === 'undefined' ? true : config.isEye;

    state.value.gFirstFeatureLayer.addFeature(gFirstFeatureCircle);
    // 多边形 添加 文本
    const textPosition = setText(data, type); // {x: 31.1640696608616, y: 48.89036653397688}
    addLayerText(id, name, textPosition);
  }
  if (type === "POLYGON") {
    const { id, name, textId, deleteMarkerId, drawingStyle } = config;
    const polygonFeature = new AILabel.Feature.Polygon(
      id, // id
      { points: data.points }, // shape
      {
        name,
        textId,
        deleteMarkerId,
      }, // props
      drawingStyle, // style
    );

    polygonFeature.operateIdx = config?.operateIdx;
    polygonFeature.isEye = `${config.isEye}` === 'undefined' ? true : config.isEye;

    state.value.gFirstFeatureLayer.addFeature(polygonFeature);
    // 多边形 添加 文本
    const textPosition = setText(data, type);
    addLayerText(id, name, textPosition);
  }
};

// 删除按钮
const deIcon = () => {
  state.value.gMap && state.value.gMap.markerLayer.removeAllMarkers();
};
// 设置文本坐标
const setText = (data: any, type: string) => {
  let textPosition = {};
  switch (type) {
    case "LINE":
      textPosition = { x: data?.start.x, y: data?.start.y };
      break;
    case "POLYLINE":
      textPosition = { x: data?.x, y: data?.y };
      break;
    case "RECT":
      textPosition = { x: data.x, y: data.y };
      break;
    case "POLYGON":
      const points = Array.isArray(data) ? data : data.points;
      textPosition = { x: points[0].x, y: points[0].y };
      break;
    case "POINT":
      textPosition = { x: data.x, y: data.y };
      break;
    case "MARKER":
      textPosition = { x: data.x, y: data.y };
      break;
    case "CIRCLE":
      textPosition = { x: data.cx, y: data.cy - data.r };
      break;
    default:
      break;
  }
  return textPosition;
};

// 半自动标注渲染点坐标
const renderPointData = async (sign: any, value: any[], id?: string | number) => {
  if (value.length == 0) return;
  visible.value = true;
  const wRadio = state.value.sWidth / state.value.nWidth;
  const hRadio = state.value.sHeight / state.value.nHeight;
  const mapValue = value.map((val) => {
    const x = Number((val[0] * wRadio).toFixed(2));
    const y = Number((val[1] * hRadio).toFixed(2));
    return [x, y];
  });
  let params;
  if (sign === "foreground_points") {
    const background_points = state.value.background_points.map((val) => {
      const x = Number((val[0] * wRadio).toFixed(2));
      const y = Number((val[1] * hRadio).toFixed(2));
      return [x, y];
    });
    params = {
      image_path: state.value.imgUrl,
      foreground_points: mapValue,
      background_points,
    };
  }
  if (sign === "background_points") {
    const foreground_points = state.value.foreground_points.map((val) => {
      const x = Number((val[0] * wRadio).toFixed(2));
      const y = Number((val[1] * hRadio).toFixed(2));
      return [x, y];
    });
    params = {
      image_path: state.value.imgUrl,
      foreground_points,
      background_points: mapValue,
    };
  }
  const res = await segmentStart(params);
  if (res.data) {
    visible.value = false;
    // 渲染前清空多边形图层
    const allFeatures = state.value.gFirstFeatureLayer.getAllFeatures();
    allFeatures.forEach((item, index) => {
      if (item.props.isAutoFit) {
        state.value.gFirstFeatureLayer.removeFeatureById(item.id);
      }
    });
    // 相当于一次图层 撤销一起撤销  随机id相同
    const revokeId = `${new Date().getTime()}-${nanoid()}`;
    mapPointData1(res.data.segmentResult, revokeId).forEach(
      async (item: any, index: any) => {
        const randomId = `${new Date().getTime()}-${nanoid()}`;
        // const randomId = "";
        await customAddFeature(item, randomId, revokeId);
      },
    );

    getFeatures();
    // 清空点信息
    state.value.allFeatures = state.value.allFeatures.filter((item) => {
      return item.type !== "POINT";
    });
    state.value.gFirstFeatureLayer.features =
      state.value.gFirstFeatureLayer.features.filter((item) => {
        return item.type !== "POINT";
      });
    state.value.gMap.refresh();
    // tagModalVisible.value = true;
    // 恢复点位标注
    state.value.gMap.setMode("POINT");
    state.value.drawingStyle = { fillStyle: "#2468f2" };
    // state.value.gMap && state.value.gMap.setDrawingStyle(drawingStyle);
  } else {
    if (id) {
      state.value.allFeatures = state.value.allFeatures.filter((item) => {
        return item.props.textId !== id;
      });
      state.value.gFirstFeatureLayer.features =
        state.value.gFirstFeatureLayer.features.filter((item) => {
          console.log('item: ', item);
          return item.props.textId !== id;
        });
      state.value.gMap.refresh();
    }
    visible.value = false;
  }
};

// 加工point坐标信息
const mapPointData1 = (
  { shapes, imageWidth, imageHeight },
  revokeId: string,
) => {
  let groupData = shapes.map((val) => {
    const id = ""; // 原逻辑 autoAno
    const props = {
      name: "",
      textId: id,
      deleteMarkerId: id,
      sign: state.value.pointSign,
      revokeId,
    };
    const style = {
      fillStyle: "#D91515", // #566eb4
      strokeStyle: "rgba(77, 101, 170)",
      fill: true, // 是否填充
      globalAlpha: 0.6,
      lineWidth: 0,
      stroke: false,
    };
    const wRadio = imageWidth / state.value.nWidth;
    const hRadio = imageHeight / state.value.nHeight;
    // 判断是否是多维数组
    const isTwoArr = val.points.every((val) => {
      return Array.isArray(val);
    });
    const type = isTwoArr ? val.shape_type.toUpperCase() : "POINT";
    const shape = {
      points: isTwoArr
        ? val.points.map((item1: any[]) => {
          return { x: item1[0] / wRadio, y: item1[1] / hRadio };
        })
        : [
          {
            x: val.points[0] / wRadio,
            y: val.points[1] / hRadio,
          },
        ],
    };
    const obj = {
      id,
      type,
      props,
      shape,
      style,
    };
    return obj;
  });
  groupData = groupData.filter((val) => {
    return val.type === "POLYGON" && val.shape.points.length >= 3;
  });
  return groupData;
};

// 增加事件
const addEvent = () => {
  const gMap = state.value.gMap;
  gMap.events.on("drawDone", (type: string, data: any) => {
    if (Boolean(state.value.isInvalid)) {
      window.$message?.warning?.("已被标注为无效数据，请先取消勾选后再标注！");
      return;
    }
    // randomId
    const randomId = `${new Date().getTime()}-${nanoid()}`;
    addFeature(data, type, randomId);
    if (type === "POINT") {
      state.value.operateWidth = state.value.nWidth;
      state.value.operateHeight = state.value.nHeight;
      // 判断前景点 后景点  pointSign  foreground_points  background_points
      if (state.value.pointSign === "foreground_points") {
        state.value.foreground_points = [
          ...state.value.foreground_points,
          // [x, y],
          [data.x, data.y],
        ];
        renderPointData("foreground_points", state.value.foreground_points, randomId);
      }
      if (state.value.pointSign === "background_points") {
        state.value.background_points = [
          ...state.value.background_points,
          // [x, y],
          [data.x, data.y],
        ];
        renderPointData("background_points", state.value.background_points, randomId);
      }
    } else {
      tagConfig.value.isShow = true;
      tagConfig.value.isEditTag = true;
      const endFeature = state.value.allFeatures[state.value.allFeatures.length - 1];
      // const index = state.value.allFeatures.findIndex(
      //   (item) => item.id === endFeature.id,
      // );
      // concatFeatureList
      const index = concatFeatureList.value.findIndex(
        (item) => item.id === endFeature.id,
      );
      tagCurrentIdx.value = index;
      state.value.editId = endFeature.id;
      operateId.value = endFeature.id;
      checkFeature.value = endFeature;
      gMap.setActiveFeature(endFeature);
    }
  });
  // 双击编辑 在绘制模式下双击feature触发选中
  gMap.events.on("featureSelected", (feature: any) => {
    const index = state.value.allFeatures.findIndex(
      (item) => item.id === feature.id,
    );
    tagCurrentIdx.value = index;
    state.value.editId = feature.id;
    operateId.value = feature.id;
    checkFeature.value = feature;
    gMap.setActiveFeature(feature);
    if (feature.type !== "POINT") {
      // 增加删除按钮
      // addDeleteIcon(feature, feature.shape);
      // const isTooltipModel = feature.id; // 弹框打标签
      const isTooltipModel = false;
      if (!isTooltipModel) {
        // tagModalVisible.value = true; // 不显示modal标签选中
        tagConfig.value.isShow = true;
        tagConfig.value.isEditTag = false;
        checkId.value = feature.props.checkId;
      }
    }
  });
  // 右键 目前只针对"点"双击选中右键触发
  gMap.events.on("featureDeleted", (feature: any) => {
    if (feature.type === "POINT") {
      // 根据id删除对应feature
      state.value.gFirstFeatureLayer.removeFeatureById(feature.id);
      // 删除对应text
      state.value.gFirstTextLayer.removeTextById(feature.props.textId);
    }
  });
  // 单机空白取消编辑
  gMap.events.on("featureUnselected", (feature) => {
    state.value.editId = "";
    deIcon();
    gMap.setActiveFeature(null);
    gMap.markerLayer.removeMarkerById(feature.props.deleteMarkerId);
  });
  // 更新完
  gMap.events.on("featureUpdated", (feature: any, shape: any) => {
    // 更新或者移动需要重新设置删除图标
    deIcon();
    feature.updateShape(shape);
    // 删除对应文本
    state.value.gFirstTextLayer.removeTextById(feature.props.textId);
    // 设置文本
    const textPosition = setText(shape, feature.type);
    addLayerText(feature.props.textId, feature.props.name, textPosition);
    if (feature.type !== "POINT") {
      // addDeleteIcon(feature, shape);
    }
    getFeatures();
  });
  // 删除
  gMap.events.on("featureDeleted", ({ id: featureId }: any) => {
    state.value.gFirstFeatureLayer.removeFeatureById(featureId);
  });
};
// 撤销
// const Revoke = () => {
//   if (revokeList.value.length > 0) {
//     console.log('revokeList.value: ', revokeList.value);
//     getFeatures();
//     const popRow = revokeList.value.pop();
//     customAddFeature(popRow);
//   } else {
//     getFeatures();
//     if (state.value.allFeatures.length == 0) return;
//     const feature = state.value.allFeatures.pop();
//     const popInfo = feature;
//     const textId = popInfo.props.textId || popInfo.id;
//     state.value.gFirstTextLayer.removeTextById(textId); // 删除文本
//     if (popInfo.props && popInfo.props.revokeId) {
//       const comFeatures = state.value.allFeatures.filter((val: any) => {
//         return val.props.revokeId == popInfo.props.revokeId;
//       });
//       // 遍历comFeatures 根据props.textId删除gFirstTextLayer
//       comFeatures.forEach((val: any) => {
//         state.value.gFirstTextLayer.removeTextById(val.props.textId);
//       });
//       // 过滤 allFeatures  删除revokeId 相同的feature
//       state.value.allFeatures = state.value.allFeatures.filter((val: any) => {
//         return val.props.revokeId !== popInfo.props.revokeId;
//       });
//     }
//     if (popInfo.type === "POINT") {
//       // filter 过滤前景背景点坐标
//       const filterArr = [popInfo.shape.x, popInfo.shape.y];
//       if (popInfo.props.sign === "foreground_points") {
//         state.value.foreground_points = state.value.foreground_points.filter(
//           (val: any) => {
//             return val[0] != filterArr[0] && val[1] != filterArr[1];
//           },
//         );
//       }
//       if (popInfo.props.sign === "background_points") {
//         state.value.background_points = state.value.background_points.filter(
//           (val: any) => {
//             return val[0] != filterArr[0] && val[1] != filterArr[1];
//           },
//         );
//       }
//     }
//     state.value.gMap.refresh(); // 刷新map
//   }
// };

// ---------------------Revoke newCode------------------------
/**
 * 撤销操作：移除最近一次添加的地图要素
 */
const Revoke = async () => {
  getFeatures();

  // 如果有 revokeList，优先 pop 并重新添加回去
  if (revokeList.value.length > 0) {
    const popRow = revokeList.value.pop();
    customAddFeature(popRow);
    return;
  }

  // 获取 allFeatures
  const features = state.value.allFeatures;

  // 如果没有要素，直接返回
  if (features.length === 0) return;

  // 获取最后一个 feature
  const feature = features[features.length - 1];
  const { props, type, shape } = feature;
  const textId = props?.textId || feature.id;

  // 移除文本和图层元素
  removeTextAndFeatureByIds(textId, feature.id, props);

  // POINT 类型特殊处理
  handlePointType(type, shape, props);

  // 更新 allFeatures 数组
  updateAllFeatures(features, props);

  // 触发地图刷新
  requestAnimationFrame(() => {
    state.value.gMap.refresh();
  });
};

// -------------------------- 辅助函数 --------------------------

/**
 * 根据 ID 删除文本和图层中的对应要素
 * @param {string|number} textId 文本 ID
 * @param {string|number} featureId 要素 ID
 * @param {Object} props 属性对象
 */
function removeTextAndFeatureByIds(textId, featureId, props) {
  if (!textId) return;

  const { gFirstTextLayer, gFirstFeatureLayer } = state.value;

  // 删除文本层和要素层
  gFirstTextLayer.removeTextById(textId);
  gFirstFeatureLayer.removeFeatureById(featureId);

  // 如果有 revokeId，删除关联的文本
  if (props?.revokeId) {
    const comFeatures = state.value.allFeatures.filter(
      val => val.props?.revokeId === props.revokeId
    );

    comFeatures.forEach(val => {
      if (val.props?.textId) {
        gFirstTextLayer.removeTextById(val.props.textId);
      }
    });
  }
}

/**
 * 处理 POINT 类型的坐标点过滤逻辑
 * @param {string} type 元素类型
 * @param {Object} shape 坐标数据
 * @param {Object} props 属性对象
 */
function handlePointType(type, shape, props) {
  if (type === "POINT" && shape?.x !== undefined && shape?.y !== undefined) {
    const filterPoint = [shape.x, shape.y];

    const pointFilters = {
      foreground_points: (val) =>
        val[0] !== filterPoint[0] || val[1] !== filterPoint[1],
      background_points: (val) =>
        val[0] !== filterPoint[0] || val[1] !== filterPoint[1]
    };

    if (props?.sign && pointFilters[props.sign]) {
      state.value[props.sign] = state.value[props.sign].filter(pointFilters[props.sign]);
    }
  }
}

/**
 * 更新 allFeatures 数组
 * @param {Array} features 当前所有要素
 * @param {Object} props 属性对象
 */
function updateAllFeatures(features, props) {
  if (props?.revokeId) {
    // 删除所有具有相同 revokeId 的要素
    state.value.allFeatures = features.filter(
      val => val.props?.revokeId !== props.revokeId
    );
  } else {
    // 否则只删除最后一个要素
    state.value.allFeatures = features.slice(0, -1);
  }
}
// ---------------------Revoke newCode------------------------

const zoomIn = () => {
  state.value.gMap.zoomIn();
  // isShowImg.value = false;
};
const zoomOut = () => {
  state.value.gMap.zoomOut();
  // isShowImg.value = false
};
// 图片复位
const restoration = () => {
  state.value.gMap.centerAndZoom({
    center: state.value.centerObj,
    zoom: state.value.zoom,
  });
};
const isCheckTag = computed(() => {
  return tagConfig.value.tagList.filter((item) => item.isCheck).length > 0;
});
// eslint-disable-next-line complexity
const setDrawingStyle = (mode: any) => {
  const drawingStyle = {};
  switch (mode) {
    case "Center":
      restoration();
      break;
    case "zoomIn":
      zoomIn();
      break;
    case "zoomOut":
      zoomOut();
      break;
    case "delete": {
      break;
    }
    // 平移
    case "PAN": {
      break;
    }
    // 点
    case "POINT": {
      let color;
      if (state.value.pointSign === "background_points") {
        color = "#2468f2";
      }
      if (state.value.pointSign === "foreground_points") {
        color = "#FF8C00";
      }

      state.value.drawingStyle = { fillStyle: color };
      state.value.gMap.setDrawingStyle(drawingStyle);
      // addEvent();
      break;
    }
    // 圆
    case "CIRCLE": {
      // if (!isCheckTag) {
      //   window.$message?.error?.("请先勾选右侧标签！");
      //   return;
      // }
      // const checkTagList = tagConfig.value.tagList.filter(
      //   (item) => item.isCheck,
      // );
      // const { color } = checkTagList[0];
      // state.value.drawingStyle = {
      //   fillStyle: color,
      //   strokeStyle: color, // #3CB371
      //   fill: true, // 是否填充
      //   globalAlpha: 0.3,
      //   lineWidth: 2,
      // };
      // state.value.gMap.setDrawingStyle(drawingStyle);
      break;
    }
    // 线段
    case "LINE": {
      state.value.drawingStyle = {
        strokeStyle: "#BA55D3",
        lineJoin: "round",
        lineCap: "round",
        lineWidth: 10,
        arrow: false,
      };
      state.value.gMap.setDrawingStyle(drawingStyle);
      break;
    }
    // 多线段
    case "POLYLINE": {
      // if (!isCheckTag) {
      //   window.$message?.error?.("请先勾选右侧标签！");
      //   return;
      // }
      state.value.drawingStyle = {
        strokeStyle: "#FF1493",
        lineJoin: "round",
        lineCap: "round",
        lineWidth: 10,
      };
      state.value.gMap.setDrawingStyle(drawingStyle);
      break;
    }
    // 矩形
    case "RECT": {
      // if (!isCheckTag) {
      //   window.$message?.error?.("请先勾选右侧标签！");
      // }
      // const checkTagList = tagConfig.value.tagList.filter(
      //   (item) => item.isCheck,
      // );
      // const { color } = checkTagList[0];
      // const drawingStyle = {
      //   fillStyle: color,
      //   strokeStyle: color, // #3CB371
      //   fill: true, // 是否填充
      //   globalAlpha: 0.3,
      //   lineWidth: 2,
      // };
      // state.value.drawingStyle = drawingStyle;
      // state.value.gMap.setDrawingStyle(drawingStyle);
      break;
    }
    // 多边形
    // eslint-disable-next-line no-fallthrough
    case "POLYGON": {
      // if (!isCheckTag) {
      //   window.$message?.error?.("请先勾选右侧标签！");
      //   return;
      // }
      // const checkTagList = tagConfig.value.tagList.filter(
      //   (item) => item.isCheck,
      // );
      // const { color } = checkTagList[0];
      // const drawingStyle = {
      //   fillStyle: color,
      //   strokeStyle: color, // #3CB371
      //   fill: true, // 是否填充
      //   globalAlpha: 0.3,
      //   lineWidth: 2,
      // };
      // state.value.drawingStyle = drawingStyle;
      // state.value.gMap.setDrawingStyle(drawingStyle);
      break;
    }
    // 涂抹
    case "DRAWMASK": {
      state.value.drawingStyle = {
        strokeStyle: "rgba(255, 0, 0, .5)",
        fillStyle: "#00f",
        lineWidth: 50,
      };
      state.value.gMap.setDrawingStyle(drawingStyle);
      break;
    }
    // 擦除
    case "CLEARMASK": {
      state.value.drawingStyle = { fillStyle: "#00f", lineWidth: 30 };
      state.value.gMap.setDrawingStyle(drawingStyle);
      break;
    }
    // 撤销
    case "Revoke": {
      Revoke();
      break;
    }
    default:
      break;
  }
};
const setMode = (mode: string, sign: any, row: any) => {
  // 每次点击前重置iconList isOperate=false
  iconList.map((val) => {
    val.isOperate = false;
    return val;
  });
  // iconList = toRefs(icons);
  row.isOperate = true;
  state.value.mode = mode;
  state.value.isLocalAno = true;
  if (mode === "CIRCLE" || mode === "RECT" || mode === "POLYGON") {
    // eslint-disable-next-line @typescript-eslint/no-shadow
    // const isCheckTag =
    //   tagConfig.value.tagList.filter((item) => item.isCheck).length > 0;
    // if (!isCheckTag) {
    //   window.$message?.error?.("请先勾选右侧标签！");
    //   return;
    // }
  }
  if (mode === "POINT") {
    // 0926 new-重新点击前景点 背景点 清空point坐标  foreground_points || background_points
    if (sign === "foreground_points") {
      state.value.foreground_points = [];
      state.value.background_points = [];
    }
    if (sign === "background_points") {
      state.value.background_points = [];
    }
    state.value.pointSign = sign;
  }
  if (mode === "import") {
    navTo("data-manage_import");
  }
  state.value.gMap.setMode(mode);
  setDrawingStyle(mode);
  updateMarkInfo();
};
const cleanMapLayer = () => {
  deIcon();
  state.value.gMap && state.value.gMap.removeLayerById("first-layer-image");
  state.value.gFirstTextLayer && state.value.gFirstTextLayer.removeAllTexts();
  state.value.gFirstFeatureLayer &&
    state.value.gFirstFeatureLayer.removeAllFeatures();
  state.value.gFirstImageLayer = null;
  state.value.gFirstFeatureLayer = null;
  state.value.gFirstTextLayer = null;
  state.value.allFeatures = [];
  // status 状态恢复默认值
  state.value.imgUrl = "";
  state.value.gMap && state.value.gMap.setActiveFeature(null);
  imgActiveIndex.value = 0;
  state.value.gMap && state.value.gMap.refresh();
};
// render layer 渲染图层
const customAddFeature = (
  { id, type, props, shape, style, isEye, operateIdx }: any,
  randomId: any,
  revokeId: any,
) => {
  // state.value.gMap && state.value.gMap.setMode(type);
  const drawingStyle = style;
  const { textId, deleteMarkerId, operateWidth, operateHeight } = props;
  let name;
  const isTagId = tagConfig.value.tagList.some(
    (val) => val.labelId == id.split("-")[0],
  );
  if (id) {
    name =
      id === "autoAno"
        ? ""
        : isTagId
          ? tagConfig.value.tagList.find(
            (val) => val.labelId == id.split("-")[0],
          ).labelName
          : "";
  } else {
    name = id === "audoAno" ? "" : "";
  }
  name = props.name
    ? props.name
    : isTagId
      ? tagConfig.value.tagList.find((val) => val.labelId == id.split("-")[0])
        .labelName
      : "";
  let data = shape; // shape
  let textPosition = setText(data, type);
  let deviceWRadio: any;
  let deviceHRadio: any;
  const isMapAno = true; // 是否是模型加强后的数据 不是人工处理过的
  if (isMapAno) {
    if (operateWidth) {
      deviceWRadio = state.value.nWidth / operateWidth;
      deviceHRadio = state.value.nHeight / operateHeight;
    } else {
      deviceWRadio = state.value.operateWidth
        ? state.value.nWidth / state.value.operateWidth
        : 1;
      deviceHRadio = state.value.operateHeight
        ? state.value.nHeight / state.value.operateHeight
        : 1;
    }
  } else {
    deviceWRadio = state.value.nWidth / state.value.operateWidth;
    deviceHRadio = state.value.nHeight / state.value.operateHeight;
  }
  // -----------------------------
  switch (type) {
    case "RECT":
      if (isMapAno) {
        data = {
          x: shape.x * deviceWRadio,
          y: shape.y * deviceHRadio,
          width: shape.width * deviceWRadio,
          height: shape.height * deviceHRadio,
        };
      } else {
        const wRadio = state.value.sWidth / state.value.nWidth;
        const hRadio = state.value.sHeight / state.value.nHeight;
        data = {
          x: (shape.x / wRadio) * deviceWRadio,
          y: (shape.y / hRadio) * deviceHRadio,
          width: (shape.width / wRadio) * deviceWRadio,
          height: (shape.height / hRadio) * deviceHRadio,
        };
      }
      textPosition = setText(data, type);
      const rectFeature = new AILabel.Feature.Rect(
        id, // id
        data, // shape
        { name, textId, deleteMarkerId }, // props
        drawingStyle, // style
      );
      rectFeature.isEye = isEye;
      rectFeature.operateIdx = operateIdx;
      state.value.gFirstFeatureLayer.addFeature(rectFeature);
      addLayerText(id, name, textPosition);
      break;
    case "CIRCLE":
      if (isMapAno) {
        textPosition = setText(
          {
            cx: data.cx * deviceWRadio,
            cy: data.cy * deviceHRadio,
            r: data.r * deviceWRadio,
          },
          type,
        );
        const gFirstFeatureCircle = new AILabel.Feature.Circle(
          id, // id
          {
            cx: data.cx * deviceWRadio,
            cy: data.cy * deviceHRadio,
            r: data.r * deviceWRadio,
          }, // shape radio
          { name, textId: id, deleteMarkerId }, // props
          drawingStyle,
        );
        gFirstFeatureCircle.isEye = isEye;
        gFirstFeatureCircle.operateIdx = operateIdx;
        state.value.gFirstFeatureLayer.addFeature(gFirstFeatureCircle);
        addLayerText(id, name, textPosition);
      } else {
        const wRadio = state.value.sWidth / state.value.nWidth;
        const hRadio = state.value.sHeight / state.value.nHeight;
        textPosition = setText(
          {
            cx: (data.cx / wRadio) * deviceWRadio,
            cy: (data.cy / hRadio) * deviceHRadio,
            r: (data.r / wRadio) * deviceWRadio,
          },
          type,
        );
        const gFirstFeatureCircle = new AILabel.Feature.Circle(
          id, // id
          {
            cx: (data.cx / wRadio) * deviceWRadio,
            cy: (data.cy / hRadio) * deviceHRadio,
            r: (data.r / wRadio) * deviceWRadio,
          }, // shape radio
          { name, textId: id, deleteMarkerId }, // props
          drawingStyle,
        );
        gFirstFeatureCircle.isEye = isEye;
        gFirstFeatureCircle.operateIdx = operateIdx;
        state.value.gFirstFeatureLayer.addFeature(gFirstFeatureCircle);
        addLayerText(id, name, textPosition);
      }
      break;
    case "POLYGON":
      if (isMapAno) {
        // deviceRadio
        const points = data.points.map((val: any) => {
          return { x: val.x * deviceWRadio, y: val.y * deviceHRadio };
        });
        textPosition = setText({ points }, type);
        const polygonId = randomId || id;
        const polygonFeature = new AILabel.Feature.Polygon(
          polygonId, // id
          { points }, // shape radio
          {
            name,
            textId: id,
            deleteMarkerId,
            isAutoFit: Boolean(randomId),
            revokeId,
          }, // props
          drawingStyle, // style
        );
        polygonFeature.isEye = isEye;
        polygonFeature.operateIdx = operateIdx;
        state.value.gFirstFeatureLayer.removeFeatureById(polygonId);
        state.value.gFirstFeatureLayer &&
          state.value.gFirstFeatureLayer.addFeature(polygonFeature);
        addLayerText(polygonId, name, textPosition);
      } else {
        const wRadio = state.value.sWidth / state.value.nWidth;
        const hRadio = state.value.sHeight / state.value.nHeight;
        const points = data.points.map((val: any) => {
          return {
            x: (val.x / wRadio) * deviceWRadio,
            y: (val.y / hRadio) * deviceHRadio,
          };
        });
        textPosition = setText({ points }, type);
        const polygonId = randomId || id;
        const polygonFeature = new AILabel.Feature.Polygon(
          polygonId, // id
          { points }, // shape radio
          { name, textId: id, deleteMarkerId, isAutoFit: Boolean(randomId) }, // props
          drawingStyle, // style
        );
        polygonFeature.isEye = isEye;
        polygonFeature.operateIdx = operateIdx;
        state.value.gFirstFeatureLayer.removeFeatureById(polygonId);
        state.value.gFirstFeatureLayer &&
          state.value.gFirstFeatureLayer.addFeature(polygonFeature);
        addLayerText(polygonId, name, textPosition);
      }
      break;
    case "POINT":
      if (isMapAno) {
        let color;
        if (state.value.pointSign === "background_points") {
          color = "#2468f2";
        }
        if (state.value.pointSign === "foreground_points") {
          color = "#FF8C00";
        }
        const gFirstFeaturePoint = new AILabel.Feature.Point(
          id,
          { name, textId: id }, // props
          {
            x: data.x * deviceWRadio,
            y: data.y * deviceHRadio,
            r: 5 * deviceWRadio,
          }, // shape radio
          { fillStyle: color, zIndex: 5, lineWidth: 2 * deviceWRadio }, // style radio
        );
        state.value.gFirstFeatureLayer.addFeature(gFirstFeaturePoint);
      } else {
        const wRadio = state.value.sWidth / state.value.nWidth;
        const hRadio = state.value.sHeight / state.value.nHeight;
        let color;
        if (state.value.pointSign === "background_points") {
          color = "#2468f2";
        }
        if (state.value.pointSign === "foreground_points") {
          color = "#FF8C00";
        }
        const gFirstFeaturePoint = new AILabel.Feature.Point(
          id,
          { name, textId: id }, // props
          {
            x: (data.x / wRadio) * deviceWRadio,
            y: (data.y / hRadio) * deviceHRadio,
            r: (5 / wRadio) * deviceWRadio,
          }, // shape radio
          {
            fillStyle: color,
            zIndex: 5,
            lineWidth: (2 / wRadio) * deviceWRadio,
          }, // style radio
        );
        state.value.gFirstFeatureLayer.addFeature(gFirstFeaturePoint);
        // addEvent();
      }

      break;
    default:
      break;
  }
};

const renderAllLayer = (markInfo: any) => {
  console.log('render');
  let parsedMarkInfo;
  try {
    parsedMarkInfo = JSON.parse(markInfo);
  } catch (error) {
    parsedMarkInfo = null;
  }

  let layerList = markInfo && Array.isArray(parsedMarkInfo) && parsedMarkInfo.length > 0
    ? parsedMarkInfo
    : [];
  layerList = layerList.map((item, index) => {
    item.isEye = item?.isEye ?? true;
    return item;
  });


  hiddenEyeList.value = layerList.filter(val => {
    return !val.isEye;
  });

  layerList.forEach((item: any) => {
    customAddFeature(item);
  });
  getFeatures();
  layerList.forEach((item: any) => {
    if (!item.isEye) {
      state.value.gFirstTextLayer.removeTextById(item.id);
      state.value.gFirstFeatureLayer.removeFeatureById(item.id);
    }
  });
  getFeatures();
};

// 修改图层标签名
const changeLayerText = (row: any) => {
  state.value.gFirstFeatureLayer.getAllFeatures().forEach((item) => {
    const customId = item.id.split("-")[0];
    if (customId == row.labelId) {
      const curText = state.value.gFirstTextLayer.getTextById(item.id);
      state.value.gFirstTextLayer.removeTextById(item.id);
      addLayerText(item.id, row.name, curText.textInfo.position);
      state.value.gMap && state.value.gMap.refresh();
    }
  });
};
// ------------------end-------------------

// ---------------newCode---------------
const tagConfig = ref<any>({
  activeIdx: -1,
  isEditTag: false,
  sign: "group",
  params: {
    val: undefined,
    englishVal: undefined,
    color: "#000000",
  },
  options: [],
  isShow: false,
  tagList: [],
  deepTagList: [],
});
const imgParams = reactive<any>({
  page: 1,
  limit: 8,
  sonId: undefined,
  state: undefined,
  modelPage: undefined
});
const selectRef = ref(null);
const imgActiveSrc = ref<string>("");
const imgActiveIndex = ref<number>(0);
const imgList: any[] = ref([]);
const imgData = ref<any>();
const tabConfig = ref<any>({
  state: '0',
  tabNum: {
    all: undefined,
    haveAno: undefined,
    noAno: undefined,
    invalid: undefined,
  },
});
const isSwiperChange = ref(false);
const visible = ref(false); // 遮罩mask
const isOperate = ref(false); // 自动保存中
const isRender = ref(false); // 图像渲染中

const targetNodeRef = ref(null); // 已打标签节点

// watch
watch(
  () => tabConfig.value.state,
  (newVal) => {
    imgParams.page = 1;
    imgActiveIndex.value = "0";
    getImgData();
  },
);
watch(
  () => imgList.value,
  (arr) => {
    if (arr.length === 0) {
      cleanMapLayer();
    } else {
      // detail to operation 指定图片跳转
      if (route.query.imgIdx >= 0 && !isSwiperChange.value) {
        // const curPageImgIdx = Number(route.query.imgIdx) % 7;
        const curPageImgIdx = Number(route.query.imgIdx) % 8;
        const curImgData = imgList.value[curPageImgIdx];
        handleImgChange(curImgData, "mounted");
      } else {
        // 重新挂载
        const imgInfo = imgList.value[imgActiveIndex.value];
        handleImgChange(imgInfo, "mounted");
      }
    }
  },
);

const imgTotal = ref(0); // 共多少张
const imgPages = ref(0);

// 标签列表index
const tagCurrentIdx = ref(null);

// add tag
const handleAddTag = (sign = "group") => {
  if (sign === 'tag') {
    tagConfig.value.params.val = undefined;
    tagConfig.value.params.englishVal = undefined;
  }
  tagConfig.value.isEditTag = true;
  tagConfig.value.sign = sign;
};

const handleCancel = () => {
  tagConfig.value.isEditTag = false;
};

const getSelectData = async () => {
  const params = {
    sonId: route.query.id,
  };
  const res = await getSelectGroupLabel(params);
  if (res.data) {
    tagConfig.value.options = res.data.map((item) => {
      return {
        label: item.label,
        value: item.id,
        count: item.count,
      };
    });
  }
};

const navToTagGroup = () => {
  router.push({
    // name: "data-ano_group",
    name: "dataset_taggroupmanager",
  });
};

const handleSelectChange = () => {
  tagConfig.value.isShow = true;
};

const handleSelectFocus = () => {
  tagConfig.value.params.val = null;
  tagConfig.value.isShow = true;
};

const handleDefine = async () => {
  if (tagConfig.value.sign === "group") {
    const params = {
      sonId: route.query.id,
      labelGroupId: tagConfig.value.params.val,
    };
    const res = await addDataSetAndLabel(params);
    if (res.data >= 1) {
      window.$message?.success?.("标签组切换成功！");
      tagConfig.value.isShow = false;
      tagConfig.value.params.val = null;
      // eslint-disable-next-line @typescript-eslint/no-use-before-define
      await getTagList();
      // usePage
      run({
        page: 1,
        limit: 10,
        sonId: route.query?.id,
      });
    }
  }
  if (tagConfig.value.sign === "tag") {
    if (!tagConfig.value.params.englishVal) {
      window.$message?.error?.('标签组英文名必填');
      return;
    }
    const regex = /^[a-zA-Z0-9_\-\/]+$/;
    if (tagConfig.value.params.englishVal && !regex.test(tagConfig.value.params.englishVal)) {
      window.$message?.error?.('请输入符合格式要求的英文名，仅允许包含字母、数字、下划线、连字符和斜杠。');
      return;
    }
    const params = {
      sonId: route.query.id,
      labelColor: tagConfig.value.params.color,
      labelName: tagConfig.value.params.val,
      englishLabelName: tagConfig.value.params.englishVal,
    };
    const res = await addSaveLabel(params);
    if (res.data >= 1) {
      window.$message?.success?.("添加标签成功！");
      tagConfig.value.isEditTag = false;
      await getTagList();
      // usePage
      run({
        page: 1,
        limit: 10,
        sonId: route.query?.id,
      });
    }
  }
};


const tagSearchVal = ref<string | undefined>(undefined);
const getTagList = async () => {
  const res = await selectDataSetLabel({ sonId: route.query.id, labelName: tagSearchVal.value });
  const dataList = res.data.map((item, index) => {
    return {
      name: item.labelName,
      color: item.labelColor,
      isOperate: true,
      isCheck: false,
      tagIdx: index,
      isHover: false,
      count: item.labelCount,
      labelId: item.labelId,
      ...item,
    };
  });
  tagConfig.value.tagList = dataList;
  tagConfig.value.deepTagList = dataList;
};

const handleInput = (val) => {
  const dataList = tagConfig.value.deepTagList.filter((item, index) => {
    return item.name.includes(val);
  });
  tagConfig.value.tagList = dataList;
};

const handleTagOperate = async (sign: string, row: any) => {
  if (sign === "delete") {
    deleteDataSetLabel({
      sonId: route.query.id,
      labelId: row.labelId,
    }).then(async (res) => {
      if (res.data >= 1) {
        window.$message?.success?.("删除成功！");
        await getTagList();
      }
    });
  }
  if (sign === "edit") {
    row.isOperate = !row.isOperate;
    // fetchLabelEdit
  }
  if (sign === "confirm") {
    const { labelId, color, labelGroupId, name } = row;
    if (!row.englishLabelName) {
      window.$message?.error?.('标签组英文名必填');
      return;
    }
    const regex = /^[a-zA-Z0-9_\-\/]+$/;
    if (row.englishLabelName && !regex.test(row.englishLabelName)) {
      window.$message?.error?.('请输入符合格式要求的英文名，仅允许包含字母、数字、下划线、连字符和斜杠。');
      return;
    }
    /* ------------------------------------ */
    const res = await fetchLabelEdit({
      id: labelId,
      labelColor: color,
      labelGroupId,
      labelName: row?.twoLabelName ?? name,
      englishLabelName: row.englishLabelName,
      sonId: route.query?.id
    });
    if (res.data >= 1) {
      window.$message?.success("修改标签成功！");
      changeLayerText(row);
      await getTagList();
      row.isOperate = !row.isOperate;
    }
  }
  if (sign === "cancel") {
    row.isOperate = !row.isOperate;
  }
};

const handleBack = () => {
  router.back();
};

const handleTagActive = (index) => {
  tagConfig.value.activeIdx = index;
  tagConfig.value.tagList = tagConfig.value.tagList.map((item) => {
    item.isCheck = false;
    return item;
  });
  tagConfig.value.tagList[index].isCheck = true;
};


const getImgData = async () => {
  const { imgIdx } = route.query;
  try {
    // 计算 imgParams.page
    if (route.query.imgIdx >= 0 && !isSwiperChange.value) {
      imgParams.page = imgIdx < 8 ? 1 : Math.ceil(imgIdx / 8);
    }

    // 构建请求参数
    const params = {
      ...imgParams,
      sonId: route.query.id,
      state: tabConfig.value.state,
      markUserId: route.query?.markUserId,
      taskId: route.query?.taskId,
      sign: route.query?.sign
    };

    // 发送请求
    const res = await getDataDetailsNoMarkFilePath(params);

    // 处理响应数据
    if (res.data) {
      imgTotal.value = res.data.total;
      imgPages.value = res.data.pages;
      imgList.value = res.data.records.map((item, index) => {
        return {
          ...item,
          defaultInfo: item.markInfo,
          imgSrc: item.imgPath,
          acticeIdx: `${index}`,
          previewImgPath: item.previewImgPath,
        };
      });

      // 更新图片 URL
      updateImgUrls();

      // 确保活动索引在有效范围内
      ensureActiveIndexInRange();


      // ----------prev load---------
      // preloadImages();
    } else {
      resetImgData();
    }
  } catch (error) {
    console.error('获取图片数据时出错:', error);
    resetImgData();
  }
};

// 更新图片 URL
const updateImgUrls = () => {
  state.value.imgUrl = imgList.value[0]?.imgSrc;
  imgActiveSrc.value = imgList.value[0]?.imgSrc;
};

// 确保活动索引在有效范围内
const ensureActiveIndexInRange = () => {
  if (imgList.value.length <= imgActiveIndex.value) {
    imgActiveIndex.value = imgList.value.length - 1;
  }

  if (route.query.anoType === 'audit' && tabConfig.value.state == 0) {
    // imgActiveIndex.value = 0;
    imgActiveIndex.value = imgActiveIndex.value == 0 ? 0 : imgActiveIndex.value - 1;
  }
};

// 重置图片数据
const resetImgData = () => {
  imgTotal.value = 0;
  imgPages.value = 1;
  imgList.value = [];
  state.value.imgUrl = "";
  imgActiveSrc.value = "";
};

// const handleImgChange = _.debounce(async (item: any, sign: any) => {
//   if (!item) {
//     state.value.isInvalid = false;
//   }

//   getFeatures();
//   const unlabelExistInFeatures = hasEmptyNameFeature(state.value.allFeatures);
//   if (!Boolean(state.value.isInvalid) && unlabelExistInFeatures) {
//     window.$message?.warning?.("所有标注必须设置标签！");
//     return;
//   }
//   if (route.query.anoType !== 'audit') {
//     state.value.gMap?.setMode("RECT");
//     isSwiperChange.value = true;
//     revokeList.value = [];
//     if (!sign) {
//       await asyncPrevImgSave();
//     }
//   }
//   if (route.query.anoType === 'audit') {
//     state.value.gMap?.setMode("PAN");
//   }
//   // -----------------------------------------------------------
//   const isArrLen = Array.isArray(imgList.value) && imgList.value.length > 0;
//   if (visible.value) return;
//   cleanMapLayer();
//   if (!isArrLen) return;
//   state.value.foreground_points = [];
//   state.value.background_points = [];
//   state.value.imgUrl = item.imgSrc;
//   state.value.sWidth = item.width ? Number(item.width) : ""; // 原图宽
//   state.value.sHeight = item.height ? Number(item.height) : ""; // 原图高
//   state.value.operateWidth = item.operateWidth ? Number(item.operateWidth) : "";
//   state.value.operateHeight = item.operateHeight
//     ? Number(item.operateHeight)
//     : "";
//   imgActiveSrc.value = item.imgSrc;
//   imgActiveIndex.value = item.acticeIdx;
//   isRender.value = true;
//   imgData.value = item;
//   // 获取图片原宽高
//   const url = state.value.imgUrl;
//   try {
//     const row = await getImageSizeByUrl(url);
//     setImgSize(row);

//     const containerWidth = main_map.value.offsetWidth;
//     const containerHeight = main_map.value.offsetHeight;
//     worker.postMessage({ type: 'getImageSize', data: { imgUrl: url, containerWidth, containerHeight } });
//   } catch (e) { }
//   // 图片切换 后台保存
//   // isOperate.value = true
//   // if (state.value.isLoadImgSuccess) {
//   //   setGFirstImageLayer(); // 图片层添加
//   //   setGFirstFeatureLayer(); // 添加矢量图层
//   //   setGFirstTextLayer(); // 添加 text 文本图层，用于展示文本
//   //   await clearAno();
//   //   setTimeout(() => {
//   //     renderAllLayer(item.markInfo);
//   //     isRender.value = false;
//   //   }, 200);
//   // }

//   if (state.value.isLoadImgSuccess) {
//     try {
//       // 图片层添加
//       setGFirstImageLayer();
//       // 添加矢量图层
//       setGFirstFeatureLayer();
//       // 添加 text 文本图层，用于展示文本
//       setGFirstTextLayer();
//       // 清除注释
//       await clearAno();

//       // 使用requestAnimationFrame确保在浏览器重绘前执行渲染
//       await new Promise(resolve => requestAnimationFrame(() => resolve()));

//       // 渲染所有图层
//       renderAllLayer(item.markInfo);
//       // 更新渲染状态
//       isRender.value = false;
//     } catch (error) {
//       console.error('图层渲染失败:', error);
//     }
//   }
//   state.value.gMap && state.value.gMap.refresh();
//   state.value.isInvalid = !Boolean(item.isInvalid);
// }, 300);


// -------------new-------------------
let activeRequestId = null; // 用于跟踪当前活跃的请求
const curFileName = ref<string | null>(null);

/**
 * 从URL中提取文件名并与扩展名拼接（保留原始大小写）
 * @param url - 需要解析的完整URL字符串
 * @returns 文件名与扩展名拼接结果，格式为 "文件名.扩展名"，如果没有扩展名则仅返回文件名
 */
function getFileNameWithExtension(url: string): string {
  try {
    // 解析URL并获取路径部分
    const pathname = new URL(url).pathname;

    // 获取最后一个路径段（可能是文件名）
    const fileNameWithExt = pathname.split('/').pop() || '';

    // 处理可能的查询参数或哈希
    const cleanFileName = fileNameWithExt.split(/[?#]/)[0];

    return cleanFileName;
  } catch (error) {
    // 处理无效URL的情况
    console.error('Invalid URL:', error);

    // 回退方案：直接从原始字符串中提取
    const urlWithoutParams = url.split('?')[0].split('#')[0];
    const pathComponents = urlWithoutParams.split('/');
    return pathComponents[pathComponents.length - 1];
  }
}

const handleImgChange = _.debounce(async (item: any, sign: any) => {
  const fileName = getFileNameWithExtension(item.imgPath);
  // curFileName.value = fileName;
  curFileName.value = item.fileName;
  const requestId = Symbol(); // 为每个请求创建唯一标识
  activeRequestId = requestId; // 设置当前活跃请求

  try {
    // 验证输入
    if (!item) {
      state.value.isInvalid = false;
      return;
    }

    // 检查特征标签
    await validateFeatures(requestId);

    // 设置地图模式
    setMapMode(sign);

    // 保存上一张图片(如果需要)
    if (!sign && route.query.anoType !== 'audit') {
      await asyncPrevImgSave(requestId);
    }

    // 清理并准备新图片
    if (visible.value) return;
    cleanMapLayer();

    const isArrLen = Array.isArray(imgList.value) && imgList.value.length > 0;
    if (!isArrLen) return;

    // 设置图片属性
    setImageProperties(item);

    // 获取图片尺寸
    await loadImageSize(requestId);

    // 渲染图层
    if (state.value.isLoadImgSuccess) {
      await renderLayers(requestId, item.markInfo);
    }

    // 刷新地图并设置有效性
    if (isActiveRequest(requestId)) {
      state.value.gMap?.refresh();
      state.value.isInvalid = !Boolean(item.isInvalid);
    }

  } catch (error) {
    if (error.name !== 'CancelledError') {
      console.error('图片切换处理失败:', error);
      window.$message?.error?.("图片切换失败，请重试");
    } else {
      console.log('请求被取消:', error.message);
    }
  }
}, 300);

// 检查请求是否仍然活跃
function isActiveRequest(requestId) {
  return activeRequestId === requestId;
}

// 抛出取消错误
function throwIfCancelled(requestId) {
  if (!isActiveRequest(requestId)) {
    throw { name: 'CancelledError', message: '请求已被新请求取代' };
  }
}

// 验证特征标签
async function validateFeatures(requestId) {
  throwIfCancelled(requestId);
  getFeatures();
  const unlabelExistInFeatures = hasEmptyNameFeature(state.value.allFeatures);
  if (!Boolean(state.value.isInvalid) && unlabelExistInFeatures) {
    throw new Error("所有标注必须设置标签！");
  }
}

// 设置地图模式
function setMapMode(sign: any) {
  if (route.query.anoType !== 'audit' && route.query.anoType !== 'result') {
    state.value.gMap?.setMode("RECT");
    isSwiperChange.value = true;
    revokeList.value = [];
  } else {
    state.value.gMap?.setMode("PAN");
  }
}

// 设置图片属性
function setImageProperties(item: any) {
  state.value.foreground_points = [];
  state.value.background_points = [];
  state.value.imgUrl = item.imgSrc;
  state.value.sWidth = item.width ? Number(item.width) : "";
  state.value.sHeight = item.height ? Number(item.height) : "";
  state.value.operateWidth = item.operateWidth ? Number(item.operateWidth) : "";
  state.value.operateHeight = item.operateHeight ? Number(item.operateHeight) : "";
  imgActiveSrc.value = item.imgSrc;
  imgActiveIndex.value = item.acticeIdx;
  isRender.value = true;
  imgData.value = item;
}

// 加载图片尺寸
async function loadImageSize(requestId) {
  throwIfCancelled(requestId);
  const url = state.value.imgUrl;
  try {
    const row = await getImageSizeByUrl(url);
    throwIfCancelled(requestId);
    setImgSize(row);

    const containerWidth = main_map.value.offsetWidth;
    const containerHeight = main_map.value.offsetHeight;
    // worker.postMessage({ type: 'getImageSize', data: { imgUrl: url, containerWidth, containerHeight } });
  } catch (e) {
    throwIfCancelled(requestId);
    console.warn('获取图片尺寸失败:', e);
  }
}

// 渲染图层
async function renderLayers(requestId, markInfo: any) {
  throwIfCancelled(requestId);
  // 图片层添加
  setGFirstImageLayer();
  // 添加矢量图层
  setGFirstFeatureLayer();
  // 添加 text 文本图层，用于展示文本
  setGFirstTextLayer();
  // 清除注释
  await clearAno();

  // 使用requestAnimationFrame确保在浏览器重绘前执行渲染
  await new Promise(resolve => requestAnimationFrame(() => resolve()));

  throwIfCancelled(requestId);
  // 渲染所有图层
  renderAllLayer(markInfo);
  // 更新渲染状态
  isRender.value = false;
}
// --------------------------------

const handleSwiperChange = async (sign: any) => {
  isSwiperChange.value = true;
  if (sign === "prev") {
    if (imgList.value.length > 0 && imgParams.page == 1) {
      window.$message?.warning?.("当前为第一页！");
      return;
    }

    await asyncPrevImgSave();
    // --------------------------------------
    imgActiveIndex.value = "0";
    state.value.gMap.removeLayerById("first-layer-image");
    imgParams.page = imgParams.page == 1 ? 1 : Number(imgParams.page) - 1;
    imgList.value = [];
    getImgData();
  }
  if (sign === "next") {
    if (imgPages.value == imgParams.page) {
      window.$message?.warning?.("当前为最后一页！");
      return;
    }

    await asyncPrevImgSave();
    // --------------------------------------
    imgActiveIndex.value = "0";
    state.value.gMap.removeLayerById("first-layer-image");
    imgParams.page =
      imgList.value.length < 8 ? imgParams.page : imgParams.page + 1;
    imgList.value = [];
    getImgData();
  }
};

const exportImg = async (data) => {
  try {
    const filename = `${new Date().getTime()}.jpg`;
    const fileBlob = new Blob([data], { type: "image/jpg" });
    const formData = new FormData();
    const { sonId, version } = imgList.value[imgActiveIndex.value];
    formData.append("file", new File([fileBlob], filename, { type: "image/jpg" }));
    formData.append("sonId", sonId);
    formData.append("version", version);

    const uploadResponse = await MarkFileUpload(formData);
    if (!uploadResponse.data) return;

    // const markDataList = state.value.gFirstFeatureLayer.getAllFeatures().map((item, index) => ({
    //   openId: item.id ? item.id.split("-")[0] : "",
    //   id: item.id ?? `${new Date().getTime()}-${nanoid()}`,
    //   type: item.type,
    //   props: {
    //     ...item.props,
    //     operateWidth: state.value.nWidth,
    //     operateHeight: state.value.nHeight,
    //   },
    //   shape: item.shape,
    //   style: item.style,
    // }));

    const markDataList = concatFeatureList.value.map((item, index) => ({
      openId: item.id ? item.id.split("-")[0] : "",
      id: item.id ?? `${new Date().getTime()}-${nanoid()}`,
      type: item.type,
      props: {
        ...item.props,
        operateWidth: state.value.nWidth,
        operateHeight: state.value.nHeight,
      },
      shape: item.shape,
      style: item.style,
      isEye: `${item.isEye}` === 'undefined' ? true : item.isEye,
      operateIdx: item?.operateIdx ?? index
    }));

    const labels = [...new Set(markDataList.map(item => item.openId))].join(",");
    const imgData = imgList.value[imgActiveIndex.value];
    const params = {
      fileId: imgData.fileId,
      sonId: route.query.id,
      markFileId: uploadResponse.data,
      markInfo: Array.isArray(markDataList) && markDataList.length > 0 ? JSON.stringify(markDataList) : "",
      labels,
      operateWidth: state.value.nWidth,
      operateHeight: state.value.nHeight,
      markUserId: route.query?.markUserId,
      taskId: route.query?.taskId,
      isInvalid: Number(!state.value.isInvalid)
    };

    imgList.value[imgActiveIndex.value].markInfo = JSON.stringify(markDataList);
    imgList.value[imgActiveIndex.value].defaultInfo = JSON.stringify(markDataList);

    const saveResponse = await addDataMarkInfo(params);
    if (saveResponse.data >= 1) {
      window.$message?.success?.("保存当前标注成功！");
      revokeList.value = [];
      isInvalidUpdate.value = false;
      // let curImgIdx: number | string = imgActiveIndex.value;
      // let curImgData = imgList.value[curImgIdx];
      // await handleImgChange(curImgData);
      /* ------------------------------------- */
      if (imgActiveIndex.value == imgList.value.length - 1) {
        if (imgParams.page < imgPages.value) {
          imgParams.page += 1;
          imgActiveIndex.value = 0;
        } else {
          window.$message?.warning?.("已经是最后一个了");
        }
      } else {
        let increment = 1;
        if (tabConfig.value.state == 1 && Boolean(state.value.isInvalid) || tabConfig.value.state == 2 && !Boolean(state.value.isInvalid)) {
          increment = 0;
        } else if (tabConfig.value.state == 0) {
          increment = 1;
        } else {
          increment = 1;
        }
        if (tabConfig.value.state == 2 && !Boolean(state.value.isInvalid) && state.value.allFeatures.length === 0) {
          increment = 1;
        }
        if (tabConfig.value.state == 1 && !Boolean(state.value.isInvalid) && state.value.allFeatures.length === 0) {
          increment = 0;
        }
        imgActiveIndex.value = Number(imgActiveIndex.value) + increment;
      }
      await getImgData();
      await getDataDetailsCount();
      await nextTick();
      /* ------------------------------------- */
    }
  } catch (error) {
    console.error("Error exporting image:", error);
  }
};

const generateImg = async () => {
  const width = state.value.nWidth;
  const height = state.value.nHeight;
  const buffer = await state.value.gMap.exportLayersToImage(
    { x: 0, y: 0, width, height },
    { type: "blob", format: "image/png" },
  );
  await exportImg(buffer);
};

const markTooltip = () => {
  window.$message?.success?.("后台标注中...");
};

const getDataDetailsCount = async () => {
  const res = await DataDetailsCount({
    sonId: route.query.id,
    markUserId: route.query?.markUserId,
    taskId: route.query.anoType === 'result' ? undefined :route.query?.taskId,
  });
  if (res.data) {
    tabConfig.value.tabNum.all = res.data.all;
    tabConfig.value.tabNum.haveAno = res.data.haveAno;
    tabConfig.value.tabNum.noAno = res.data.noAno;
    tabConfig.value.tabNum.invalid = res.data?.invalid ?? 0;
  }
};

const setCurTabNum = (val: string | number) => {
  //
  const strVal = String(val); // 将 val 转换为字符串
  switch (strVal) {
    case "1":
      curTabNum.value = tabConfig.value.tabNum.haveAno;
      break;
    case "2":
      curTabNum.value = tabConfig.value.tabNum.noAno;
      break;
    case "0":
      curTabNum.value = tabConfig.value.tabNum.all;
      break;
    default:
      // 可以根据需要处理其他情况，这里不做处理
      break;
  }
};

const saveAno = async (sign: number | string) => {
  getFeatures();
  const unlabelExistInFeatures = hasEmptyNameFeature(state.value.allFeatures);
  if (!Boolean(state.value.isInvalid) && unlabelExistInFeatures) {
    window.$message?.warning?.("所有标注必须设置标签！");
    return;
  }

  // 无效数据不能保存
  // if (Boolean(state.value.isInvalid)) {
  //   window.$message?.warning?.("已被标注为无效数据，请先取消勾选后再标注！");
  //   return;
  // }



  if (sign === "0") {
    await handleInvalidMark();
  }
  if (sign === "1") {
    await generateImg();
    await getTagList();
  }
};

const clearAno = async () => {
  console.log('clearAno');
  // state.value.gMap && state.value.gMap.removeAllLayers();
  state.value.gFirstFeatureLayer &&
    state.value.gFirstFeatureLayer.removeAllFeatures();
  state.value.gFirstTextLayer && state.value.gFirstTextLayer.removeAllTexts();
  state.value.gMap && state.value.gMap.setActiveFeature(null);
  getFeatures();
};

const handleTabChange = async (val: number | any) => {
  tabConfig.value.state = val;
  curFileName.value = null;

  state.value.isInvalid = false;
  imgActiveIndex.value = "0";
  await getDataDetailsCount();
  await getImgData();
  await setCurTabNum(val);
};
const handleTabBefore = async () => {
  // const imgInfo = imgList.value[imgActiveIndex.value];
  // await handleImgChange(imgInfo);

  await asyncPrevImgSave();
  return true;
};

// invalid data
const handleInvalidMark = async () => {
  const curImgData = imgList.value[imgActiveIndex.value];
  const params = {
    fileId: curImgData.fileId,
    sonId: route.query.id,
    markInfo: "",
    operateWidth: state.value.nWidth,
    operateHeight: state.value.nHeight,
    markUserId: route.query?.markUserId,
    taskId: route.query?.taskId,
    isInvalid: Number(!state.value.isInvalid)
  };
  const res = await addDataMarkInfo(params);
  if (res.data >= 1) {
    window.$message?.success?.("保存当前标注成功！");
    isInvalidUpdate.value = false;
    // let curImgIdx: number | string = imgActiveIndex.value;
    // let curImgData = imgList.value[curImgIdx];
    // await handleImgChange(curImgData);
    /* ------------------------------------- */
    if (imgActiveIndex.value == imgList.value.length - 1) {
      if (imgParams.page < imgPages.value) {
        imgParams.page += 1;
        imgActiveIndex.value = 0;
      } else {
        window.$message?.warning?.("已经是最后一个了");
      }
    } else {
      // imgActiveIndex.value = Number(imgActiveIndex.value) + 1;
      let increment = 1;
      if (tabConfig.value.state == 1 && Boolean(state.value.isInvalid) || tabConfig.value.state == 2 && !Boolean(state.value.isInvalid)) {
        increment = 0;
      } else if (tabConfig.value.state == 0) {
        increment = 1;
      } else {
        increment = 1;
      }
      if (tabConfig.value.state == 2 && !Boolean(state.value.isInvalid) && state.value.allFeatures.length === 0) {
        increment = 1;
      }
      if (tabConfig.value.state == 1 && !Boolean(state.value.isInvalid) && state.value.allFeatures.length === 0) {
        increment = 0;
      }
      imgActiveIndex.value = Number(imgActiveIndex.value) + increment;
    }
    await getImgData();
    await getDataDetailsCount();
    await nextTick();
    /* ------------------------------------- */
  }
};

const handleModalCancel = (sign: string) => {
  if (sign === "tag") {
    tagModalVisible.value = false;
  }
  if (sign === "validate") {
    isShowValidate.value = false;
  }
  if (sign === "pass") {
    noPassVisable.value = false;
    handleApprove("2");
  }
  if (sign === "pass0") {
    noPassVisable.value = false;
    validateModel.value.message = undefined;
    handleApprove("2");
  }
};

const updateMarkInfo = () => {
  // update markInfo
  const markDataList = state.value.gFirstFeatureLayer
    .getAllFeatures()
    .map((item, index) => {
      const randomId = `${new Date().getTime()}-${nanoid()}`;
      return {
        openId: item.id ? item.id.split("-")[0] : "",
        id: item.id ?? randomId,
        type: item.type,
        // props: item.props,
        props: {
          ...item.props,
          operateWidth: state.value.nWidth,
          operateHeight: state.value.nHeight,
        },
        shape: item.shape,
        style: item.style,
      };
    });
  imgList.value[imgActiveIndex.value].markInfo =
    Array.isArray(markDataList) && markDataList.length > 0
      ? JSON.stringify(markDataList)
      : "";
};

const handleFeaOperate = (sign: any, row: any, index: any) => {
  if (route.query.anoType === 'result' || route.query.anoType === 'audit') return;
  if (sign === "delete") {
    // 记录删除操作
    revokeList.value = [...revokeList.value, row];
    if (row.id) {
      state.value.allFeatures = state.value.allFeatures.filter((item) => {
        return item.id !== row.id;
      });
      state.value.gFirstFeatureLayer.features =
        state.value.gFirstFeatureLayer.features.filter((item) => {
          return item.id !== row.id;
        });
      const textId = row.props?.textId.split("-")[-1];
      state.value.gFirstTextLayer.removeTextById(row.id);
      // new 2024.12.02
      const markDataList = state.value.gFirstFeatureLayer
        .getAllFeatures()
        .map((item, index) => {
          const randomId = `${new Date().getTime()}-${nanoid()}`;
          return {
            openId: item.id ? item.id.split("-")[0] : "",
            id: item.id ?? randomId,
            type: item.type,
            props: {
              ...item.props,
              operateWidth: state.value.nWidth,
              operateHeight: state.value.nHeight,
            },
            shape: item.shape,
            style: item.style,
          };
        });
      imgList.value[imgActiveIndex.value].markInfo =
        Array.isArray(markDataList) && markDataList.length > 0
          ? JSON.stringify(markDataList)
          : "";

      hiddenEyeList.value = hiddenEyeList.value.filter(
        (val) => val.id !== row.id,
      );
    } else {
      getFeatures();
      // 根据index删除state.value.allFeatures
      state.value.allFeatures.splice(index, 1);
      // state.value.gFirstFeatureLayer.features.splice(index, );
    }
    state.value.gMap.setActiveFeature(null);
    state.value.gMap.refresh();
    getFeatures();
  }
  if (sign === "eye") {
    const fRow = row; // 过滤feature object
    const isEye = !row.isEye;
    row.isEye = isEye;
    if (row.isEye) {
      hiddenEyeList.value = hiddenEyeList.value.filter(
        (val) => val.id !== row.id,
      );

      const { id, type, shape, props, style, operateIdx, isEye } = fRow;
      const config = {
        id,
        textId: props.textId,
        deleteMarkerId: props.deleteMarkerId,
        type,
        name: props.name,
        drawingStyle: style,
        operateIdx,
        isEye
      };
      againAddFeature(type, shape, config);
      state.value.gMap.refresh();
      getFeatures();
      state.value.allFeatures.splice(index, 0, state.value.allFeatures.pop());
    } else {
      if (row.id) {
        const eyeFeature = state.value.gFirstFeatureLayer.getFeatureById(row.id);
        hiddenEyeList.value = _.uniqBy([...hiddenEyeList.value, eyeFeature], 'id');

        state.value?.gFirstFeatureLayer?.removeFeatureById(row.id);
        state.value.gFirstTextLayer.removeTextById(row.id);
      } else {
        // 根据index删除state.value.allFeatures
        state.value.allFeatures.splice(index, 1);
      }
      state.value.gMap.setActiveFeature(null);
    }
  }
};

function adjustRatio(ratio) {
  const screenWidth = window.screen.width; // 获取屏幕宽度
  const maxWidth = 1920; // 假设我们的最大宽度是1920px
  const newRatio = (screenWidth / maxWidth) * ratio; // 根据屏幕宽度和最大宽度计算新的比例
  return newRatio;
}

// ---------------------newCode--------------------------
const isInvalidUpdate = ref<Boolean>(false);
// watch(() => state.value.isInvalid, (newVal, oldVal) => {
//   // 判断新旧值
//   isInvalidUpdate.value = newVal !== oldVal ? true: false;
// });

const handleInvalidCheck = (value) => {
  isInvalidUpdate.value = !isInvalidUpdate.value;
  state.value.isInvalid = value;
}

// 异步保存 切换图片用到
async function asyncPrevImgSave() {
  const isArrLen = Array.isArray(imgList.value) && imgList.value.length > 0;
  if (!isSwiperChange.value || !isArrLen) {
    return;
  }
  // 判断是否编辑过
  const prevImgData = imgList.value[imgActiveIndex.value];
  const markIds =
    prevImgData.markInfo &&
      Array.isArray(JSON.parse(prevImgData.markInfo)) &&
      JSON.parse(prevImgData.markInfo).length > 0
      ? JSON.parse(prevImgData.markInfo).map((item: any) => {
        return item.id;
      })
      : [];
  const defaultIds =
    prevImgData.defaultInfo &&
      Array.isArray(JSON.parse(prevImgData.defaultInfo)) &&
      JSON.parse(prevImgData.defaultInfo).length > 0
      ? JSON.parse(prevImgData.defaultInfo).map((item: any) => {
        return item.id;
      })
      : [];
  const feaIds =
    Array.isArray(state.value.allFeatures) && state.value.allFeatures.length > 0
      ? state.value.allFeatures.map((item: any) => {
        return item.id;
      })
      : [];
  // lodash 比较俩个数组markIds feaIds是否相同
  const isEdit = _.isEqual(markIds, feaIds); // 作废？？ 涉及到ui渲染卡顿 导致状态没及时更新
  const isEdit1 = _.isEqual(markIds, defaultIds);
  if (isInvalidUpdate.value || !isEdit1) {
    isOperate.value = true;
    const imgInfo = await getImageSizeByUrl(prevImgData.imgSrc); // 不同屏幕图片宽高
    const buffer = await state.value.gMap.exportLayersToImage(
      { x: 0, y: 0, width: imgInfo.width, height: imgInfo.height },
      { type: "blob", format: "image/png" },
    );
    getFeatures();
    if (
      Array.isArray(state.value.allFeatures) &&
      state.value.allFeatures.length > 0
    ) {
      // 图片上传 + 保存markInfo
      const filename = `${new Date().getTime()}.jpg`;
      const file = new File([buffer], filename, { type: "image/jpg" });
      const formData = new FormData();
      formData.append("file", file);
      formData.append("sonId", prevImgData.sonId);
      formData.append("version", prevImgData.version);
      const res = await MarkFileUpload(formData);
      if (res.data) {
        const markDataList = concatFeatureList.value.map((item, index) => ({
          openId: item.id ? item.id.split("-")[0] : "",
          id: item.id ?? `${new Date().getTime()}-${nanoid()}`,
          type: item.type,
          props: {
            ...item.props,
            operateWidth: state.value.nWidth,
            operateHeight: state.value.nHeight,
          },
          shape: item.shape,
          style: item.style,
          isEye: `${item.isEye}` === 'undefined' ? true : item.isEye,
          operateIdx: item?.operateIdx ?? index
        }));
        const labels = markDataList.map((item) => item.openId);
        const params = {
          fileId: prevImgData.fileId,
          sonId: route.query.id,
          markFileId: res.data,
          markInfo: JSON.stringify(markDataList),
          labels: [...new Set(labels)].join(","),
          operateWidth: imgInfo.width,
          operateHeight: imgInfo.height,
          markUserId: route.query?.markUserId,
          taskId: route.query?.taskId,
          isInvalid: Number(!state.value.isInvalid)
        };
        const res1 = await addDataMarkInfo(params);
        if (res1.data) {
          isInvalidUpdate.value = false;
          imgList.value[imgActiveIndex.value].markInfo =
            JSON.stringify(markDataList);
          imgList.value[imgActiveIndex.value].defaultInfo =
            JSON.stringify(markDataList);
          window.$message?.success?.("后台保存成功！");
          revokeList.value = [];
          // ---------------------------------------------
          if (imgActiveIndex.value == imgList.value.length - 1) {
            if (imgParams.page < imgPages.value) {
              imgParams.page += 1;
              imgActiveIndex.value = 0;
            } else {
              window.$message?.warning?.("已经是最后一个了");
            }
          } else {
            // imgActiveIndex.value = Number(imgActiveIndex.value) + 1;
            let increment = 1;
            if (tabConfig.value.state == 1 && Boolean(state.value.isInvalid) || tabConfig.value.state == 2 && !Boolean(state.value.isInvalid)) {
              increment = 0;
            } else if (tabConfig.value.state == 0) {
              increment = 1;
            } else {
              increment = 1;
            }
            if (tabConfig.value.state == 2 && !Boolean(state.value.isInvalid) && state.value.allFeatures.length === 0) {
              increment = 1;
            }
            if (tabConfig.value.state == 1 && !Boolean(state.value.isInvalid) && state.value.allFeatures.length === 0) {
              increment = 0;
            }
            if (operateADKey.value) {
              if (operateADKey.value === 'a') {
                imgActiveIndex.value = Number(imgActiveIndex.value) - increment;
              };
              if (operateADKey.value === 'd') {
                imgActiveIndex.value = Number(imgActiveIndex.value) + increment;
              };
              operateADKey.value = undefined;
            } else {
              imgActiveIndex.value = Number(imgActiveIndex.value) + increment;
            }
          }
          await getImgData();
          await getDataDetailsCount();
          await nextTick();
          isOperate.value = false;
        } else {
          isOperate.value = false;
        }
      }
    } else {
      // 图片上传 + 保存markInfo
      const filename = `${new Date().getTime()}.jpg`;
      const file = new File([buffer], filename, { type: "image/jpg" });
      const formData = new FormData();
      formData.append("file", file);
      formData.append("sonId", prevImgData.sonId);
      formData.append("version", prevImgData.version);
      const res = await MarkFileUpload(formData);
      if (res.data) {
        const labels = [];
        const params = {
          fileId: prevImgData.fileId,
          sonId: route.query.id,
          markFileId: res.data,
          markInfo: "",
          labels: labels.length > 0 ? [...new Set(labels)].join(",") : "",
          operateWidth: imgInfo.width,
          operateHeight: imgInfo.height,
          markUserId: route.query?.markUserId,
          taskId: route.query?.taskId,
          isInvalid: Number(!state.value.isInvalid)
        };
        const res1 = await addDataMarkInfo(params);
        if (res1.data) {
          isInvalidUpdate.value = false;
          imgList.value[imgActiveIndex.value].markInfo = JSON.stringify("");
          imgList.value[imgActiveIndex.value].defaultInfo = JSON.stringify("");
          window.$message?.success?.("后台保存成功！");
          if (imgActiveIndex.value == imgList.value.length - 1) {
            if (imgParams.page < imgPages.value) {
              imgParams.page += 1;
              imgActiveIndex.value = 0;
            } else {
              window.$message?.warning?.("已经是最后一个了");
            }
          } else {
            // imgActiveIndex.value = Number(imgActiveIndex.value) + 1;
            let increment = 1;
            if (tabConfig.value.state == 1 && Boolean(state.value.isInvalid) || tabConfig.value.state == 2 && !Boolean(state.value.isInvalid)) {
              increment = 0;
            } else if (tabConfig.value.state == 0) {
              increment = 1;
            } else {
              increment = 1;
            }
            if (tabConfig.value.state == 2 && !Boolean(state.value.isInvalid) && state.value.allFeatures.length === 0) {
              increment = 1;
            }
            if (tabConfig.value.state == 1 && !Boolean(state.value.isInvalid) && state.value.allFeatures.length === 0) {
              increment = 0;
            }
            if (operateADKey.value) {
              if (operateADKey.value === 'a') {
                imgActiveIndex.value = Number(imgActiveIndex.value) - increment;
              };
              if (operateADKey.value === 'd') {
                imgActiveIndex.value = Number(imgActiveIndex.value) + increment;
              };
              operateADKey.value = undefined;
            } else {
              imgActiveIndex.value = Number(imgActiveIndex.value) + increment;
            }
          }
          await getImgData();
          await getDataDetailsCount();
          await nextTick();
          isOperate.value = false;
        } else {
          isOperate.value = false;
        }
      }
    }
  }
  return true;
}

function handleTagClick(feature: any, index: any) {
  if (route.query.anoType === 'audit' || route.query.anoType === 'result') return;
  tagCurrentIdx.value = index;
  state.value.editId = feature.id;
  operateId.value = feature.id;
  checkId.value = feature.id;
  activeFId.value = feature.id;
  // 设置编辑feature
  const gMap = state.value.gMap;
  gMap.setActiveFeature(feature);
  if (feature.type !== "POINT") {
    // 增加删除按钮
    // addDeleteIcon(feature, feature.shape);
    // const isTooltipModel = feature.id; // 弹框打标签
    const isTooltipModel = false;
    if (!isTooltipModel) {
      // if(feature.props.textId) {
      //   state.value.gFirstTextLayer && state.value.gFirstTextLayer.removeTextById(feature.props.textId);
      //   state.value.gMap.refresh(); // 刷新map
      // }
      tagConfig.value.isShow = true;
      tagModalVisible.value = true;
      tagConfig.value.isEditTag = false;
      checkId.value = feature.props.checkId;
    }
  }
}

const handleDataTagClick = (item: any) => {
  if (route.query.anoType === 'audit' || route.query.anoType === 'result') return;
  tagKeyCurrentIdx.value = item.idx;
  const activeFeature = state.value.gMap.getActiveFeature();
  if (activeFeature) {
    removeFeatureById(activeFeature.id, checkId.value);
    state.value.gMap.refresh();
    const operateIdx = item.idx == 0 ? '9' : item.idx - 1;
    const { labelColor, labelName } = pagTagList.value[operateIdx];
    const config = {
      id: activeFeature.id,
      textId: activeFeature.id,
      deleteMarkerId: activeFeature.id,
      type: activeFeature.type,
      name: labelName,
      drawingStyle: Object.assign(activeFeature.style, {
        fillStyle: labelColor,
        strokeStyle: labelColor,
      }),
    };
    againAddFeature(activeFeature.type, activeFeature.shape, config);
    state.value.gMap.refresh();
    getFeatures();
  }
};

// 辅助函数：根据 checkId 查找特征
const findFeatureByCheckId = (checkId) => {
  return state.value.allFeatures.find((item) => item.props.checkId === checkId);
};

// 辅助函数：根据 ID 查找特征
const findFeatureById = (id) => {
  return state.value.allFeatures.find((item) => item.id === id);
};

// 辅助函数：移除特征
const removeFeatureById = (id, checkId) => {
  state.value.allFeatures = state.value.allFeatures.filter(
    (item) => item.id !== id && item.props.checkId !== checkId,
  );
  state.value.gFirstFeatureLayer.features =
    state.value.gFirstFeatureLayer.features.filter(
      (item) => item.id !== id && item.props.checkId !== checkId,
    );

  if (state.value.gFirstTextLayer) {
    state.value.gFirstTextLayer.removeTextById(checkId);
    state.value.gFirstTextLayer.removeTextById(id);
  }
};

// ----------------------------------------------------------
// utils function
type opState = "ArrowUp" | "ArrowDown" | "ArrowLeft" | "ArrowRight";
type shapeType = "POLYGON" | "RECT" | "CIRCLE";

function copyFeature(feature: any, id: string, name: string) {
  const { type, shape, style, props } = feature;
  if (type === "RECT") {
    feature = new AILabel.Feature.Rect(id, shape, props, style);
  } else if (type === "CIRCLE") {
    feature = new AILabel.Feature.Circle(id, shape, props, style);
  } else if (type === "POLYGON") {
    feature = new AILabel.Feature.Polygon(id, shape, props, style);
  }
  if (feature) {
    state.value.gFirstFeatureLayer.addFeature(feature);
    const textPosition = setText(shape, type);
    addLayerText(id, name, textPosition);
  }
}

function getStepX(state: opState, step: string | number = 1) {
  let opStep;
  if (state === "ArrowLeft") {
    opStep = -step;
  }
  if (state === "ArrowRight") {
    opStep = step;
  }
  return opStep;
}

function getStepY(state: opState, step: string | number = 1) {
  let opStep;
  if (state === "ArrowUp") {
    opStep = -step;
  }
  if (state === "ArrowDown") {
    opStep = step;
  }
  return opStep;
}

function getOpShape(
  type: shapeType,
  shape: any,
  step: string | number = 1,
  state: opState,
) {
  let stepX = getStepX(state, step) ?? 0;
  let stepY = getStepY(state, step) ?? 0;
  switch (type) {
    case "POLYGON":
      return {
        points: shape.points.map((val: any) => {
          return {
            x: val.x + stepX,
            y: val.y + stepY,
          };
        }),
      };
    case "RECT":
      return {
        width: shape.width,
        height: shape.height,
        x: shape.x + stepX,
        y: shape.y + stepY,
      };
    case "CIRCLE":
      return {
        cx: shape.cx + stepX,
        cy: shape.cy + stepY,
        r: shape.r,
      };
    default:
      return {};
  }
}

function getCopyShape(type: shapeType, shape: any, step: string | number = 50) {
  switch (type) {
    case "POLYGON":
      return {
        points: shape.points.map((val: any) => {
          return {
            x: val.x + step,
            y: val.y,
          };
        }),
      };
    case "RECT":
      return {
        width: shape.width,
        height: shape.height,
        x: shape.x + step,
        y: shape.y,
      };
    case "CIRCLE":
      return {
        cx: shape.cx + step,
        cy: shape.cy,
        r: shape.r,
      };
    default:
      return {};
  }
}

function handleMouseWheel(event: MouseEvent) {
  event = event || window.event;
  var delta = 0;
  if (event.wheelDelta) {
    // IE和Opera
    delta = -event.wheelDelta / 120;
  } else if (event.detail) {
    // Firefox
    delta = event.detail / 3;
  }
  if (delta) {
    if (delta < 0) {
      console.log("滚轮向上滚动");
      zoomIn();
    } else {
      console.log("滚轮向下滚动");
      zoomOut();
    }
  }
}

function handleKeyDown(e: KeyboardEvent) {
  if (tagModalVisible.value || isShowValidate.value || noPassVisable.value || submitShowModal.value) return;
  if (route.query.anoType === "audit" || route.query.anoType === "result") {
    switch (e.key) {
      case "a":
      case "d":
        if (e.ctrlKey) {
          handleCtrlDKey(e, e.key);
        } else {
          handleADKeys(e.key);
        }
        break;
    }
  } else {
    switch (e.key) {
      case "Delete":
        handleDeleteKey();
        break;
      case "ArrowUp":
      case "ArrowDown":
      case "ArrowLeft":
      case "ArrowRight":
        handleArrowKeys(e.key);
        break;
      case "a":
      case "d":
        if (e.ctrlKey) {
          handleCtrlDKey(e, e.key);
        } else {
          handleADKeys(e.key);
        }
        break;
      case "Escape":
        tagCurrentIdx.value = -1;
        handleEscapeKey();
        break;
      case "s":
        if (e.ctrlKey) {
          handleCtrlSKey(e);
        }
        break;
      case "+":
      case "-":
        if (e.ctrlKey) {
          handleCtrlPlusMinusKeys(e, e.key);
        }
        break;
      // Y N
      case "y":
      case "n":
        handleCtrlYNKey(e, e.key);
        break;
      // z
      case "z":
        if (e.ctrlKey) {
          handleCtrlZKey(e, e.key);
        }
        break;
      case "0":
      case "1":
      case "2":
      case "3":
      case "4":
      case "5":
      case "6":
      case "7":
      case "8":
      case "9":
        try {
          let operateKeyIdx = e.key == 0 ? 9 : +e.key - 1;
          if (!pagTagList.value[operateKeyIdx]) return;
          tagKeyCurrentIdx.value = e.key == 0 ? 0 : +e.key;
          const activeFeature = state.value.gMap.getActiveFeature();
          const removeId = state.value.allFeatures[tagCurrentIdx.value]?.id;

          if (activeFeature) {
            // 移除旧的特征
            getFeatures();
            let featureConfig, featureId;
            if (checkId.value) {
              featureConfig = findFeatureByCheckId(checkId.value);
              if (!featureConfig) return; // 如果没有找到匹配的特征，直接返回
              featureId = featureConfig.id;
              removeFeatureById(featureId, checkId.value);
            } else {
              featureConfig = findFeatureById(operateId.value);
              if (!featureConfig) return; // 如果没有找到匹配的特征，直接返回
              featureId = featureConfig.id;
              removeFeatureById(featureId, operateId.value);
            }

            // 更新地图显示
            state.value.gMap.refresh();

            // 获取新的标签信息
            const operateIdx = e.key == 0 ? '9' : +e.key - 1;
            const { labelColor, labelName } = pagTagList.value[operateIdx];

            // 创建新的配置
            const config = {
              id: activeFeature.id,
              textId: activeFeature.id,
              deleteMarkerId: activeFeature.id,
              type: activeFeature.type,
              name: labelName,
              drawingStyle: Object.assign({}, activeFeature.style, {
                fillStyle: labelColor,
                strokeStyle: labelColor,
              }),
            };

            // 再次添加特征
            againAddFeature(activeFeature.type, activeFeature.shape, config);

            checkId.value = ""
            getFeatures();
            state.value.gMap.refresh();

            // 根据tagCurrentIdx.value 重新排序 state.value.allFeatures
            state.value.allFeatures.splice(
              tagCurrentIdx.value,
              0,
              state.value.allFeatures.pop(),
            );
          }
        } catch (error) {
          console.error("Error handling tag change:", error);
        }
        break;
    }
  };
}

function handleDeleteKey() {
  if (state.value.editId) {
    const feature = state.value.gFirstFeatureLayer.getFeatureById(
      state.value.editId,
    );
    revokeList.value = [...revokeList.value, feature];
    const deleteId = state.value.editId;
    state.value.gFirstFeatureLayer &&
      state.value.gFirstFeatureLayer.removeFeatureById(deleteId);
    state.value.gFirstTextLayer &&
      state.value.gFirstTextLayer.removeTextById(deleteId);
    tagModalVisible.value = false;

    hiddenEyeList.value = hiddenEyeList.value.filter(
      (val) => val.id !== state.value.editId,
    );

    getFeatures();
  }
}

function handleArrowKeys(key) {
  if (visible.value || isOperate.value || isRender.value) return;
  let activeFeature;
  // if (activeFId.value) {
  //   activeFeature = state.value.allFeatures.find(
  //     (val) => val.id === activeFId.value,
  //   );
  //   state.value.gMap.setActiveFeature(null);
  // } else {
  //   activeFeature = state.value.gMap.getActiveFeature();
  // }
  activeFeature = state.value.gMap.getActiveFeature();
  const opShape = getOpShape(activeFeature.type, activeFeature.shape, 1, key);
  activeFeature.updateShape(opShape);
  // textFeature
  const { props } = activeFeature;
  const textId = props.textId;
  const activeTextFeature = state.value.gFirstTextLayer.getTextById(textId);
  activeTextFeature.updatePosition({
    x: opShape.x,
    y: opShape.y,
  });
}

// async function handleADKeys(key) {
//   operateADKey.value = key;
//   // 检查是否有未打的标签
//   getFeatures();
//   const mapFeatures = state.value.allFeatures
//     .filter((val: any) => val.type !== "POINT")
//     .map((val: any) => {
//       val.isEye = val?.isEye ?? true;
//       return val;
//     });
//   const unlabelExistInFeatures = hasEmptyNameFeature(state.value.allFeatures);
//   if (unlabelExistInFeatures) {
//     window.$message?.warning?.("所有标注必须设置标签！");
//     return;
//   }

//   if (visible.value || isOperate.value || isRender.value) return;
//   const curIndex = Number(imgActiveIndex.value);
//   let nextIndex = key === "a" ? curIndex - 1 : curIndex + 1;
//   if (nextIndex < 0 || nextIndex >= imgList.value.length) {
//     console.log(nextIndex);
//     return;
//   };
//   const nextImgData = imgList.value[nextIndex];
//   handleImgChange(nextImgData);
// }

// ----------------NEW ad快捷键------------------
async function handleADKeys(key: string) {
  // 存储当前操作的按键
  operateADKey.value = key;

  // 获取并处理特征数据
  await getFeatures();

  // 过滤非点类型特征并确保isEye属性存在
  const mapFeatures = state.value.allFeatures
    .filter((feature: FeatureType) => feature.type !== "POINT")
    .map((feature: FeatureType) => ({
      ...feature,
      isEye: feature.isEye ?? true,
    }));

  // 检查是否存在未命名的标注
  if (hasEmptyNameFeature(state.value.allFeatures)) {
    window.$message?.warning("所有标注必须设置标签！");
    return;
  }

  // 检查界面状态是否允许切换图片
  if (visible.value || isOperate.value || isRender.value) {
    return;
  }

  // 计算下一张图片的索引
  const currentIndex = Number(imgActiveIndex.value);
  const nextIndex = key === "a" ? currentIndex - 1 : currentIndex + 1;

  // 索引越界处理 - 增加翻页逻辑
  if (nextIndex < 0) {
    // 尝试上一页
    if (imgParams.page > 1) {
      await navigateToPage(imgParams.page - 1, true); // 上一页时设置为最后一张
    } else {
      window.$message?.warning("已经是第一张图片！");
    }
    return;
  }

  if (nextIndex >= imgList.value.length) {
    // 尝试下一页
    if (imgParams.page < imgPages.value) {
      await navigateToPage(imgParams.page + 1, false); // 下一页时设置为第一张
    } else {
      window.$message?.warning("已经是最后一张图片！");
    }
    return;
  }

  // 正常切换图片
  const nextImgData = imgList.value[nextIndex];
  await handleImgChange(nextImgData);
}

// 新增：页面导航函数
async function navigateToPage(pageNum: number, isPrevPage: boolean) {
  try {
    // 保存当前图片
    await asyncPrevImgSave();

    // 重置状态并准备加载新页面
    state.value.gMap.removeLayerById("first-layer-image");
    imgParams.page = pageNum;
    imgList.value = [];

    // 获取新页面的图片数据
    await getImgData();

    // 如果成功加载了图片，则显示对应位置的图片
    if (imgList.value.length > 0) {
      // 计算要显示的图片索引
      const targetIndex = isPrevPage
        ? Math.min(7, imgList.value.length - 1) // 上一页显示最后一张(索引7或最后一张)
        : 0; // 下一页显示第一张

      imgActiveIndex.value = String(targetIndex);
      await handleImgChange(imgList.value[targetIndex]);
    }
  } catch (error) {
    console.error("翻页过程中出错:", error);
    window.$message?.error("加载页面失败，请重试");
  }
}

// 假设的特征类型定义
type FeatureType = {
  type: string;
  isEye?: boolean;
  name?: string;
  // 其他可能的属性
}
// ----------------------------------

function handleCtrlZKey(e, key) {
  if (key === "z") {
    Revoke();
  }
}

function handleEscapeKey() {
  state.value.gMap.setMode("PAN");
  iconList.forEach((val) => {
    val.isOperate = false;
  });
  state.value.foreground_points = [];
  state.value.background_points = [];
}

async function handleCtrlSKey(e: KeyboardEvent) {
  if (route.query.anoType === 'audit' || route.query.anoType === 'result') return;
  e.preventDefault();
  saveAno("1");
}

function handleCtrlDKey(e: KeyboardEvent, key: any) {
  e.preventDefault();
  if (key === "d") {
    const activeFeature = state.value.gMap.getActiveFeature();
    if (activeFeature) {
      const { id, props, type, style, shape } = activeFeature;
      let data = getCopyShape(type, shape);
      const feature = Object.assign({}, activeFeature, {
        shape: data,
      });
      const randomId = `${new Date().getTime()}-${nanoid()}`;
      const textName = tagConfig.value.tagList.find(
        (val) => val.labelId == id.split("-")[0],
      )?.labelName;
      copyFeature(feature, randomId, textName);
    }
  }
}

function handleCtrlPlusMinusKeys(e: KeyboardEvent, key) {
  e.preventDefault();
  const zoom = state.value.zoom;
  if (key === "+") {
    zoomIn();
  } else {
    zoomOut();
  }
}

function handleCtrlYNKey(e: KeyboardEvent, key: string) {
  if (route.query.anoType === 'audit' || route.query.anoType === 'validateUser' || route.query.anoType === 'validate' || route.query.anoType === 'result') return;
  if (key === "y") {
    handlePass("1");
  }
  if (key === "n") {
    handlePass("2");
  }
}

function fIconList(iconList: any) {
  if (anoType.value == 0) {
    return iconList.filter((item: any) => {
      return item.permList.length === 2 && item.permList[0] === "0";
    });
  }
  if (anoType.value == 1) {
    return iconList;
  }
}

const handleWheel = (event) => {
  const targetNode = targetNodeRef.value;
  if (targetNode && targetNode.contains(event.target)) {
    event.preventDefault();
    state.value.gMap.disableZoomWhenDrawing(); // 禁用绘制时可鼠标滑轮缩放
  }
};

onBeforeUnmount(() => {
  state.value.gMap?.destroy();
  window.onresize = null;
  window.onkeydown = null;
  window.removeEventListener("keydown", handleKeyDown);
  clearTimeout();

  const targetNode = targetNodeRef.value;
  if (targetNode) {
    targetNode.removeEventListener('wheel', handleWheel);
  }
});
onMounted(async () => {
  const anoStore = useAnoStore();
  anoType.value = anoStore.anoType;
  const originalRadio = 16 / 9; // 假设我们的原始比例是16:9
  const sourceRadio = 1.42;
  const newRadio = adjustRatio(originalRadio).toFixed(2);
  const radio = newRadio / sourceRadio;
  const zoom = state.value.zoom * radio;
  state.value.zoom = zoom;
  await getDataDetailsCount();
  await getSelectData();
  await getTagList();
  await getImgData();
  await setCurTabNum("0");
  // 获取图片原宽高
  try {
    const row = await getImageSizeByUrl();
    setImgSize(row);
  } catch (e) { }
  initMap(); // 初始化实例
  setGFirstImageLayer(); // 图片层添加
  setGFirstFeatureLayer(); // 添加矢量图层
  setGFirstTextLayer(); // 添加 text 文本图层，用于展示文本
  addEvent(); // 添加事件
  state.value.gMap.disablePanWhenDrawing(); // 禁用绘制时鼠标达到边界外自动平移
  // state.value.gMap.disableZoomWhenDrawing(); // 禁用绘制时可鼠标滑轮缩放
  window.onresize = () => {
    state.value.gMap && state.value.gMap.resize();
  };

  // 监听键盘事件
  window.addEventListener("keydown", handleKeyDown);

  // 监听图片层区域鼠标事件
  const mainMap = main_map.value;
  if (mainMap) {
    mainMap.addEventListener("wheel", handleMouseWheel);
  }
});

// newCode
const isShowValidate = ref<Boolean>(false);

const validateStatus = ref<String>("0");
const validateModel = ref<any>({
  verifyState: null,
  returnState: "",
  message: null,
});
const curTabNum = ref<string | number>("0");

const noPassVisable = ref<Boolean>(false);
const statusOptions = ref([
  { value: "1", label: "保存全部数据 " },
  { value: "2", label: "仅保存验收通过的数据 " },
]);
const repulseOptions = ref([
  { value: "1", label: "未验收的数据 " },
  { value: "2", label: "验收不通过的数据 " },
  { value: "3", label: "未验收+验收不通过的数据" },
]);

const submitShowModal = ref<Boolean>(false);
const submitTooltipText = ref<string>("");
const submitRow = ref<any>(null);

const operateADKey = ref<string | undefined>(undefined);

const validateTitle = computed(() => {
  if (validateStatus.value === "0") {
    return "剩余验收通过";
  } else if (validateStatus.value === "1") {
    return "验收完成";
  } else if (validateStatus.value === "2") {
    return "打回任务";
  }
});

const shouldShowButtons = computed(() => {
  const anoType = route.query.anoType;
  const isValidAnoType = anoType === "validate" || anoType === "audit";
  return isValidAnoType && !!curTabNum.value;
});

const notPassMessage = ref<string>("");

const curTabVal = ref<string>("0");

watchEffect(() => {
  const value = imgList.value[imgActiveIndex.value]?.notPassMessage;
  notPassMessage.value = value;
});

const handleValidate = async (statusCode: string) => {
  validateStatus.value = statusCode;
  if (statusCode === "0") {
    const res = await remainingApprove({
      taskId: route.query?.taskId,
      id: route.query?.markUserId,
    });
    if (res.data) {
      window.$message?.success?.("操作成功！");
      await getDataDetailsCount();
      await getTagList();
      await getSelectData();
      await getImgData();
    }
  }
  if (statusCode === "1") {
    isShowValidate.value = true;
  }
  if (statusCode === "2") {
    isShowValidate.value = true;
  }
  if (statusCode === "3") {
    const res = await submitTaskPrompt({ id: route.query.taskId });
    if (res.data) {
      submitRow.value = row;
      submitShowModal.value = true;
      submitTooltipText.value = res.data;
    }
  }
};

const handlePass = async (sign: string) => {
  switch (sign) {
    case "1":
      await handleApprove("1");
      break;
    case "2":
      noPassVisable.value = true;
      break;
    default:
      break;
  }
};

const handleApprove = async (
  sign: string | number,
  tooltipText: string = "自动保存成功！",
) => {
  try {
    const params = {
      taskId: route.query?.taskId,
      isApprove: sign,
      fileId: imgList.value[imgActiveIndex.value]?.fileId,
      notPassMessage: validateModel.value.message,
    };

    const res = await fileIsApprove(params);

    if (res.data) {
      validateModel.value.message = undefined;
      if (tooltipText) {
        window.$message?.success?.(tooltipText);
      }
      let curImgIdx: number | string = imgActiveIndex.value;
      let curImgData = imgList.value[curImgIdx];
      await handleImgChange(curImgData);
      /* ------------------------------------- */
      if (imgActiveIndex.value == imgList.value.length - 1) {
        if (imgParams.page < imgPages.value) {
          imgParams.page += 1;
          imgActiveIndex.value = -1;
        } else {
          window.$message?.warning?.("已经是最后一个了");
        }
      }
      await nextTick(() => {
        let increment = 0;
        if (tabConfig.value.state == 1 && sign == 2 || tabConfig.value.state == 2 && sign == 1) {
          increment = 0;
        } else if (tabConfig.value.state == 0) {
          increment = 1;
        } else {
          increment = 1;
        }
        imgActiveIndex.value = Number(imgActiveIndex.value) + increment
      });
      await getImgData();
      await getTagList();
      await getDataDetailsCount();
      /* ------------------------------------- */
    }
  } catch (error) {
    console.error("Error handling approve:", error);
  }
};



const handleSubmitDefine = async () => {
  const res = await submitTask({ id: route.query.taskId });
  if (res.data) {
    window.$message?.success?.("任务提交成功！");
    submitShowModal.value = false;
    router.back();
  }
};

const handleSubmitClose = () => {
  submitShowModal.value = false;
};

const handlePassSuccess = async () => {
  noPassVisable.value = false;
  // validateModel.value.message = null;
  handleApprove("2", "验收意见保存成功！");
};

const handleValidateSuccess = async () => {
  if (validateStatus.value === "1") {
    const res = await verifyComplete({
      taskId: route.query?.taskId,
      verifyState: validateModel.value.verifyState,
    });
    if (res.data) {
      window.$message?.success?.(`操作成功`);
      router.back();
    }
  }
  if (validateStatus.value === "2") {
    const res = await returnTask({
      taskId: route.query?.taskId,
      returnState: validateModel.value.returnState,
      id: route.query?.markUserId,
    });
    if (res.data) {
      window.$message?.success?.(`操作成功`);
      router.back();
    }
  }
};

const handleModalDefine = async (sign: string) => {
  switch (sign) {
    case "tag":
      // handleTag();
      const checkedTags = tagConfig.value.tagList.filter(
        (item) => item.isCheck,
      );

      if (checkedTags.length === 0) {
        window.$message?.warning?.("请先勾选标签！");
        return;
      }
      getFeatures();
      let featureConfig, shape, type, featureId;
      if (checkId.value) {
        featureConfig = findFeatureByCheckId(checkId.value);
        if (!featureConfig) return; // 如果没有找到匹配的特征，直接返回
        featureId = featureConfig.id;
        shape = featureConfig.shape;
        type = featureConfig.type;
        removeFeatureById(featureId, checkId.value);
      } else {
        featureConfig = findFeatureById(operateId.value);
        if (!featureConfig) return; // 如果没有找到匹配的特征，直接返回
        featureId = featureConfig.id;
        shape = featureConfig.shape;
        type = featureConfig.type;
        removeFeatureById(featureId, operateId.value);
      }
      state.value.gMap.refresh();
      const [firstCheckedTag] = checkedTags;
      const customId = nanoid();
      const id = `${firstCheckedTag.labelId}-${customId}`;
      activeFId.value = id;
      const drawingStyle = {
        fillStyle: firstCheckedTag.color,
        strokeStyle: firstCheckedTag.color,
        fill: true,
        globalAlpha: 0.3,
        lineWidth: 2,
      };
      const config = {
        id,
        textId: id,
        deleteMarkerId: id,
        type,
        name: firstCheckedTag.name,
        drawingStyle,
      };
      againAddFeature(type, shape, config);
      state.value.gMap.refresh();
      getFeatures();
      // 根据tagCurrentIdx.value 重新排序 state.value.allFeatures
      state.value.allFeatures.splice(
        tagCurrentIdx.value,
        0,
        state.value.allFeatures.pop(),
      );
      const activeFeature = state.value.allFeatures[tagCurrentIdx.value];
      state.value.gMap.setActiveFeature(activeFeature);
      checkId.value = ""
      operateId.value = activeFeature.id;
      getFeatures();
      state.value.gMap.refresh();
      tagModalVisible.value = false;
      break;
    case "validate":
      await handleValidateSuccess();
      break;
    case "pass":
      await handlePassSuccess();
      break;
    default:
      break;
  }
};

const handleRTabChange = (sign: stirng) => {
  switch (sign) {
    case "0":
      curTabVal.value = "0";
      break;
    case "1":
      curTabVal.value = "1";
      break;
    default:
      break;
  }
};

// 分页标签
const tagKeyCurrentIdx = ref<string>("-1");
const { data, run, current, totalPage, pageSize } = usePagination(
  getSelectGroupLabelPage,
  {
    defaultParams: [
      {
        limit: 10,
        sonId: route.query?.id,
      },
    ],
    pagination: {
      currentKey: "page",
      pageSizeKey: "limit",
      totalKey: "data.total",
    },
  },
);

const handleTabSearch = () => {
  run({
    page: 1,
    limit: 10,
    sonId: route.query?.id,
    labelName: tagSearchVal.value
  })
};

const handleTagStick = async (rowData: any) => {
  const res = await topUpLabel({
    labelId: rowData.labelId,
    sonId: route.query?.id,
  });
  if (res.data) {
    tagSearchVal.value = "";
    window.$message?.success?.(`置顶成功`);
    run({
      page: 1,
      limit: 10,
      sonId: route.query?.id,
      labelName: tagSearchVal.value
    })
  }
};

const pagTagList = computed(
  () =>
    data.value?.data.records.map((val, idx) => {
      let newIdx = idx + 1;
      if (newIdx >= 10) {
        newIdx = parseInt(String(newIdx).slice(-1));
      }
      return {
        idx: newIdx,
        ...val,
      };
    }) || [],
);

function hasEmptyNameFeature(features: any[]) {
  return features.some((feature: any, index: number) => {
    return feature.props && feature.props.name === '';
  });
}

// 重构 没时间 算了
const hiddenEyeList = ref<any>([]);
const concatFeatureList = computed(() => {
  getFeatures();

  const insertItemsByIndex = (arr1: any, arr2: any) => {
    arr1.forEach((item, idx) => {
      const index = item?.operateIdx ?? idx;
      arr2.splice(index, 0, item);
    });
    return arr2;
  };

  let list = insertItemsByIndex(hiddenEyeList.value, state.value.allFeatures).map((item, index) => {
    item.isEye = `${item.isEye}` === 'undefined' ? true : item.isEye;
    item.operateIdx = index;
    return item;
  }).filter((val: any) => val.type !== "POINT");
  const mapList = _.uniqBy(list, 'id');
  return mapList;

})
const message = useMessage();


// 验收结果
const shouldShowResult = computed(() => {
  const anoType = route.query.anoType;
  const showTypes = ['validate', 'validateUser', 'audit'];
  return showTypes.includes(anoType) && imgList.value.length;
});

const resultColor = computed(() => {
  /*   const state = tabConfig.value.state;
    if (state === '0') {
      return '';
    }
    return state === '2' ? 'red' : 'green'; */
  return ""
});

const getResultText = (defaultText) => {
  const anoType = route.query.anoType;
  if (anoType === 'validateUser' && !notPassMessage.value) {
    return '';
  }
  return notPassMessage.value ? notPassMessage.value : defaultText;
};

const resultText = computed(() => {
  const state = tabConfig.value.state;
  if (state === '2') {
    return getResultText('');
  }
  return getResultText('');
});

const validateEnglishVal = () => {
  const regex = /^[a-zA-Z0-9_\-\/]+$/;
  if (!regex.test(tagConfig.value.params.englishVal)) {
    window.$message?.error?.('请输入符合格式要求的英文名，仅允许包含字母、数字、下划线、连字符和斜杠。');
  }
};

// ----------------------图片预加载 cache----------------------------
const batchSize = 5; // 每批加载的图片数量
const loadedImages = new Set(); // 用于记录已经加载过的图片路径

const preloadImage = (imageUrl) => {
  if (loadedImages.has(imageUrl)) {
    return Promise.resolve(); // 如果已经加载过，直接返回一个已解决的 Promise
  }
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.src = imageUrl;
    img.onload = () => {
      loadedImages.add(imageUrl); // 图片加载成功后，将其路径添加到已加载集合中
      resolve();
    };
    img.onerror = (error) => reject(error);
  });
};

const preloadBatch = async (batch) => {
  const promises = batch.map((item) => preloadImage(item.imgPath));
  await Promise.all(promises);
};

const preloadImages = async () => {
  for (let i = 0; i < imgList.value.length; i += batchSize) {
    const batch = imgList.value.slice(i, i + batchSize);
    await new Promise((resolve) => requestAnimationFrame(() => resolve(preloadBatch(batch))));
  }
};


// 页码跳转函数
const jumpToPage = async (page: number) => {
  // 页码验证
  if (page < 1 || page > imgPages.value) {
    window.$message?.warning(`页码必须在 1 到 ${imgPages.value} 之间`);
    return;
  }

  // 如果当前已经是目标页码，直接返回
  if (page === imgParams.page) {
    return;
  }

  // 保存当前图片的标记信息
  await asyncPrevImgSave();

  // 重置图片索引和地图图层
  imgActiveIndex.value = "0";
  state.value.gMap.removeLayerById("first-layer-image");

  // 更新页码并加载数据
  imgParams.page = page;
  imgList.value = [];
  await getImgData();

  // 提示用户跳转成功
  window.$message?.success(`已跳转到第 ${page} 页`);
};

const handlePageJump = async (pageInput: string) => {
  const page = parseInt(pageInput, 10);
  if (!isNaN(page)) {
    await jumpToPage(page);
  } else {
    window.$message?.warning("请输入有效的页码");
  }
};

</script>

<template>
  <div style="padding: 0 !important"
    class="wrap_main h-full w-full flex flex-col items-center justify-start bg-[#f7f7f9] p-0">
    <div class="header box-border box-border h-[48px] w-full flex items-center bg-[#fff] px-16px py-0">
      <div class="item_return h-full w-auto flex cursor-pointer items-center" @click="handleBack()">
        <SvgIcon local-icon="oui--return-key" class="inline-block align-text-bottom text-18px color-[#000]"></SvgIcon>
        <span class="ml-[4px] block h-full w-auto flex items-center text-[12px] text-[#84868c]">返回</span>
      </div>
      <div class="item_name ml-[12px] h-full w-auto flex items-center text-16px text-[#151b26] font-[500]">
        详情
      </div>
      <div class="item_rBtn_con h-full flex-1">
        <!-- anoType === 'validate' -->
        <div class="h-full w-full flex items-center justify-end gap-[24px]" v-if="
          route.query.anoType === 'validate' ||
          route.query.anoType === 'audit'
        ">
          <n-button type="primary" @click="handleValidate('0')">
            剩余验收通过
          </n-button>
          <n-button type="primary" @click="handleValidate('1')" v-show="route.query.anoType !== 'audit'">
            验收完成
          </n-button>
          <n-button type="primary" @click="handleValidate('2')">
            打回任务
          </n-button>
        </div>
        <!-- anoType === 'validateUser' || anoType === 'audit' -->
        <!-- <div class="h-full w-full flex items-center justify-end gap-[24px]"
          v-if="route.query.anoType === 'validateUser'">
          <n-button type="primary" @click="handleValidate('3')">
            提交任务
          </n-button>
        </div> -->
      </div>
    </div>
    <div class="header1 box-border box-border h-[48px] w-full flex items-center bg-[#fff] px-16px py-0">
      <NTabs type="line" animated class="wrap_tabs ml-14px w-auto" @update:value="handleTabChange"
        @before-leave="handleTabBefore" v-if="
          route.query.anoType === 'validateUser' ||
          route.query.anoType === 'online' ||
          route.query.anoType === 'setOnline' ||
          !route.query.anoType
        ">
        <NTabPane name="0" tab="全部">
          <template #tab>全部({{ tabConfig.tabNum.all }})</template>
        </NTabPane>
        <NTabPane name="1" tab="有标注信息">
          <template #tab>有标注信息({{ tabConfig.tabNum.haveAno }})</template>
        </NTabPane>
        <NTabPane name="2" tab="无标注信息">
          <template #tab>无标注信息({{ tabConfig.tabNum.noAno }})</template>
        </NTabPane>
        <NTabPane name="3" tab="无效数据信息">
          <template #tab>无效数据信息({{ tabConfig.tabNum.invalid }})</template>
        </NTabPane>
      </NTabs>
      <NTabs type="line" animated class="wrap_tabs ml-14px w-auto" @update:value="handleTabChange"
        @before-leave="handleTabBefore" v-if="route.query.anoType === 'validate'">
        <NTabPane name="0" tab="全部">
          <template #tab>未经过验收({{ tabConfig.tabNum.all }})</template>
        </NTabPane>
        <NTabPane name="1" tab="有标注信息">
          <template #tab>验收通过({{ tabConfig.tabNum.haveAno }})</template>
        </NTabPane>
        <NTabPane name="2" tab="无标注信息">
          <template #tab>验收不通过({{ tabConfig.tabNum.noAno }})</template>
        </NTabPane>
        <NTabPane name="3" tab="无效数据信息">
          <template #tab>无效数据信息({{ tabConfig.tabNum.invalid }})</template>
        </NTabPane>
      </NTabs>
      <NTabs type="line" animated class="wrap_tabs ml-14px w-auto" @update:value="handleTabChange"
        @before-leave="handleTabBefore" v-if="route.query.anoType === 'audit'">
        <NTabPane name="0" tab="全部">
          <template #tab>未审核信息({{ tabConfig.tabNum.all }})</template>
        </NTabPane>
        <NTabPane name="1" tab="有标注信息">
          <template #tab>审核通过({{ tabConfig.tabNum.haveAno }})</template>
        </NTabPane>
        <NTabPane name="2" tab="无标注信息">
          <template #tab>审核未通过({{ tabConfig.tabNum.noAno }})</template>
        </NTabPane>
        <NTabPane name="3" tab="无效数据信息">
          <template #tab>无效数据信息({{ tabConfig.tabNum.invalid }})</template>
        </NTabPane>
      </NTabs>
      <NTabs type="line" animated class="wrap_tabs ml-14px w-auto" @update:value="handleTabChange"
        @before-leave="handleTabBefore" v-if="route.query.anoType === 'result'">
        <NTabPane name="0" tab="全部">
          <template #tab>全部({{ tabConfig.tabNum.all }})</template>
        </NTabPane>
      </NTabs>
    </div>
    <div class="content box-border w-full flex-1 p-[16px] overflow-auto relative">
      <NCard title="" class="h-full w-full pt-24px">
        <div class="absolute left-50% top-10px -translate-x-[50%]" v-show="curFileName">当前图片名称: {{ curFileName }}</div>
        <div class="h-full w-full flex items-center justify-start border-1 border-[#eee] relative">
          <div class="absolute left-24px -top-36px" v-if="shouldShowResult" v-cloak>
            <span v-if="!!resultText">验收结果：</span>
            <span :style="{ color: resultColor }" v-if="!!resultText">{{ resultText }}</span>
          </div>
          <!-- end -->
          <div class="h-full w-20% flex flex-col items-center justify-start border-r-1 border-r-[#eee]" :style="{
            'cursor': route.query.anoType === 'audit' ? 'not-allowed' : 'pointer'
          }">
            <!-- 数据集标签 -->
            <div class="boder-b-[#eee] h-[42px] w-full flex items-center gap-8px border-b-1 border-t-1 px-24px">
              <div class="w-full flex items-center justify-center gap-4px">
                <NButton type="primary" quaternary class="prev" size="small" @click="handleRTabChange('1')">数据集标签
                </NButton>
              </div>
            </div>
            <!-- 搜索 -->
            <div class="boder-b-[#eee] h-[42px] w-full flex items-center gap-8px border-b-1 border-t-1 px-4px">
              <div class="w-full flex items-center justify-center gap-4px">
                <NInput v-model:value="tagSearchVal" class="!flex-1" placeholder="请输入标签名" clearable />
                <NButton type="primary" quaternary class="prev" size="small" @click="handleTabSearch">搜索
                </NButton>
              </div>
            </div>
            <div class="w-full flex-1 min-h-0 overflow-y-auto relative flex-col justify-start items-center">
              <div
                class="flex-1 min-h-0 box-border w-full flex flex-nowrap flex-col items-start justify-start gap-4px px-8px py-8px">
                <div v-for="(item, index) in pagTagList" :key="index" :class="[
                  'box-border h-40px h-auto w-full flex items-center justify-between overflow-hidden rounded-[4px] px-4px py-4px',
                  {
                    'border-2 border-[#2d7a67]': tagKeyCurrentIdx === item.idx,
                    'border-1 border-[#eee]': tagKeyCurrentIdx !== item.idx
                  }
                ]" @click="handleDataTagClick(item, index)">
                  <div class="h-full w-full flex items-center cursor-pointer">
                    <div class="mr-8px w-10px flex justify-center items-center">
                      {{ item.idx }}
                    </div>
                    <div class="mr-8px w-10px h-14px" :style="{ background: item.labelColor }"></div>
                    <div class="flex-1 min-w-0 text-13px line-clamp-2">
                      {{ item.labelName || '暂无标签' }}
                    </div>
                    <!-- 置顶 -->
                    <div class="w-24px">
                      <n-tooltip trigger="hover">
                        <template #trigger>
                          <div class="w-24px" @click.stop="(event) => handleTagStick(item)">
                            <SvgIcon local-icon="icon-park-outline--to-top"
                              class="inline-block align-text-bottom text-18px text-[#000]"></SvgIcon>
                          </div>
                        </template>
                        点击置顶
                      </n-tooltip>
                    </div>
                  </div>
                </div>
              </div>
              <div
                class="pagination-container__page w-full h-46px flex justify-center py-4px box-border boder-b-[#eee] border-t-1 z-666">
                <NPagination v-model:page="current" v-model:page-size="pageSize" :page-count="totalPage"
                  v-if="pagTagList.length > 0" :page-slot="5" />
              </div>
            </div>
          </div>
          <div class="h-full w-60% flex flex-col items-center justify-start relative">
            <div class="boder-b-[#eee] h-[42px] w-full flex items-center gap-8px border-b-1 px-24px">
              <div v-if="route.query.anoType !== 'audit' && route.query.anoType !== 'result'" v-for="(item, index) of fIconList(iconList)" :key="index"
                :style="{ 'background-color': item.isOperate ? '#eaf1f0' : '' }" class="box-border p-4px">
                <NPopover trigger="hover">
                  <template #trigger>
                    <div class="item_operate_icon" @click="setMode(item.mode, item.sign, item)">
                      <SvgIcon :local-icon="item.localIcon" class="text-[20px]" :style="{
                        color: item.isOperate ? '#2d7a67' : '',
                      }"></SvgIcon>
                    </div>
                  </template>
                  <span>{{ item.name }}</span>
                </NPopover>
              </div>
              <div class="flex-1">
                <div class="w-full flex items-center justify-end">
                  <div class="w-auto flex items-center">
                    <!-- <span>第{{ imgParams.page }}页</span>
                    <span class="mx-4px">/</span>
                    <span v-if="imgPages">共{{ imgPages }}页</span>
                    <span v-else>共{{ 1 }}页</span> -->
                    <n-input-group>
                      <div class="flex items-center mr-8px">{{ imgParams.page }} / {{ imgPages }}页</div>
                      <n-input v-model:value="imgParams.modelPage" type="text" placeholder="请输入页码"
                        class="!w-100px ml-8px" />
                      <n-button type="primary" ghost @click="handlePageJump(imgParams.modelPage)">
                        跳转
                      </n-button>
                    </n-input-group>
                  </div>
                  <div class="flex justify-center items-center" v-if="
                    route.query.anoType === 'online' ||
                    route.query.anoType === 'setOnline' ||
                    route.query.anoType === 'validateUser' ||
                    !route.query.anoType
                  ">
                    <div class="ml-4px h-full flex items-center">
                      <NCheckbox v-model:checked="state.isInvalid" class="flex items-center"
                        @update:checked="handleInvalidCheck">
                        <!-- <NButton quaternary class="px-0">标记为无效数据</NButton> -->
                        <n-tooltip>
                          <template #trigger>
                            <NButton quaternary class="px-0">无效</NButton>
                          </template>
                          标记为无效数据
                        </n-tooltip>
                      </NCheckbox>
                    </div>
                    <n-tooltip v-if="state.isInvalid">
                      <template #trigger>
                        <NButton size="small" quaternary type="info" @click="saveAno('1')">保存
                        </NButton>
                      </template>
                      点击保存当前标注
                    </n-tooltip>
                    <n-tooltip v-else>
                      <template #trigger>
                        <NButton quaternary size="small" type="info" @click="saveAno('1')">保存</NButton>
                      </template>
                      点击保存当前标注
                    </n-tooltip>
                    <n-tooltip>
                      <template #trigger>
                        <NButton quaternary type="default" size="small" @click="clearAno()">取消</NButton>
                      </template>
                      点击取消当前标注
                    </n-tooltip>
                  </div>
                  <div class="flex justify-center items-center" v-if="shouldShowButtons">
                    <!-- <NButton quaternary type="info" @click="handlePass('1')">通过</NButton>
                    <NButton quaternary type="default" @click="handlePass('2')">不通过</NButton> -->
                    <n-popconfirm @positive-click="handlePass('1')">
                      <template #trigger>
                        <NButton quaternary type="info">通过</NButton>
                      </template>
                      是否确认通过？
                    </n-popconfirm>
                    <NButton quaternary type="default" @click="handlePass('2')">不通过</NButton>
                  </div>
                </div>
              </div>
            </div>
            <div class="main_map h-[72%] w-full flex items-center justify-center overflow-hidden">
              <div id="main_map" ref="main_map"
                class="relative h-full w-[88%] flex items-center justify-center object-cover">
                <div v-show="visible" class="mask-layer">
                  <div class="loading-spinner">
                    <NSpin size="large" description="自动标注中... 请稍等" />
                  </div>
                </div>
                <div v-show="isOperate" class="mask-layer">
                  <div class="loading-spinner">
                    <NSpin size="large" description="后台保存中.. 请稍等" />
                  </div>
                </div>
                <div v-show="isRender" class="mask-layer">
                  <div class="loading-spinner">
                    <NSpin size="large" description="" />
                  </div>
                </div>
                <div id="map" ref="map" :style="GetWindowInfo">
                  <ContextMenu v-model:show="ctxMenuConfig.show" :options="ctxMenuConfig.optionsComponent">
                    <ContextMenuItem label="Simple item" @click="onMenuClick(1)" />
                    <ContextMenuSperator />
                  </ContextMenu>
                </div>
              </div>
              <div v-if="state.isInvalid" class="invalid-tip">无效数据</div>
            </div>
            <!-- imgList change -->
            <div class="flex-1 w-full flex items-center justify-center border-t-1 border-t-[#eee]">
              <!-- left -->
              <div class=" left w-[10%] flex items-center justify-center" @click="handleSwiperChange('prev')">
                <div>
                  <SvgIcon local-icon="teenyicons--left-solid" class="text-[32px]"></SvgIcon>
                </div>
              </div>
              <!-- img container -->
              <div class="center h-full w-0 flex flex-1 overflow-x-auto py-8px box-border">
                <div class="grid grid-cols-2 gap-4 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-8">
                  <div v-for="(item, index) in imgList" :key="index"
                    :class="['bg-white p-8px rounded-lg shadow-md flex items-center justify-center', imgActiveIndex === item.acticeIdx ? 'border-2 border-[#2468f2]' : '']"
                    @click="handleImgChange(item)">
                    <img v-if="!!item.previewImgPath" :src="item.previewImgPath"
                      class="w-full h-auto object-contain rounded-lg" />
                    <img v-else :src="item.imgSrc" class="w-full h-auto object-contain rounded-lg" />
                  </div>
                </div>
              </div>
              <!-- right -->
              <div class="right w-[10%] flex items-center justify-center" @click="handleSwiperChange('next')">
                <div>
                  <SvgIcon local-icon="teenyicons--right-solid" class="text-[32px]"></SvgIcon>
                </div>
              </div>
            </div>
          </div>
          <!-- 标签列表 已打 -->
          <div class="h-full w-20% flex flex-col items-center justify-start border-l-1 border-l-[#eee]" :style="{
            'cursor': route.query.anoType === 'audit' ? 'not-allowed' : 'pointer'
          }">
            <!-- 已打标签列表 -->
            <div class="boder-b-[#eee] h-[42px] w-full flex items-center gap-8px border-b-1 px-24px">
              <div class="w-full flex items-center justify-center gap-4px">
                <NButton type="primary" quaternary class="prev" size="small" @click="handleRTabChange('0')">
                  已打标签
                </NButton>
              </div>
            </div>
            <div class="w-full flex-1 overflow-y-auto" ref="targetNodeRef">
              <div
                class="box-border w-full h-auto flex flex-nowrap flex-col items-start justify-start gap-8px px-8px py-16px">
                <div v-for="(item, index) of concatFeatureList" :key="index" :style="{
                  border:
                    tagCurrentIdx == index
                      ? '2px solid #2d7a67'
                      : '1px solid #eee',
                }"
                  class="mb-8px box-border h-40px h-auto w-full flex items-center justify-between overflow-hidden border-1 border-[#eee] rounded-[4px] px-4px py-4px">
                  <div @click="handleTagClick(item, index)" class="h-full w-auto flex items-center cursor-pointer">
                    <div class="mr-8px w-10px h-14px" :style="{ background: item.style.fillStyle }"></div>
                    <span v-if="item.props.name" class="">{{
                      item.props.name
                      }}</span>
                    <span v-else class="text-[red] cursor-pointer">暂无标签</span>
                  </div>
                  <div class="flex justify-center items-center gap-8px"
                    v-if="route.query.anoType !== 'audit'">
                    <div @click="handleFeaOperate('eye', item, index)" v-show="route.query.anoType !== 'result'">
                      <SvgIcon :localIcon="item.isEye ? 'mdi--eye' : 'ion--eye-off-sharp'
                        " class="text-[20px]">
                      </SvgIcon>
                    </div>
                    <div @click="handleFeaOperate('delete', item, index)" v-show="route.query.anoType !== 'result'">
                      <SvgIcon localIcon="material-symbols-light--delete-outline" class="text-[20px]"></SvgIcon>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
          <!-- 标签选择 -->
          <div v-if="false" class="h-full w-20% border-l-1 border-l-[#eee]">
            <div class="h-[10%] w-full border-b-1 border-b-[#eeee]">
              <div v-if="!tagConfig.isEditTag" class="h-full w-full flex items-center justify-around">
                <div class="flex items-center text-[16px] text-[#000]">
                  标签栏
                </div>
                <div class="flex items-center gap-4px">
                  <NButton type="primary" class="add" size="small" @click="handleAddTag('tag')">自定义标签</NButton>
                  <NButton type="primary" class="add" size="small" @click="handleAddTag()">选择标签组</NButton>
                </div>
              </div>
              <div v-else class="box-border h-full w-full flex items-center justify-center">
                <div v-if="tagConfig.sign === 'group'" class="select w-[94%]">
                  <NSelect ref="selectRef" v-model:value="tagConfig.params.val" :show="tagConfig.isShow"
                    placeholder="请选择标签组" :options="tagConfig.options" @update:value="handleSelectChange"
                    @focus="handleSelectFocus">
                    <template #action>
                      <div class="w-full flex items-center justify-between">
                        <div class="l w-60% flex items-center">
                          <NButton quaternary type="info" size="tiny" @click="navToTagGroup()">创建标签组</NButton>
                        </div>
                        <div class="r w-35% flex items-center justify-end">
                          <NButton quaternary type="info" size="tiny" @click="handleDefine()">确定</NButton>
                          <NButton quaternary size="tiny" @click="handleCancel()">取消</NButton>
                        </div>
                      </div>
                    </template>
                  </NSelect>
                </div>
                <div v-if="tagConfig.sign === 'tag'" class="select w-[94%] flex items-center">
                  <div class="flex items-center">
                    <NColorPicker v-model:value="tagConfig.params.color" :show-alpha="false" class="custom-color-picker"
                      :actions="['confirm']" />
                  </div>
                  <div class="w-[70%]">
                    <NInput v-model:value="tagConfig.params.val" class="w-full" placeholder="请输入标签名" />
                  </div>
                  <div class="ml-4px w-[25%] flex items-center">
                    <NButton quaternary type="info" size="tiny" @click="handleDefine()">确定</NButton>
                    <NButton quaternary size="tiny" @click="handleCancel()">取消</NButton>
                  </div>
                </div>
              </div>
            </div>
            <div class="box-border h-[12%] w-full flex items-center border-b-1 border-b-[#eee] px-[24px]">
              <NInput placeholder="搜索" class="h-50% w-full flex items-center" @input="handleInput">
                <template #suffix>
                  <NIcon :component="FlashOutline" />
                </template>
              </NInput>
            </div>
            <div v-if="tagConfig.tagList.length !== 0"
              class="box-border h-[75%] w-full overflow-y-auto px-24px py-24px">
              <div v-for="(item, index) of tagConfig.tagList" :key="index"
                class="mb-8px box-border h-40px w-full flex items-center overflow-hidden border-1 border-[#eee] rounded-[4px]"
                @click="handleTagActive(index)">
                <div class="mr-8px h-full w-10px" :style="{ background: item.color }"></div>
                <!--default-->
                <div v-show="item.isOperate" class="default h-full w-full flex items-center"
                  @mouseenter="handleTagMoute('enter', item)" @mouseleave="handleTagMoute('leave', item)">
                  <div class="mr-8px flex items-center">
                    <NCheckbox v-model:checked="item.isCheck" />
                  </div>
                  <div class="w-[80%]">
                    <span>{{ item.name }}</span>
                    <span class="ml-4px">(ID: {{ item.lableId }})</span>
                  </div>
                  <div v-show="item.isHover" class="h-full w-[20%] flex items-center gap-[8px]">
                    <div @click="handleTagOperate('edit', item)">
                      <!--<SvgIcon icon="lucide:edit" class="text-[16px]"></SvgIcon>-->
                      <SvgIcon local-icon="lucide--edit" class="text-[16px]"></SvgIcon>
                    </div>
                    <div @click="handleTagOperate('delete', item)">
                      <SvgIcon local-icon="delete" class="text-[20px]"></SvgIcon>
                    </div>
                  </div>
                  <!--
 <div v-show="!item.isHover" class="mr-16px h-full w-[20%] flex items-center justify-end gap-[8px]">
                    {{ item.count }}
                  </div>
-->
                </div>
                <div v-show="!item.isOperate" class="h-full w-full flex items-center">
                  <div class="item_ipt_con h-full w-[80%] flex items-center">
                    <NInput v-model:value="item.name" type="text" placeholder="" class="border-none outline-none" />
                  </div>
                  <div class="item_ipt_con h-full w-[20%] flex items-center">
                    <NButton quaternary type="info" size="tiny" @click="handleTagOperate('confirm', item)">
                      确定
                    </NButton>
                    <NButton quaternary type="default" size="tiny" @click="handleTagOperate('cancel', item)">
                      取消
                    </NButton>
                  </div>
                </div>
              </div>
            </div>
            <div v-else class="box-border h-[75%] w-full flex flex-col items-center justify-center px-24px py-24px">
              <img :src="noTag" alt="" />
              <div class="mt-24px text-[14px] text-[#666]">
                暂无可用标签 ，请点击上方按钮添加！
              </div>
              <!--<div class="cursor-pointer" @click="navToTagGroup()">跳转</div>-->
            </div>
          </div>
        </div>
      </NCard>
    </div>
    <!-- 打标签 -->
    <NModal v-model:show="tagModalVisible" :close-on-esc="false" class="wrap-tag-modal">
      <NCard style="width: 720px" title="" :bordered="false" size="huge" role="dialog" aria-modal="true">
        <template #header-extra></template>
        <div class="h-full w-full">
          <div class="my-16px h-[15%] w-full">
            <div v-if="!tagConfig.isEditTag" class="h-full w-full flex items-center justify-around">
              <div class="flex items-center text-[16px] text-[#000]">
                标签组选择
              </div>
              <div class="flex items-center gap-4px" v-show="route.query.anoType !== 'validateUser'">
                <NButton type="primary" class="add" size="small" @click="handleAddTag('tag')">自定义标签</NButton>
                <NButton type="primary" class="add" size="small" @click="handleAddTag()">选择标签组</NButton>
              </div>
            </div>
            <div v-else class="box-border h-full w-full flex items-center justify-center">
              <div v-if="tagConfig.sign === 'group'" class="select w-full">
                <NSelect ref="selectRef" v-model:value="tagConfig.params.val" placeholder="请选择标签组"
                  :show="tagConfig.isShow" :options="tagConfig.options" @update:value="handleSelectChange"
                  @focus="handleSelectFocus">
                  <template #action>
                    <div class="w-full flex items-center justify-between">
                      <div class="l w-60% flex items-center">
                        <NButton quaternary type="info" size="tiny" @click="navToTagGroup()">标签组管理</NButton>
                      </div>
                      <div class="r w-35% flex items-center justify-end">
                        <NButton quaternary type="info" size="tiny" @click="handleDefine()">确定</NButton>
                        <NButton quaternary size="tiny" @click="handleCancel()">取消</NButton>
                      </div>
                    </div>
                  </template>
                </NSelect>
              </div>
              <div v-if="tagConfig.sign === 'tag'" class="select w-[94%] flex items-center">
                <div class="flex items-center">
                  <NColorPicker v-model:value="tagConfig.params.color" :show-alpha="false" class="custom-color-picker"
                    :actions="['confirm']" />
                </div>
                <div class="w-[70%] ml-16px">
                  <NInput v-model:value="tagConfig.params.val" class="w-full" placeholder="请输入标签名" />
                  <!-- @blur.enter="validateEnglishVal" -->
                  <NInput v-model:value="tagConfig.params.englishVal" class="w-full mt-4px" placeholder="请输入标签名(英文名)" />
                </div>
                <div class="ml-4px w-[25%] flex-col items-center">
                  <NButton quaternary type="info" size="" @click="handleDefine()">确定</NButton>
                  <NButton quaternary size="" @click="handleCancel()">取消</NButton>
                </div>
              </div>
            </div>
          </div>

          <!-- <div class="box-border h-[12%] w-full flex items-center">
            <NInput placeholder="搜索" class="h-50% w-full flex items-center" @input="handleInput">
              <template #suffix>
                <NIcon :component="FlashOutline" />
              </template>
            </NInput>
          </div> -->

          <div v-if="tagConfig.tagList.length !== 0" class="mt-24px box-border h-[360px] w-full overflow-y-auto">
            <div v-for="(item, index) of tagConfig.tagList" :key="index"
              class="mb-8px box-border h-40px w-full flex items-center overflow-hidden border-1 border-[#eee] rounded-[4px]"
              @click="handleTagActive(index)">
              <div class="mr-8px h-full w-10px" :style="{ background: item.color }"></div>
              <!--default-->
              <div v-show="item.isOperate" class="default h-full w-full flex items-center"
                @mouseenter="handleTagMoute('enter', item)" @mouseleave="handleTagMoute('leave', item)">
                <div class="mr-8px flex items-center">
                  <NCheckbox v-model:checked="item.isCheck" />
                </div>
                <div class="w-[80%]">
                  <span>{{ item.name }}</span>
                  <span class="ml-4px">(ID: {{ item.labelId }})</span>
                </div>
                <div v-show="item.isHover && route.query.anoType !== 'validateUser'
                  " class="h-full w-[20%] flex items-center gap-[8px]">
                  <div @click="handleTagOperate('edit', item)">
                    <SvgIcon local-icon="lucide--edit" class="text-[16px]"></SvgIcon>
                  </div>
                  <div @click="handleTagOperate('delete', item)">
                    <SvgIcon icon="material-symbols-light:delete-outline" class="text-[20px]"></SvgIcon>
                  </div>
                </div>
              </div>
              <div v-show="!item.isOperate" class="h-full w-full flex items-center">
                <div class="item_ipt_con h-full w-[80%] flex items-center gap-8px">
                  <NInput v-model:value="item.twoLabelName" type="text" placeholder="请输入标签名"
                    class="border-none outline-none" />
                  <NInput v-model:value="item.englishLabelName" type="text" placeholder="请输入标签英文名"
                    class="border-none outline-none" />
                </div>
                <div class="item_ipt_con h-full w-[20%] flex items-center ml-8px">
                  <NButton quaternary type="info" size="tiny" @click="handleTagOperate('confirm', item)">确定</NButton>
                  <NButton quaternary type="default" size="tiny" @click="handleTagOperate('cancel', item)">
                    取消
                  </NButton>
                </div>
              </div>
            </div>
          </div>
          <div v-else class="box-border h-[75%] w-full flex flex-col items-center justify-center px-24px py-24px">
            <img :src="noTag" alt="" />
            <div class="mt-24px text-[14px] text-[#666]">
              暂无可用标签 ，请点击上方按钮添加！
            </div>
            <!--<div class="cursor-pointer" @click="navToTagGroup()">跳转</div>-->
          </div>
        </div>
        <template #footer>
          <div class="h-auto w-full flex items-center justify-end gap-24px">
            <NButton type="primary" @click="handleModalDefine('tag')">确定</NButton>
            <NButton @click="handleModalCancel('tag')">取消</NButton>
          </div>
        </template>
      </NCard>
    </NModal>
    <!--验收-->
    <NModal v-model:show="isShowValidate" :close-on-esc="false" class="wrap-tag-modal">
      <NCard :title="validateTitle" class="w-640px" :bordered="false" size="huge" role="dialog" aria-modal="true">
        <NForm ref="formRef" :model="validateModel">
          <div class="h-full w-full">
            <div v-if="validateStatus === '1'">
              <NFormItem label="保存类型" path="type">
                <NRadioGroup v-model:value="validateModel.verifyState">
                  <NRadio v-for="item in statusOptions" :key="item.value" :value="item.value" :label="item.label" />
                </NRadioGroup>
              </NFormItem>
              <div class="bg-[#fff4e6] p-8px py-8px">
                注:验收完成后任务就会结束 不能再进行操作
              </div>
            </div>
            <div v-if="validateStatus === '2'">
              <NFormItem label="打回类型" path="repulseType">
                <NRadioGroup v-model:value="validateModel.returnState">
                  <NRadio v-for="item in repulseOptions" :key="item.value" :value="item.value" :label="item.label" />
                </NRadioGroup>
              </NFormItem>
            </div>
          </div>
        </NForm>
        <template #footer>
          <div class="h-auto w-full flex items-center justify-end gap-24px">
            <NButton type="primary" @click="handleModalDefine('validate')">确定</NButton>
            <NButton @click="handleModalCancel('validate')">取消</NButton>
          </div>
        </template>
      </NCard>
    </NModal>
    <!-- 不通过意见modal -->
    <NModal v-model:show="noPassVisable" :close-on-esc="false" class="wrap-tag-modal">
      <NCard style="width: 520px" title="验收意见" :bordered="false" size="huge" role="dialog" aria-modal="true">
        <div class="w-full h-full">
          <n-input v-model:value="validateModel.message" type="textarea" placeholder="请输入验收意见" />
        </div>
        <template #footer>
          <div class="h-auto w-full flex items-center justify-end gap-24px">
            <NButton type="primary" @click="handleModalDefine('pass')">添加意见</NButton>
            <NButton @click="handleModalCancel('pass0')">无意见</NButton>
            <!-- 取消 -->
            <NButton @click="() => noPassVisable = false">取消</NButton>
          </div>
        </template>
      </NCard>
    </NModal>
    <!-- 提交任务 -->
    <n-modal :show="submitShowModal">
      <n-card style="width: 600px" title="提交任务" size="huge" :bordered="false" role="dialog" aria-modal="true">
        <div>{{ submitTooltipText }}</div>
        <template #footer>
          <div class="h-auto w-full flex items-center justify-end gap-24px">
            <NButton type="primary" @click="handleSubmitDefine()">确定提交</NButton>
            <NButton @click="handleSubmitClose()">关闭窗口</NButton>
          </div>
        </template>
      </n-card>
    </n-modal>
  </div>
</template>

<style scoped lang="scss">
:deep(.wrap_tabs) {
  .n-tabs-nav-scroll-content {
    border: none !important;
  }

  .n-tabs-pane-wrapper {
    display: none !important;
  }
}

#icon_delete {
  width: 24px;
  height: 24px;
}

:deep(.custom-color-picker) {
  width: 24px !important;
  height: 24px !important;

  .n-color-picker-trigger {
    border: none !important;
  }

  .n-color-picker-trigger__value {
    display: none;
  }
}

.main_map {
  position: relative;

  .invalid-tip {
    position: absolute;
    z-index: 1;
    left: 0;
    top: 0;
    width: 160px;
    height: 160px;
    background-color: rgba(0, 0, 0, 0.4);
    color: #fff;
    font-size: 32px;
    padding: 40px;
    text-align: center;
    line-height: 40px;

    &:before {
      position: absolute;
      content: " ";
      width: 40px;
      height: 20px;
      bottom: -80px;
      overflow: hidden;
      left: 0;
      border-left: 0;
      border-right: 80px solid transparent;
      border-top: 40px solid rgba(0, 0, 0, 0.4);
      border-bottom: 40px solid transparent;
    }

    &:after {
      position: absolute;
      content: " ";
      width: 40px;
      height: 20px;
      bottom: -80px;
      overflow: hidden;
      right: 0;
      border-left: 80px solid transparent;
      border-right: 0;
      border-top: 40px solid rgba(0, 0, 0, 0.4);
      border-bottom: 40px solid transparent;
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

:deep(.n-card__content) {
  height: 100% !important;
  // padding: 0 !important;
}
</style>
