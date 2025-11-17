<script setup lang="ts">
import AILabel from "ailabel";
import { nanoid } from "nanoid";
import { NButton } from "naive-ui";
// import { useMouseInElement } from "@vueuse/core";
import _ from "lodash";
// import WebSocket from 'ws';
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
  selectLabelList,
} from "@/service/api/ano";
import { fetchLabelEdit } from "@/service/api/tag";

// ts 后期做抽离吧 留给你们大佬做吧  我3.5年经验给开几千块钱 没那个必要给他做
interface IconInfo {
  name: string;
  icon: string;
  mode: string;
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
  },
  {
    name: "背景点标注",
    icon: "ic:sharp-auto-fix-off",
    localIcon: "ic--sharp-auto-fix-off",
    mode: "POINT",
    sign: "background_points",
    isOperate: false,
  },
  // {name: '图像居中', icon: 'carbon:center-to-fit', mode: 'Center'},
  {
    name: "多边形",
    icon: "uil:polygon",
    mode: "POLYGON",
    localIcon: "uil--polygon",
    isOperate: false,
  },
  {
    name: "矩形",
    icon: "ph:rectangle-bold",
    mode: "RECT",
    localIcon: "ph--rectangle-bold",
    isOperate: false,
  },
  {
    name: "圆形",
    icon: "material-symbols:circle-outline",
    mode: "CIRCLE",
    localIcon: "material-symbols--circle-outline",
    isOperate: false,
  },
  {
    name: "撤销",
    icon: "fluent:arrow-undo-16-filled",
    mode: "Revoke",
    localIcon: "fluent--arrow-undo-16-filled",
    isOperate: false,
  },
  {
    name: "移动",
    localIcon: "ic--baseline-pan-tool",
    mode: "PAN",
    isOperate: false,
  },
  // { name: '重做', icon: 'fluent:arrow-redo-32-filled', mode: 'Back' },
  {
    name: "放大",
    icon: "gg:add",
    mode: "zoomIn",
    localIcon: "gg--add",
    isOperate: false,
  },
  {
    name: "缩小",
    icon: "icon-park-outline:reduce-one",
    mode: "zoomOut",
    localIcon: "icon-park-outline--reduce-one",
    isOperate: false,
  },
  // { name: "删除", icon: "material-symbols:delete-outline", mode: "delete" },
  {
    name: "导入图片到该数据集",
    localIcon: "fa6-solid--file-import",
    mode: "import",
    isOperate: false,
  },
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

// 标签modal
const tagModalVisible = ref(false);
const revokeList = ref<any>([]); // 撤销列表-记录用户撤销操作operate

// socket
const socket = ref(null);
const messages = ref([]);
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
// 上一个图片信息
const prevImgInfo = ref<any>({
  fileId: undefined,
  sonId: undefined,
  markFileId: undefined,
  markInfo: undefined,
  labels: undefined,
  operateWidth: undefined,
  operateHeight: undefined,
});

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

// 初始化实例
const initMap = () => {
  const gMap = new AILabel.Map(state.value.divId, {
    center: state.value.centerObj, // 为了让图片居中
    zoom: state.value.zoom, // 初始缩放级别
    mode: "", // 绘制线段
    refreshDelayWhenZooming: true, // 缩放时是否允许刷新延时，性能更优
    zoomWhenDrawing: false, // 绘制时可滑轮缩放
    panWhenDrawing: true, // 绘制时可到边界外自动平移
    zoomWheelRatio: 5, // 控制滑轮缩放缩率[0, 10), 值越小，则缩放越快，反之越慢
    withHotKeys: false, // 关闭快捷键
  });
  state.value.gMap = gMap;
};
// 添加 text 文本图层，用于展示文本
const setGFirstTextLayer = () => {
  const gFirstTextLayer = new AILabel.Layer.Text(
    "first-layer-text", // id
    { name: "第一个文本图层" }, // props
    { zIndex: 12, opacity: 1 } // style
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
    { zIndex: 10 } // style
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
    { zIndex: 5 } // style
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
  state.value.allFeatures = state.value.gFirstFeatureLayer.getAllFeatures();
};
// 添加文本
const addLayerText = (
  textId: string | number,
  textName: any,
  textPosition: any
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
    }
  );
  state.value.gFirstTextLayer.addText(gFirstText);
};
// 添加图形
const addFeature = (data: any, type: string, id: any) => {
  const drawingStyle = state.value.drawingStyle;
  // const name: any = id;
  // let checkTagList;
  // let color;
  // let name;
  // let relatedDeleteMarkerId;
  // if (type === "CIRCLE" || type === "RECT" || type === "POLYGON") {
  //   checkTagList = tagConfig.value.tagList.filter((item) => item.isCheck);
  //   name = checkTagList[0].name;
  //   color = checkTagList[0].color;
  //   const customId = nanoid();
  //   id = `${checkTagList[0].labelId}-${customId}`;
  //   relatedDeleteMarkerId = `${checkTagList[0].labelId}-${customId}`;
  // }
  // ----------------------------------------------------------------------------------------
  // newCode
  // 线
  if (type === "LINE") {
    const scale = state.value.gMap.getScale();
    const width = drawingStyle.lineWidth / scale;
    const lineFeature = new AILabel.Feature.Line(
      feaDefaultCon.value.id,
      { ...data, width },
      { name: feaDefaultCon.value.name, textId: id },
      drawingStyle
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
      drawingStyle // style
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
      drawingStyle // style
    );
    state.value.gFirstFeatureLayer.addFeature(rectFeature);
    // 多边形 添加 文本
    const textPosition = setText(data, type);
    addLayerText(
      feaDefaultCon.value.id,
      feaDefaultCon.value.name,
      textPosition
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
      drawingStyle // style
    );
    state.value.gFirstFeatureLayer.addFeature(polygonFeature);
    // 多边形 添加 文本
    const textPosition = setText(data, type);
    addLayerText(
      feaDefaultCon.value.id,
      feaDefaultCon.value.name,
      textPosition
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
      drawingStyle // style
    );
    state.value.gFirstFeatureLayer.addFeature(gFirstFeatureCircle);
    // 多边形 添加 文本
    const textPosition = setText(data, type); // {x: 31.1640696608616, y: 48.89036653397688}
    addLayerText(
      feaDefaultCon.value.id,
      feaDefaultCon.value.name,
      textPosition
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
      { fillStyle: color, zIndex: 5, lineWidth: 2 } // style
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
      { name: "第一个marker注记", textId: id } // props
    );
    state.value.gFirstFeatureLayer.addFeature(gFirstMarker);
    // 多边形 添加 文本
    const textPosition = setText(data, type);
    addLayerText(id, name, textPosition);
  }
  updateMarkInfo();
  getFeatures();
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
  imgList.value[imgActiveIndex.value].markInfo = Array.isArray(markDataList) && markDataList.length > 0
    ? JSON.stringify(markDataList)
    : ""
}

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
      drawingStyle
    );
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
      drawingStyle // style
    );
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
      drawingStyle // style
    );
    state.value.gFirstFeatureLayer.addFeature(polygonFeature);
    // 多边形 添加 文本
    const textPosition = setText(data, type);
    addLayerText(id, name, textPosition);
  }
};

// 增加删除图标
// eslint-disable-next-line complexity
const addDeleteIcon = (feature: any, shape: any) => {
  const iconDelete = state.value.gMap.markerLayer.getMarkerById("icon_delete");
  if (iconDelete) {
    state.value.gMap.markerLayer.removeMarkerById("icon_delete");
  }
  // 添加delete-icon
  const x =
    (shape?.x || shape?.cx || shape?.start?.x || shape?.points[0].x) +
    (shape?.width || shape?.r || 0);
  const y =
    (shape?.y || shape?.cy || shape?.start?.y || shape?.points[0].y) - 15;
  const gFirstMarker = new AILabel.Marker(
    "icon_delete", // id  feature.props.deleteMarkerId
    {
      // src: "./iconDelete.png",
      src: "https://s1.ax1x.com/2022/06/20/XvFRbT.png",
      position: { x, y }, // 矩形右上角 根据图形动态调整
      // position: feature.getPoints()[1], // 矩形右上角 根据图形动态调整
      offset: {
        x: -20,
        y: -4,
      },
      featureId: feature.id,
    }, // markerInfo
    { name: "delete" } // props
  );
  gFirstMarker.image.style.width = "18px";
  gFirstMarker.image.style.minWidth = "18px";
  gFirstMarker.image.style.height = "18px";
  gFirstMarker.image.style.src = "./iconDelete.png";
  gFirstMarker.events.on("click", (marker) => {
    // 首先删除当前marker
    state.value.gMap.markerLayer.removeMarkerById(marker.id);
    // 删除对应text
    state.value.gFirstTextLayer.removeTextById(feature.props.textId);
    // 删除对应feature
    state.value.gFirstFeatureLayer.removeFeatureById(feature.id);
    getFeatures();
  });
  const gMap = state.value.gMap;
  gMap.markerLayer.addMarker(gFirstMarker);
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
const renderPointData = async (sign: any, value: any[]) => {
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
      }
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
    visible.value = false;
  }
};

// 加工point坐标信息
const mapPointData1 = (
  { shapes, imageWidth, imageHeight },
  revokeId: string
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

// tooltip offset config
const hTooltipOffsetConfig = (type: any, data: any) => {
  let offsetX = 0;
  let offsetY = 0;
  switch (type) {
    case "POINT":
      offsetX = -data.width / 2;
      offsetY = -data.height / 2;
      break;
    case "CIRCLE":
      offsetX = data.cx + 250 + data.r;
      offsetY = data.cy;
      break;
    case "RECT":
      offsetX = data.x;
      offsetY = data.y;
      break;
    case "POLYGON":
      // 找到data数组中最大的x y
      const maxX = Math.max(...data.map((item) => item.x));
      const maxY = Math.max(...data.map((item) => item.y));
      offsetX = maxX;
      offsetY = maxY;
      break;
    default:
      break;
  }
  return { offsetX, offsetY };
};
// 增加事件
const addEvent = () => {
  const gMap = state.value.gMap;
  gMap.events.on("drawDone", (type: string, data: any) => {
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
        renderPointData("foreground_points", state.value.foreground_points);
      }
      if (state.value.pointSign === "background_points") {
        state.value.background_points = [
          ...state.value.background_points,
          // [x, y],
          [data.x, data.y],
        ];
        renderPointData("background_points", state.value.background_points);
      }
    } else {
      // setTimeout(() => {
      //   tagConfig.value.isShow = true;
      //   tagConfig.value.isEditTag = false;
      //   // tagModalVisible.value = true;
      // }, 50);
      tagConfig.value.isShow = true;
      tagConfig.value.isEditTag = false;
    }
    // vue3 context menu
    // const { offsetX, offsetY } = hTooltipOffsetConfig(type, data);
    // const mapData = {
    //   x: offsetX,
    //   y: offsetY,
    // };
    // showContextMenu(mapData);
    // --------------------------------------
  });
  // 双击编辑 在绘制模式下双击feature触发选中
  gMap.events.on("featureSelected", (feature: any) => {
    state.value.editId = feature.id;
    operateId.value = feature.id;
    // 设置编辑feature
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
const Revoke = () => {
  if (revokeList.value.length > 0) {
    getFeatures();
    // const descRevokeList = revokeList.value.reverse().filter(val => val);
    // descRevokeList.forEach((item, index) => {
    //   customAddFeature(item);
    //   revokeList.value.splice(index, 1);
    // })
    const popRow = revokeList.value.pop();
    customAddFeature(popRow);
  } else {
    getFeatures();
    if (state.value.allFeatures.length == 0) return;
    const feature = state.value.allFeatures.pop();
    const popInfo = feature;
    state.value.gFirstTextLayer.removeTextById(popInfo.props.textId); // 删除文本
    if (popInfo.props && popInfo.props.revokeId) {
      const comFeatures = state.value.allFeatures.filter((val: any) => {
        return val.props.revokeId == popInfo.props.revokeId;
      });
      // 遍历comFeatures 根据props.textId删除gFirstTextLayer
      comFeatures.forEach((val: any) => {
        state.value.gFirstTextLayer.removeTextById(val.props.textId);
      });
      // 过滤 allFeatures  删除revokeId 相同的feature
      state.value.allFeatures = state.value.allFeatures.filter((val: any) => {
        return val.props.revokeId !== popInfo.props.revokeId;
      });
    }
    if (popInfo.type === "POINT") {
      // filter 过滤前景背景点坐标
      const filterArr = [popInfo.shape.x, popInfo.shape.y];
      if (popInfo.props.sign === "foreground_points") {
        state.value.foreground_points = state.value.foreground_points.filter(
          (val: any) => {
            return val[0] != filterArr[0] && val[1] != filterArr[1];
          }
        );
      }
      if (popInfo.props.sign === "background_points") {
        state.value.background_points = state.value.background_points.filter(
          (val: any) => {
            return val[0] != filterArr[0] && val[1] != filterArr[1];
          }
        );
      }
    }
    state.value.gMap.refresh(); // 刷新map
  }
};
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
      updateMarkInfo();
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
};
const cleanMapLayer = async () => {
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
  { id, type, props, shape, style }: any,
  randomId: any,
  revokeId: any
) => {
  state.value.gMap && state.value.gMap.setMode(type);
  const drawingStyle = style;
  const { textId, deleteMarkerId, operateWidth, operateHeight } = props;
  let name;
  const isTagId = tagConfig.value.tagList.some(
    (val) => val.labelId == id.split("-")[0]
  );
  if (id) {
    name =
      id === "autoAno"
        ? ""
        : isTagId
          ? tagConfig.value.tagList.find((val) => val.labelId == id.split("-")[0])
            .labelName
          : "";
  } else {
    name = id === "audoAno" ? "" : "";
  }
  name = props.name ? props.name : isTagId
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
      // deviceRadio
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
        drawingStyle // style
      );
      state.value.gFirstFeatureLayer.addFeature(rectFeature);
      addLayerText(id, name, textPosition);
      // addDelIcon(id, textId, shape);
      break;
    case "CIRCLE":
      if (isMapAno) {
        textPosition = setText(
          {
            cx: data.cx * deviceWRadio,
            cy: data.cy * deviceHRadio,
            r: data.r * deviceWRadio,
          },
          type
        );
        const gFirstFeatureCircle = new AILabel.Feature.Circle(
          id, // id
          // { cx: data.cx, cy: data.cy, r: data.r }, // shape
          {
            cx: data.cx * deviceWRadio,
            cy: data.cy * deviceHRadio,
            r: data.r * deviceWRadio,
          }, // shape radio
          { name, textId: id, deleteMarkerId }, // props
          drawingStyle
        );
        state.value.gFirstFeatureLayer.addFeature(gFirstFeatureCircle);
        addLayerText(id, name, textPosition);
        // addDelIcon(id, textId, shape);
      } else {
        const wRadio = state.value.sWidth / state.value.nWidth;
        const hRadio = state.value.sHeight / state.value.nHeight;
        textPosition = setText(
          {
            cx: (data.cx / wRadio) * deviceWRadio,
            cy: (data.cy / hRadio) * deviceHRadio,
            r: (data.r / wRadio) * deviceWRadio,
          },
          type
        );
        const gFirstFeatureCircle = new AILabel.Feature.Circle(
          id, // id
          // { cx: data.cx, cy: data.cy, r: data.r }, // shape
          {
            cx: (data.cx / wRadio) * deviceWRadio,
            cy: (data.cy / hRadio) * deviceHRadio,
            r: (data.r / wRadio) * deviceWRadio,
          }, // shape radio
          { name, textId: id, deleteMarkerId }, // props
          drawingStyle
        );
        state.value.gFirstFeatureLayer.addFeature(gFirstFeatureCircle);
        addLayerText(id, name, textPosition);
        // addDelIcon(id, textId, shape);
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
          // { points: data.points }, // shape
          { points }, // shape radio
          {
            name,
            textId: id,
            deleteMarkerId,
            isAutoFit: Boolean(randomId),
            revokeId,
          }, // props
          drawingStyle // style
        );
        state.value.gFirstFeatureLayer.removeFeatureById(polygonId);
        state.value.gFirstFeatureLayer &&
        state.value.gFirstFeatureLayer.addFeature(polygonFeature);
        addLayerText(polygonId, name, textPosition);
        // addDelIcon(id, textId, shape);
      } else {
        const wRadio = state.value.sWidth / state.value.nWidth;
        const hRadio = state.value.sHeight / state.value.nHeight;
        // const wRadio = state.value.operateWidth / state.value.nWidth;
        // const hRadio = state.value.operateHeight / state.value.nHeight;
        // deviceRadio
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
          // { points: data.points }, // shape
          { points }, // shape radio
          { name, textId: id, deleteMarkerId, isAutoFit: Boolean(randomId) }, // props
          drawingStyle // style
        );
        state.value.gFirstFeatureLayer.removeFeatureById(polygonId);
        state.value.gFirstFeatureLayer &&
        state.value.gFirstFeatureLayer.addFeature(polygonFeature);
        addLayerText(polygonId, name, textPosition);
        // addDelIcon(id, textId, shape);
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
          // { x: data.x, y: data.y, r: 5 }, // shape
          // { fillStyle: color, zIndex: 5, lineWidth: 2 }, // style
          {
            x: data.x * deviceWRadio,
            y: data.y * deviceHRadio,
            r: 5 * deviceWRadio,
          }, // shape radio
          { fillStyle: color, zIndex: 5, lineWidth: 2 * deviceWRadio } // style radio
        );
        state.value.gFirstFeatureLayer.addFeature(gFirstFeaturePoint);
        // addEvent();
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
          // { x: data.x, y: data.y, r: 5 }, // shape
          // { fillStyle: color, zIndex: 5, lineWidth: 2 }, // style
          {
            x: (data.x / wRadio) * deviceWRadio,
            y: (data.y / hRadio) * deviceHRadio,
            r: (5 / wRadio) * deviceWRadio,
          }, // shape radio
          {
            fillStyle: color,
            zIndex: 5,
            lineWidth: (2 / wRadio) * deviceWRadio,
          } // style radio
        );
        state.value.gFirstFeatureLayer.addFeature(gFirstFeaturePoint);
        // addEvent();
      }

      break;
    default:
      break;
  }
};

function flushMicrotasks() {
  return new Promise((resolve) => {
    Promise.resolve().then(() => {
      resolve();
    });
  });
}

const renderAllLayer = async (markInfo: any) => {
  const layerList =
    markInfo &&
    Array.isArray(JSON.parse(markInfo)) &&
    JSON.parse(markInfo).length > 0
      ? JSON.parse(markInfo)
      : [];
  layerList.forEach((item: any) => {
    console.log(item);
    customAddFeature(item);
  });
  console.log("-------------------")
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
    color: "#000000",
  },
  options: [],
  isShow: false,
  tagList: [],
  deepTagList: [],
});
const imgParams = reactive({
  page: 1,
  limit: 8,
  sonId: undefined,
  state: undefined,
});
const selectRef = ref(null);
const imgActiveSrc = ref<string>("");
const imgActiveIndex = ref<string>("0");
const imgList: any[] = ref([]);
const imgListTotal = ref("");
const imgData = ref<any>();
const tabConfig = ref<any>({
  state: 0,
  tabNum: {
    all: undefined,
    haveAno: undefined,
    noAno: undefined,
  },
});
const isSwiperChange = ref(false);
const visible = ref(false); // 遮罩mask
const isOperate = ref(false); // 自动保存中

// watch
watch(
  () => tabConfig.value.state,
  (newVal) => {
    imgParams.page = 1;
    imgActiveIndex.value = "0";
    getImgData();
  }
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
  }
);

// active 保存当前标注 button
const isActiveCurMarkBtn = computed(() => {
  return state.value.allFeatures.length === 0;
});

// 第几张
const curImgNum = computed(() => {
  return imgParams.page * (Number(imgActiveIndex.value) + 1);
});
// 共多少张
const imgTotal = ref(0);
const imgPages = ref(0);

// add tag
const handleAddTag = (sign = "group") => {
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
    name: "data-ano_group",
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
    }
  }
  if (tagConfig.value.sign === "tag") {
    const params = {
      sonId: route.query.id,
      labelColor: tagConfig.value.params.color,
      labelName: tagConfig.value.params.val,
    };
    const res = await addSaveLabel(params);
    if (res.data >= 1) {
      window.$message?.success?.("添加标签成功！");
      tagConfig.value.isEditTag = false;
      await getTagList();
    }
  }
};

const getTagList = async () => {
  const res = await selectDataSetLabel({ sonId: route.query.id });
  // const res = await selectLabelList({sonId: route.query.id});
  // const tagObj = {name: '测试001', isOperate: true, tagIdx: `${tagIdx}`, isHover: false};
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
  // deepTagList
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
    const res = await fetchLabelEdit({
      id: labelId,
      labelColor: color,
      labelGroupId,
      labelName: name,
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

// 图片预请求
function preloadImage(url) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.src = url;
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error(`Failed to load image at ${url}`));
  });
}

const getImgData = async () => {
  const { imgIdx } = route.query;
  if (route.query.imgIdx >= 0 && !isSwiperChange.value) {
    // imgParams.page = imgIdx < 7 ? 1 : Math.floor(imgIdx / 7) + 1;
    imgParams.page = imgIdx < 8 ? 1 : Math.floor(imgIdx / 8) + 1;
  }
  // getDataDetails 带分页   getImgDataDetails 无分页
  const params = {
    ...imgParams,
    sonId: route.query.id,
    state: tabConfig.value.state,
  };
  const res = await getDataDetailsNoMarkFilePath(params);
  if (res.data) {
    imgTotal.value = res.data.total;
    imgPages.value = res.data.pages;
    imgList.value = res.data.records.map((item, index) => {
      // prev img request
      // preloadImage(item.imgPath)
      return {
        ...item,
        defaultInfo: item.markInfo,
        imgSrc: item.imgPath,
        acticeIdx: `${index}`,
        previewImgPath: item.previewImgPath,
      };
    });
    state.value.imgUrl = imgList.value[0]?.imgSrc;
    imgActiveSrc.value = imgList.value[0]?.imgSrc;

    // detail to operation 指定图片跳转
    // if (route.query.imgIdx >= 0 && !isSwiperChange.value) {
    //   // const curPageImgIdx = Number(route.query.imgIdx) % 7;
    //   const curPageImgIdx = Number(route.query.imgIdx) % 8;
    //   const curImgData = imgList.value[curPageImgIdx];
    //   handleImgChange(curImgData, "mounted");
    // }
  } else {
    imgTotal.value = 0;
    imgPages.value = 1;
    imgList.value = [];
    state.value.imgUrl = "";
    imgActiveSrc.value = "";
  }
};

const clearAllFeatures = async () => {
  state.value.gFirstFeatureLayer.removeAllFeatures();
}

const handleImgChange = async (item: any, sign: any) => {
  await cleanMapLayer();
  isSwiperChange.value = true;
  revokeList.value = [];
  if (!sign) {
    await asyncPrevImgSave();
  }
  // -----------------------------------------------------------
  const isArrLen = Array.isArray(imgList.value) && imgList.value.length > 0;
  if (visible.value) return;
  if (!isArrLen) return;
  state.value.foreground_points = [];
  state.value.background_points = [];
  state.value.imgUrl = item.imgSrc;
  state.value.sWidth = item.width ? Number(item.width) : ""; // 原图宽
  state.value.sHeight = item.height ? Number(item.height) : ""; // 原图高
  state.value.operateWidth = item.operateWidth ? Number(item.operateWidth) : "";
  state.value.operateHeight = item.operateHeight
    ? Number(item.operateHeight)
    : "";
  imgActiveSrc.value = item.imgSrc;
  imgActiveIndex.value = item.acticeIdx;
  imgData.value = item;
  // 获取图片原宽高
  const url = state.value.imgUrl;
  try {
    const row = await getImageSizeByUrl(url);
    setImgSize(row);
  } catch (e) { }
  // 图片切换 后台保存
  // isOperate.value = true
  if (state.value.isLoadImgSuccess) {
    await setGFirstImageLayer(); // 图片层添加
    await setGFirstFeatureLayer(); // 添加矢量图层
    await setGFirstTextLayer(); // 添加 text 文本图层，用于展示文本
    // setTimeout(() => {
    //   renderAllLayer(item.markInfo);
    // }, 200);
    await renderAllLayer(item.markInfo);
  }
  state.value.gMap && state.value.gMap.refresh();
};

const handleSwiperChange = async (sign: any) => {
  isSwiperChange.value = true;
  if (sign === "prev") {
    if (imgList.value.length > 0 && imgParams.page == 1) {
      window.$message?.error?.("当前为第一页！");
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
    // if (imgList.value.length < 7) {
    if (imgPages.value == imgParams.page) {
      window.$message?.error?.("当前为最后一页！");
      return;
    }

    await asyncPrevImgSave();
    // --------------------------------------
    imgActiveIndex.value = "0";
    state.value.gMap.removeLayerById("first-layer-image");
    imgParams.page =
      // imgList.value.length < 7 ? imgParams.page : imgParams.page + 1;
      imgList.value.length < 8 ? imgParams.page : imgParams.page + 1;
    imgList.value = [];
    getImgData();
  }
};

const handlePrevNextImg = (sign: any) => {
  let imgInfo;
  if (sign === "prev") {
    const index = imgActiveIndex.value ? Number(imgActiveIndex.value) - 1 : 0;
    imgInfo = imgList.value[index];
  }
  if (sign === "next") {
    const index =
      imgActiveIndex.value >= imgList.value.length - 1
        ? imgActiveIndex.value
        : Number(imgActiveIndex.value) + 1;
    imgInfo = imgList.value[index];
  }
  handleImgChange(imgInfo);
};

const exportImg = async (data: any) => {
  const filename = `${new Date().getTime()}.jpg`;
  const file2 = new File([data], filename, { type: "image/jpg" });
  const formData = new FormData();
  const { sonId, version } = imgList.value[imgActiveIndex.value];
  formData.append("file", file2);
  formData.append("sonId", sonId);
  formData.append("version", version);
  const res = await MarkFileUpload(formData);
  if (res.data) {
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
    console.log("markDataList");
    console.log(markDataList);
    const labels = markDataList.map((item) => item.openId);
    const imgData = imgList.value[imgActiveIndex.value];
    const params = {
      // fileId: route.query.fileId || imgData.fileId,
      fileId: imgData.fileId,
      sonId: route.query.id,
      markFileId: res.data,
      markInfo:
        Array.isArray(markDataList) && markDataList.length > 0
          ? JSON.stringify(markDataList)
          : "",
      labels: [...new Set(labels)].join(","),
      operateWidth: state.value.nWidth,
      operateHeight: state.value.nHeight,
    };
    imgList.value[imgActiveIndex.value].markInfo = JSON.stringify(markDataList);
    imgList.value[imgActiveIndex.value].defaultInfo = JSON.stringify(markDataList);
    await addDataMarkInfo(params);
    window.$message?.success?.("保存成功！");
    revokeList.value = [];
    await getDataDetailsCount();
  }
};

const generateImg = async () => {
  const width = state.value.nWidth;
  const height = state.value.nHeight;
  const buffer = await state.value.gMap.exportLayersToImage(
    { x: 0, y: 0, width, height },
    { type: "blob", format: "image/png" }
  );
  exportImg(buffer);
};

const markTooltip = () => {
  window.$message?.success?.("后台标注中...");
};

const getDataDetailsCount = async () => {
  const res = await DataDetailsCount({ sonId: route.query.id });
  if (res.data) {
    tabConfig.value.tabNum.all = res.data.all;
    tabConfig.value.tabNum.haveAno = res.data.haveAno;
    tabConfig.value.tabNum.noAno = res.data.noAno;
  }
};

const saveAno = async (sign: number | string) => {
  if (sign === "0") {
    handleInvalidMark();
  }
  if (sign === "1") {
    await markTooltip();
    await generateImg();
    await getTagList();
  }
};

const clearAno = () => {
  // state.value.gMap && state.value.gMap.removeAllLayers();
  state.value.gFirstFeatureLayer &&
  state.value.gFirstFeatureLayer.removeAllFeatures();
  state.value.gFirstTextLayer && state.value.gFirstTextLayer.removeAllTexts();
  state.value.gMap && state.value.gMap.setActiveFeature(null);
  getFeatures();
};

const handleTabChange = (val: number | any) => {
  tabConfig.value.state = val;
  getDataDetailsCount();
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
    // fileId: route.query.fileId || curImgData.fileId,
    fileId: curImgData.fileId,
    sonId: route.query.id,
    markInfo: "",
    operateWidth: state.value.nWidth,
    operateHeight: state.value.nHeight,
  };
  const res = await addDataMarkInfo(params);
  if (res.data >= 1) {
    window.$message?.success?.("图片标记成功");
    // next
    const nextIdx = Number(imgActiveIndex.value) + 1;
    if (nextIdx === imgList.value.length) {
      window.$message?.error?.("已经是最后一个了");
      return;
    }
    const nextImgData = imgList.value[nextIdx];
    handleImgChange(nextImgData);
  }
};

// vue3 context menu
const showContextMenu = (data: any) => {
  ctxMenuConfig.value.show = true;
  ctxMenuConfig.value.optionsComponent.x = data.x;
  ctxMenuConfig.value.optionsComponent.y = data.y;
};

// modal
const handleModalDefine = async () => {
  const isCheckTag =
    tagConfig.value.tagList.filter((item) => item.isCheck).length > 0;
  if (!isCheckTag) {
    window.$message?.error?.("请先勾选标签！");
    return;
  }
  getFeatures();
  let shape;
  let type;
  let featureId;
  if (checkId.value) {
    // 根据checkId找state.value.allFeatures
    const featureConfig = state.value.allFeatures.find(
      (item) => item.props.checkId === checkId.value
    );
    featureId = featureConfig.id;
    shape = featureConfig.shape;
    type = featureConfig.type;
    // 过滤 allFeatures  删除checkId 相同的feature
    state.value.allFeatures = state.value.allFeatures.filter((item) => {
      return item.props.checkId !== checkId.value;
    });
    state.value.gFirstFeatureLayer.features =
      state.value.gFirstFeatureLayer.features.filter((item) => {
        return item.props.checkId !== checkId.value;
      });
    state.value.gFirstTextLayer &&
    state.value.gFirstTextLayer.removeTextById(checkId.value);
  } else {
    // const featureConfig = state.value.allFeatures[state.value.allFeatures.length - 1];
    const featureConfig = state.value.allFeatures.find((val) => {
      return val.id === operateId.value;
    });
    featureId = featureConfig.id;
    shape = featureConfig.shape;
    type = featureConfig.type;
    const deleteId = operateId.value;
    state.value.gFirstFeatureLayer &&
    state.value.gFirstFeatureLayer.removeFeatureById(deleteId);
    state.value.gFirstTextLayer &&
    state.value.gFirstTextLayer.removeTextById(deleteId);
    getFeatures();
  }
  state.value.gMap.refresh();
  const checkTagList = tagConfig.value.tagList.filter((item) => item.isCheck);
  const name = checkTagList[0].name;
  const color = checkTagList[0].color;
  const customId = nanoid();
  const id = `${checkTagList[0].labelId}-${customId}`;
  const textId = id;
  const deleteMarkerId = `${checkTagList[0].labelId}-${customId}`;
  const drawingStyle = {
    fillStyle: color,
    strokeStyle: color, // #3CB371
    fill: true, // 是否填充
    globalAlpha: 0.3,
    lineWidth: 2,
  };
  const config = {
    id,
    textId,
    deleteMarkerId,
    type,
    name,
    drawingStyle,
  };
  againAddFeature(type, shape, config);
  state.value.gMap.refresh();
  checkId.value = "";
  getFeatures();
  tagModalVisible.value = false;
};

const handleModalCancel = () => {
  tagModalVisible.value = false;
};

const getFeatureList = (dataList: any) => {
  const list = dataList.filter((val) => val.type !== "POINT");
  // const anoList = list
  //   .filter(val => val.props.revokeId)
  //   .map(item => {
  //     return { ...item, revokeId: item.props.revokeId };
  //   });
  // const mapAnoList = Object.entries(_.groupBy(anoList, 'revokeId')).map(([key, value]) => {
  //   return value[0];
  // });
  // const otherList = list.filter(val => !val.props.revokeId);
  // return Array.isArray(dataList) ? [...mapAnoList, ...otherList] : [];
  return list;
};

const handleFeaOperate = (sign: any, row: any, index: any) => {
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
      state.value.gFirstTextLayer.removeTextById(row.props.textId);
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
      imgList.value[imgActiveIndex.value].markInfo = Array.isArray(markDataList) && markDataList.length > 0
        ? JSON.stringify(markDataList)
        : ""
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
};

function adjustRatio(ratio) {
  const screenWidth = window.screen.width; // 获取屏幕宽度
  const maxWidth = 1920; // 假设我们的最大宽度是1920px
  const newRatio = (screenWidth / maxWidth) * ratio; // 根据屏幕宽度和最大宽度计算新的比例
  return newRatio;
}

// ---------------------newCode--------------------------
// 异步保存 切换图片用到
async function asyncPrevImgSave() {
  const isArrLen = Array.isArray(imgList.value) && imgList.value.length > 0;
  if (!isSwiperChange.value || !isArrLen) {
    return;
  }
  // 判断是否编辑过
  const prevImgData = imgList.value[imgActiveIndex.value];
  // getFeatures();
  const markIds =
    prevImgData.markInfo &&
    Array.isArray(JSON.parse(prevImgData.markInfo)) &&
    JSON.parse(prevImgData.markInfo).length > 0
      ? JSON.parse(prevImgData.markInfo).map((item: any) => {
        return item.id;
      })
      : [];
  const defaultIds = prevImgData.defaultInfo &&
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
  if (!isEdit1) {
    isOperate.value = true;
    const imgInfo = await getImageSizeByUrl(prevImgData.imgSrc); // 不同屏幕图片宽高
    const buffer = await state.value.gMap.exportLayersToImage(
      { x: 0, y: 0, width: imgInfo.width, height: imgInfo.height },
      { type: "blob", format: "image/png" }
    );
    // getFeatures();
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
        const markDataList = state.value.allFeatures.map((item, index) => {
          const randomId = `${new Date().getTime()}-${nanoid()}`;
          return {
            openId: item.id ? item.id.split("-")[0] : "",
            id: item.id ?? randomId,
            type: item.type,
            props: {
              ...item.props,
              operateWidth: imgInfo.width,
              operateHeight: imgInfo.height,
            },
            shape: item.shape,
            style: item.style,
          };
        });
        const labels = markDataList.map((item) => item.openId);
        const params = {
          fileId: prevImgData.fileId,
          sonId: route.query.id,
          markFileId: res.data,
          markInfo: JSON.stringify(markDataList),
          labels: [...new Set(labels)].join(","),
          operateWidth: imgInfo.width,
          operateHeight: imgInfo.height,
        };
        const res1 = await addDataMarkInfo(params);
        if (res1.data) {
          imgList.value[imgActiveIndex.value].markInfo =
            JSON.stringify(markDataList);
          imgList.value[imgActiveIndex.value].defaultInfo =
            JSON.stringify(markDataList);
          window.$message?.success?.("后台保存成功！");
          revokeList.value = [];
          // getImgData();
          getDataDetailsCount();
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
        };
        const res1 = await addDataMarkInfo(params);
        if (res1.data) {
          imgList.value[imgActiveIndex.value].markInfo = JSON.stringify("");
          imgList.value[imgActiveIndex.value].defaultInfo = JSON.stringify("");
          window.$message?.success?.("后台保存成功！");
          // getImgData();
          getDataDetailsCount();
          isOperate.value = false;
        } else {
          isOperate.value = false;
        }
      }
    }
  }
  return true;
}

function handleTagClick(feature: any) {
  state.value.editId = feature.id;
  operateId.value = feature.id;
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

// ----------------------------------------------------------

onBeforeUnmount(() => {
  state.value.gMap?.destroy();
  window.onresize = null;
  window.onkeydown = null;
  clearTimeout();
});
onMounted(async () => {
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
  // 获取图片原宽高
  try {
    const row = await getImageSizeByUrl();
    setImgSize(row);
  } catch (e) { }
  await initMap(); // 初始化实例
  await setGFirstImageLayer(); // 图片层添加
  await setGFirstFeatureLayer(); // 添加矢量图层
  await setGFirstTextLayer(); // 添加 text 文本图层，用于展示文本
  await addEvent(); // 添加事件
  state.value.gMap.disablePanWhenDrawing(); // 禁用绘制时鼠标达到边界外自动平移
  state.value.gMap.disableZoomWhenDrawing(); // 禁用绘制时可鼠标滑轮缩放
  window.onresize = () => {
    state.value.gMap && state.value.gMap.resize();
  };

  // 监听键盘事件
  document.addEventListener("keydown", (e) => {
    // 删除
    if (e.key === "Delete") {
      if (state.value.editId) {
        // 记录撤销操作列表
        const feature = state.value.gFirstFeatureLayer.getFeatureById(state.value.editId);
        revokeList.value = [...revokeList.value, feature];
        // ----------------------------------------------
        const deleteId = state.value.editId;
        state.value.gFirstFeatureLayer &&
        state.value.gFirstFeatureLayer.removeFeatureById(deleteId);
        state.value.gFirstTextLayer &&
        state.value.gFirstTextLayer.removeTextById(deleteId);
        tagModalVisible.value = false;
        getFeatures();
      }
    }
    // 监听键盘上下左右
    if (
      e.key === "ArrowUp" ||
      e.key === "ArrowDown" ||
      e.key === "ArrowLeft" ||
      e.key === "ArrowRight"
    ) {
      if (visible.value || isOperate.value) return;
      const curIndex = Number(imgActiveIndex.value);
      const nextIndex =
        e.key === "ArrowUp" || e.key === "ArrowLeft"
          ? curIndex - 1
          : curIndex + 1;
      if (nextIndex < 0 || nextIndex >= imgList.value.length) return;
      const nextImgData = imgList.value[nextIndex];
      handleImgChange(nextImgData);
    }

    // 监听Esc
    if (e.key === "Escape") {
      state.value.gMap.setMode("PAN");
      iconList.map((val) => {
        val.isOperate = false;
        return val;
      });
      state.value.foreground_points = [];
      state.value.background_points = [];
    }
  });
});
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
        <div class="h-full w-full flex items-center justify-end gap-[24px]">
          <!--<n-button @click="handleOperation(OperateType.import)">导入</n-button>-->
          <!--<n-button @click="handleOperation(OperateType.annotation)">数据标注</n-button>-->
        </div>
      </div>
    </div>
    <div class="header1 box-border box-border h-[48px] w-full flex items-center bg-[#fff] px-16px py-0">
      <!--
<NPopover trigger="click" placement="bottom-end" class="w-[350px]">
        <template #trigger>
          <div>
            <SvgIcon
              icon="mdi:tag-search-outline"
              class="inline-block align-text-bottom text-18px color-[#84868c]"
            ></SvgIcon>
          </div>
        </template>
<div class="wrap_iconSearch h-auto w-full flex flex-col items-start justify-center gap-12px p-24px">
  <div class="item_search_row w-full flex items-center justify-start">
    <span class="block w-86px flex items-center font-500">名称搜索</span>
    <NInput v-model:value="searchObj.nameSearch" type="text" placeholder="" />
  </div>
  <div class="item_search_row w-full flex items-center justify-start">
    <span class="block w-86px flex items-center font-500">标签搜索</span>
    <NSelect v-model:value="searchObj.tagsSearch" options="[]" />
  </div>
  <div class="icon_footer mt-[8px] h-auto w-full flex justify-end gap-[24px]">
    <NButton type="info">重置</NButton>
    <NButton>确定</NButton>
  </div>
</div>
</NPopover>
-->
      <!--<div class="line ml-8px h-40% w-2px bg-[#ededed]"></div>-->
      <!--@before-leave="handleTabBefore"-->
      <NTabs type="line" animated class="wrap_tabs ml-14px w-auto" @update:value="handleTabChange"
             @before-leave="handleTabBefore">
        <NTabPane name="0" tab="全部">
          <template #tab>全部({{ tabConfig.tabNum.all }})</template>
        </NTabPane>
        <NTabPane name="1" tab="有标注信息">
          <template #tab>有标注信息({{ tabConfig.tabNum.haveAno }})</template>
        </NTabPane>
        <NTabPane name="2" tab="无标注信息">
          <template #tab>无标注信息({{ tabConfig.tabNum.noAno }})</template>
        </NTabPane>
      </NTabs>
    </div>
    <div class="content box-border w-full flex-1 p-[16px] overflow-auto">
      <NCard title="" class="h-full w-full py-24px">
        <div class="h-full w-full flex items-center justify-start border-1 border-[#eee]">
          <!-- 图片列表 -->
          <div class="h-full w-17% flex flex-col items-center justify-start border-r-1 border-r-[#eee]">
            <div class="boder-b-[#eee] h-[42px] w-full flex items-center gap-8px border-b-1 px-24px">
              <!--上一页 下一页-->
              <div class="w-full flex items-center justify-between gap-4px">
                <NButton type="primary" quaternary class="prev" size="small" @click="handleSwiperChange('prev')">
                  上一页
                </NButton>
                <div class="flex items-center">
                  <span>第{{ imgParams.page }}页</span>
                  <span class="mx-4px">/</span>
                  <span v-if="imgPages">共{{ imgPages }}页</span>
                  <span v-else>共{{ 1 }}页</span>
                </div>
                <NButton type="primary" class="next" size="small" quaternary @click="handleSwiperChange('next')">
                  下一页
                </NButton>
              </div>
            </div>
            <div class="w-full flex-1">
              <div
                class="box-border box-border h-full w-full flex flex-col items-center gap-8px overflow-hidden px-24px"
                :class="[
                  imgList.length === 8 ? 'justify-between' : 'justify-start',
                ]">
                <!-- class="box-border h-75px w-[100px] flex items-center justify-center object-cover py-[4px]" -->
                <div v-for="(item, index) of imgList" :key="index"
                     class="relative box-border h-12% w-72% flex items-center justify-center overflow-hidden object-cover"
                     :class="[
                    imgActiveIndex === item.acticeIdx
                      ? 'border-2 border-[#2468f2]'
                      : 'border-2 border-[#eee]',
                  ]" @click="handleImgChange(item)">
                  <!--<img :src="item.imgSrc" class="h-full w-auto" />-->
                  <img v-if="!!item.previewImgPath" v-lazy="item.previewImgPath"
                       class="absolute left-50% top-50% block h-full w-auto -translate-x-1/2 -translate-y-1/2" />
                  <img v-else v-lazy="item.imgSrc"
                       class="absolute left-50% top-50% block h-full w-auto -translate-x-1/2 -translate-y-1/2" />
                </div>
                <div v-show="false"
                     class="h-75px w-[100px] flex items-center justify-center border-1 border-[#eee] py-[4px]"
                     @click="navTo('data-manage_import')">
                  <SvgIcon local-icon="teenyicons--add-solid" class="text-[24px] text-[#1b20bb]"></SvgIcon>
                </div>
              </div>
            </div>
          </div>
          <div class="h-full w-68% flex flex-col items-center justify-start">
            <div class="boder-b-[#eee] h-[42px] w-full flex items-center gap-8px border-b-1 px-24px">
              <div v-for="(item, index) of iconList" :key="index" :style="{
                'background-color': item.isOperate ? '#eaf1f0' : '',
              }" class="box-border p-4px">
                <NPopover trigger="hover">
                  <template #trigger>
                    <div class="item_operate_icon" @click="setMode(item.mode, item.sign, item)">
                      <!--<SvgIcon :icon="item.icon" class="text-[20px]"></SvgIcon>-->
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
                  <div v-if="false" class="flex items-center">
                    <div class="prev" @click="handlePrevNextImg('prev')">
                      <SvgIcon icon="fluent:previous-20-regular" class="text-[18px]"></SvgIcon>
                    </div>
                    <div class="mx-4px flex items-center">
                      <div>
                        第
                        <span class="mx-2px">{{ curImgNum }}</span>
                        张
                      </div>
                      <span>/</span>
                      <div class="">
                        共
                        <span class="mx-2px">{{ imgTotal }}</span>
                        张
                      </div>
                    </div>
                    <div class="next" @click="handlePrevNextImg('next')">
                      <SvgIcon icon="fluent:next-20-regular" class="text-[18px]"></SvgIcon>
                    </div>
                  </div>
                  <!-- <div class="ml-16px h-full flex items-center">
                    <NCheckbox
                      v-model:checked="state.isInvalid"
                      class="flex items-center"
                    >
                      <NButton quaternary class="px-0">标记为无效数据</NButton>
                    </NCheckbox>
                  </div> -->
                  <NButton v-if="state.isInvalid" quaternary type="info" @click="saveAno('0')">保存当前标注</NButton>
                  <!-- :disabled="isActiveCurMarkBtn"  -->
                  <NButton v-else quaternary type="info" @click="saveAno('1')">保存当前标注</NButton>
                  <NButton quaternary type="info" @click="clearAno()">取消当前标注</NButton>
                </div>
              </div>
            </div>
            <!--boder-b-[#eee] main_map h-[92%] w-full flex items-center justify-center border-b-1-->
            <div class="main_map h-[92%] w-full flex items-center justify-center overflow-hidden">
              <div id="main_map" ref="main_map"
                   class="relative h-full w-[88%] flex items-center justify-center object-cover">
                <div v-show="visible" class="mask-layer">
                  <div class="loading-spinner">
                    <!-- 这里可以放置任何你想要的加载动画 -->
                    <NSpin size="large" description="自动标注中... 请稍等" />
                  </div>
                </div>
                <div v-show="isOperate" class="mask-layer">
                  <div class="loading-spinner">
                    <!-- 这里可以放置任何你想要的加载动画 -->
                    <NSpin size="large" description="后台保存中.. 请稍等" />
                  </div>
                </div>
                <div id="map" ref="map" :style="GetWindowInfo">
                  <ContextMenu v-model:show="ctxMenuConfig.show" :options="ctxMenuConfig.optionsComponent">
                    <ContextMenuItem label="Simple item" @click="onMenuClick(1)" />
                    <ContextMenuSperator />
                  </ContextMenu>
                </div>
              </div>
              <div v-show="state.isInvalid" class="invalid-tip">无效数据</div>
            </div>
            <!-- imgList change -->
            <div v-if="false" class="h-[22%] w-full flex items-center justify-center">
              <div class="left w-[10%] flex items-center justify-center" @click="handleSwiperChange('prev')">
                <div>
                  <SvgIcon local-icon="teenyicons--left-solid" class="text-[32px]"></SvgIcon>
                </div>
              </div>
              <div class="center h-full w-0 flex flex-1 overflow-x-auto">
                <div class="h-full w-[100%] flex flex-nowrap items-center justify-start gap-24px overflow-x-auto">
                  <div v-for="(item, index) of imgList" :key="index"
                       class="box-border h-75px w-[100px] flex items-center justify-center object-cover" :class="[
                      imgActiveIndex === item.acticeIdx
                        ? 'border-2 border-[#2468f2]'
                        : 'border-none',
                    ]" @click="handleImgChange(item)">
                    <!-- <img :src="item.imgSrc" class="h-full w-auto" /> -->
                    <img v-if="!!item.previewImgPath" :src="item.previewImgPath" class="h-full w-auto" />
                    <img v-else :src="item.imgSrc" class="h-full w-auto" />
                  </div>
                  <div class="h-75px w-[100px] flex items-center justify-center border-1 border-[#eee] py-[4px]"
                       @click="navTo('data-manage_import')">
                    <SvgIcon local-icon="teenyicons--add-solid" class="text-[24px] text-[#1b20bb]"></SvgIcon>
                  </div>
                </div>
              </div>
              <div class="right w-[10%] flex items-center justify-center" @click="handleSwiperChange('next')">
                <div>
                  <!--
<SvgIcon
                    icon="teenyicons:right-solid"
                    class="text-[32px]"
                  ></SvgIcon>
-->
                  <SvgIcon local-icon="teenyicons--right-solid" class="text-[32px]"></SvgIcon>
                </div>
              </div>
            </div>
          </div>
          <!-- 标签列表 已打 -->
          <div class="h-full w-15% flex flex-col items-center justify-start border-l-1 border-l-[#eee]">
            <div class="boder-b-[#eee] h-[42px] w-full flex items-center gap-8px border-b-1 px-24px">
              <div class="w-full flex items-center justify-center gap-4px">
                <NButton type="primary" quaternary class="prev" size="small">标签列表</NButton>
              </div>
            </div>
            <div class="w-full flex-1 overflow-y-auto">
              <div
                class="box-border w-full h-auto flex flex-nowrap flex-col items-start justify-start gap-8px px-8px py-16px">
                <div v-for="(item, index) of getFeatureList(state.allFeatures)" :key="index"
                     class="mb-8px box-border h-40px h-auto w-full flex items-center justify-between overflow-hidden border-1 border-[#eee] rounded-[4px] px-4px py-4px">
                  <div class="h-full w-auto flex items-center cursor-pointer" @click="handleTagClick(item)">
                    <div class="mr-8px w-10px h-14px" :style="{ background: item.style.fillStyle }"></div>
                    <span v-if="item.props.name" class="">{{
                        item.props.name
                      }}</span>
                    <span v-else class="text-[red] cursor-pointer">暂无标签</span>
                  </div>
                  <div @click="handleFeaOperate('delete', item, index)">
                    <SvgIcon icon="material-symbols-light:delete-outline" class="text-[20px]"></SvgIcon>
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
                  <div class="w-[80%]">{{ item.name }}</div>
                  <div v-show="item.isHover" class="h-full w-[20%] flex items-center gap-[8px]">
                    <div @click="handleTagOperate('edit', item)">
                      <!--<SvgIcon icon="lucide:edit" class="text-[16px]"></SvgIcon>-->
                      <SvgIcon local-icon="lucide--edit" class="text-[16px]"></SvgIcon>
                    </div>
                    <div @click="handleTagOperate('delete', item)">
                      <SvgIcon icon="material-symbols-light:delete-outline" class="text-[20px]"></SvgIcon>
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
                  <div class="item_ipt_con h-full w-[40%] flex items-center">
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
      <NCard style="width: 460px" title="" :bordered="false" size="huge" role="dialog" aria-modal="true">
        <template #header-extra></template>
        <div class="h-full w-full">
          <div class="my-16px h-[15%] w-full">
            <div v-if="!tagConfig.isEditTag" class="h-full w-full flex items-center justify-around">
              <div class="flex items-center text-[16px] text-[#000]">
                标签选择
              </div>
              <div class="flex items-center gap-4px">
                <NButton type="primary" class="add" size="small" @click="handleAddTag('tag')">自定义标签</NButton>
                <NButton type="primary" class="add" size="small" @click="handleAddTag()">选择标签组</NButton>
              </div>
            </div>
            <div v-else class="box-border h-full w-full flex items-center justify-center">
              <div v-if="tagConfig.sign === 'group'" class="select w-full">
                <!--:show="tagConfig.isShow"-->
                <NSelect ref="selectRef" v-model:value="tagConfig.params.val" placeholder="请选择标签组"
                         :show="tagConfig.isShow" :options="tagConfig.options" @update:value="handleSelectChange"
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
          <!--
<div class="box-border h-[12%] w-full flex items-center">
            <NInput placeholder="搜索" class="h-50% w-full flex items-center" @input="handleInput">
              <template #suffix>
                <NIcon :component="FlashOutline" />
              </template>
            </NInput>
          </div>
-->
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
                <div class="w-[80%]">{{ item.name }}</div>
                <div v-show="item.isHover" class="h-full w-[20%] flex items-center gap-[8px]">
                  <div @click="handleTagOperate('edit', item)">
                    <!--<SvgIcon icon="lucide:edit" class="text-[16px]"></SvgIcon>-->
                    <SvgIcon local-icon="lucide--edit" class="text-[16px]"></SvgIcon>
                  </div>
                  <div @click="handleTagOperate('delete', item)">
                    <SvgIcon icon="material-symbols-light:delete-outline" class="text-[20px]"></SvgIcon>
                  </div>
                </div>
              </div>
              <div v-show="!item.isOperate" class="h-full w-full flex items-center">
                <div class="item_ipt_con h-full w-[80%] flex items-center">
                  <NInput v-model:value="item.name" type="text" placeholder="" class="border-none outline-none" />
                </div>
                <div class="item_ipt_con h-full w-[40%] flex items-center">
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
            <NButton type="primary" @click="handleModalDefine()">确定</NButton>
            <NButton @click="handleModalCancel()">取消</NButton>
          </div>
        </template>
      </NCard>
    </NModal>
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
