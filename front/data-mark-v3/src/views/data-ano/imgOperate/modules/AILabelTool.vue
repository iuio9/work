<template>
  <div id="map" ref="containerRef" class="w-full h-full overflow-hidden relative border-dashed border border-gray-300">
    <!-- 内部元素已清空 -->
  </div>
</template>

<script lang="ts" setup>
import { onMounted, ref, defineProps, withDefaults } from "vue";
import AILabel from "ailabel";

// 定义类型
interface Props {
  imageUrl: string;
  mode:
    | "POINT"
    | "RECT"
    | "CIRCLE"
    | "REVOKE"
    | "PAN"
    | "ZOOM_IN"
    | "ZOOM_OUT"
    | "POLYGON";
}

type Point = { x: number; y: number };
type Shape = Record<string, any>;
type Style = Record<string, any>;

interface ImageInfo {
  deviceWidth: number;
  deviceHeight: number;
  previousDeviceWidth: number;
  previousDeviceHeight: number;
}

// 矩形形状类型定义
interface IRectShape {
  x: number;
  y: number;
  width: number;
  height: number;
}

// 点形状类型定义，这里可用于多边形的点相关定义基础，也可单独使用
interface IPointShape {
  x: number;
  y: number;
  r?: number;
  sr?: number;
}

// 圆形形状类型定义
interface ICircleShape {
  cx: number;
  cy: number;
  r?: number;
  sr?: number;
}

// 多段线形状类型定义
interface IPolylineShape {
  points: IPointShape[];
  width?: number;
}

// 定义绘制类型对应的形状数据类型
type ShapeData = IRectShape | ICircleShape | { points: IPointShape[] } | IPolylineShape;

// 定义 Config 类型
interface Config {
  id: string;
  name: string;
  textId: string;
  deleteMarkerId: string;
  drawingStyle: Style;
  operateIdx?: number;
}

// 假设这些类型定义在相应的命名空间或模块中
namespace AILabel {
  export namespace Feature {
    export class Rect {
      constructor(
        public id: string,
        public shape: IRectShape,
        public props: { name: string; textId: string; deleteMarkerId: string },
        public style: any
      ) {}
      public operateIdx?: number | undefined;
      public isEye?: boolean | undefined;
    }
    export class Circle {
      constructor(
        public id: string,
        public shape: ICircleShape,
        public props: { name: string; textId: string; deleteMarkerId: string },
        public style: any
      ) {}
      public operateIdx?: number;
      public isEye?: boolean;
    }
    // 这里假设多边形相关类型定义，由于没给完整多边形定义，简单以点数组示意
    export class Polygon {
      constructor(
        public id: string,
        public shape: { points: IPointShape[] },
        public props: { name: string; textId: string; deleteMarkerId: string },
        public style: any
      ) {}
      public operateIdx?: number;
      public isEye?: boolean;
    }
    export class Polyline {
      constructor(
        public id: string,
        public shape: IPolylineShape,
        public props: { name: string; textId: string; deleteMarkerId: string },
        public style: any
      ) {}
      public operateIdx?: number;
      public isEye?: boolean;
    }
  }
}

// 定义 props
const props = withDefaults(defineProps<Props>(), {
  mode: "PAN", // 设置 mode 的默认值为 "PAN"
});

// 抽象默认样式
const DEFAULT_STYLES: Record<string, Style> = {
  point: { fillStyle: "#9370DB" },
  circle: { fillStyle: "#9370DB", strokeStyle: "#0000FF", lineWidth: 2 },
  rect: { strokeStyle: "#0f0", lineWidth: 1 },
  polygon: {
    strokeStyle: "#00f",
    fillStyle: "#0f0",
    globalAlpha: 0.3,
    lineWidth: 1,
    fill: true,
    stroke: true,
  },
};

// 定义 ref
const drawingStyle = ref<Style>({}); // 绘制过程中样式
const gMap = ref<AILabel.Map | null>(null);
const gFirstFeatureLayer = ref<AILabel.Layer.Feature | null>(null);
const gFirstMaskLayer = ref<AILabel.Layer.Mask | null>(null);
const containerRef = ref<HTMLElement | null>(null);
const imageInfo = ref<ImageInfo>({
  deviceWidth: 0,
  deviceHeight: 0,
  previousDeviceWidth: 0,
  previousDeviceHeight: 0,
});

// 工具函数
const getImageSizeByUrl = async (
  imgUrl: string | undefined,
  containerElement: HTMLElement | null
): Promise<{ width: number; height: number; centerObj: Point }> => {
  return new Promise((resolve, reject) => {
    if (!containerElement) {
      reject(new Error("Container element is not available."));
      return;
    }
    const containerWidth = containerElement.offsetWidth;
    const containerHeight = containerElement.offsetHeight;
    const image = new Image();
    image.crossOrigin = "Anonymous";
    image.onload = () => {
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
      });
    };
    image.onerror = () => {
      reject(new Error("Failed to load image"));
    };
    image.src = imgUrl ?? props.imageUrl;
  });
};

const getWidthAndHeightRatios = (info: typeof imageInfo): { widthRatio: number; heightRatio: number } => {
  const { deviceWidth, deviceHeight, previousDeviceWidth, previousDeviceHeight } = info.value;

  // 计算宽的比例
  const widthRatio = previousDeviceWidth !== 0 ? deviceWidth / previousDeviceWidth : 1;

  // 计算高的比例
  const heightRatio = previousDeviceHeight !== 0 ? deviceHeight / previousDeviceHeight : 1;

  return {
    widthRatio,
    heightRatio,
  };
};

// 提取创建要素实例的逻辑
const createFeatureInstance = (
  type: Props["mode"] | "POLYLINE",
  id: string,
  data: ShapeData,
  props: { name: string; textId: string; deleteMarkerId: string },
  style: Style
) => {
  let feature: any;
  switch (type) {
    case "POINT":
      feature = new AILabel.Feature.Point(
        id,
        { ...data, sr: 3 },
        props,
        style
      );
      break;
    case "CIRCLE":
      feature = new AILabel.Feature.Circle(
        id,
        data as ICircleShape,
        props,
        style
      );
      break;
    case "RECT":
      feature = new AILabel.Feature.Rect(
        id,
        data as IRectShape,
        props,
        style
      );
      break;
    case "POLYGON":
      feature = new AILabel.Feature.Polygon(
        id,
        data as { points: IPointShape[] },
        props,
        style
      );
      break;
    case "POLYLINE":
      feature = new AILabel.Feature.Polyline(
        id,
        data as IPolylineShape,
        props,
        style
      );
      break;
    default:
      break;
  }
  return feature;
};

// 要素创建函数
const createFeature = (
  type: Props["mode"],
  data: Shape
) => {
  const id = `${Date.now()}`;
  const props = { name: "第一个矢量图层" };
  const style = type === "CIRCLE" ? DEFAULT_STYLES.circle : drawingStyle.value;
  return createFeatureInstance(type, id, data, props, style);
};

// 绘制要素函数
const drawFeature = (type: "RECT" | "CIRCLE" | "POLYGON" | "POLYLINE", data: ShapeData, config: Config) => {
  if (!gFirstFeatureLayer.value) {
    console.error("First feature layer is not initialized.");
    return;
  }
  const { id, name, textId, deleteMarkerId, drawingStyle } = config;
  const props = { name, textId, deleteMarkerId };
  const feature = createFeatureInstance(type, id, data, props, drawingStyle);
  if (feature) {
    gFirstFeatureLayer.value.addFeature(feature);
  }
};

// 事件处理函数
const handleDrawDone = (type: string, data: Shape) => {
  console.log("--type, data--", type, data);
  const feature = createFeature(type as Props["mode"], data);
  if (feature && gFirstFeatureLayer.value) {
    gFirstFeatureLayer.value.addFeature(feature);
  }
};

// 图层创建及添加的通用函数
const createAndAddLayer = <T extends AILabel.Layer.Base>(
  layer: T,
  eventHandlers: Partial<Record<string, (a: unknown, b: unknown) => void>> = {}
) => {
  if (gMap.value) {
    gMap.value.addLayer(layer);
  } else {
    console.error("Map is not initialized.");
    return;
  }
  Object.entries(eventHandlers).forEach(([eventName, handler]) => {
    layer.events.on(eventName, handler);
  });
};

// 图层设置函数
const setupLayers = async () => {
  try {
    const handleLoadEvent = (eventName: string) => (a: unknown, b: unknown) => {
      console.log(`--${eventName}--`, a, b);
    };

    // 创建并添加第一个图片图层
    const firstImageLayer = new AILabel.Layer.Image(
      "first-layer-image",
      {
        src: props.imageUrl,
        width: imageInfo.value.deviceWidth,
        height: imageInfo.value.deviceHeight,
        crossOrigin: false,
        position: { x: 0, y: 0 },
      },
      { name: "第一个图片图层" },
      { zIndex: 5 }
    );
    createAndAddLayer(firstImageLayer, {
      loadStart: handleLoadEvent("loadStart"),
      loadEnd: handleLoadEvent("loadEnd"),
      loadError: handleLoadEvent("loadError"),
    });

    // 创建并添加第一个矢量图层
    gFirstFeatureLayer.value = new AILabel.Layer.Feature(
      "first-layer-feature",
      { name: "第一个矢量图层" },
      { zIndex: 10 }
    );
    createAndAddLayer(gFirstFeatureLayer.value);

    // 创建并添加第一个涂抹图层
    gFirstMaskLayer.value = new AILabel.Layer.Mask(
      "first-layer-mask",
      { name: "第一个涂抹图层" },
      { zIndex: 11, opacity: 0.5 }
    );
    createAndAddLayer(gFirstMaskLayer.value);
  } catch (error) {
    console.error("在设置图层时出错:", error);
  }
};

// 事件监听设置函数
const setupEvents = () => {
  if (!gMap.value) return;
  const eventHandlers: Record<string, (a: any, b?: any) => void> = {
    click: (point: Point) => {
      console.log('--click--', point);
    },
    draging: (activeFeature: any, toUpdateShape: Shape) => {
      console.log("--draging--", activeFeature, toUpdateShape);
    },
    drawDone: (type: string, data: Shape) => {
      handleDrawDone(type, data);
    },
    boundsChanged: (data: any) => {
      return 2222;
    },
    featureSelected: (feature: any) => {
      gMap.value?.setActiveFeature(feature);
    },
    featureUnselected: () => {
      gMap.value?.setActiveFeature(null);
    },
    featureUpdated: (feature: any, shape: Shape) => {
      feature.updateShape(shape);
    },
    featureDeleted: ({ id: featureId }: { id: string }) => {
      if (gFirstFeatureLayer.value) {
        gFirstFeatureLayer.value.removeFeatureById(featureId);
      }
    },
  };
  Object.entries(eventHandlers).forEach(([eventName, handler]) => {
    gMap.value?.events.on(eventName, handler);
  });
};

// 地图初始化函数
const initMap = async () => {
  try {
    if (!containerRef.value) {
      throw new Error("Container element is not available.");
    }
    const { width, height, centerObj } = await getImageSizeByUrl(props.imageUrl, containerRef.value);
    imageInfo.value.deviceWidth = width;
    imageInfo.value.deviceHeight = height;

    gMap.value = new AILabel.Map("map", {
      center: centerObj,
      zoom: 800,
      mode: props.mode || "PAN",
      refreshDelayWhenZooming: true,
      zoomWhenDrawing: true,
      panWhenDrawing: true,
      zoomWheelRatio: 5,
      withHotKeys: true,
    });

    await setupLayers();
    setupEvents();
  } catch (error) {
    console.error("地图初始化出错:", error);
  }
};

// 缩放函数
const zoomIn = () => {
  gMap.value?.zoomIn();
};

const zoomOut = () => {
  gMap.value?.zoomOut();
};

// 生命周期钩子
onMounted(async () => {
  await initMap();
});
</script>
