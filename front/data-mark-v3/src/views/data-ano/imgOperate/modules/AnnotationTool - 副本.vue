<template>
  <div ref="containerRef" class="w-full h-full relative">
    <canvas ref="canvasRef"></canvas>
  </div>
</template>

<script lang="ts" setup>
import { ref, onMounted, watch, onBeforeUnmount } from "vue";
import { fabric } from "fabric";

// Props 定义
interface ImageItem {
  id: null | number;
  fileId: string;
  version: number;
  markFileId: null | number;
  imgPath: string;
  previewImgPath: string;
  isMark: string;
  labels: string;
  markInfo: null;
  labelMarkInfo: null;
  width: number;
  height: number;
  operateWidth: null | number;
  operateHeight: null | number;
  fileName: null | string;
  notPassMessage: null | string;
}

type Mode =
  | "POINT"
  | "POLYGON"
  | "RECT"
  | "CIRCLE"
  | "REVOKE"
  | "PAN"
  | "ZOOM_IN"
  | "ZOOM_OUT";

const props = defineProps<{
  mode: Mode;
  imageUrl: string;
  selectedImageInfo: ImageItem;
}>();

const containerRef = ref<HTMLDivElement | null>(null);
const canvasRef = ref<HTMLCanvasElement | null>(null);
const canvas = ref<fabric.Canvas | null>(null);
const state = ref({
  isLoadImgSuccess: false,
  imgUrl: props.imageUrl,
});

// 多边形点集合
const polygonPoints = ref<{ x: number; y: number }[]>([]);
// 矩形起始点
const rectStartPoint = ref<{ x: number; y: number } | null>(null);
// 圆形起始点
const circleStartPoint = ref<{ x: number; y: number } | null>(null);
// 选中的对象
const selectedObject = ref<fabric.Object | null>(null);
// 移动相关变量
const isDragging = ref(false);
const lastPointer = ref<{ x: number; y: number } | null>(null);
// 节流间隔时间（毫秒）
const throttleInterval = 32;
// 移动速度因子
const panSpeedFactor = ref(0.6);
// 历史操作记录
const history = ref<fabric.Object[]>([]);
// 临时多边形线条对象
const tempPolygonLines = ref<fabric.Line[]>([]);
// 临时圆形对象
const tempCircle = ref<fabric.Circle | null>(null);
// 临时矩形对象
const tempRect = ref<fabric.Rect | null>(null);
// 图像对象
const imageObj = ref<fabric.Image | null>(null);
// 新增：是否允许绘制的开关
const canDraw = ref(true);

// 标注框默认配置
const annotationConfig = {
  point: {
    mode: "point",
    radius: 5,
    stroke: "#06d902",
    fill: "#06d902",
    opacity: "1.0",
    selectable: false,
  },
  rect: {
    mode: "rect",
    stroke: "#6c8fe2",
    fill: "#6c8fe2",
    opacity: "0.8",
    strokeWidth: 2,
    selectable: true,
    scalable: true, // 允许缩放
    rotatable: true, // 允许旋转
    hasControls: true, // 显示控制点
    hasBorders: true, // 显示边框
  },
  circle: {
    mode: "circle",
    stroke: "#6c8fe2",
    fill: "#6c8fe2",
    opacity: "0.8",
    strokeWidth: 2,
    hasControls: true, // 支持拉伸
    hasRotatingPoint: true, // 支持旋转
  },
  polygon: {
    mode: "polygon",
    fill: "#6c8fe2",
    opacity: "0.8",
    stroke: "#6c8fe2",
    strokeWidth: 2, // 增加多边形边框的宽度
    selectable: true,
    scalable: true,
    rotatable: true,
    hasControls: true,
    hasBorders: true,
  },
  tempPolygonLine: {
    mode: "tempPolygonLine",
    stroke: "#6c8fe2",
    fill: "#6c8fe2",
    opacity: "0.8",
    strokeWidth: 2,
  },
};

// 节流函数
const throttle = (func: (...args: any[]) => void, delay: number) => {
  let timer: NodeJS.Timeout | null = null;
  return (...args: any[]) => {
    if (!timer) {
      func(...args);
      timer = setTimeout(() => {
        timer = null;
      }, delay);
    }
  };
};

// 使用Image对象获取图片真实宽高
const getImageSizeByUrl = (imgUrl: string) => {
  return new Promise<{
    width: number;
    height: number;
    centerObj: { x: number; y: number };
  }>((resolve, reject) => {
    if (!containerRef.value) {
      reject(new Error("Container reference is not available."));
      return;
    }
    const containerWidth = containerRef.value.offsetWidth;
    const containerHeight = containerRef.value.offsetHeight;
    const image = new Image();
    image.crossOrigin = "Anonymous";
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
      });
    };
    image.onerror = () => {
      state.value.isLoadImgSuccess = false;
      reject(new Error("Failed to load image."));
    };
    image.src = imgUrl;
  });
};

// 加载图像到画布
const loadImageToCanvas = async (imgUrl: string) => {
  if (!canvas.value || !containerRef.value) return;
  try {
    const { width, height } = await getImageSizeByUrl(imgUrl);
    const containerWidth = containerRef.value.offsetWidth;
    const containerHeight = containerRef.value.offsetHeight;
    const left = (containerWidth - width) / 2;
    const top = (containerHeight - height) / 2;
    fabric.Image.fromURL(imgUrl, (img) => {
      img.set({
        left,
        top,
        scaleX: width / img.width!,
        scaleY: height / img.height!,
        selectable: false,
      });
      canvas.value!.add(img);
      img.sendToBack();
      imageObj.value = img;
      canvas.value!.renderAll();
    });
  } catch (error) {
    console.error("Error loading image:", error);
  }
};

// 初始化画布
const initCanvas = async () => {
  if (containerRef.value && canvasRef.value) {
    canvas.value = new fabric.Canvas(canvasRef.value, {
      width: containerRef.value.offsetWidth,
      height: containerRef.value.offsetHeight,
      selection: true,
    });
    await loadImageToCanvas(props.imageUrl);
  }
};

// 调整画布大小
const resizeCanvas = () => {
  if (containerRef.value && canvas.value) {
    canvas.value.setWidth(containerRef.value.offsetWidth);
    canvas.value.setHeight(containerRef.value.offsetHeight);
    canvas.value.renderAll();
  }
};

// 监听窗口大小变化
const handleWindowResize = () => {
  resizeCanvas();
};
// 通用绘制函数
const drawShape = (
  shapeType: "point" | "rect" | "circle" | "polygon",
  options: any,
) => {
  if (canvas.value && !selectedObject.value && canDraw.value) {
    let shape;
    switch (shapeType) {
      case "point":
        shape = new fabric.Circle({
          left: options.x,
          top: options.y,
          ...annotationConfig.point,
        });
        break;
      case "rect":
        shape = new fabric.Rect({
          left: options.x,
          top: options.y,
          width: options.width,
          height: options.height,
          ...annotationConfig.rect,
        });
        break;
      case "circle":
        shape = new fabric.Circle({
          left: options.x - options.radius,
          top: options.y - options.radius,
          radius: options.radius,
          ...annotationConfig.circle,
        });
        break;
      case "polygon":
        shape = new fabric.Polygon(options.points, {
          ...annotationConfig.polygon,
        });
        break;
    }
    if (shape) {
      canvas.value.add(shape);
      shape.bringToFront();
      canvas.value.renderAll();
      history.value.push(shape);
    }
  }
};

// 绘制点
const drawPoint = (x: number, y: number) => {
  drawShape("point", { x, y });
};

// 绘制矩形
const drawRect = (x: number, y: number, width: number, height: number) => {
  drawShape("rect", { x, y, width, height });
};

// 绘制圆形
const drawCircle = (x: number, y: number, radius: number) => {
  drawShape("circle", { x, y, radius });
};

// 绘制多边形
const drawPolygon = (points: { x: number; y: number }[]) => {
  drawShape("polygon", { points });
};

// 撤销操作
const revokeLastAction = () => {
  if (canvas.value && history.value.length > 0) {
    const lastObject = history.value.pop();
    if (lastObject) {
      canvas.value.remove(lastObject);
      canvas.value.renderAll();
    }
  }
};

const updateTempLines = (pointer: fabric.Point) => {
  // 清除之前的临时线条和圆点
  tempPolygonLines.value.forEach((item) => canvas.value?.remove(item));
  tempPolygonLines.value = [];

  if (polygonPoints.value.length > 0) {
    polygonPoints.value.forEach((point, index) => {
      let nextPoint: fabric.Point;
      if (index < polygonPoints.value.length - 1) {
        nextPoint = new fabric.Point(
          polygonPoints.value[index + 1].x,
          polygonPoints.value[index + 1].y,
        );
      } else {
        nextPoint = pointer; // 最后一条线连接到最后一个点和当前鼠标位置
      }

      // 创建线条时直接使用圆点的坐标，确保线条从圆心开始和结束
      const line = new fabric.Line(
        [point.x, point.y, nextPoint.x, nextPoint.y],
        {
          stroke: "#6c8fe2", // 设置线条颜色
          strokeWidth: 2, // 设置线条宽度
          selectable: false,
          strokeLineCap: "round",
        },
      );
      tempPolygonLines.value.push(line);
      canvas.value?.add(line);

      // 在每个端点添加一个小圆点，其位置正好是顶点的位置
      const circle = new fabric.Circle({
        radius: 4, // 圆点半径
        fill: "white", // 设置填充色为白色
        stroke: "#6c8fe2", // 设置边框颜色为#6c8fe2
        strokeWidth: 2, // 设置边框宽度
        left: point.x,
        top: point.y,
        originX: "center",
        originY: "center",
        selectable: true, // 允许选择和移动
        hasControls: false, // 不显示控制手柄
      });
      circle.on("moving", () => updatePolygon(circle, index)); // 监听移动事件
      circle.on("moved", finalizeChanges); // 移动结束后执行的操作
      tempPolygonLines.value.push(circle);
      canvas.value?.add(circle);
    });
  }
};

const updatePolygon = (circle: fabric.Circle, index: number) => {
  if (!polygon || !canvas.value) return;

  const newPosition = circle.getCenterPoint();
  polygon.set(
    "points",
    polygon.points.map((point, i) =>
      i === index ? { x: newPosition.x, y: newPosition.y } : point,
    ),
  );
  canvas.value.renderAll();
};

const finalizeChanges = () => {
  // 如果需要，在对象移动结束后执行一些额外的操作
  canvas.value?.renderAll();
};

const clearDrawing = () => {
  polygonPoints.value = [];
  tempPolygonLines.value.forEach((item) => canvas.value?.remove(item));
  tempPolygonLines.value = [];
};

// 处理鼠标按下事件
const handleMouseDown = (options: { e: MouseEvent }) => {
  if (!canvas.value || !canDraw.value) return;
  const event = options.e;
  const pointer = canvas.value.getPointer(event);
  const activeObject = canvas.value.getActiveObject();
  const target = canvas.value.findTarget(pointer, true);

  // 处理对象选择
  handleObjectSelection(activeObject, target);

  const x = pointer.x;
  const y = pointer.y;

  if (selectedObject.value) return; // 若有选中对象，不进行绘制操作

  switch (props.mode) {
    case "PAN":
      startPanning(x, y);
      break;
    case "POINT":
      drawPoint(x, y);
      break;
    case "RECT":
      startDrawingRect(x, y);
      break;
    case "CIRCLE":
      startDrawingCircle(x, y);
      break;
    case "POLYGON":
      handlePolygonClick(x, y, pointer);
      break;
  }
};

// 处理对象选择
const handleObjectSelection = (
  activeObject: fabric.Object | null,
  target: fabric.Object | null,
) => {
  if (activeObject) {
    if (selectedObject.value) {
      selectedObject.value.set({ strokeWidth: 1 });
    }
    selectedObject.value = target;
    target?.set({ strokeWidth: 3 });
    canvas.value?.renderAll();
  } else {
    if (selectedObject.value) {
      selectedObject.value.set({ strokeWidth: 1 });
      selectedObject.value = null;
      canvas.value?.renderAll();
    }
  }
};

// 开始平移操作
const startPanning = (x: number, y: number) => {
  isDragging.value = true;
  lastPointer.value = { x, y };
};

// 开始绘制矩形
const startDrawingRect = (x: number, y: number) => {
  rectStartPoint.value = { x, y };
  tempRect.value = new fabric.Rect({
    left: x,
    top: y,
    width: 0,
    height: 0,
    ...annotationConfig.rect,
  });
  canvas.value?.add(tempRect.value);
  tempRect.value?.bringToFront();
  canvas.value?.renderAll();
};

// 开始绘制圆形
const startDrawingCircle = (x: number, y: number) => {
  circleStartPoint.value = { x, y };
  tempCircle.value = new fabric.Circle({
    left: x,
    top: y,
    ...annotationConfig.circle,
    radius: 0,
  });
  canvas.value?.add(tempCircle.value);
  tempCircle.value?.bringToFront();
  canvas.value?.renderAll();
};

// 处理多边形点击事件
const handlePolygonClick = (x: number, y: number, pointer: fabric.Point) => {
  if (polygonPoints.value.length === 0) {
    polygonPoints.value.push({ x, y });
  } else {
    polygonPoints.value.push({ x, y });
    const firstPoint = polygonPoints.value[0];
    if (
      Math.abs(firstPoint.x - pointer.x) < 5 &&
      Math.abs(firstPoint.y - pointer.y) < 5
    ) {
      drawPolygon(polygonPoints.value);
      clearDrawing();
    }
  }
};

// 处理鼠标移动事件
const handleMouseMove = throttle((options: { e: MouseEvent }) => {
  if (!canvas.value || !canDraw.value) return;
  const event = options.e;
  const pointer = canvas.value.getPointer(event);
  const x = pointer.x;
  const y = pointer.y;

  switch (props.mode) {
    case "PAN":
      handlePanning(x, y);
      break;
    case "RECT":
      updateRect(x, y);
      break;
    case "CIRCLE":
      updateCircle(x, y);
      break;
    case "POLYGON":
      if (polygonPoints.value.length > 0) {
        updateTempLines(pointer);
      }
      break;
  }
}, throttleInterval);

// 处理平移操作
const handlePanning = (x: number, y: number) => {
  if (props.mode === "PAN" && isDragging.value && lastPointer.value) {
    const deltaX = (x - lastPointer.value!.x) * panSpeedFactor.value;
    const deltaY = (y - lastPointer.value!.y) * panSpeedFactor.value;
    const smoothFactor = 0.8;
    const smoothedDeltaX = deltaX * smoothFactor;
    const smoothedDeltaY = deltaY * smoothFactor;
    canvas.value!.relativePan({ x: smoothedDeltaX, y: smoothedDeltaY });
    canvas.value!.renderAll();
    lastPointer.value = { x, y };
  }
};

// 更新矩形
const updateRect = (x: number, y: number) => {
  if (props.mode === "RECT" && rectStartPoint.value && tempRect.value) {
    const startX = rectStartPoint.value.x;
    const startY = rectStartPoint.value.y;
    const width = x - startX;
    const height = y - startY;
    tempRect.value.set({
      left: startX,
      top: startY,
      width,
      height,
    });
    canvas.value.renderAll();
  }
};

// 更新圆形
const updateCircle = (x: number, y: number) => {
  if (props.mode === "CIRCLE" && circleStartPoint.value && tempCircle.value) {
    const startX = circleStartPoint.value.x;
    const startY = circleStartPoint.value.y;
    const radius = Math.sqrt((x - startX) ** 2 + (y - startY) ** 2);
    tempCircle.value.set({
      left: startX - radius,
      top: startY - radius,
      radius,
    });
    canvas.value.renderAll();
  }
};

// 处理鼠标抬起事件
const handleMouseUp = (options: { e: MouseEvent }) => {
  if (!canvas.value || !canDraw.value) return;
  const event = options.e;
  const pointer = canvas.value.getPointer(event);
  const x = pointer.x;
  const y = pointer.y;

  switch (props.mode) {
    case "PAN":
      stopPanning();
      break;
    case "RECT":
      finishDrawingRect(x, y);
      break;
    case "CIRCLE":
      finishDrawingCircle(x, y);
      break;
  }
};

// 停止平移操作
const stopPanning = () => {
  isDragging.value = false;
  lastPointer.value = null;
};

// 完成绘制矩形
const finishDrawingRect = (x: number, y: number) => {
  if (props.mode === "RECT" && rectStartPoint.value && tempRect.value) {
    const startX = rectStartPoint.value.x;
    const startY = rectStartPoint.value.y;
    const width = x - startX;
    const height = y - startY;
    drawRect(startX, startY, width, height);
    canvas.value.remove(tempRect.value);
    tempRect.value = null;
    rectStartPoint.value = null;
  }
};

// 完成绘制圆形
const finishDrawingCircle = (x: number, y: number) => {
  if (props.mode === "CIRCLE" && circleStartPoint.value && tempCircle.value) {
    const startX = circleStartPoint.value.x;
    const startY = circleStartPoint.value.y;
    const radius = Math.sqrt((x - startX) ** 2 + (y - startY) ** 2);
    drawCircle(startX, startY, radius);
    canvas.value.remove(tempCircle.value);
    tempCircle.value = null;
    circleStartPoint.value = null;
  }
};

// 处理鼠标双击事件
const handleMouseDblClick = () => {
  if (
    props.mode === "POLYGON" &&
    polygonPoints.value.length >= 3 &&
    !selectedObject.value &&
    canDraw.value
  ) {
    drawPolygon(polygonPoints.value);
    clearDrawing();
  }
};

// 处理鼠标点击选中事件
const handleMouseClick = (options: { e: MouseEvent }) => {
  if (!canvas.value) return;
  const event = options.e;
  const pointer = canvas.value.getPointer(event);
  const target = canvas.value.findTarget(pointer, true);

  if (target) {
    if (selectedObject.value) {
      selectedObject.value.set({ strokeWidth: 1 });
    }
    selectedObject.value = target;
    target.set({ strokeWidth: 3 });
    canvas.value.renderAll();
  } else {
    if (selectedObject.value) {
      selectedObject.value.set({ strokeWidth: 1 });
      selectedObject.value = null;
      canvas.value.renderAll();
    }
  }
};

// 处理鼠标滚动事件
const handleMouseWheel = (options: { e: WheelEvent }) => {
  if (!canvas.value) return;
  const event = options.e;
  const delta = event.deltaY;
  const zoom = canvas.value.getZoom();
  const zoomFactor = 0.1;
  const minZoom = 0.1;
  const maxZoom = 5;

  if (delta > 0) {
    if (zoom > minZoom) {
      canvas.value.zoomToPoint(
        { x: event.offsetX, y: event.offsetY },
        zoom - zoomFactor,
      );
    }
  } else {
    if (zoom < maxZoom) {
      canvas.value.zoomToPoint(
        { x: event.offsetX, y: event.offsetY },
        zoom + zoomFactor,
      );
    }
  }
  canvas.value.renderAll();
};

// 监听 props 变化
watch(
  () => props.mode,
  (newMode) => {
    // 重置状态
    polygonPoints.value = [];
    rectStartPoint.value = null;
    circleStartPoint.value = null;
    clearDrawing();

    // 移除临时对象
    if (tempCircle.value) {
      canvas.value?.remove(tempCircle.value);
      tempCircle.value = null;
    }
    if (tempRect.value) {
      canvas.value?.remove(tempRect.value);
      tempRect.value = null;
    }

    // 取消选中对象
    if (selectedObject.value) {
      selectedObject.value.set({ strokeWidth: 1 });
      selectedObject.value = null;
      canvas.value?.renderAll();
    }

    // 重置移动状态
    isDragging.value = false;
    lastPointer.value = null;

    // 执行撤销操作
    if (newMode === "REVOKE") {
      revokeLastAction();
    }
  },
);

watch(
  () => props.imageUrl,
  async (newImageUrl) => {
    if (canvas.value) {
      // 清空画布和历史记录
      canvas.value.clear();
      history.value = [];

      // 清除临时多边形线条
      clearDrawing();

      // 移除临时圆形和矩形
      if (tempCircle.value) {
        canvas.value.remove(tempCircle.value);
        tempCircle.value = null;
      }
      if (tempRect.value) {
        canvas.value.remove(tempRect.value);
        tempRect.value = null;
      }

      // 加载新图像
      await loadImageToCanvas(newImageUrl);

      // 取消选中对象
      if (selectedObject.value) {
        selectedObject.value.set({ strokeWidth: 1 });
        selectedObject.value = null;
      }
    }
  },
);

// 组件挂载时初始化画布和监听事件
onMounted(() => {
  initCanvas();
  window.addEventListener("resize", handleWindowResize);
  if (canvas.value) {
    canvas.value.on("mouse:down", handleMouseDown);
    canvas.value.on("mouse:move", handleMouseMove);
    canvas.value.on("mouse:up", handleMouseUp);
    canvas.value.on("mouse:dblclick", handleMouseDblClick);
    canvas.value.on("mouse:click", handleMouseClick);
    canvas.value.on("mouse:wheel", handleMouseWheel);
  }
});

// 组件卸载时移除事件监听
onBeforeUnmount(() => {
  window.removeEventListener("resize", handleWindowResize);
  if (canvas.value) {
    canvas.value.off("mouse:down", handleMouseDown);
    canvas.value.off("mouse:move", handleMouseMove);
    canvas.value.off("mouse:up", handleMouseUp);
    canvas.value.off("mouse:dblclick", handleMouseDblClick);
    canvas.value.off("mouse:click", handleMouseClick);
    canvas.value.off("mouse:wheel", handleMouseWheel);
  }
});
</script>
