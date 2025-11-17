<template>
  <div class="relative w-full h-full">
    <!-- 加载状态下显示加载动画 -->
    <n-spin
      v-if="isLoading"
      class="absolute inset-0 flex items-center justify-center bg-white/50 z-10"
      size="large"
    />
    <div ref="canvasContainer" class="w-full h-full">
      <canvas ref="canvasElement" class="w-full h-full" />
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted, onUnmounted, watch, defineExpose } from "vue";
import { NSpin } from "naive-ui";
import { fabric } from "fabric";

// 定义 Props
const props = defineProps<{
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
}>();

// 状态和引用
const isLoading = ref(true);
const canvasContainer = ref<HTMLDivElement | null>(null);
const canvasElement = ref<HTMLCanvasElement | null>(null);
const canvasInstance = ref<fabric.Canvas | null>(null);
const selectedObject = ref<fabric.Object | null>(null);
const canDraw = ref<boolean>(true);
const history = ref<fabric.Object[]>([]);
const isDrawingRect = ref<boolean>(false);
const rectStartPoint = ref<{ x: number; y: number } | null>(null);
const tempRect = ref<fabric.Rect | null>(null);
const isDrawingCircle = ref<boolean>(false);
const circleStartPoint = ref<{ x: number; y: number } | null>(null);
const tempCircle = ref<fabric.Circle | null>(null);
const isPanning = ref(false);
const panStart = ref<{ x: number; y: number } | null>(null);

// 注释样式配置
const annotationConfig = {
  point: {
    radius: 5,
    fill: "red",
    stroke: "black",
    strokeWidth: 2,
    selectable: false,
  },
  rect: {
    fill: "rgba(255,0,0,0.3)",
    stroke: "#ff0000",
    strokeWidth: 2,
    cornerSize: 8,
    hasRotatingPoint: false,
    hasControls: true,
    hasBorders: true,
  },
  circle: {
    fill: "rgba(0,255,0,0.3)",
    stroke: "#00ff00",
    strokeWidth: 2,
    hasControls: true,
    hasBorders: true,
    hasRotatingPoint: true,
  },
  vertexCircle: {
    radius: 5,
    fill: "#ffffff",
    stroke: "",
    strokeWidth: 2,
    originX: "center",
    originY: "center",
    hasControls: false,
    hasBorders: false,
    selectable: true,
    hoverCursor: "pointer",
  },
  polygon: {
    fill: "rgba(127, 153, 223 ,0.8)",
    stroke: "#7f99df",
    strokeWidth: 2,
    selectable: false, // 设置多边形不可选中
    hasControls: false,
    hasBorders: false,
  },
};

// 定义应用状态的接口
interface DrawingState {
  // 存储所有已绘制完成的多边形
  drawnPolygons: fabric.Polygon[];
  // 当前正在绘制的多边形的顶点坐标
  currentDrawingPoints: fabric.Point[];
  // 当前正在绘制的多边形的顶点对应的圆点对象
  currentDrawingCircles: fabric.Circle[];
  // 当前正在绘制的多边形对象
  currentDrawingPolygon: fabric.Polygon | null;
  // 当前选中的多边形的索引
  selectedPolygonIndex: number;
  // 是否处于绘制状态
  isDrawingInProgress: boolean;
}

// 初始化绘制状态
const drawingState = ref<DrawingState>({
  drawnPolygons: [],
  currentDrawingPoints: [],
  currentDrawingCircles: [],
  currentDrawingPolygon: null,
  selectedPolygonIndex: -1,
  isDrawingInProgress: true,
});

// 封装绘制形状的公共方法
const createShape = (type: "rect" | "circle", options: any) => {
  switch (type) {
    case "rect":
      return new fabric.Rect({ ...annotationConfig.rect, ...options });
    case "circle":
      return new fabric.Circle({ ...annotationConfig.circle, ...options });
    default:
      return null;
  }
};

// 矩形绘制逻辑封装
const rectDrawing = {
  start: (pointer: fabric.IPoint) => {
    isDrawingRect.value = true;
    rectStartPoint.value = { x: pointer.x, y: pointer.y };
    tempRect.value = createShape("rect", {
      left: pointer.x,
      top: pointer.y,
      width: 0,
      height: 0,
    });
    if (tempRect.value) {
      canvasInstance.value?.add(tempRect.value);
    }
  },
  update: (pointer: fabric.IPoint) => {
    if (!rectStartPoint.value || !tempRect.value) return;
    const startX = rectStartPoint.value.x;
    const startY = rectStartPoint.value.y;
    const width = pointer.x - startX;
    const height = pointer.y - startY;
    tempRect.value.set({
      width: Math.abs(width),
      height: Math.abs(height),
      angle: 0,
      originX: width >= 0 ? "left" : "right",
      originY: height >= 0 ? "top" : "bottom",
    });
    canvasInstance.value?.renderAll();
  },
  end: () => {
    if (!isDrawingRect.value || !rectStartPoint.value || !tempRect.value)
      return;
    const rect = tempRect.value;
    rect.set({
      left: rectStartPoint.value.x,
      top: rectStartPoint.value.y,
      selectable: true,
      evented: true,
    });
    history.value.push(rect);
    tempRect.value = null;
    rectStartPoint.value = null;
    isDrawingRect.value = false;
  },
};

// 圆形绘制逻辑封装
const circleDrawing = {
  start: (pointer: fabric.IPoint) => {
    isDrawingCircle.value = true;
    circleStartPoint.value = { x: pointer.x, y: pointer.y };
    tempCircle.value = createShape("circle", {
      left: pointer.x,
      top: pointer.y,
      radius: 0,
    });
    if (tempCircle.value) {
      canvasInstance.value?.add(tempCircle.value);
    }
  },
  update: (pointer: fabric.IPoint) => {
    if (!circleStartPoint.value || !tempCircle.value) return;
    const startX = circleStartPoint.value.x;
    const startY = circleStartPoint.value.y;
    const dx = pointer.x - startX;
    const dy = pointer.y - startY;
    const radius = Math.sqrt(dx * dx + dy * dy);
    tempCircle.value.set({
      radius,
      left: startX - radius,
      top: startY - radius,
    });
    canvasInstance.value?.renderAll();
  },
  end: () => {
    if (!isDrawingCircle.value || !circleStartPoint.value || !tempCircle.value)
      return;
    const circle = tempCircle.value;
    circle.set({
      selectable: true,
      evented: true,
    });
    history.value.push(circle);
    tempCircle.value = null;
    circleStartPoint.value = null;
    isDrawingCircle.value = false;
  },
};

// 封装多边形绘制逻辑
const polygonDrawing = {
  start: (pointer: fabric.IPoint) => {
    if (drawingState.value.isDrawingInProgress) {
      const newPoint = new fabric.Point(pointer.x, pointer.y);

      // 检查是否可以闭合当前正在绘制的多边形
      if (drawingState.value.currentDrawingPoints.length > 0) {
        const firstPoint = drawingState.value.currentDrawingPoints[0];
        const distanceToFirstPoint = Math.hypot(
          newPoint.x - firstPoint.x,
          newPoint.y - firstPoint.y,
        );
        if (distanceToFirstPoint < 10) {
          polygonDrawing.end();
          return;
        }
      }

      // 开始新绘制时隐藏所有已绘制多边形的顶点圆点
      drawingState.value.drawnPolygons.forEach((polygon) => {
        (polygon as any).vertexCircles.forEach((circle) => {
          circle.visible = false;
        });
      });
      drawingState.value.selectedPolygonIndex = -1;
      canvasInstance.value?.renderAll();

      // 添加新的顶点坐标
      drawingState.value.currentDrawingPoints.push(newPoint);

      // 创建新的顶点圆点
      const vertexCircle = new fabric.Circle({
        left: pointer.x,
        top: pointer.y,
        ...annotationConfig.vertexCircle,
      });
      // 为顶点圆点添加移动事件监听器
      vertexCircle.on("moving", (e) => {
        const draggedCircle = e.target as fabric.Circle;
        const selectedPolygon =
          drawingState.value.drawnPolygons[
            drawingState.value.selectedPolygonIndex
          ];
        if (selectedPolygon) {
          const vertexIndex = (selectedPolygon as any).vertexCircles.indexOf(
            draggedCircle,
          );
          if (vertexIndex !== -1) {
            const newPoint = new fabric.Point(
              draggedCircle.left!,
              draggedCircle.top!,
            );
            selectedPolygon.points[vertexIndex] = newPoint;
            selectedPolygon.set({ points: selectedPolygon.points });
            selectedPolygon.setCoords();
            canvasInstance.value?.renderAll();
          }
        }
      });
      drawingState.value.currentDrawingCircles.push(vertexCircle);
      canvasInstance.value?.add(vertexCircle);
      vertexCircle.bringToFront();

      // 为第一个顶点添加鼠标悬停放大和移出恢复的效果
      if (drawingState.value.currentDrawingCircles.length === 1) {
        const firstVertexCircle = drawingState.value.currentDrawingCircles[0];
        firstVertexCircle.on("mouseover", function () {
          this.scale(1.2);
          canvasInstance.value?.renderAll();
        });
        firstVertexCircle.on("mouseout", function () {
          this.scale(1 / 1.2);
          canvasInstance.value?.renderAll();
        });
      }

      // 当有足够的顶点时，绘制临时多边形
      if (drawingState.value.currentDrawingPoints.length > 2) {
        if (drawingState.value.currentDrawingPolygon)
          canvasInstance.value?.remove(
            drawingState.value.currentDrawingPolygon,
          );
        drawingState.value.currentDrawingPolygon = new fabric.Polygon(
          drawingState.value.currentDrawingPoints,
          {
            ...annotationConfig.polygon,
          },
        );
        // 修改插入层级为 1
        canvasInstance.value?.insertAt(
          drawingState.value.currentDrawingPolygon,
          1,
        );
      }
    }
  },
  update: (options: fabric.IEvent) => {
    // 这里暂时没有需要更新的逻辑，可根据实际需求添加
  },
  end: () => {
    if (drawingState.value.isDrawingInProgress) {
      drawingState.value.isDrawingInProgress = false;
      const newPolygon = drawingState.value.currentDrawingPolygon;
      if (newPolygon) {
        (newPolygon as any).vertexCircles =
          drawingState.value.currentDrawingCircles;
        drawingState.value.drawnPolygons.push(newPolygon);
      }
      drawingState.value.currentDrawingPoints = [];
      drawingState.value.currentDrawingCircles.forEach(
        (circle) => (circle.visible = false),
      );
      drawingState.value.currentDrawingCircles = [];
      drawingState.value.currentDrawingPolygon = null;
      // 完成当前多边形绘制后可以开始新的绘制
      drawingState.value.isDrawingInProgress = true;
    }
  },
};

// 清除临时形状
const clearDrawing = () => {
  if (tempRect.value) {
    canvasInstance.value?.remove(tempRect.value);
    tempRect.value = null;
  }
  if (tempCircle.value) {
    canvasInstance.value?.remove(tempCircle.value);
    tempCircle.value = null;
  }
};

// 点击事件处理
const handleMouseDown = (options: fabric.IEvent) => {
  if (!canvasInstance.value) return;
  const pointer = canvasInstance.value.getPointer(options.e);
  const target = canvasInstance.value.findTarget(pointer);
  if (props.mode === "PAN") {
    isPanning.value = true;
    panStart.value = { x: pointer.x, y: pointer.y };
    canvasInstance.value.selection = false;
    return;
  }
  if (target) {
    selectedObject.value = target;
  } else {
    if (props.mode === "RECT") {
      rectDrawing.start(pointer);
    } else if (props.mode === "POINT") {
      drawPoint(pointer.x, pointer.y);
    } else if (props.mode === "CIRCLE") {
      circleDrawing.start(pointer);
    }
  }

  if (props.mode === "POLYGON") {
    if (pointer) {
      const clickedObject = canvasInstance.value?.findTarget(options.e);
      const isPolygonClicked =
        clickedObject &&
        drawingState.value.drawnPolygons.includes(
          clickedObject as fabric.Polygon,
        );
      const isCircleClicked =
        clickedObject &&
        drawingState.value.drawnPolygons.some((polygon) => {
          return (polygon as any).vertexCircles.includes(
            clickedObject as fabric.Circle,
          );
        });

      if (isPolygonClicked) {
        // 如果点击的是已绘制的多边形，切换选中状态并回显圆点
        drawingState.value.selectedPolygonIndex =
          drawingState.value.drawnPolygons.indexOf(
            clickedObject as fabric.Polygon,
          );
        updateVertexVisibility();
        return;
      }

      if (isCircleClicked && drawingState.value.selectedPolygonIndex !== -1) {
        // 若点击的是选中多边形的圆点，不进行新多边形绘制
        return;
      }

      polygonDrawing.start(pointer);
    }
  }
};

// 处理鼠标双击事件
const handleMouseDoubleClick = () => {
  if (
    drawingState.value.isDrawingInProgress &&
    drawingState.value.currentDrawingPoints.length > 2
  ) {
    polygonDrawing.end();
  }
};

// 处理对象移动事件
const handleObjectDrag = (event: fabric.IEvent) => {
  const draggedObject = event.target as fabric.Circle;
  const selectedPolygon: any =
    drawingState.value.drawnPolygons[drawingState.value.selectedPolygonIndex];

  if (
    selectedPolygon &&
    (selectedPolygon as any).vertexCircles.includes(draggedObject)
  ) {
    const vertexIndex = (selectedPolygon as any).vertexCircles.indexOf(
      draggedObject,
    );
    selectedPolygon.points[vertexIndex] = new fabric.Point(
      draggedObject.left!,
      draggedObject.top!,
    );
    draggedObject.bringToFront();

    // 删除当前选中的多边形
    canvasInstance.value?.remove(selectedPolygon);
    const indexToRemove =
      drawingState.value.drawnPolygons.indexOf(selectedPolygon);
    if (indexToRemove !== -1) {
      drawingState.value.drawnPolygons.splice(indexToRemove, 1);
    }

    // 创建新的多边形
    const newPolygon = new fabric.Polygon(selectedPolygon.points, {
      ...annotationConfig.polygon,
    });
    (newPolygon as any).vertexCircles = (selectedPolygon as any).vertexCircles;

    // 插入新的多边形，层级设置为 1
    canvasInstance.value?.insertAt(newPolygon, 1);
    drawingState.value.drawnPolygons.push(newPolygon);
    drawingState.value.selectedPolygonIndex =
      drawingState.value.drawnPolygons.length - 1;

    // 更新画布
    canvasInstance.value?.renderAll();
  }
};

const updateVertexVisibility = () => {
  drawingState.value.drawnPolygons.forEach((polygon, index) => {
    (polygon as any).vertexCircles.forEach((circle) => {
      circle.visible = index === drawingState.value.selectedPolygonIndex;
      circle.bringToFront();
    });
  });
  canvasInstance.value?.renderAll();
};

// 移动事件处理
const originalHandleMouseMove = (options: fabric.IEvent) => {
  if (!canvasInstance.value) return;
  if (isPanning.value) {
    const pointer = canvasInstance.value.getPointer(options.e);
    if (panStart.value) {
      const deltaX = pointer.x - panStart.value.x;
      const deltaY = pointer.y - panStart.value.y;
      canvasInstance.value.relativePan({ x: deltaX, y: deltaY });
      panStart.value = { x: pointer.x, y: pointer.y };
    }
    return;
  }
  if (isDrawingRect.value) {
    rectDrawing.update(canvasInstance.value.getPointer(options.e));
  }
  if (isDrawingCircle.value) {
    circleDrawing.update(canvasInstance.value.getPointer(options.e));
  }
};

// 封装节流函数
const throttle = (func: Function, delay: number) => {
  let timer: number | null = null;
  return function (this: any, ...args: any[]) {
    if (!timer) {
      func.apply(this, args);
      timer = setTimeout(() => {
        timer = null;
      }, delay);
    }
  };
};

// 使用节流函数优化 handleMouseMove
const handleMouseMove = throttle(originalHandleMouseMove, 20);

// 释放事件处理
const handleMouseUp = () => {
  if (isPanning.value) {
    canvasInstance.value.selection = true;
  }
  isPanning.value = false;
  panStart.value = null;
  if (isDrawingRect.value) {
    rectDrawing.end();
  }
  if (isDrawingCircle.value) {
    circleDrawing.end();
  }
};

// 绘制点的方法
const drawPoint = (x: number, y: number) => {
  const point = new fabric.Circle({
    left: x - annotationConfig.point.radius,
    top: y - annotationConfig.point.radius,
    radius: annotationConfig.point.radius,
    ...annotationConfig.point,
  });
  canvasInstance.value?.add(point);
  history.value.push(point);
};

// 加载背景图
const loadBackgroundImage = async (imgUrl: string) => {
  if (!canvasInstance.value || !imgUrl) return;
  // 清除旧背景
  canvasInstance.value.getObjects().forEach((obj) => {
    if (obj.type === "image") canvasInstance.value.remove(obj);
  });
  try {
    const { width, height } = await getImageSizeByUrl(imgUrl);
    fabric.Image.fromURL(imgUrl, (img) => {
      img.scaleToWidth(width);
      img.scaleToHeight(height);
      img.selectable = false;
      img.evented = false;
      const left = (canvasInstance.value!.width - img.getScaledWidth()) / 2;
      const top = (canvasInstance.value!.height - img.getScaledHeight()) / 2;
      img.set({ left, top }).sendToBack();
      canvasInstance.value!.add(img);
      isLoading.value = false;
    });
  } catch (error) {
    console.error("Failed to load image:", error);
    isLoading.value = false;
  }
};

// 获取图片尺寸
const getImageSizeByUrl = (
  imgUrl: string,
): Promise<{ width: number; height: number }> => {
  return new Promise((resolve, reject) => {
    if (!canvasContainer.value) return reject(new Error("Container not found"));
    const containerWidth = canvasContainer.value.offsetWidth;
    const containerHeight = canvasContainer.value.offsetHeight;
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
      } else {
        imgHeight = containerHeight;
        imgWidth = containerHeight * imgRatio;
      }
      resolve({ width: imgWidth, height: imgHeight });
    };
    image.onerror = () => reject(new Error("Failed to load image"));
    image.src = imgUrl;
  });
};

// 初始化画布
const initCanvas = () => {
  if (canvasElement.value && canvasContainer.value) {
    canvasElement.value.width = canvasContainer.value.offsetWidth;
    canvasElement.value.height = canvasContainer.value.offsetHeight;
    canvasInstance.value = new fabric.Canvas(canvasElement.value, {
      selection: true,
      width: canvasContainer.value.offsetWidth,
      height: canvasContainer.value.offsetHeight,
    });
    // 注册事件监听
    canvasInstance.value.on("mouse:down", handleMouseDown);
    canvasInstance.value.on("mouse:move", handleMouseMove);
    canvasInstance.value.on("mouse:up", handleMouseUp);
    canvasInstance.value.on("mouse:up", clearDrawing);
    canvasInstance.value.on("mouse:dblclick", handleMouseDoubleClick);
    canvasInstance.value.on("object:moving", handleObjectDrag);
    canvasInstance.value.on("mouse:wheel", handleMouseWheel);
  }
};

// 容器尺寸变化处理
let resizeObserver: ResizeObserver | null = null;
const handleResize = () => {
  if (!canvasInstance.value || !canvasContainer.value) return;
  const width = canvasContainer.value.offsetWidth;
  const height = canvasContainer.value.offsetHeight;
  canvasElement.value?.setAttribute("width", String(width));
  canvasElement.value?.setAttribute("height", String(height));
  canvasInstance.value.setDimensions({ width, height });
  // 重新加载背景图
  if (props.imageUrl) {
    loadBackgroundImage(props.imageUrl);
  }
};

// 处理鼠标滚动事件
const handleMouseWheel = (options: { e: WheelEvent }) => {
  if (!canvasInstance.value) return;
  const event = options.e;
  const delta = event.deltaY;
  const zoom = canvasInstance.value.getZoom();
  const zoomFactor = 0.1;
  const minZoom = 0.1;
  const maxZoom = 5;

  if (delta > 0) {
    if (zoom > minZoom) {
      canvasInstance.value.zoomToPoint(
        { x: event.offsetX, y: event.offsetY },
        zoom - zoomFactor,
      );
    }
  } else {
    if (zoom < maxZoom) {
      canvasInstance.value.zoomToPoint(
        { x: event.offsetX, y: event.offsetY },
        zoom + zoomFactor,
      );
    }
  }
  canvasInstance.value.renderAll();
};

// 封装画布放大、缩小、撤回功能
const zoomIn = () => {
  if (canvasInstance.value) {
    canvasInstance.value.setZoom(canvasInstance.value.getZoom() * 1.2);
  }
};

const zoomOut = () => {
  if (canvasInstance.value) {
    canvasInstance.value.setZoom(canvasInstance.value.getZoom() * 0.8);
  }
};

const revoke = () => {
  if (history.value.length > 0) {
    const lastShape = history.value.pop();
    if (lastShape) canvasInstance.value?.remove(lastShape);
  }
};

// 删除选中的标注框
const deleteSelectedObject = () => {
  if (selectedObject.value) {
    canvasInstance.value?.remove(selectedObject.value);
    const index = history.value.indexOf(selectedObject.value);
    if (index !== -1) {
      history.value.splice(index, 1);
    }
    selectedObject.value = null;
  }
};

// 生命周期钩子
onMounted(() => {
  initCanvas();
  if (canvasContainer.value) {
    resizeObserver = new ResizeObserver(() => handleResize());
    resizeObserver.observe(canvasContainer.value);
  }
  window.addEventListener("resize", handleResize);
  if (props.imageUrl) {
    loadBackgroundImage(props.imageUrl);
  }
  // 监听键盘事件
  window.addEventListener("keydown", (event) => {
    if (event.ctrlKey && event.key === "z") {
      revoke();
    } else if (event.key === "Delete") {
      deleteSelectedObject();
    }
  });
});

onUnmounted(() => {
  if (resizeObserver && canvasContainer.value) {
    resizeObserver.unobserve(canvasContainer.value);
  }
  window.removeEventListener("resize", handleResize);
  if (canvasInstance.value) {
    canvasInstance.value.off("mouse:down", handleMouseDown);
    canvasInstance.value.off("mouse:move", handleMouseMove);
    canvasInstance.value.off("mouse:up", handleMouseUp);
    canvasInstance.value.off("mouse:up", clearDrawing);
    // 移除 mouse:wheel 事件监听
    canvasInstance.value.off("mouse:wheel");
  }
  // 移除键盘事件监听
  window.removeEventListener("keydown", (event) => {
    if (event.ctrlKey && event.key === "z") {
      revoke();
    } else if (event.key === "Delete") {
      deleteSelectedObject();
    }
  });
});

// 监听模式变化
watch(
  () => props.mode,
  (newMode) => {
    if (!canvasInstance.value) return;
    switch (newMode) {
      case "PAN":
        canvasInstance.value.defaultCursor = "grab";
        break;
      case "ZOOM_IN":
      case "ZOOM_OUT":
        canvasInstance.value.defaultCursor = "zoom-in";
        break;
      default:
        canDraw.value = true;
        canvasInstance.value.defaultCursor = "default";
        break;
    }
  },
);

// 监听图片变化
watch(
  () => props.imageUrl,
  (newUrl) => {
    if (newUrl) {
      isLoading.value = true;
      // 清空之前的标注框
      history.value.forEach((obj) => {
        canvasInstance.value?.remove(obj);
      });
      history.value = [];
      loadBackgroundImage(newUrl);
    }
  },
);

// 通过 defineExpose 暴露方法
defineExpose({
  zoomIn,
  zoomOut,
  revoke,
  deleteSelectedObject,
});
</script>

<style scoped>
/* Tailwind CSS 样式 */
</style>
