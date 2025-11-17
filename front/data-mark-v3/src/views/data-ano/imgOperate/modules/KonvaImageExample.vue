<template>
  <div ref="container" class="w-full h-full"></div>
</template>

<script setup lang="ts">
import { ref, onMounted, defineProps, watch } from 'vue';
import * as Konva from 'konva';

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

type AnnotationShapeInfo = {
  shape: Konva.Shape;
  show: boolean;
  type: string;
  text: Konva.Text;
  tag?: string;
};

// 定义 mode 类型
type Mode = 'POINT' | 'POLYGON' | 'RECT' | 'CIRCLE' | 'Revoke' | 'PAN' | 'zoomIn' | 'zoomOut';

const props = defineProps<{
  imageUrl: string;
  mode: Mode;
  selectedImageInfo: ImageItem;
}>();

const container = ref<HTMLDivElement | null>(null);
const annotationShapes = ref<AnnotationShapeInfo[]>([]);
const undoStack = ref<{ type: 'add' | 'delete'; shapeInfo: AnnotationShapeInfo; index?: number }[]>([]);
const selectedShape = ref<{ shapeInfo: AnnotationShapeInfo } | null>(null);
const showModal = ref(false);
const modalX = ref(0);
const modalY = ref(0);

// 用于绘制多边形时存储点
const polygonPoints = ref<number[]>([]);
// 记录鼠标按下时的位置
const startPoint = ref<{ x: number; y: number } | null>(null);
// 用于临时存储正在绘制的形状
let tempShape: Konva.Shape | null = null;
let tempText: Konva.Text | null = null;
// 标记是否正在绘制多边形
let isDrawingPolygon = false;
// 标记是否正在移动选中的形状
let isMovingShape = false;
// 记录移动形状时鼠标的起始位置
let moveStartPoint: { x: number; y: number } | null = null;

// 使用 Image 对象获取图片真实宽高
const getImageSizeByUrl = (imgUrl: string): Promise<{ width: number; height: number; x: number; y: number }> => {
  return new Promise((resolve, reject) => {
    if (!container.value) {
      reject(new Error('容器元素未找到'));
      return;
    }
    const containerWidth = container.value.offsetWidth;
    const containerHeight = container.value.offsetHeight;
    const image = new Image();
    image.crossOrigin = 'Anonymous';
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
        x: (containerWidth - imgWidth) / 2,
        y: (containerHeight - imgHeight) / 2
      });
    };
    image.onerror = () => {
      reject(new Error('图片加载失败'));
    };
    image.src = imgUrl;
  });
};

const loadImage = async (imageUrl: string) => {
  if (!container.value) return;

  // 创建舞台
  const stage = new Konva.Stage({
    container: container.value,
    width: container.value.offsetWidth,
    height: container.value.offsetHeight
  });

  // 创建图层
  const layer = new Konva.Layer();
  stage.add(layer);

  try {
    const { width, height, x, y } = await getImageSizeByUrl(imageUrl);
    const img = new Image();
    img.crossOrigin = 'Anonymous';
    img.src = imageUrl;
    await new Promise((resolve, reject) => {
      img.onload = resolve;
      img.onerror = reject;
    });

    // 创建图像对象，使用计算后的尺寸和位置
    const konvaImage = new Konva.Image({
      x,
      y,
      width,
      height,
      image: img
    });

    // 将图像添加到图层
    layer.add(konvaImage);
    // 绘制图层
    layer.draw();

    const addAnnotationBasedOnMode = (x: number, y: number, width?: number, height?: number, points?: number[], radius?: number) => {
      switch (props.mode) {
        case 'POINT':
          addPoint(x, y);
          break;
        case 'RECT':
          if (width !== undefined && height !== undefined) {
            addRectangle(x, y, width, height);
          }
          break;
        case 'POLYGON':
          if (points !== undefined) {
            addPolygon(points);
          }
          break;
        case 'CIRCLE':
          if (radius !== undefined) {
            addCircle(x, y, radius);
          }
          break;
      }
    };

    const addPoint = (x: number, y: number) => {
      const point = new Konva.Circle({
        x,
        y,
        radius: 5,
        fill: 'red',
        stroke: 'black',
        strokeWidth: 1
      });
      layer.add(point);
      const text = new Konva.Text({
        x: x + 10,
        y: y - 10,
        text: '',
        fontSize: 12,
        fill: 'black'
      });
      layer.add(text);
      const shapeInfo: AnnotationShapeInfo = { shape: point, show: true, type: '点', text };
      annotationShapes.value.push(shapeInfo);
      undoStack.value.push({ type: 'add', shapeInfo });
      layer.draw();
      point.on('click', () => {
        selectShape({ shapeInfo });
        showModalForShape(point);
        console.log('点标注被点击');
      });
      point.on('dblclick', () => {
        startMovingShape({ shapeInfo });
      });
    };

    const addRectangle = (x: number, y: number, width: number, height: number) => {
      const rectangle = new Konva.Rect({
        x,
        y,
        width,
        height,
        fill: 'transparent',
        stroke: 'blue',
        strokeWidth: 2
      });
      layer.add(rectangle);
      const text = new Konva.Text({
        x: x + width + 10,
        y: y,
        text: '',
        fontSize: 12,
        fill: 'black'
      });
      layer.add(text);
      const shapeInfo: AnnotationShapeInfo = { shape: rectangle, show: true, type: '矩形', text };
      annotationShapes.value.push(shapeInfo);
      undoStack.value.push({ type: 'add', shapeInfo });
      layer.draw();
      rectangle.on('click', () => {
        selectShape({ shapeInfo });
        showModalForShape(rectangle);
        console.log('矩形标注被点击');
      });
      rectangle.on('dblclick', () => {
        startMovingShape({ shapeInfo });
      });
    };

    const addPolygon = (points: number[]) => {
      const polygon = new Konva.Line({
        points,
        fill: 'transparent',
        stroke: 'green',
        strokeWidth: 2,
        closed: true
      });
      layer.add(polygon);
      const centroid = polygon.getClientRect();
      const text = new Konva.Text({
        x: centroid.x + centroid.width + 10,
        y: centroid.y,
        text: '',
        fontSize: 12,
        fill: 'black'
      });
      layer.add(text);
      const shapeInfo: AnnotationShapeInfo = { shape: polygon, show: true, type: '多边形', text };
      annotationShapes.value.push(shapeInfo);
      undoStack.value.push({ type: 'add', shapeInfo });
      layer.draw();
      polygon.on('click', () => {
        selectShape({ shapeInfo });
        showModalForShape(polygon);
        console.log('多边形标注被点击');
      });
      polygon.on('dblclick', () => {
        startMovingShape({ shapeInfo });
      });
    };

    const addCircle = (x: number, y: number, radius: number) => {
      const circle = new Konva.Circle({
        x,
        y,
        radius,
        fill: 'transparent',
        stroke: 'yellow',
        strokeWidth: 2
      });
      layer.add(circle);
      const text = new Konva.Text({
        x: x + radius + 10,
        y: y,
        text: '',
        fontSize: 12,
        fill: 'black'
      });
      layer.add(text);
      const shapeInfo: AnnotationShapeInfo = { shape: circle, show: true, type: '圆形', text };
      annotationShapes.value.push(shapeInfo);
      undoStack.value.push({ type: 'add', shapeInfo });
      layer.draw();
      circle.on('click', () => {
        selectShape({ shapeInfo });
        showModalForShape(circle);
        console.log('圆形标注被点击');
      });
      circle.on('dblclick', () => {
        startMovingShape({ shapeInfo });
      });
    };

    let isDragging = false;
    let lastPointerPosition: Konva.Vector2d | null = null;

    stage.on('mousedown touchstart', (e: Konva.KonvaEventObject<MouseEvent | TouchEvent>) => {
      if (props.mode === 'PAN') {
        isDragging = true;
        lastPointerPosition = stage.getPointerPosition();
      } else if (['POINT', 'RECT', 'CIRCLE', 'POLYGON'].includes(props.mode)) {
        // 如果已经选中了某个形状，不允许开始新的绘制
        if (selectedShape.value) {
          return;
        }
        const pos = stage.getPointerPosition();
        if (pos) {
          startPoint.value = { x: pos.x, y: pos.y };
          if (props.mode === 'POINT') {
            addPoint(pos.x, pos.y);
          } else if (props.mode === 'POLYGON') {
            if (!isDrawingPolygon) {
              isDrawingPolygon = true;
              polygonPoints.value = [pos.x, pos.y];
              tempShape = new Konva.Line({
                points: polygonPoints.value,
                fill: 'transparent',
                stroke: 'green',
                strokeWidth: 2,
                closed: false
              });
              layer.add(tempShape);
              layer.draw();
            } else {
              polygonPoints.value.push(pos.x, pos.y);
              if (tempShape) {
                tempShape.points(polygonPoints.value);
                layer.draw();
              }
            }
          } else {
            if (props.mode === 'RECT') {
              tempShape = new Konva.Rect({
                x: pos.x,
                y: pos.y,
                width: 0,
                height: 0,
                fill: 'transparent',
                stroke: 'blue',
                strokeWidth: 2
              });
            } else if (props.mode === 'CIRCLE') {
              tempShape = new Konva.Circle({
                x: pos.x,
                y: pos.y,
                radius: 0,
                fill: 'transparent',
                stroke: 'yellow',
                strokeWidth: 2
              });
            }
            if (tempShape) {
              layer.add(tempShape);
              layer.draw();
            }
          }
        }
      }
    });

    stage.on('mousemove touchmove', (e: Konva.KonvaEventObject<MouseEvent | TouchEvent>) => {
      if (isDragging) {
        const newPointerPosition = stage.getPointerPosition();
        if (lastPointerPosition) {
          const dx = newPointerPosition.x - lastPointerPosition.x;
          const dy = newPointerPosition.y - lastPointerPosition.y;
          stage.position({
            x: stage.x() + dx,
            y: stage.y() + dy
          });
          lastPointerPosition = newPointerPosition;
          layer.draw();
        }
      } else if (startPoint.value && ['RECT', 'CIRCLE'].includes(props.mode)) {
        const pos = stage.getPointerPosition();
        if (pos) {
          if (props.mode === 'RECT') {
            const width = pos.x - startPoint.value.x;
            const height = pos.y - startPoint.value.y;
            if (tempShape) {
              tempShape.setAttrs({
                width,
                height
              });
              layer.draw();
            }
          } else if (props.mode === 'CIRCLE') {
            const dx = pos.x - startPoint.value.x;
            const dy = pos.y - startPoint.value.y;
            const radius = Math.sqrt(dx * dx + dy * dy);
            if (tempShape) {
              tempShape.setAttrs({
                radius
              });
              layer.draw();
            }
          }
        }
      }
      // 移动选中的形状
      if (isMovingShape && selectedShape.value) {
        const newPointerPosition = stage.getPointerPosition();
        if (moveStartPoint && newPointerPosition) {
          const dx = newPointerPosition.x - moveStartPoint.x;
          const dy = newPointerPosition.y - moveStartPoint.y;
          const shape = selectedShape.value.shapeInfo.shape;
          shape.position({
            x: shape.x() + dx,
            y: shape.y() + dy
          });
          const text = selectedShape.value.shapeInfo.text;
          text.position({
            x: text.x() + dx,
            y: text.y() + dy
          });
          moveStartPoint = newPointerPosition;
          layer.draw();
        }
      }
    });

    stage.on('dblclick dbltap', (e: Konva.KonvaEventObject<MouseEvent | TouchEvent>) => {
      if (props.mode === 'POLYGON' && isDrawingPolygon) {
        if (polygonPoints.value.length >= 6) {
          addPolygon(polygonPoints.value);
          if (tempShape) {
            tempShape.destroy();
            tempText?.destroy();
            tempShape = null;
            tempText = null;
          }
          polygonPoints.value = [];
          isDrawingPolygon = false;
        }
      }
    });

    stage.on('mouseup touchend', (e: Konva.KonvaEventObject<MouseEvent | TouchEvent>) => {
      if (isDragging) {
        isDragging = false;
      } else if (startPoint.value) {
        const pos = stage.getPointerPosition();
        if (pos) {
          if (props.mode === 'RECT') {
            const width = pos.x - startPoint.value.x;
            const height = pos.y - startPoint.value.y;
            addRectangle(startPoint.value.x, startPoint.value.y, width, height);
          } else if (props.mode === 'CIRCLE') {
            const dx = pos.x - startPoint.value.x;
            const dy = pos.y - startPoint.value.y;
            const radius = Math.sqrt(dx * dx + dy * dy);
            addCircle(startPoint.value.x, startPoint.value.y, radius);
          }
          if (tempShape) {
            tempShape.destroy();
            tempText?.destroy();
            tempShape = null;
            tempText = null;
          }
          startPoint.value = null;
        }
      }
      // 停止移动选中的形状
      if (isMovingShape) {
        isMovingShape = false;
        moveStartPoint = null;
      }
    });

    stage.on('wheel', (e: Konva.KonvaEventObject<WheelEvent>) => {
      if (props.mode === 'zoomIn' || props.mode === 'zoomOut') {
        const oldScale = stage.scaleX();
        const pointer = stage.getPointerPosition();
        const mousePointTo = {
          x: (pointer.x - stage.x()) / oldScale,
          y: (pointer.y - stage.y()) / oldScale
        };
        const newScale = props.mode === 'zoomIn' ? oldScale * 1.1 : oldScale * 0.9;
        stage.scale({ x: newScale, y: newScale });
        const newPos = {
          x: pointer.x - mousePointTo.x * newScale,
          y: pointer.y - mousePointTo.y * newScale
        };
        stage.position(newPos);
        layer.draw();
      }
    });

    const selectShape = (shapeObj: { shapeInfo: AnnotationShapeInfo }) => {
      if (selectedShape.value) {
        const prevShape = selectedShape.value.shapeInfo.shape;
        if (prevShape instanceof Konva.Circle) {
          prevShape.stroke('black');
          prevShape.strokeWidth(1);
        } else if (prevShape instanceof Konva.Rect) {
          prevShape.stroke('blue');
          prevShape.strokeWidth(2);
        } else if (prevShape instanceof Konva.Polygon) {
          prevShape.stroke('green');
          prevShape.strokeWidth(2);
        }
      }
      selectedShape.value = shapeObj;
      const currentShape = shapeObj.shapeInfo.shape;
      if (currentShape instanceof Konva.Circle) {
        currentShape.stroke('purple');
        currentShape.strokeWidth(3);
      } else if (currentShape instanceof Konva.Rect) {
        currentShape.stroke('purple');
        currentShape.strokeWidth(4);
      } else if (currentShape instanceof Konva.Polygon) {
        currentShape.stroke('purple');
        currentShape.strokeWidth(4);
      }
      layer.draw();
    };

    const showModalForShape = (shape: Konva.Shape) => {
      const { x, y } = shape.getAbsolutePosition();
      const containerRect = (container.value as HTMLElement).getBoundingClientRect();
      const modalElement = document.querySelector('.absolute.bg-white') as HTMLElement;
      const modalWidth = modalElement ? modalElement.offsetWidth : 0;
      const modalHeight = modalElement ? modalElement.offsetHeight : 0;

      let newX = x;
      if (x + modalWidth > containerRect.right) {
        newX = containerRect.right - modalWidth;
      }
      if (newX < containerRect.left) {
        newX = containerRect.left;
      }

      let newY = y + (shape as any).height() + 10;
      if (newY + modalHeight > containerRect.bottom) {
        newY = y - modalHeight - 10;
        if (newY < containerRect.top) {
          newY = containerRect.top;
        }
      }

      modalX.value = newX;
      modalY.value = newY;
      showModal.value = true;
    };

    const toggleSingleAnnotation = (index: number) => {
      const shapeInfo = annotationShapes.value[index];
      shapeInfo.show = !shapeInfo.show;
      shapeInfo.shape.visible(shapeInfo.show);
      if (shapeInfo.text) {
        shapeInfo.text.visible(shapeInfo.show);
      }
      layer.draw();
    };

    const deleteAnnotation = (index: number) => {
      const shapeInfo = annotationShapes.value[index];
      shapeInfo.shape.destroy();
      if (shapeInfo.text) {
        shapeInfo.text.destroy();
      }
      annotationShapes.value.splice(index, 1);
      undoStack.value.push({ type: 'delete', shapeInfo, index });
      layer.draw();
    };

    const undoOperation = () => {
      if (undoStack.value.length === 0) return;
      const lastOperation = undoStack.value.pop()!;
      if (lastOperation.type === 'add') {
        lastOperation.shapeInfo.shape.destroy();
        if (lastOperation.shapeInfo.text) {
          lastOperation.shapeInfo.text.destroy();
        }
        const index = annotationShapes.value.findIndex(
          (info) => info.shape === lastOperation.shapeInfo.shape
        );
        if (index !== -1) {
          annotationShapes.value.splice(index, 1);
        }
      } else if (lastOperation.type === 'delete') {
        const { shapeInfo, index } = lastOperation;
        layer.add(shapeInfo.shape);
        if (shapeInfo.text) {
          layer.add(shapeInfo.text);
        }
        annotationShapes.value.splice(index, 0, shapeInfo);
      }
      layer.draw();
    };

    const saveTag = (index: number) => {
      const shapeInfo = annotationShapes.value[index];
      if (shapeInfo.text) {
        shapeInfo.text.text(shapeInfo.tag || '');
        layer.draw();
      }
      // 这里可以添加保存标签到后端的逻辑
      console.log(`保存标签: ${shapeInfo.tag} 到 ${shapeInfo.type} 标注框`);
    };

    const startMovingShape = (shapeObj: { shapeInfo: AnnotationShapeInfo }) => {
      selectShape(shapeObj);
      isMovingShape = true;
      moveStartPoint = stage.getPointerPosition();
    };

    watch(() => props.mode, (newMode) => {
      if (newMode === 'Revoke') {
        undoOperation();
      }
    });

    // 这里可以添加调用 addAnnotationBasedOnMode 的逻辑，根据实际需求触发
  } catch (error) {
    console.error('图片加载或处理失败:', error);
  }
};

onMounted(() => {
  loadImage(props.imageUrl);
});

watch(() => props.imageUrl, (newImageUrl) => {
  loadImage(newImageUrl);
});
</script>

<style scoped>
/* 可以在这里添加样式 */
</style>    