// useImageAnnotation.ts
import { onMounted, onUnmounted, reactive, ref, Ref, computed } from 'vue';
import type { Tool } from './types';
import { useRoute } from 'vue-router';

import AILabel from "ailabel";
import { nanoid } from "nanoid";
import { segmentStart, getDataSonType } from '../../../service/api/ano';
export interface AnnotationOptions {
  containerId: string;
  containerRef: Ref<HTMLElement | null>;
  defaultImageUrl?: string;
  defaultZoom?: number;
}

export function useImageAnnotation(options: AnnotationOptions) {
  const { containerId, containerRef, defaultImageUrl = '', defaultZoom = 1200 } = options as any;

  // 地图和图层引用
  const gMap = ref<any>(null);
  const featureLayer = ref<any>(null);
  const maskLayer = ref<any>(null);
  const imageLayer = ref<any>(null);
  const textLayer = ref<any>(null);

  let latestImageRequest: string | null = null;  // 跟踪最新的图片请求

  // 状态管理
  const state = reactive<any>({
    isInitialized: false,
    centerObj: { x: 0, y: 0 },
    allFeatures: [] as any[],
    allTexts: [] as any[],
    undoStack: [] as any[],
    redoStack: [] as any[],
    drawingStyle: {},
    textStyle: {
      fillStyle: "#F4A460",
      strokeStyle: "#D2691E",
      background: true,
      globalAlpha: 1,
      fontColor: "white",
      fontSize: 16,
      fontWeight: 'normal',
      fontFamily: 'Arial, sans-serif'
    },
    currentMode: 'RECT',
    originalImageWidth: 0, // 原图实际宽度
    originalImageHeight: 0, // 原图实际高度
    currentViewWidth: 0, // 当前视图宽度
    currentViewHeight: 0, // 当前视图高度
    activeImageUrl: defaultImageUrl,
    isImgRender: false,
    isAutoLabel: false,
    foreground_points: [],
    background_points: [],
    isImgInvalid: 1, // 是否为无效数据 0：无效 1：有效
    anoType: 1,   // 1显示baseTools 2显示segmentTools
  });

  const tools = ref<Tool[]>([]);

  // { mode: 'POINT', icon: 'ph:dot-fill', label: '点' },
  // { mode: 'LINE', icon: 'humbleicons:minus', label: '线段' },
  // { mode: 'DRAWMASK', icon: 'material-symbols:brush', label: '涂抹' },

  // 工具列表
  // const tools: Tool[] = [
  //   { mode: 'PAN', icon: 'humbleicons:arrows', localIcon: "humbleicons--arrows", label: '平移' },
  //   { mode: 'POINT-FOREGROUND', icon: "mdi:auto-fix", localIcon: "mdi--auto-fix", label: '前景点标注' },
  //   { mode: 'POINT-BACKGROUND', icon: "ic:sharp-auto-fix-off", localIcon: "ic--sharp-auto-fix-off", label: '背景点标注' },
  //   { mode: 'CIRCLE', icon: 'humbleicons:circle', localIcon: "material-symbols--circle-outline", label: '圆' },
  //   { mode: 'RECT', icon: 'material-symbols:rectangle-outline', localIcon: "ph--rectangle-bold", label: '矩形' },
  //   { mode: 'POLYGON', icon: 'tabler:polygon', localIcon: "tabler--polygon", label: '多边形' },
  //   { mode: 'UNDO', icon: 'icon-park-outline:undo', localIcon: 'icon-park-outline--undo', label: '撤销' },
  //   { mode: 'REDO', icon: 'icon-park-outline:redo', localIcon: 'icon-park-outline--redo', label: '重做' }
  // ];

  // 获取路由实例
  const route = useRoute();

  // 基础工具列表（提取为基础数据，避免重复定义）
  const baseTools: Tool[] = [
    { mode: 'PAN', icon: 'humbleicons:arrows', localIcon: "humbleicons--arrows", label: '平移' },
    { mode: 'POINT-FOREGROUND', icon: "mdi:auto-fix", localIcon: "mdi--auto-fix", label: '前景点标注' },
    { mode: 'POINT-BACKGROUND', icon: "ic:sharp-auto-fix-off", localIcon: "ic--sharp-auto-fix-off", label: '背景点标注' },
    { mode: 'CIRCLE', icon: 'humbleicons:circle', localIcon: "material-symbols--circle-outline", label: '圆' },
    { mode: 'RECT', icon: 'material-symbols:rectangle-outline', localIcon: "ph--rectangle-bold", label: '矩形' },
    { mode: 'POLYGON', icon: 'tabler:polygon', localIcon: "tabler--polygon", label: '多边形' },
    { mode: 'UNDO', icon: 'icon-park-outline:undo', localIcon: 'icon-park-outline--undo', label: '撤销' },
    { mode: 'REDO', icon: 'icon-park-outline:redo', localIcon: 'icon-park-outline--redo', label: '重做' }
  ];

  // 需要在markType=0时显示的工具mode集合
  const allowedModesWhenMarkType0 = ['PAN', 'RECT', 'UNDO', 'REDO'];

  // 计算属性：根据markType动态返回工具列表
  // const tools = computed<Tool[]>(() => {
  //   const markType = route.query.markType === undefined
  //     ? 1
  //     : Number(route.query.markType);

  //   return markType == 1
  //     ? baseTools  // markType=1时显示全部工具
  //     : baseTools.filter(tool => allowedModesWhenMarkType0.includes(tool.mode));  // markType=0时只显示指定工具
  // });

  // watch state.anoType
  watch(() => state.anoType, (newVal) => {
    if (newVal == 1) {
      tools.value = baseTools;
    } else {
      // anoType=0时只显示指定工具
      tools.value = baseTools.filter(tool => allowedModesWhenMarkType0.includes(tool.mode));;
    }
  }, {
    immediate: true
  });

  onMounted(async () => {
    await getDataSonType({
      sonId: route.query?.id,
    }).then(res => {
      if (res?.data) {
        const { anoType } = res.data;
        state.anoType = anoType.toString();
      }
    });

    if (route.query.anoType === 'audit' || route.query.anoType === 'result') {
      setMode('PAN')
    }
  });

  onUnmounted(() => {
    // 清理资源
    if (gMap.value) {
      gMap.value.destroy();
    }
    document.removeEventListener('keydown', handleKeydown);

    // onresize
    window.removeEventListener('resize', () => {
      resize()
    })
  });

  async function initAnnotation() {
    if (state.isInitialized) return;
    try {
      await loadImage(state.activeImageUrl);
      initMap();
      initLayers();
      bindEvents();

      // onresize
      window.addEventListener('resize', () => resize());
      state.isInitialized = true;
    } catch (error) {
      console.error('初始化失败:', error);
    }
  }

  // 初始化方法
  function initMap() {
    if (!containerRef.value) return;

    gMap.value = new AILabel.Map(containerId, {
      center: state.centerObj,
      zoom: defaultZoom,
      mode: state.currentMode,
      zoomWhenDrawing: true,
      refreshDelayWhenZooming: true,
      panWhenDrawing: true,
      zoomWheelRatio: 5,
      withHotKeys: true,
    });
  }

  function initLayers() {
    imageLayer.value = createImageLayer();
    featureLayer.value = createFeatureLayer();
    textLayer.value = createTextLayer();
    maskLayer.value = createMaskLayer();

    gMap.value.addLayer(imageLayer.value);
    gMap.value.addLayer(featureLayer.value);
    gMap.value.addLayer(textLayer.value);
    gMap.value.addLayer(maskLayer.value);
  }

  // 图层创建方法
  function createFeatureLayer() {
    return new AILabel.Layer.Feature('feature-layer', { name: '矢量图层' }, { zIndex: 16 });
  }

  function createMaskLayer() {
    return new AILabel.Layer.Mask(
      'first-layer-mask',
      { name: '第一个涂抹图层' },
      { zIndex: 11, opacity: .5 }
    );
  }

  function createImageLayer() {
    return new AILabel.Layer.Image('image-layer', {
      src: state.activeImageUrl,
      width: state.currentViewWidth,
      height: state.currentViewHeight,
      crossOrigin: true,
    }, { name: '图片图层' }, { zIndex: 1 });
  }

  function createTextLayer() {
    return new AILabel.Layer.Text(
      'text-layer',
      { name: '文本图层' },
      { zIndex: 12, opacity: 1 }
    );
  }

  // 图片加载方法
  async function loadImage(url: string) {
    latestImageRequest = url;
    return new Promise((resolve, reject) => {
      if (!containerRef.value) {
        reject(new Error('标注容器未找到'));
        return;
      }

      if (!url) {
        if (imageLayer.value && latestImageRequest === url) {
          // 更新图片图层信息
          imageLayer.value.updateImageInfo({
            src: url,
            width: 0,
            height: 0,
          });
        }
        // reject(new Error('图片路径未找到'));
        resolve("图片路径未找到");
        return;
      }

      state.isImgRender = true;

      const img = new Image();

      img.onload = () => {
        // 检查是否是最新请求
        if (latestImageRequest !== url) {
          resolve(img); // 虽然加载完成，但不是最新请求，不更新UI
          return;
        }

        state.isImgRender = false;
        try {
          const containerWidth = containerRef.value.offsetWidth;
          const containerHeight = containerRef.value.offsetHeight;
          const imgWidth = img.width;
          const imgHeight = img.height;

          // 计算图片和容器的宽高比
          const imgRatio = imgWidth / imgHeight;
          const containerRatio = containerWidth / containerHeight;

          // 计算等比例缩放的尺寸
          let newWidth = imgWidth, newHeight = imgHeight;
          if (imgRatio > containerRatio) {
            newWidth = containerWidth;
            newHeight = containerWidth / imgRatio;
          } else if (imgRatio < containerRatio) {
            newHeight = containerHeight;
            newWidth = containerHeight * imgRatio;
          } else {
            newWidth = containerWidth;
            newHeight = containerHeight;
          }

          // 更新原图的实际宽度和高度
          state.originalImageWidth = imgWidth;
          state.originalImageHeight = imgHeight;

          // 更新当前用户界面中图片的显示宽度和高度
          state.currentViewWidth = newWidth;
          state.currentViewHeight = newHeight;

          state.centerObj = {
            x: newWidth / 2,
            y: newHeight / 2
          };

          if (imageLayer.value) {
            // 更新图片图层信息
            imageLayer.value.updateImageInfo({
              src: url,
              width: newWidth,
              height: newHeight,
            });
          }

          if (gMap.value) {
            gMap.value.setCenter({
              x: newWidth / 2,
              y: newHeight / 2
            });

            gMap.value.zoom = 1200;
          }

          state.activeImageUrl = url;
          resolve(img);
        } catch (error) {
          console.error('图片加载后处理失败:', error);
          reject(error);
        }
      };

      img.onerror = () => {
        // 检查是否是最新请求
        if (latestImageRequest !== url) {
          reject(new Error(`图片加载失败但非最新请求: ${url}`));
          return;
        }

        state.isImgRender = false;
        console.error('图片加载失败:', url);
        reject(new Error(`图片加载失败: ${url}`));
      };

      img.crossOrigin = "Anonymous";
      img.src = url;
    });
  }

  // 模式设置方法
  function setMode(mode: string) {
    if (mode === 'UNDO') {
      undo();
      return;
    }

    if (mode === 'REDO') {
      redo();
      return;
    }

    const gMapMode = mode === 'POINT-FOREGROUND' || mode === 'POINT-BACKGROUND' ? 'POINT' : mode;

    if (mode === 'POINT-FOREGROUND') {
      state.foreground_points = [];
      state.background_points = [];
    }

    if (mode === 'POINT-BACKGROUND') {
      state.background_points = [];
    }

    if (!Boolean(state.isImgInvalid)) {
      window.$message?.warning?.("已被标注为无效数据，请先取消勾选后再标注！");
      return;
    }

    state.currentMode = mode;
    if (gMap.value) {
      gMap.value.setMode(gMapMode);
      setDrawingStyle(mode);
    }
  }

  // 获取模式名称
  function getModeName(mode: string) {
    const modeNames: Record<string, string> = {
      'PAN': '平移',
      'POINT': '点',
      'POINT-FOREGROUND': '前景点',
      'POINT-BACKGROUND': '背景点',
      'LINE': '线段',
      'CIRCLE': '圆',
      'RECT': '矩形',
      'POLYGON': '多边形',
      'DRAWMASK': '涂抹'
    };

    return modeNames[mode] || mode;
  }

  // 设置绘制样式
  function setDrawingStyle(mode: string) {
    // 为不同类型的标注定义独特的样式
    const styleMap: any = {
      'POINT': {
        fillStyle: '#409EFF',
        strokeStyle: '#FFFFFF',
        lineWidth: 1,
        radius: 6,
        shadowColor: 'rgba(64, 158, 255, 0.5)',
        shadowBlur: 5,
        cursor: 'crosshair'
      },
      'POINT-FOREGROUND': {
        fillStyle: '#409EFF',
        strokeStyle: '#FFFFFF',
        lineWidth: 1,
        radius: 6,
        shadowColor: 'rgba(64, 158, 255, 0.5)',
        shadowBlur: 5,
        cursor: 'crosshair'
      },
      'POINT-BACKGROUND': {
        fillStyle: '#F5222D',
        strokeStyle: '#FFFFFF',
        lineWidth: 1,
        radius: 6,
        shadowColor: 'rgba(245, 34, 45, 0.5)',
        shadowBlur: 5,
        cursor: 'crosshair'
      },
      'LINE': {
        strokeStyle: '#52C41A',
        lineJoin: 'round',
        lineCap: 'round',
        lineWidth: 3,
        dashPattern: [10, 0],
        cursor: 'crosshair'
      },
      "CIRCLE": {
        "opacity": 1,
        "fillStyle": "#E6F7FF",
        "lineWidth": 2,
        "strokeStyle": "#1890FF",
        "fill": true,
        "globalAlpha": 0.3
      },
      "RECT": {
        "opacity": 1,
        "fillStyle": "#FFF7E6",
        "lineWidth": 2,
        "strokeStyle": "#FA8C16",
        "fill": true,
        "globalAlpha": 0.3
      },
      "POLYGON": {
        "opacity": 1,
        "fillStyle": "#F6FFED",
        "lineWidth": 2,
        "strokeStyle": "#52C41A",
        "fill": true,
        "globalAlpha": 0.3
      },
      'DRAWMASK': {
        strokeStyle: 'rgba(248, 114, 114, 0.8)',
        fillStyle: 'rgba(248, 114, 114, 0.3)',
        lineWidth: 20,
        lineCap: 'round',
        lineJoin: 'round',
        shadowColor: 'rgba(248, 114, 114, 0.5)',
        shadowBlur: 10,
        cursor: 'crosshair'
      },
      default: {
        "opacity": 1,
        "fillStyle": "#FFF7E6",
        "lineWidth": 2,
        "strokeStyle": "#FA8C16",
        "fill": true,
        "globalAlpha": 0.3
      }
    };

    state.drawingStyle = styleMap[mode] || styleMap.default;
    if (gMap.value) {
      gMap.value.setDrawingStyle(state.drawingStyle);
    }
  }

  // 事件绑定
  function bindEvents() {
    if (!gMap.value) return;

    gMap.value.events.on('drawDone', (mapMode: string, shape: any) => {
      if (!Boolean(state.isImgInvalid)) {
        window.$message?.warning?.("已被标注为无效数据，请先取消勾选后再标注！");
        return;
      }

      const feature = createFeature(mapMode, shape);
      if (feature) {
        addFeature(feature);
        state.allFeatures = featureLayer.value.getAllFeatures();
        state.allTexts = textLayer.value.getAllTexts();

        // active feature
        gMap.value.setActiveFeature(feature);
      }

      if (mapMode === "POINT") {
        if (state.currentMode === "POINT-FOREGROUND") {
          state.foreground_points.push([shape.x, shape.y]);
          renderPointData(state.foreground_points);
        }
        if (state.currentMode === "POINT-BACKGROUND") {
          state.background_points.push([shape.x, shape.y]);
          renderPointData(state.background_points);
        }
      }
    });

    gMap.value.events.on('draging', (feature: any, shape: any) => {
      if (feature !== null) {
        const { type, id } = feature;
        const activeText = textLayer.value.getTextById(feature.id);
        const textShape = getTextShapePosition(type, shape);
        activeText.updatePosition(textShape);
        refresh();
        const index = state.allFeatures.findIndex(item => item.id === id);
        if (index !== -1) {
          state.allFeatures[index] = feature;
        }
      }
    });

    gMap.value.events.on('featureSelected', (feature: any) => {
      gMap.value.setActiveFeature(feature);
    });

    gMap.value.events.on('featureUnselected', () => {
      gMap.value.setActiveFeature(null);
    });

    gMap.value.events.on('featureUpdated', (feature: any, shape: any) => {
      feature.updateShape(shape);
    });

    gMap.value.events.on('featureDeleted', ({ id: featureId }: { id: string }) => {
      const feature = featureLayer.value.getFeatureById(featureId);
      if (feature) {
        state.undoStack.push({
          type: 'removeFeature',
          layer: 'feature',
          id: featureId,
          data: feature
        });
        state.redoStack = [];
      }
      featureLayer.value.removeFeatureById(featureId);
    });

    // 键盘事件
    document.addEventListener('keydown', handleKeydown);
  }

  // 键盘事件处理
  function handleKeydown(e: KeyboardEvent) {
    if (e.key === 'Delete') {
      deleteSelectedFeature();
    }
  }

  // 缩放控制
  function zoomIn() {
    if (gMap.value) {
      gMap.value.zoomIn();
    }
  }

  function zoomOut() {
    if (gMap.value) {
      gMap.value.zoomOut();
    }
  }

  // 创建标注要素
  function createFeature(mode: string, shape: any, id?: string, drawingStyle = state.drawingStyle) {
    const timestamp = Date.now();
    const randomId = nanoid();
    const featureId = id !== undefined ? id : `feature-${randomId}-${timestamp}`;
    // const props = { name: getModeName(mode), timestamp: Date.now() };
    const props = { name: undefined, timestamp: timestamp };

    const featureMap: Record<string, any> = {
      'POINT': () => new AILabel.Feature.Point(featureId, { ...shape, sr: 6 }, props, drawingStyle),
      'POINT-FOREGROUND': () => new AILabel.Feature.Point(featureId, { ...shape, sr: 6 }, props, drawingStyle),
      'POINT-BACKGROUND': () => new AILabel.Feature.Point(featureId, { ...shape, sr: 6 }, props, drawingStyle),
      'LINE': () => new AILabel.Feature.Line(featureId, { ...shape, width: 10 }, props, drawingStyle),
      'POLYLINE': () => new AILabel.Feature.Polyline(featureId, shape, props, drawingStyle),
      'RECT': () => new AILabel.Feature.Rect(featureId, shape, props, drawingStyle),
      'CIRCLE': () => new AILabel.Feature.Circle(featureId, shape, props, drawingStyle),
      'POLYGON': () => new AILabel.Feature.Polygon(featureId, { points: shape }, props, drawingStyle),
      'DRAWMASK': () => new AILabel.Mask.Draw(
        featureId,
        '铅笔',
        { points: shape, width: 20 },
        { name: '港币', price: '1元' },
        { strokeStyle: '#FF0000' }
      )
    };

    const feature = featureMap[mode] ? featureMap[mode]() : null;
    if (feature) {
      feature.isEye = true;
      feature.operateIdx = new Date().getTime();
    }

    return feature;
  }

  function createEyeFeature(mode: string, shape: any, config: { isEye: boolean, operateIdx: string | number, [key: string]: any }, id?: string, drawingStyle = state.drawingStyle) {
    const timestamp = Date.now();
    const randomId = nanoid();
    const featureId = id !== undefined ? id : `feature-${randomId}-${timestamp}`;
    const props = { name: undefined, timestamp: timestamp, isAutoFit: config.isAutoFit };

    const featureMap: Record<string, any> = {
      'POINT': () => new AILabel.Feature.Point(featureId, { ...shape, sr: 6 }, props, drawingStyle),
      'POINT-FOREGROUND': () => new AILabel.Feature.Point(featureId, { ...shape, sr: 6 }, props, drawingStyle),
      'POINT-BACKGROUND': () => new AILabel.Feature.Point(featureId, { ...shape, sr: 6 }, props, drawingStyle),
      'LINE': () => new AILabel.Feature.Line(featureId, { ...shape, width: 10 }, props, drawingStyle),
      'POLYLINE': () => new AILabel.Feature.Polyline(featureId, shape, props, drawingStyle),
      'RECT': () => new AILabel.Feature.Rect(featureId, shape, props, drawingStyle),
      'CIRCLE': () => new AILabel.Feature.Circle(featureId, shape, props, drawingStyle),
      'POLYGON': () => new AILabel.Feature.Polygon(featureId, shape, props, drawingStyle),
      'DRAWMASK': () => new AILabel.Mask.Draw(
        featureId,
        '铅笔',
        { points: shape, width: 20 },
        { name: '港币', price: '1元' },
        { strokeStyle: '#FF0000' }
      )
    };

    const feature = featureMap[mode] ? featureMap[mode]() : null;
    if (feature) {
      feature.isEye = config.isEye;
      feature.operateIdx = config.operateIdx;
      Object.keys(config).forEach(key => {
        feature[key] = config[key];
      });
    }

    return feature;
  }

  // 添加/删除要素
  function addFeature(feature: any) {
    if (featureLayer.value) {
      featureLayer.value.addFeature(feature);
      // 记录添加要素操作
      state.undoStack.push({
        type: 'addFeature',
        layer: 'feature',
        id: feature.id,
        data: feature
      });
      state.redoStack = [];
    }
  }

  function removeFeature(featureId: string) {
    const feature = featureLayer.value?.getFeatureById(featureId);
    if (feature) {
      state.undoStack.push({
        type: 'removeFeature',
        layer: 'feature',
        id: featureId,
        data: feature
      });
      featureLayer.value.removeFeatureById(featureId);
      state.redoStack = [];
    }
  }

  function selectFeature(featureId: string) {
    const feature = featureLayer.value?.getFeatureById(featureId);
    if (feature) {
      gMap.value.setActiveFeature(feature);
    }
  }

  function getActiveFeature() {
    return gMap.value?.getActiveFeature();
  }

  // 添加/删除文本
  /**
   * 创建文本标注对象
   * @param textId 文本唯一标识
   * @param textName 文本内容
   * @param textPosition 文本位置坐标 {x: number, y: number}
   * @param textStyle 文本样式，默认使用全局textStyle
   * @returns 创建的文本对象
   */
  function createText(textId: string | number, textName: string, textPosition: { x: number, y: number }, textStyle = state.textStyle) {
    return new AILabel.Text(
      textId.toString(),
      {
        text: textName,
        position: textPosition,
        offset: { x: 0, y: 0 }
      },
      { name: textName },
      textStyle
    );
  }
  /**
   * 添加文本标注到图层并记录操作历史
   * @param text 文本对象
   */
  function addText(text: any) {
    if (textLayer.value) {
      textLayer.value.addText(text);
      state.undoStack.push({
        type: 'addText',
        layer: 'text',
        id: text.id,
        data: text
      });
      state.redoStack = [];
    }
  }

  // 判断是否存在文本标注
  function hasText(textId: string) {
    return textLayer.value?.getTextById(textId) !== null;
  }

  /**
   * 从图层移除文本标注并记录操作历史
   * @param textId 文本唯一标识
   */
  function removeText(textId: string) {
    if (textLayer.value) {
      const text = textLayer.value.getTextById(textId);
      if (text) {
        state.undoStack.push({
          type: 'removeText',
          layer: 'text',
          id: textId,
          data: text
        });
        textLayer.value.removeTextById(textId);
        state.redoStack = [];
      }
    }
  }

  // 撤销/重做
  function undo() {
    if (state.undoStack.length > 0) {
      const action = state.undoStack.pop();
      if (action) {
        state.redoStack.push(action);

        if (action.type === 'addFeature') {
          featureLayer.value?.removeFeatureById(action.id);
        } else if (action.type === 'removeFeature') {
          featureLayer.value?.addFeature(action.data);
        } else if (action.type === 'addText') {
          textLayer.value?.removeTextById(action.id);
        } else if (action.type === 'removeText') {
          textLayer.value?.addText(action.data);
        }
      }
    }
  }

  function redo() {
    if (state.redoStack.length > 0) {
      const action = state.redoStack.pop();
      if (action) {
        state.undoStack.push(action);

        if (action.type === 'addFeature') {
          featureLayer.value?.addFeature(action.data);
        } else if (action.type === 'removeFeature') {
          featureLayer.value?.removeFeatureById(action.id);
        } else if (action.type === 'addText') {
          textLayer.value?.addText(action.data);
        } else if (action.type === 'removeText') {
          textLayer.value?.removeTextById(action.id);
        }
      }
    }
  }

  // 删除选中标注
  function deleteSelectedFeature() {
    const activeFeature = gMap.value?.getActiveFeature();
    if (activeFeature) {
      // 记录删除操作
      state.undoStack.push({
        type: 'removeFeature',
        layer: 'feature',
        id: activeFeature.id,
        data: activeFeature
      });
      featureLayer.value?.removeFeatureById(activeFeature.id);
      gMap.value.setActiveFeature(null);
      // 清空重做栈
      state.redoStack = [];
    }
  }

  // 清空当前所有标注
  function clearAnnotations() {
    gMap.value?.setActiveFeature(null);
    state.allFeatures = [];
    state.allTexts = [];
    featureLayer.value?.removeAllFeatures();
    textLayer.value?.removeAllTexts();
    // 清空重做栈
    state.redoStack = [];
    // 清空撤销栈
    state.undoStack = [];

    refresh();
  }

  // 获取标注图片FormData
  async function getAnnotationFormData(sonId: string, version: string | number, fileSuffix: string) {
    if (!gMap.value) {
      throw new Error('地图实例未初始化');
    }

    // 导出当前标注图层为图片 blob
    const imagedata = await gMap.value.exportLayersToImage(
      { x: 0, y: 0, width: state.currentViewWidth, height: state.currentViewHeight },
      { type: "blob", format: `image/${fileSuffix}`, layers: [imageLayer.value, textLayer.value, featureLayer.value, maskLayer.value] },
    );

    const filename = `${new Date().getTime()}.${fileSuffix}`;
    const fileBlob = new Blob([imagedata], { type: `image/${fileSuffix}` });
    const formData = new FormData();
    formData.append('file', new File([fileBlob], filename, { type: `image/${fileSuffix}` }));
    formData.append('sonId', sonId);
    formData.append('version', version);

    return formData;
  }

  // 获取标注列表
  function getAnnotations() {
    const features = featureLayer.value?.getAllFeatures().map((item: any, index: number) => {
      return {
        ...item,
        isEye: item.isEye,
        operateIdx: item.operateIdx ?? 0,
      }
    });

    return features || [];
  }

  /**
 * 根据图形数据获取文本标注位置坐标
 * @param data 图形数据对象
 * @param type 图形类型
 * @returns 坐标对象 { x, y }
 */
  function getLabelTextPosition(data: any, type: string) {
    switch (type) {
      case "LINE":
        return { x: data.start.x, y: data.start.y };

      case "POLYLINE":
        return { x: data.x, y: data.y };

      case "RECT":
      case "POINT":
      case "POINT-FOREGROUND":
      case "POINT-BACKGROUND":
      case "MARKER":
        return { x: data.x, y: data.y };

      case "POLYGON":
        const points = Array.isArray(data) ? data : data.points;
        return { x: points[0].x, y: points[0].y };

      case "CIRCLE":
        return { x: data.cx, y: data.cy - data.r };

      default:
        // 默认返回原点或安全坐标
        return { x: 0, y: 0 };
    }
  }

  // 渲染自定义标注
  function renderCustomAnnotations(annotations: any[]) {
    annotations.forEach(annotation => {
      let { shape, type, props, style: drawingStyle, textStyle } = annotation;

      // 计算宽高比例
      const widthRatio = calculateRatio(
        state.currentViewWidth,
        state.originalImageWidth,
        props.operateWidth,
      );

      const heightRatio = calculateRatio(
        state.currentViewHeight,
        state.originalImageHeight,
        props.operateHeight,
      );

      // 缩放形状
      const scaledShape = scaleShapeByRatio(shape, type, widthRatio, heightRatio);

      // 后续逻辑保持不变
      let featureShape = scaledShape;
      if (type === 'POLYGON') {
        featureShape = { points: scaledShape.points };
      }

      const feature = createEyeFeature(type, featureShape, { isEye: true, operateIdx: Date.now(), isAutoFit: props?.isAutoFit ?? false }, annotation.id, drawingStyle);
      feature.props.name = annotation.props.name;
      addFeature(feature);

      if (annotation.id) {
        const textPosition = getLabelTextPosition(scaledShape, type);
        const text = createText(
          annotation.id,
          annotation.props.name,
          textPosition,
          textStyle
        );
        addText(text);
      }
    });
    state.allFeatures = getAnnotations();
  }

  function calculateRatio(
    currentViewSize: number,  // 当前视图尺寸
    originalImageSize: number,  // 原图宽高
    operateSize?: number  // 操作宽高（可选参数）
  ): number {
    const currentWidth = !operateSize
      ? currentViewSize
      : (currentViewSize ?? originalImageSize);

    const originalWidth = operateSize ? operateSize : originalImageSize;

    // 返回比例计算结果
    return currentWidth / originalWidth;
  }

  // 根据类型对shape进行比例缩放
  function scaleShapeByRatio(shape: any, type: string, widthRatio: number, heightRatio: number): any {
    switch (type) {
      case 'POLYGON':
        const points = Array.isArray(shape) ? shape : shape.points;
        return {
          points: points.map((point: any) => ({
            x: point.x * widthRatio,
            y: point.y * heightRatio
          }))
        };
      case 'RECT':
        return {
          x: shape.x * widthRatio,
          y: shape.y * heightRatio,
          width: shape.width * widthRatio,
          height: shape.height * heightRatio
        };
      case 'CIRCLE':
        return {
          cx: shape.cx * widthRatio,
          cy: shape.cy * heightRatio,
          r: shape.r * widthRatio
        };
      case 'POINT':
        return {
          x: shape.x * widthRatio,
          y: shape.y * heightRatio,
          r: shape.r ? shape.r * widthRatio : 5 * widthRatio
        };
      default:
        return shape;
    }
  }

  /**
 * 刷新地图
 */
  function refresh() {
    gMap.value?.refresh();
  }

  /**
   * 重设地图大小
   * @param {object} size - 可选，包含width和height的尺寸对象
   */
  function resize() {
    gMap.value?.resize();
  }

  // 导出图片上护具
  async function exportImage(type: 'base64' | 'blob' = 'base64') {
    const imagedata = await gMap.value?.exportLayersToImage(
      { x: 0, y: 0, width: state.currentViewWidth, height: state.currentViewHeight },
      { type, format: 'image/png', layers: [imageLayer.value, textLayer.value, featureLayer.value, maskLayer.value] }
    );

    const imageDom = new Image();
    if (type === 'base64') {
      // 导出base64格式
      imageDom.src = imagedata;
    }
    else {
      // 导出blob格式
      const url = URL.createObjectURL(imagedata);
      imageDom.src = url;
      imageDom.onload = () => { URL.revokeObjectURL(url); }
    }

    let aLink = document.createElement('a');
    aLink.style.display = 'none';
    aLink.href = imageDom.src;
    aLink.download = 'export.png';
    // 触发点击-然后移除
    document.body.appendChild(aLink);
    aLink.click();
    document.body.removeChild(aLink);
  }

  // 将箭头函数改为普通函数形式
  function renderPointData(value: any[]) {
    return new Promise<void>(async (resolve) => {
      if (value.length === 0) {
        resolve();
        return;
      }

      // 显示加载状态
      state.isAutoLabel = false;

      // 计算宽高缩放比例（使用hooks中的计算方法）
      const widthRatio = calculateRatio(state.currentViewWidth, state.originalImageWidth, state.originalImageWidth);
      const heightRatio = calculateRatio(state.currentViewHeight, state.originalImageHeight, state.originalImageHeight);

      // 缩放点坐标
      const mapValue = value.map((val) => {
        const x = Number((val[0] * widthRatio).toFixed(2));
        const y = Number((val[1] * heightRatio).toFixed(2));
        return [x, y];
      });

      let params;
      if (state.currentMode === "POINT-FOREGROUND") {
        // 处理背景点坐标缩放
        const background_points = state.background_points.map((val: any) => {
          const x = Number((val[0] * widthRatio).toFixed(2));
          const y = Number((val[1] * heightRatio).toFixed(2));
          return [x, y];
        });
        params = {
          image_path: state.activeImageUrl,
          foreground_points: mapValue,
          background_points,
        };
      } else if (state.currentMode === "POINT-BACKGROUND") {
        // 处理前景点坐标缩放
        const foreground_points = state.foreground_points.map((val: any) => {
          const x = Number((val[0] * widthRatio).toFixed(2));
          const y = Number((val[1] * heightRatio).toFixed(2));
          return [x, y];
        });
        params = {
          image_path: state.activeImageUrl,
          foreground_points,
          background_points: mapValue,
        };
      }

      try {
        // 显示加载状态
        state.isAutoLabel = true;
        const res = await segmentStart(params);
        // const res = { data: true };
        if (res.data) {
          // 隐藏加载状态
          state.isAutoLabel = false;

          // 清除自动生成的多边形标注
          const allFeatures = getAnnotations();
          allFeatures.forEach((item: any) => {
            if (item.props?.isAutoFit) {
              removeFeature(item.id);
            }
          });

          // 清除点
          const activeFeature = getActiveFeature();
          if (activeFeature) {
            removeFeature(activeFeature.id);
          }
          refresh();

          const segmentResult: any = res.data.segmentResult;
          // const segmentResult = GroundCorrosion_17;
          const annotations = transformSegmentResults(segmentResult);
          renderCustomAnnotations(annotations);
          refresh();
        } else {
          const activeFeature = getActiveFeature();
          if (activeFeature) {
            removeFeature(activeFeature.id);
          }
          state.isAutoLabel = false;
        }
      } catch (error) {
        console.error("处理标注数据失败:", error);
        // 错误处理：清除指定ID的标注
        const activeFeature = getActiveFeature();
        if (activeFeature) {
          removeFeature(activeFeature.id);
        }
        state.isAutoLabel = false;
      } finally {
        resolve();
      }
    });
  }

  /**
   * 转换分割结果为标注系统可用的形状数据
   * @param param 包含形状数组和图像尺寸的参数对象
   * @returns 处理后的标注形状数组
   */
  function transformSegmentResults({
    shapes,
    imageWidth,
    imageHeight
  }: {
    shapes: any[];
    imageWidth: number;
    imageHeight: number;
  }) {
    // 验证输入参数
    if (!shapes || !Array.isArray(shapes) || imageWidth <= 0 || imageHeight <= 0) {
      console.warn('无效的输入参数，返回空数组');
      return [];
    }

    // 计算缩放比例（使用hooks中的计算方法）
    // const widthRatio = calculateRatio(state.currentViewWidth, imageWidth, imageWidth);
    // const heightRatio = calculateRatio(state.currentViewHeight, imageHeight, imageHeight);
    const widthRatio = 1;
    const heightRatio = 1;

    return shapes
      .map((shapeData, index) => {
        // 生成唯一ID，包含时间戳和索引确保唯一性
        const id = undefined;

        // 确定标注类型
        const isTwoArr = shapeData.points?.every((point: any) => Array.isArray(point)) ?? false;
        const type = isTwoArr
          ? (shapeData.shape_type?.toUpperCase() || "POLYGON")
          : "POINT";

        // 处理坐标点
        const points = shapeData.points || [];
        const transformedPoints = isTwoArr
          ? points.map((point: any[]) => ({
            x: Number((point[0] * widthRatio).toFixed(2)),
            y: Number((point[1] * heightRatio).toFixed(2))
          }))
          : [
            {
              x: Number((points[0] * widthRatio).toFixed(2)),
              y: Number((points[1] * heightRatio).toFixed(2))
            }
          ];

        return {
          id,
          type,
          props: {
            name: "",
            textId: id,
            deleteMarkerId: id,
            sign: state.currentMode === 'POINT-FOREGROUND' ? 'foreground' : 'background',
            isAutoFit: true,
            operateWidth: imageWidth,
            operateHeight: imageHeight
          },
          shape: { points: transformedPoints },
          style: {
            fillStyle: "#D91515",
            strokeStyle: "rgba(77, 101, 170, 0.8)", // 增加透明度以提升视觉效果
            fill: true,
            globalAlpha: 0.6,
            lineWidth: 1, // 轻微增加线宽，使边界更清晰
            stroke: true, // 启用描边，增强多边形轮廓
          }
        };
      })
      // 过滤无效多边形
      .filter(item =>
        item.type === "POLYGON" &&
        item.shape.points.length >= 3 &&
        // 确保所有点都有有效坐标
        item.shape.points.every((p: any) => !isNaN(p.x) && !isNaN(p.y))
      );
  }

  /**
 * 根据形状类型获取文本位置
 * @param {string} type - 形状类型 (CIRCLE, POLYGON, RECT等)
 * @param {Object} opShape - 形状对象
 * @returns {Object} 文本位置 {x, y}
 */
  function getTextShapePosition(type: string, opShape: any) {
    const positionMap: any = {
      CIRCLE: { x: opShape.cx, y: opShape.cy - opShape.r },
      POLYGON: {
        x: opShape.points?.[0]?.x || '',
        y: opShape.points?.[0]?.y || ''
      },
      RECT: { x: opShape.x, y: opShape.y }
    };

    // 返回对应类型的位置，默认返回空值
    return positionMap[type] || { x: '', y: '' };
  }

  // 导出公共方法和状态
  return {
    // 状态
    state,
    tools,

    // 初始化
    initAnnotation,

    // 方法
    setMode,
    getModeName,
    zoomIn,
    zoomOut,
    loadImage,
    getAnnotations,
    deleteSelectedFeature,
    undo,
    redo,
    clearAnnotations,
    getAnnotationFormData,
    createEyeFeature,
    addFeature,
    removeFeature,
    selectFeature,
    createText,
    addText,
    removeText,
    hasText,
    refresh,
    resize,
    renderCustomAnnotations, // 导出新增方法
    getActiveFeature,
    getLabelTextPosition,
    exportImage,

    calculateRatio,
    scaleShapeByRatio,

    // 内部引用 - 谨慎使用
    _gMap: gMap,
    _featureLayer: featureLayer,
    _textLayer: textLayer,
    _state: state
  };
}
