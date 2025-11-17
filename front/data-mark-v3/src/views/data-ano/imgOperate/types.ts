// types.ts
export interface Tag {
  id: number;
  name: string;
  color: string;
}

export interface Annotation {
  id: string;
  name: string;
  type: string;
  shape: any;
}

export interface Tool {
  mode: string;
  icon: string;
  label: string;
  [key: string]: any;
}

// 标注形状样式接口
export interface MarkStyle {
  opacity: number;
  fillStyle: string;
  lineWidth: number;
  strokeStyle: string;
  fill: boolean;
  globalAlpha: number;
}

// 标注形状接口
export interface MarkShape {
  x?: number;
  y?: number;
  width?: number;
  height?: number;
  cx?: number;
  cy?: number;
  r?: number;
  points?: Array<{ x: number; y: number }>;
}

// 标注属性接口
export interface MarkProps {
  name: string;
  textId: string;
  deleteMarkerId: string;
  operateWidth: number;
  operateHeight: number;
}

// 标注信息项接口
export interface MarkInfoItem {
  openId: string;
  id: string;
  type: 'RECT' | 'CIRCLE' | 'POLYGON' | string;
  props: MarkProps;
  shape: MarkShape;
  style: MarkStyle;
  isEye: boolean;
  operateIdx: number;
}

// 标签形状接口
export interface LabelShape {
  label: string;
  points: number[][];
  shape_type: string;
}

// 标签标注信息接口
export interface LabelMarkInfo {
  imageWidth: number;
  imagePath: string;
  imageHeight: number;
  shapes: LabelShape[];
}

// 图片数据主接口
export interface ImageData {
  id: number | null;
  sonId: string;
  version: number;
  fileId: number;
  markFileId: number | null;
  imgPath: string;
  previewImgPath: string;
  isMark: string;
  labels: string;
  isInvalid: number;
  markInfo: string; // JSON字符串，解析后为MarkInfoItem[]
  labelMarkInfo: string | null; // JSON字符串，解析后为LabelMarkInfo
  width: number;
  height: number;
  operateWidth: number | null;
  operateHeight: number | null;
  fileName: string | null;
  notPassMessage: string | null;
}

// 图片列表类型
export type ImageList = ImageData[];

// 标签接口定义（替代原Tag接口）
export interface Label {
  labelId: number;
  onlyId: string;
  labelName: string;
  englishLabelName: string;
  labelColor: string;
  labelCount: number;
  labelGroupId: number;
  twoLabelName: string;
  labelSort: number;
}
export interface AnnotationItem {
  id: string | number;
  textId: string | number;
  isEye: boolean;
  operateIdx: number;
  props: {
    textId: string | number;
    operateWidth?: number;
    operateHeight?: number;
    [key: string]: any; // 允许其他属性
  };
  shape: string;
  style: Record<string, any>;
  type: string;
}

export interface ValidationResult {
  isValid: boolean;
  hasChanges: boolean;
  hasEmptyLabels: boolean;
}

export interface SaveResult {
  success: boolean;
  hasChanges: boolean;
  [key: string]: any;
}

export interface CurrentImage {
  imgPath?: string;
  sonId: string | number;
  version: string | number;
  fileId: string | number;
  [key: string]: any; // 允许其他属性
}

// 定义标签数量类型
export interface TabNumbers {
  all: number;
  haveAno: number;
  noAno: number;
  invalid: number;
}

// 定义标签配置类型
export interface TabConfig {
  tabNum: TabNumbers;
}

// 定义标签页项类型
export interface TabPane {
  name: string;
  tab: string;
  label: string;
}

// 定义路由参数中anoType的可能值
export type AnoType = 'validateUser' | 'online' | 'setOnline' | 'validate' | 'audit' | 'result' | undefined;
