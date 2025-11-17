import _ from 'lodash';

// 定义文件对象的接口类型，包含常见的文件属性及自定义的状态和错误信息属性
interface FileObject {
  type: string;
  size: number;
  status?: string;
  errMessage?: string;
}

// 辅助函数，用于校验文件扩展名是否在允许的列表中
const isExtensionAllowed = (file: FileObject, allowedExtensions: string[]): boolean => {
  return allowedExtensions.includes(file.type);
};

// 文件校验，针对图片类型的文件校验（如限制大小、格式等）
const fileValidation = (file: FileObject): boolean => {
  const allowedImageExtensions = ['image/jpg', 'image/jpeg', 'image/png', 'image/webp', 'image/bmp'];
  const isImage = file.type.startsWith('image/');
  if (!isImage) {
    file.status = 'error';
    file.errMessage = '只能上传图片';
    return false;
  }
  if (!isExtensionAllowed(file, allowedImageExtensions)) {
    file.status = 'error';
    file.errMessage = '只能上传jpg、jpeg、png、webp或bmp格式的图片';
    return false;
  }
  const MAX_FILE_SIZE = 20 * 1024 * 1024; // 20MB
  if (file.size > MAX_FILE_SIZE) {
    file.status = 'error';
    file.errMessage = '文件大小不能超过 20MB';
    console.error('文件大小不能超过 20MB');
    return false;
  }
  return true;
};

// 文件校验，针对图片及标注信息（json、xml等格式）的文件校验
const fileValidationForImageAndAnnotation = (file: FileObject): boolean => {
  const allowedExtensions = ['jpg', 'jpeg', 'png', 'webp', 'bmp', 'json', 'xml'];
  if (!isExtensionAllowed(file, allowedExtensions)) {
    file.status = 'error';
    file.errMessage = '只能上传json、xml、jpg、jpeg、png、webp或bmp格式的文件';
    return false;
  }
  return true;
};

addEventListener('message', async (e: any) => {
  const { files, markStatus } = e.data;
  // _.uniqBy(files, 'name')
  const uploaderFiles = files.map((item: any, index) => {
    const postfix = item.name.split('.')[1].toLowerCase();
    item.sortIdx = index + 1;
    item.isHover = false;
    item.errMessage = '暂未上传，请点击下方确认按钮开始上传！';
    if (markStatus === '0') {
      item.type = `image/${postfix}`;
      fileValidation(item);
    }
    if (markStatus === '1') {
      item.lastModified = item.raw.lastModified;
      item.type = item.name.split('.')[1].toLowerCase();
      item.prevfix = item.name.split('.')[0];
      fileValidationForImageAndAnnotation(item);
    }
    return item;
  });
  postMessage({ uploaderFiles });
});
