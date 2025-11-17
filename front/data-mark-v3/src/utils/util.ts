export function formattedNumber(num: string) {
  return num.endsWith(".00") ? Number(num.slice(0, -3)) : Number(num);
}

export function computeSize(size: number): string {
  const num = 1024.0;
  if (size < num) return `${size} B`;
  if (size < num ** 2) return `${formattedNumber((size / num).toFixed(2))} KB`;
  if (size < num ** 3)
    return `${formattedNumber((size / num ** 2).toFixed(2))} MB`;
  if (size < num ** 4)
    return `${formattedNumber((size / num ** 3).toFixed(2))} GB`;
  return `${formattedNumber((size / num ** 4).toFixed(2))} TB`;
}


/**
 * 定义 DownloadFileParams 接口，用于描述请求参数和 headers 的结构
 */
interface DownloadFileParams {
  url: string; // 请求地址
  params?: Record<string, any>; // GET 参数
  headers?: Record<string, string>; // 自定义请求头
}

/**
 * 使用 fetch 下载文件（Blob 流），自动解析后端返回的文件名
 * @param {DownloadFileParams} options - 包含 url、params 和 headers 的对象
 */
export async function downloadFile({ url, params = {}, headers = {} }: DownloadFileParams): Promise<void> {
  try {
    // 拼接 GET 参数
    const paramsStr = new URLSearchParams(params).toString();
    const requestUrl = paramsStr ? `${url}?${paramsStr}` : url;

    // 发起 fetch 请求
    const response = await fetch(requestUrl, {
      method: 'GET',
      headers: {
        ...headers
      }
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    // 提取文件名
    const disposition = response.headers.get('Content-Disposition');
    let filename = 'downloaded-file';

    if (disposition && disposition.includes('filename=')) {
      const fileNameMatch = disposition.match(/filename="?([^"]+)"?/);
      if (fileNameMatch?.length > 1) {
        try {
          // 尝试解码 UTF-8 编码的文件名
          filename = decodeURIComponent(fileNameMatch[1]);
        } catch (e) {
          // 兼容旧浏览器或 ISO-8859-1 编码的文件名
          filename = unescape(fileNameMatch[1]);
        }
      }
    }

    // 获取 Blob 数据并触发下载
    const blob = await response.blob();
    const downloadUrl = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = downloadUrl;
    a.download = filename;
    document.body.appendChild(a);
    a.click();

    // 清理资源
    a.remove();
    window.URL.revokeObjectURL(downloadUrl);
  } catch (error) {
    console.error('文件下载失败:', error);
    throw error;
  }
}


import axios, { AxiosRequestConfig, AxiosResponse } from 'axios';

interface DownloadFileParams {
  url: string;
  params?: Record<string, any>;
  headers?: Record<string, string>;
}

export async function downloadFile1({ url, params = {}, headers = {} }: DownloadFileParams): Promise<void> {
  try {
    // 配置 Axios 请求
    const config: AxiosRequestConfig = {
      url,
      method: 'GET',
      params, // 自动拼接 GET 参数
      headers,
      responseType: 'blob' // 必须声明响应类型为 blob
    };

    const response: AxiosResponse<Blob> = await axios(config);
    console.log('response: ', response);

    if (response.status !== 200) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    // 提取文件名
    const disposition = response.headers['content-disposition'];
    console.log('disposition: ', disposition);
    let filename = 'downloaded-file';

    if (disposition && typeof disposition === 'string' && disposition.includes('filename=')) {
      const fileNameMatch = disposition.match(/filename="?([^"]+)"?/);
      if (fileNameMatch?.[1]) {
        try {
          // 尝试解码 UTF-8 编码的文件名
          filename = decodeURIComponent(fileNameMatch[1]);
        } catch (e) {
          // 兼容旧浏览器或 ISO-8859-1 编码的文件名
          filename = unescape(fileNameMatch[1]);
        }
      }
    }

    // 创建 Blob URL 并触发下载
    const blob = new Blob([response.data]);
    const downloadUrl = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = downloadUrl;
    a.download = filename;
    document.body.appendChild(a);
    a.click();

    // 清理资源
    a.remove();
    window.URL.revokeObjectURL(downloadUrl);
  } catch (error: any) {
    console.error('文件下载失败:', error);
    throw error;
  }
}
