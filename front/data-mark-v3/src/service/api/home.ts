import { request } from "../request";

// ano查看
export function getComputerInfo(params?: any) {
  return request<any>({
    url: "/home/getComputerInfo",
    method: "get",
    params,
  });
}

// getDaysComputerInfo
export function getDaysComputerInfo(params?: any) {
  return request<any>({
    url: "/home/getDaysComputerInfo",
    method: "get",
    params,
  });
}
