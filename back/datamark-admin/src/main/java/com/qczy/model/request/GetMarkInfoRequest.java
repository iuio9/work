package com.qczy.model.request;

import lombok.Data;

/**
 * @Author: hh
 * @Version: 1.0
 * @Date: 2024/8/24 15:01
 * @Description:
 */
@Data
public class GetMarkInfoRequest {


    // 数据集组id
    private String groupId;

    // 数据集id
    private String sonId;

    // 版本号
    private String version;

    // 转义字符   \\ /
    private String  escaping;

    // 文件id
    private String fileIds;

    // 文件访问地址
    private String accessAddress;

}
