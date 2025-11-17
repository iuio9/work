package com.qczy.model.request;

import lombok.Data;

/**
 * @Author: hh
 * @Version: 1.0
 * @Date: 2025/4/28 9:52
 * @Description:
 */
@Data
public class DownloadRequest {

    // 数据集id
    private String sonId;
    // 类型
    private Integer type;
    // json 或者 xml
    private Integer anoType;

    // 任务id
    private Integer id;


}
