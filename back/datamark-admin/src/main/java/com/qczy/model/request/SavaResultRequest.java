package com.qczy.model.request;

import lombok.Data;

@Data
public class SavaResultRequest {


    /**
     * 任务id
     */
    private Integer taskId;

    /**
     * 保存类型   1:保存到当前版本  2：保存新版本
     */
    private Integer type;

    /**
     * 数据集类型
     */
    private Integer dataTypeId;

    /**
     * 数据集名称
     */
    private String groupName;

    /**
     *  标注类型
     */
    private Integer anoType;


    /**
     * 文件id
     */
    private String fileIds;

}
