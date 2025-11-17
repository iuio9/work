package com.qczy.model.response;

import lombok.Data;

/**
 * @Author: hh
 * @Version: 1.0
 * @Date: 2025/3/19 14:46
 * @Description:
 */
@Data
public class ExamineTeamInfoResponse {

    /**
     *  数据集id
     */
    private String sonId;

    /**
     *  任务总量 (文件总量)
     */
    private String fileNumber;

    /**
     *  任务名称
     */
    private String taskName;

    /**
     *  标注团队名称
     */
    private String markTeamName;

    /**
     *  标注团队人数
     */
    private Integer markTeamNumber;


    /**
     *  审核团队名称
     */
    private String auditTeamName;

    /**
     *  审核团队人数
     */
    private Integer auditTeamNumber;

}
