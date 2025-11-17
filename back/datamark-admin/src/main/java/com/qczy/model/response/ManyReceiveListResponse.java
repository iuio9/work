package com.qczy.model.response;

import com.fasterxml.jackson.annotation.JsonFormat;
import lombok.Data;
import org.springframework.format.annotation.DateTimeFormat;

import java.util.Date;

/**
 * @Author: hh
 * @Version: 1.0
 * @Date: 2025/2/28 10:55
 * @Description:
 */
@Data
public class ManyReceiveListResponse {

    /**
     * id
     */
    private Integer id;

    /**
     *  任务id
     */
    private Integer taskId;


    /**
     * 任务名称
     */
    private String taskName;

    /**
     *  标注员名称
     */
    private String markName;

    /**
     * 标注进度
     */
    private String progress;


    /**
     *  审核数量    未审核/已审核
     */
    private String auditNum;


    /**
     * 标注状态
     */
    private String markState;

    /**
     *  任务创建者
     */
    private String creator;

    /**
     *  创建时间
     */
    @DateTimeFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss", timezone = "GMT+8")
    private Date createTime;

    /**
     *  数据集id
     */
    private String sonId;

    /**
     *  是否隐藏按钮  0:正常 、 1：隐藏
     */
    private Integer isHide;


    /**
     *  用户id
     */
    private Integer userId;




}
