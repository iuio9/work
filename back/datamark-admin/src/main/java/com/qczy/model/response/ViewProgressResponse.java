package com.qczy.model.response;

import com.fasterxml.jackson.annotation.JsonFormat;
import lombok.Data;
import org.springframework.format.annotation.DateTimeFormat;

import java.util.Date;

/**
 * @Author: hh
 * @Version: 1.0
 * @Date: 2025/2/27 14:34
 * @Description:
 */
@Data
public class ViewProgressResponse {

    /**
     * 序号
     */
    private Integer id;


    /**
     * 标注员
     */
    private String nickName;


    /**
     * 已标注/待标注
     */
    private String markNum;


    /**
     * 标注进度
     */
    private String progress;

    /**
     * 标注状态
     */
    private String markState;


    /**
     *  创建时间
     */
    @DateTimeFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss", timezone = "GMT+8")
    private Date createTime;

    /**
     *  用户id
     */
    private Integer userId;


    /**
     *  是否隐藏按钮  0:正常 、 1：隐藏
     */
    private Integer isHide;
}
