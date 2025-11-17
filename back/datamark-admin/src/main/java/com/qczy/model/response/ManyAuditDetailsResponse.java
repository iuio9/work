package com.qczy.model.response;

import com.fasterxml.jackson.annotation.JsonFormat;
import lombok.Data;
import org.springframework.format.annotation.DateTimeFormat;

import java.util.Date;

/**
 * @Author: hh
 * @Version: 1.0
 * @Date: 2025/3/17 17:04
 * @Description:
 */
@Data
public class ManyAuditDetailsResponse {

    /**
     *  id
     */
    private Integer id;

    /**
     *  标注员
     */
    private String markName;

    /**
     *  审核员
     */
    private String auditName;

    /**
     *  审核数量    未审核/已审核
     */
    private String auditNum;

    /**
     *  进度
     */
    private String progress;

    /**
     *  审核状态
     */
    private String auditState;

    /**
     *  创建时间
     */
    @DateTimeFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss", timezone = "GMT+8")
    private Date createTime;

    /**
     *  是否隐藏按钮  0:正常 、 1：隐藏
     */
    private Integer isHide;
}
