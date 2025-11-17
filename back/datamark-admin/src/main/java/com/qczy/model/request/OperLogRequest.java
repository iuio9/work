package com.qczy.model.request;

import com.fasterxml.jackson.annotation.JsonFormat;
import lombok.Data;
import org.springframework.format.annotation.DateTimeFormat;

import java.util.Date;

/**
 * @Author: hh
 * @Version: 1.0
 * @Date: 2024/8/5 15:57
 * @Description:
 */
@Data
public class OperLogRequest {




    /**
     * 操作模块
     */
    private String title;


    /**
     * 操作人员
     */
    private String operName;


    /**
     * 业务类型（0其它 1新增 2修改 3删除）
     */
    private Integer businessType;


    /**
     * 操作状态（0正常 1异常）
     */
    private Integer status;



    /**
     * 操作时间 开始
     */

    private String operTimeStart;



    /**
     * 操作时间 开始
     */

    private String operTimeEnd;

}
