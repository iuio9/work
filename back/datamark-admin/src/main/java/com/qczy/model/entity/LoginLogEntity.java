package com.qczy.model.entity;

import com.baomidou.mybatisplus.annotation.TableField;
import com.baomidou.mybatisplus.annotation.TableName;
import com.fasterxml.jackson.annotation.JsonFormat;
import lombok.Data;
import org.springframework.format.annotation.DateTimeFormat;

import java.util.Date;

/**
 * @Author: hh
 * @Version: 1.0
 * @Date: 2024/8/5 16:44
 * @Description:
 */
@TableName("qczy_login_log")
@Data
public class LoginLogEntity {


    /**
     * 主键
     */
    private Integer id;

    /**
     * 登录账号
     */
    private String loginName;

    /**
     * 登录IP地址
     */
    private String ipaddr;

    /**
     * 浏览器
     */
    private String browser;

    /**
     *  操作系统
     */
    private String os;

    /**
     *  访问时间
     */
    @DateTimeFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss", timezone = "GMT+8")
    private Date loginTime;

    // 前端额外传入的字段（这个字段没用）
    @TableField(exist = false)
    private Integer index;
}
