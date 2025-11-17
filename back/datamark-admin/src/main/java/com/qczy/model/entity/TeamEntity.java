package com.qczy.model.entity;


import com.baomidou.mybatisplus.annotation.TableField;
import com.baomidou.mybatisplus.annotation.TableName;
import com.fasterxml.jackson.annotation.JsonFormat;
import com.qczy.model.request.AddTeamUser;
import io.swagger.annotations.ApiModelProperty;
import lombok.Data;
import org.hibernate.validator.constraints.Length;
import org.springframework.format.annotation.DateTimeFormat;

import javax.validation.constraints.Size;
import java.io.Serializable;
import java.util.Date;
import java.util.List;

/**
 * @Author: hh
 * @Version: 1.0
 * @Date: 2025/2/24 9:48
 * @Description: 团队表
 */
@Data
@TableName("qczy_team")
public class TeamEntity implements Serializable {


    /**
     * 自增id
     */
    @ApiModelProperty("自增id")
    private Integer id;
    /**
     * 团队名称
     */
    @Size(max = 200, message = "编码长度不能超过200")
    @ApiModelProperty("团队名称")
    @Length(max = 200, message = "编码长度不能超过200")
    private String teamName;
    /**
     * 团队描述
     */
    @Size(max = 500, message = "编码长度不能超过500")
    @ApiModelProperty("团队描述")
    @Length(max = 500, message = "编码长度不能超过500")
    private String teamDec;
    /**
     * 创建者id
     */
    @ApiModelProperty("创建者id")
    private Integer creator;
    /**
     * 创建时间
     */
    @ApiModelProperty("创建时间")
    @DateTimeFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss", timezone = "GMT+8")
    private Date createTime;
    /**
     * 修改时间
     */
    @ApiModelProperty("修改时间")
    @DateTimeFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss", timezone = "GMT+8")
    private Date updateTime;

    // 团队类型
    private Integer teamType;

    @TableField(exist = false)
    // 用户列表
    private List<AddTeamUser> userList;

    @TableField(exist = false)
    // 团队总数量
    private Integer teamCount;


}
