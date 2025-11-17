package com.qczy.model.entity;

import com.baomidou.mybatisplus.annotation.FieldFill;
import com.baomidou.mybatisplus.annotation.TableField;
import com.baomidou.mybatisplus.annotation.TableName;
import com.fasterxml.jackson.annotation.JsonFormat;
import io.swagger.annotations.ApiModelProperty;
import lombok.Data;
import org.springframework.format.annotation.DateTimeFormat;

import javax.validation.constraints.*;
import java.io.Serializable;
import java.util.Date;

/**
 * @Author: hh
 * @Version: 1.0
 * @Date: 2024/8/6 14:39
 * @Description:
 */
@Data
@TableName("qczy_dept")
public class DeptEntity implements Serializable {


    /** 自增id */
    private Integer id;

    /** 部门名称 */
    @NotBlank(message = "部门名称不能为空")
    @Size(min = 1, max = 20, message = "部门名称长度必须在 1 到 20 个字符之间")
    private String deptName;

    /** 排序 */
    @Min(value = 1, message = "排序不能小于 0")
    private Integer sort;

    /** 负责人 */
    @NotBlank(message = "负责人不能为空")
    @Size(min = 1, max = 20, message = "部门长度必须在 1 到 20 个字符之间")
    private String supt;

    /** 联系电话 */
    @Pattern(regexp = "^1[3-9]\\d{9}$", message = "手机号码格式不正确")
    private String telePhone;

    /** 邮箱 */
    @Email(message = "邮箱格式不正确")
    private String email;

    /** 状态: 1-> 正常、 2->停用 */
    private Integer status;

    /** 是否允许删除：1-> 不允许 、 2->允许 */
    private Integer isAllowDeletion;

    @ApiModelProperty(value = "创建时间")
    @DateTimeFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss", timezone = "GMT+8")
    @TableField(fill = FieldFill.INSERT)
    private Date createTime;


    @ApiModelProperty(value = "更新时间")
    @DateTimeFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss", timezone = "GMT+8")
    @TableField(fill = FieldFill.INSERT_UPDATE)
    private Date updateTime;



    // 前端额外传入的字段（这个字段没用）
    @TableField(exist = false)
    private Integer index;
}
